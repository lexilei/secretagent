"""Iterative ptool induction pipeline.

Discovers reasoning primitives by:
1. Running ReAct on MUSR murder mysteries (or loading existing traces)
2. Analyzing traces to find frequently used action patterns
3. Merging synonym categories, selecting the top uncaptured pattern
4. Synthesizing that pattern into a formal ptool (named structured prompt)
5. Re-running ReAct with the new ptool available
6. Repeating until convergence

Usage:
    source .env && export TOGETHER_API_KEY
    uv run python experiments/induction/iterate.py run --n-cases 75 --n-iters 5
    uv run python experiments/induction/iterate.py run --n-cases 10 --n-iters 2  # smoke test
"""

import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / 'src'))
sys.path.insert(0, str(_PROJECT_ROOT / 'benchmarks' / 'musr'))

import typer
from litellm import completion, completion_cost
from secretagent import config
from secretagent.llm_util import llm as llm_cached

# Import reusable pieces from existing code (not modifying them)
from run_react import load_murder_cases, MAX_STEPS

ITER_DIR = Path(__file__).parent / 'iterations'


# ═══════════════════════════════════════════════════════════════
# Ptool data management
# ═══════════════════════════════════════════════════════════════

def load_ptools(ptools_dir: Path) -> list[dict]:
    """Load all ptool JSON specs from a directory."""
    if not ptools_dir.exists():
        return []
    ptools = []
    for f in sorted(ptools_dir.glob('ptool_*.json')):
        with open(f) as fh:
            ptools.append(json.load(fh))
    return ptools


def save_ptool(ptool: dict, ptools_dir: Path):
    """Save a ptool spec to JSON."""
    ptools_dir.mkdir(parents=True, exist_ok=True)
    path = ptools_dir / f'{ptool["id"]}.json'
    with open(path, 'w') as f:
        json.dump(ptool, f, indent=2)
    print(f'  Saved ptool to {path}')


def format_ptool_actions(ptools: list[dict]) -> str:
    """Format ptools as action descriptions for the ReAct system prompt."""
    lines = []
    for i, p in enumerate(ptools, start=2):
        lines.append(
            f'({i}) {p["display_name"]}[focus], which {p["description"]}. '
            f'The system will read the narrative and respond with structured analysis.'
        )
    return '\n'.join(lines)


# ═══════════════════════════════════════════════════════════════
# Modified ReAct loop (ptool-aware)
# ═══════════════════════════════════════════════════════════════

def _llm_with_stop(prompt: str, model: str, stop: list[str]) -> tuple[str, dict]:
    """Call LLM with stop tokens (direct litellm, no cache)."""
    messages = [dict(role='user', content=prompt)]
    timeout = config.get('llm.timeout', 300)
    start_time = time.time()
    response = completion(
        model=model, messages=messages, stop=stop,
        max_tokens=512, timeout=timeout, temperature=0,
    )
    latency = time.time() - start_time
    text = response.choices[0].message.content or ''
    try:
        cost = completion_cost(completion_response=response)
    except Exception:
        cost = 0.0
    stats = dict(
        input_tokens=response.usage.prompt_tokens,
        output_tokens=response.usage.completion_tokens,
        latency=latency, cost=cost,
    )
    return text, stats


def _execute_action(action_arg: str, narrative: str, model: str,
                    ptool: dict | None = None) -> tuple[str, dict]:
    """Environment: read narrative and answer the agent's request.

    If ptool is provided, use its structured description to enhance the request.
    Otherwise, use the generic prompt (same as original run_react.py).
    """
    if ptool:
        request = f'{ptool["description"]}. Focus: {action_arg}'
    else:
        request = action_arg

    env_prompt = (
        f"You are a careful reader assisting a detective. Read the narrative below "
        f"and respond to this request:\n\n"
        f"Request: {request}\n\n"
        f"Narrative:\n{narrative}\n\n"
        f"Respond concisely with only information directly stated in or inferable "
        f"from the narrative. Do not speculate beyond what the text supports."
    )
    response, stats = llm_cached(env_prompt, model)
    if len(response) > 1500:
        response = response[:1500] + '...'
    return response.strip(), stats


def _make_system_prompt(ptools: list[dict]) -> str:
    """Build the system prompt with ptool descriptions."""
    base = (
        "Solve a murder mystery question with interleaving Thought, Action, Observation steps. "
        "Thought can reason about the current situation. Action can be:\n"
        "(1) Do[description], which performs any reasoning step you describe — "
        "the system will read the narrative and return the relevant information. "
        "You decide what to investigate.\n"
    )
    if ptools:
        base += format_ptool_actions(ptools) + '\n'
        finish_num = len(ptools) + 2
        base += (
            f'({finish_num}) Finish[answer], which returns the 0-based index of '
            f'the correct answer choice and finishes the task.\n\n'
            f'IMPORTANT: Prefer using the specialized tools above over Do[] when '
            f'they match your reasoning need. Use Do[] only for steps not covered '
            f'by a specialized tool.\n'
        )
    else:
        base += (
            '(2) Finish[answer], which returns the 0-based index of the correct '
            'answer choice and finishes the task.\n'
        )
    base += 'Here are some examples.\n'
    return base


FEW_SHOT = """\
Question: In a small town, a baker named Tom was found dead in his shop. \
The suspects are: Alice (his business partner), Bob (a disgruntled employee), \
and Carol (his ex-wife). Alice claims she was at home all evening. Bob says he \
was at a bar with friends. Carol says she was visiting her sister in another city. \
A witness saw someone matching Bob's description near the shop at 10pm. \
Tom's financial records show Alice would inherit the business. \
A kitchen knife from the shop was the murder weapon, with no fingerprints. \
The choices are: (0) Alice, (1) Bob, (2) Carol.
Thought 1: I need to understand who the suspects are and what evidence exists.
Action 1: Do[identify all suspects, their relationship to the victim, and any motives mentioned]
Observation 1: Three suspects: Alice is Tom's business partner and would inherit the business (financial motive). Bob is a disgruntled employee (revenge/grudge motive). Carol is Tom's ex-wife (personal motive). Tom's financial records confirm Alice stands to gain the most financially.
Thought 2: Now I know the suspects. I should check their alibis against the physical evidence.
Action 2: Do[check each suspect's alibi and whether any evidence contradicts them]
Observation 2: Alice claims she was at home all evening — no witnesses confirm or deny. Bob claims he was at a bar with friends — but a witness saw someone matching Bob's description near the shop at 10pm, contradicting his alibi. Carol claims she was visiting her sister in another city — unverified but no contradicting evidence.
Thought 3: Bob's alibi is contradicted by a witness. He had access to shop knives as an employee. The wiped fingerprints show premeditation. Bob is the most likely suspect.
Action 3: Finish[1]

"""


def _parse_action(text: str) -> tuple[str, str]:
    """Parse 'ToolName[arg]' from action text. Returns (name_lower, arg)."""
    match = re.match(r'\s*(\w+)\[(.+)\]\s*$', text.strip(), re.DOTALL)
    if match:
        return match.group(1).lower(), match.group(2).strip()
    return '', text.strip()


def _find_ptool(action_name: str, ptools: list[dict]) -> dict | None:
    """Find a ptool by matching the action name (case-insensitive)."""
    for p in ptools:
        if p['display_name'].lower() == action_name:
            return p
    return None


def run_react_on_case(narrative: str, question: str, choices: list,
                      ptools: list[dict], model: str) -> dict:
    """Run ReAct loop with optional ptools on a single case."""
    system = _make_system_prompt(ptools)
    choices_str = ', '.join(f'({i}) {c}' for i, c in enumerate(choices))
    prompt = (
        system + FEW_SHOT
        + f'Question: {narrative}\n{question}\nThe choices are: {choices_str}.\n'
    )

    steps = []
    total_stats = {'input_tokens': 0, 'output_tokens': 0, 'latency': 0, 'cost': 0}

    def accum(s):
        for k in total_stats:
            total_stats[k] += s.get(k, 0)

    for i in range(1, MAX_STEPS + 1):
        llm_input = prompt + f'Thought {i}:'
        stop = [f'\nObservation {i}:']
        response, stats = _llm_with_stop(llm_input, model, stop=stop)
        accum(stats)

        response = response.strip()
        action_match = re.search(r'Action\s*\d+\s*:\s*(.+)', response, re.DOTALL)
        if action_match:
            thought = response[:action_match.start()].strip()
            action_str = action_match.group(1).strip()
        else:
            thought = response
            action_str = ''

        action_name, action_arg = _parse_action(action_str)

        # Finish
        if action_name == 'finish':
            try:
                answer = int(action_arg.strip())
            except ValueError:
                answer = -1
            steps.append(dict(
                step=i, thought=thought, action=action_str,
                action_type='finish', action_arg=action_arg,
                observation=None, ptool_used=None,
            ))
            return dict(
                answer=answer, steps=steps, n_steps=i,
                termination='finish', stats=total_stats,
            )

        # Check if it's a ptool action
        ptool = _find_ptool(action_name, ptools) if action_name != 'do' else None

        if ptool:
            observation, env_stats = _execute_action(action_arg, narrative, model, ptool=ptool)
            accum(env_stats)
            steps.append(dict(
                step=i, thought=thought, action=action_str,
                action_type='ptool', action_arg=action_arg,
                observation=observation, ptool_used=ptool['id'],
            ))
        elif action_name == 'do' and action_arg:
            observation, env_stats = _execute_action(action_arg, narrative, model)
            accum(env_stats)
            steps.append(dict(
                step=i, thought=thought, action=action_str,
                action_type='do', action_arg=action_arg,
                observation=observation, ptool_used=None,
            ))
        elif action_str:
            observation, env_stats = _execute_action(action_str, narrative, model)
            accum(env_stats)
            steps.append(dict(
                step=i, thought=thought, action=action_str,
                action_type='do', action_arg=action_str,
                observation=observation, ptool_used=None,
            ))
        else:
            observation = 'No action provided. Please use Do[description] or Finish[answer_index].'
            steps.append(dict(
                step=i, thought=thought, action=action_str,
                action_type='none', action_arg='',
                observation=observation, ptool_used=None,
            ))

        prompt += (
            f'Thought {i}: {thought}\n'
            f'Action {i}: {action_str}\n'
            f'Observation {i}: {observation}\n'
        )

    return dict(
        answer=-1, steps=steps, n_steps=MAX_STEPS,
        termination='max_steps', stats=total_stats,
    )


# ═══════════════════════════════════════════════════════════════
# Pattern analysis (from traces)
# ═══════════════════════════════════════════════════════════════

def extract_free_actions(traces: list[dict]) -> list[dict]:
    """Extract all free-form Do[] actions (not ptool actions) from traces."""
    actions = []
    for t in traces:
        for s in t['steps']:
            if s.get('action_type') == 'do' and s.get('action_arg', '').strip():
                actions.append(dict(
                    case=t['case_name'], step=s['step'],
                    text=s['action_arg'].strip(),
                ))
    return actions


def categorize_actions(actions: list[dict], model: str,
                       batch_size: int = 30) -> list[tuple[str, int, list]]:
    """Categorize free-form actions via LLM, then merge synonyms.

    Returns: list of (category_name, count, example_items) sorted by count desc.
    """
    if not actions:
        return []

    # Step 1: LLM categorization (batched)
    cat_map = {}
    for batch_start in range(0, len(actions), batch_size):
        batch = actions[batch_start:batch_start + batch_size]
        items_text = ''
        for i, item in enumerate(batch):
            idx = batch_start + i
            text = item['text'][:300]
            items_text += f'\n[{idx}] ({item["case"]}, step {item["step"]}): {text}\n'

        prompt = f"""You are analyzing actions from an AI agent solving murder mystery puzzles.
Below are Do[] action descriptions the agent used. Categorize each into a short,
reusable REASONING ACTION TYPE (3-6 words max).

Rules:
- Use consistent, canonical names (merge synonyms into one category)
- Categories should be general enough to appear across different cases
- Focus on WHAT the agent is doing, not case-specific details
- Output ONLY a JSON array with "index" and "category" fields

Actions to categorize:
{items_text}

<answer>
[{{"index": {batch_start}, "category": "your category"}}, ...]
</answer>"""

        response, _ = llm_cached(prompt, model)
        json_str = None
        match = re.search(r'<answer>(.*)</answer>', response, re.DOTALL)
        if match:
            json_str = match.group(1).strip()
        if not json_str:
            match = re.search(r'\[.*\]', response, re.DOTALL)
            if match:
                json_str = match.group(0)
        if json_str:
            try:
                for c in json.loads(json_str):
                    cat_map[c['index']] = c['category']
            except (json.JSONDecodeError, KeyError):
                pass

    # Attach categories
    for i, item in enumerate(actions):
        item['category'] = cat_map.get(i, 'unknown')

    # Count raw frequencies
    freq = Counter(item['category'] for item in actions)

    # Step 2: Merge synonym categories via LLM
    categories = [cat for cat, _ in freq.most_common()]
    if len(categories) <= 5:
        # Few enough categories, no merging needed
        merged = freq
    else:
        merge_prompt = f"""Below are {len(categories)} category names from analyzing reasoning actions.
Many are synonyms. Merge them into 5-10 canonical groups.

Categories:
{json.dumps(categories, indent=2)}

Output a JSON object mapping each original category to its canonical group name.
Use short, clear names (3-6 words).

<answer>
{{"original category": "canonical group", ...}}
</answer>"""

        response, _ = llm_cached(merge_prompt, model)
        merge_map = {}
        match = re.search(r'<answer>(.*)</answer>', response, re.DOTALL)
        if match:
            try:
                merge_map = json.loads(match.group(1).strip())
            except json.JSONDecodeError:
                pass
        if not merge_map:
            match = re.search(r'\{.*\}', response, re.DOTALL)
            if match:
                try:
                    merge_map = json.loads(match.group(0))
                except json.JSONDecodeError:
                    pass

        # Apply merge map
        if merge_map:
            for item in actions:
                item['merged_category'] = merge_map.get(
                    item['category'], item['category'])
            merged = Counter(item.get('merged_category', item['category'])
                             for item in actions)
        else:
            print('  WARNING: synonym merging failed, using raw categories')
            merged = freq

    # Build result: (category, count, examples)
    result = []
    for cat, count in merged.most_common():
        examples = [a for a in actions
                    if a.get('merged_category', a['category']) == cat]
        result.append((cat, count, examples))
    return result


def select_top_pattern(categories: list[tuple[str, int, list]],
                       existing_ptools: list[dict],
                       min_count: int = 3) -> tuple[str, int, list] | None:
    """Select the top pattern not yet captured by an existing ptool."""
    existing_names = {p['source_pattern'].lower() for p in existing_ptools}
    existing_names |= {p['display_name'].lower() for p in existing_ptools}

    for cat, count, examples in categories:
        if count < min_count:
            return None  # no more frequent patterns
        if cat.lower() not in existing_names:
            return (cat, count, examples)
    return None


# ═══════════════════════════════════════════════════════════════
# Ptool synthesis
# ═══════════════════════════════════════════════════════════════

def synthesize_ptool(pattern_name: str, examples: list[dict],
                     model: str, ptool_id: str) -> dict:
    """Given a frequent pattern and examples, synthesize a formal ptool spec."""
    examples_text = ''
    for ex in examples[:5]:
        examples_text += f'  - "{ex["text"][:200]}"\n'

    prompt = f"""You are designing a reusable reasoning tool for solving murder mystery puzzles.

The tool should capture this frequently used reasoning action: "{pattern_name}"

Here are examples of how agents described this action in free text:
{examples_text}

Create a tool specification:
1. display_name: A CamelCase name for the tool (e.g., SummarizeSuspectEvidence)
2. description: One sentence describing what this tool does (will be shown to the agent)
3. The tool takes one argument: [focus] — a short instruction about what to focus on

Output as JSON:
<answer>
{{"display_name": "ToolName", "description": "what this tool does"}}
</answer>"""

    response, _ = llm_cached(prompt, model)
    spec = None
    match = re.search(r'<answer>(.*)</answer>', response, re.DOTALL)
    if match:
        try:
            spec = json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            pass
    if not spec:
        match = re.search(r'\{.*\}', response, re.DOTALL)
        if match:
            try:
                spec = json.loads(match.group(0))
            except json.JSONDecodeError:
                pass

    if not spec:
        # Fallback: create spec manually
        name = pattern_name.title().replace(' ', '')
        spec = {'display_name': name, 'description': pattern_name.lower()}

    return {
        'id': ptool_id,
        'display_name': spec['display_name'],
        'description': spec['description'],
        'source_pattern': pattern_name,
        'examples': [ex['text'][:200] for ex in examples[:5]],
    }


# ═══════════════════════════════════════════════════════════════
# Main iteration loop
# ═══════════════════════════════════════════════════════════════

app = typer.Typer()


@app.command()
def run(
    n_cases: int = typer.Option(75, help='Number of MUSR cases'),
    n_iters: int = typer.Option(5, help='Max iterations'),
    model: str = typer.Option(
        'together_ai/deepseek-ai/DeepSeek-V3', help='LLM model'),
    seed: int = typer.Option(42, help='Random seed'),
    min_count: int = typer.Option(3, help='Min pattern frequency to induce'),
    reuse_iter0: bool = typer.Option(True, help='Use existing traces for iter 0'),
):
    """Run the iterative ptool induction pipeline."""
    config.configure(cfg={
        'llm': {'model': model, 'timeout': 300},
        'cachier': {
            'cache_dir': str(ITER_DIR / 'llm_cache'),
            'enable_caching': True,
        },
    })

    ITER_DIR.mkdir(parents=True, exist_ok=True)
    ptools_dir = ITER_DIR / 'ptools'
    cases = load_murder_cases(n_cases, seed)
    ptools: list[dict] = []
    iteration_stats = []

    print(f'Iterative Ptool Induction')
    print(f'  Cases: {n_cases}, Max iterations: {n_iters}, Model: {model}')
    print(f'  Output: {ITER_DIR}\n')

    for iteration in range(n_iters):
        iter_path = ITER_DIR / f'iter_{iteration}'
        iter_path.mkdir(parents=True, exist_ok=True)
        traces_file = iter_path / 'traces.json'

        print(f'\n{"="*60}')
        print(f'ITERATION {iteration}  |  ptools so far: {len(ptools)}')
        print(f'{"="*60}')

        # --- Step 1: Run or load traces ---
        if iteration == 0 and reuse_iter0:
            existing = Path(__file__).parent / 'traces' / 'react_traces.json'
            if existing.exists():
                with open(existing) as f:
                    traces = json.load(f)
                # Subset to match n_cases
                case_names = {c['name'] for c in cases}
                traces = [t for t in traces if t['case_name'] in case_names]
                print(f'  Loaded {len(traces)} existing traces (iter 0)')
            else:
                print(f'  No existing traces found, running ReAct...')
                traces = _run_all_cases(cases, ptools, model)
        else:
            print(f'  Running ReAct with {len(ptools)} ptools...')
            traces = _run_all_cases(cases, ptools, model)

        # Save traces
        with open(traces_file, 'w') as f:
            json.dump(traces, f, indent=2, default=str)

        # --- Step 2: Compute metrics ---
        n_correct = sum(1 for t in traces if t.get('correct', False))
        accuracy = n_correct / len(traces) if traces else 0
        total_cost = sum(t.get('stats', {}).get('cost', 0) for t in traces)
        total_latency = sum(t.get('stats', {}).get('latency', 0) for t in traces)
        avg_cost = total_cost / len(traces) if traces else 0
        avg_latency = total_latency / len(traces) if traces else 0

        # Count ptool usage
        ptool_uses = sum(
            1 for t in traces for s in t['steps']
            if s.get('ptool_used'))
        do_uses = sum(
            1 for t in traces for s in t['steps']
            if s.get('action_type') == 'do')

        stats = dict(
            iteration=iteration, n_ptools=len(ptools),
            accuracy=accuracy, n_correct=n_correct, n_total=len(traces),
            avg_cost=avg_cost, avg_latency=avg_latency,
            ptool_uses=ptool_uses, do_uses=do_uses,
        )
        iteration_stats.append(stats)

        print(f'\n  Accuracy: {n_correct}/{len(traces)} ({accuracy:.1%})')
        print(f'  Avg cost: ${avg_cost:.4f}  |  Avg latency: {avg_latency:.1f}s')
        print(f'  Ptool uses: {ptool_uses}  |  Do[] uses: {do_uses}')

        # --- Step 3: Analyze free-form actions ---
        print(f'\n  Analyzing free-form actions...')
        free_actions = extract_free_actions(traces)
        print(f'  Found {len(free_actions)} free-form Do[] actions')

        if not free_actions:
            print('  No free-form actions left — all captured by ptools!')
            break

        categories = categorize_actions(free_actions, model)
        print(f'  Top patterns:')
        for cat, count, _ in categories[:5]:
            print(f'    {count:3d}x  {cat}')

        # --- Step 4: Select top pattern & synthesize ptool ---
        top = select_top_pattern(categories, ptools, min_count=min_count)
        if top is None:
            print(f'\n  No frequent uncaptured pattern (min_count={min_count}). Converged!')
            break

        pattern_name, count, examples = top
        print(f'\n  Inducing ptool from: "{pattern_name}" ({count} occurrences)')

        ptool_id = f'ptool_{len(ptools)+1:03d}'
        new_ptool = synthesize_ptool(pattern_name, examples, model, ptool_id)
        save_ptool(new_ptool, ptools_dir)

        print(f'  New ptool: {new_ptool["display_name"]} — {new_ptool["description"]}')
        ptools.append(new_ptool)

        # Save iteration metadata
        meta = {**stats, 'new_ptool': new_ptool}
        with open(iter_path / 'meta.json', 'w') as f:
            json.dump(meta, f, indent=2, default=str)

    # --- Final summary ---
    print(f'\n\n{"="*60}')
    print(f'FINAL SUMMARY')
    print(f'{"="*60}')
    print(f'{"Iter":>4}  {"Ptools":>6}  {"Accuracy":>8}  {"Avg$":>8}  {"AvgLat":>8}  {"Ptool":>5}  {"Do[]":>5}')
    for s in iteration_stats:
        print(f'{s["iteration"]:4d}  {s["n_ptools"]:6d}  '
              f'{s["accuracy"]:7.1%}  ${s["avg_cost"]:.4f}  '
              f'{s["avg_latency"]:7.1f}s  {s["ptool_uses"]:5d}  {s["do_uses"]:5d}')

    print(f'\nInduced ptools:')
    for p in ptools:
        print(f'  {p["id"]}: {p["display_name"]} — {p["description"]}')

    # Save overall results
    summary = {
        'iteration_stats': iteration_stats,
        'ptools': ptools,
        'config': dict(n_cases=n_cases, n_iters=n_iters, model=model,
                        seed=seed, min_count=min_count),
    }
    with open(ITER_DIR / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f'\nSummary saved to {ITER_DIR / "summary.json"}')


def _run_all_cases(cases: list[dict], ptools: list[dict],
                   model: str) -> list[dict]:
    """Run ReAct on all cases, return traces."""
    results = []
    for idx, case in enumerate(cases):
        print(f'  [{idx+1}/{len(cases)}] {case["name"]}...', end=' ', flush=True)
        try:
            result = run_react_on_case(
                case['narrative'], case['question'], case['choices'],
                ptools, model)
        except Exception as ex:
            print(f'ERROR: {ex}')
            result = dict(answer=-1, steps=[], n_steps=0,
                          termination='error', stats={})

        correct = (result['answer'] == case['expected'])
        result['case_name'] = case['name']
        result['expected'] = case['expected']
        result['correct'] = correct
        results.append(result)

        status = 'OK' if correct else 'WRONG'
        print(f'{status} (pred={result["answer"]}, exp={case["expected"]}, '
              f'steps={result["n_steps"]})')
    return results


if __name__ == '__main__':
    app()
