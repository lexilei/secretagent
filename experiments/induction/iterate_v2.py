"""Iterative ptool induction pipeline v2.

Improvements over v1:
- Orthogonal ptool enforcement via cosine similarity check
- Configurable max_steps (default 14)
- Batch induction (induce N ptools per iteration)
- Source selection: actions vs thoughts
- Saves sample traces (10 success + 10 failure) for human review

Usage:
    source .env && export TOGETHER_API_KEY

    # Exp A: orthogonal ptools + max_steps=14
    uv run python experiments/induction/iterate_v2.py --max-steps 14 --sim-threshold 0.75 --output-dir iterations_v2a

    # Exp B: batch induction (3 ptools from iter 0)
    uv run python experiments/induction/iterate_v2.py --batch-size 3 --max-steps 14 --output-dir iterations_v2b

    # Exp C: thoughts-based induction
    uv run python experiments/induction/iterate_v2.py --source thoughts --max-steps 14 --output-dir iterations_v2c
"""

import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / 'src'))
sys.path.insert(0, str(_PROJECT_ROOT / 'benchmarks' / 'musr'))

import typer
from litellm import completion, completion_cost
from secretagent import config
from secretagent.llm_util import llm as llm_cached
from sentence_transformers import SentenceTransformer

from run_react import load_murder_cases

BASE_DIR = Path(__file__).parent


# ═══════════════════════════════════════════════════════════════
# Embedding similarity for orthogonal ptool check
# ═══════════════════════════════════════════════════════════════

_EMBED_MODEL = None

def _get_embed_model():
    global _EMBED_MODEL
    if _EMBED_MODEL is None:
        _EMBED_MODEL = SentenceTransformer('all-MiniLM-L6-v2')
    return _EMBED_MODEL


def ptool_similarity(new_desc: str, existing_ptools: list[dict]) -> float:
    """Max cosine similarity between new ptool description and all existing ones."""
    if not existing_ptools:
        return 0.0
    model = _get_embed_model()
    texts = [new_desc] + [p['description'] for p in existing_ptools]
    embeddings = model.encode(texts, normalize_embeddings=True)
    sims = embeddings[0] @ embeddings[1:].T
    return float(np.max(sims))


# ═══════════════════════════════════════════════════════════════
# Ptool data management (same as v1)
# ═══════════════════════════════════════════════════════════════

def save_ptool(ptool: dict, ptools_dir: Path):
    ptools_dir.mkdir(parents=True, exist_ok=True)
    path = ptools_dir / f'{ptool["id"]}.json'
    with open(path, 'w') as f:
        json.dump(ptool, f, indent=2)
    print(f'  Saved ptool to {path}')


def format_ptool_actions(ptools: list[dict]) -> str:
    lines = []
    for i, p in enumerate(ptools, start=2):
        lines.append(
            f'({i}) {p["display_name"]}[focus], which {p["description"]}. '
            f'The system will read the narrative and respond with structured analysis.'
        )
    return '\n'.join(lines)


# ═══════════════════════════════════════════════════════════════
# ReAct loop (ptool-aware, configurable max_steps)
# ═══════════════════════════════════════════════════════════════

def _llm_with_stop(prompt: str, model: str, stop: list[str]) -> tuple[str, dict]:
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
    match = re.match(r'\s*(\w+)\[(.+)\]\s*$', text.strip(), re.DOTALL)
    if match:
        return match.group(1).lower(), match.group(2).strip()
    return '', text.strip()


def _find_ptool(action_name: str, ptools: list[dict]) -> dict | None:
    for p in ptools:
        if p['display_name'].lower() == action_name:
            return p
    return None


def run_react_on_case(narrative: str, question: str, choices: list,
                      ptools: list[dict], model: str,
                      max_steps: int = 14) -> dict:
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

    for i in range(1, max_steps + 1):
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
        answer=-1, steps=steps, n_steps=max_steps,
        termination='max_steps', stats=total_stats,
    )


# ═══════════════════════════════════════════════════════════════
# Pattern analysis — supports actions OR thoughts
# ═══════════════════════════════════════════════════════════════

def extract_items(traces: list[dict], source: str) -> list[dict]:
    """Extract free-form items from traces. source='actions' or 'thoughts'."""
    items = []
    for t in traces:
        for s in t['steps']:
            if source == 'actions':
                if s.get('action_type') == 'do' and s.get('action_arg', '').strip():
                    items.append(dict(
                        case=t['case_name'], step=s['step'],
                        text=s['action_arg'].strip(),
                    ))
            elif source == 'thoughts':
                if s.get('thought', '').strip() and s.get('action_type') != 'finish':
                    items.append(dict(
                        case=t['case_name'], step=s['step'],
                        text=s['thought'].strip(),
                    ))
    return items


def categorize_items(items: list[dict], source: str, model: str,
                     batch_size: int = 30) -> list[tuple[str, int, list]]:
    """Categorize items via LLM + merge synonyms. Returns (category, count, examples)."""
    if not items:
        return []

    cat_map = {}
    for batch_start in range(0, len(items), batch_size):
        batch = items[batch_start:batch_start + batch_size]
        items_text = ''
        for i, item in enumerate(batch):
            idx = batch_start + i
            text = item['text'][:300]
            items_text += f'\n[{idx}] ({item["case"]}, step {item["step"]}): {text}\n'

        prompt = f"""You are analyzing {source} from an AI agent solving murder mystery puzzles.
Below are {source} the agent produced. Categorize each into a short,
reusable REASONING ACTION TYPE (3-6 words max).

Rules:
- Use consistent, canonical names (merge synonyms into one category)
- Categories should be FUNCTIONALLY DISTINCT from each other
- Focus on WHAT the agent is doing, not case-specific details
- Output ONLY a JSON array with "index" and "category" fields

{source.capitalize()} to categorize:
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

    for i, item in enumerate(items):
        item['category'] = cat_map.get(i, 'unknown')

    freq = Counter(item['category'] for item in items)

    # Merge synonyms
    categories = [cat for cat, _ in freq.most_common()]
    if len(categories) <= 5:
        merged = freq
    else:
        merge_prompt = f"""Below are {len(categories)} category names from analyzing reasoning {source}.
Many are synonyms. Merge them into 5-10 canonical groups that are FUNCTIONALLY DISTINCT.
Each group should represent a genuinely different reasoning step.

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

        if merge_map:
            for item in items:
                item['merged_category'] = merge_map.get(
                    item['category'], item['category'])
            merged = Counter(item.get('merged_category', item['category'])
                             for item in items)
        else:
            print('  WARNING: synonym merging failed, using raw categories')
            merged = freq

    result = []
    for cat, count in merged.most_common():
        examples = [a for a in items
                    if a.get('merged_category', a['category']) == cat]
        result.append((cat, count, examples))
    return result


# ═══════════════════════════════════════════════════════════════
# Ptool synthesis with orthogonality check
# ═══════════════════════════════════════════════════════════════

def synthesize_ptool(pattern_name: str, examples: list[dict],
                     model: str, ptool_id: str) -> dict:
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
        name = pattern_name.title().replace(' ', '')
        spec = {'display_name': name, 'description': pattern_name.lower()}

    return {
        'id': ptool_id,
        'display_name': spec['display_name'],
        'description': spec['description'],
        'source_pattern': pattern_name,
        'examples': [ex['text'][:200] for ex in examples[:5]],
    }


def select_and_synthesize(categories: list[tuple[str, int, list]],
                          existing_ptools: list[dict],
                          model: str, batch_size: int,
                          sim_threshold: float,
                          min_count: int,
                          next_id: int) -> list[dict]:
    """Select top patterns and synthesize ptools, enforcing orthogonality."""
    new_ptools = []
    all_ptools = list(existing_ptools)  # copy for sim checking

    for cat, count, examples in categories:
        if len(new_ptools) >= batch_size:
            break
        if count < min_count:
            break

        # Skip if name matches existing ptool
        existing_names = {p['source_pattern'].lower() for p in all_ptools}
        existing_names |= {p['display_name'].lower() for p in all_ptools}
        if cat.lower() in existing_names:
            continue

        # Synthesize candidate
        ptool_id = f'ptool_{next_id + len(new_ptools):03d}'
        candidate = synthesize_ptool(cat, examples, model, ptool_id)

        # Orthogonality check
        sim = ptool_similarity(candidate['description'], all_ptools)
        if sim > sim_threshold:
            print(f'  SKIP "{cat}" — too similar to existing ptool (cosine={sim:.2f} > {sim_threshold})')
            continue

        print(f'  ACCEPT "{cat}" — {candidate["display_name"]} (cosine={sim:.2f})')
        new_ptools.append(candidate)
        all_ptools.append(candidate)

    return new_ptools


# ═══════════════════════════════════════════════════════════════
# Sample trace output for human review
# ═══════════════════════════════════════════════════════════════

def save_sample_traces(traces: list[dict], iter_path: Path, n: int = 10):
    """Save n success + n failure traces with full detail."""
    successes = [t for t in traces if t.get('correct')]
    failures = [t for t in traces if not t.get('correct')]
    sample = {
        'successes': successes[:n],
        'failures': failures[:n],
    }
    path = iter_path / 'sample_traces.json'
    with open(path, 'w') as f:
        json.dump(sample, f, indent=2, default=str)
    return sample


def format_trace_for_review(trace: dict) -> str:
    """Format a single trace as readable text."""
    lines = [f"**{trace['case_name']}** — {'CORRECT' if trace.get('correct') else 'WRONG'} "
             f"(pred={trace['answer']}, exp={trace['expected']}, steps={trace['n_steps']})"]
    for s in trace['steps']:
        lines.append(f"  **Thought {s['step']}:** {s['thought'][:200]}")
        lines.append(f"  **Action {s['step']}:** {s['action'][:200]}")
        if s.get('ptool_used'):
            lines.append(f"  *(used ptool: {s['ptool_used']})*")
        if s.get('observation'):
            obs = s['observation'][:200] + ('...' if len(s.get('observation', '')) > 200 else '')
            lines.append(f"  **Observation {s['step']}:** {obs}")
        lines.append('')
    return '\n'.join(lines)


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

app = typer.Typer()


@app.command()
def run(
    n_cases: int = typer.Option(75, help='Number of MUSR cases'),
    n_iters: int = typer.Option(5, help='Max iterations'),
    model: str = typer.Option('together_ai/deepseek-ai/DeepSeek-V3', help='LLM model'),
    seed: int = typer.Option(42, help='Random seed'),
    min_count: int = typer.Option(3, help='Min pattern frequency to induce'),
    reuse_iter0: bool = typer.Option(True, help='Use existing traces for iter 0'),
    max_steps: int = typer.Option(14, help='Max ReAct steps per case'),
    sim_threshold: float = typer.Option(0.75, help='Cosine similarity threshold for orthogonality'),
    batch_size: int = typer.Option(1, help='Number of ptools to induce per iteration'),
    source: str = typer.Option('actions', help='Induction source: actions or thoughts'),
    output_dir: str = typer.Option('iterations_v2', help='Output directory name'),
):
    """Run iterative ptool induction v2."""
    out_dir = BASE_DIR / output_dir
    config.configure(cfg={
        'llm': {'model': model, 'timeout': 300},
        'cachier': {
            'cache_dir': str(out_dir / 'llm_cache'),
            'enable_caching': True,
        },
    })

    out_dir.mkdir(parents=True, exist_ok=True)
    ptools_dir = out_dir / 'ptools'
    cases = load_murder_cases(n_cases, seed)
    ptools: list[dict] = []
    iteration_stats = []

    print(f'Iterative Ptool Induction v2')
    print(f'  Cases: {n_cases}, Max iters: {n_iters}, Max steps: {max_steps}')
    print(f'  Source: {source}, Batch size: {batch_size}, Sim threshold: {sim_threshold}')
    print(f'  Model: {model}')
    print(f'  Output: {out_dir}\n')

    for iteration in range(n_iters):
        iter_path = out_dir / f'iter_{iteration}'
        iter_path.mkdir(parents=True, exist_ok=True)

        print(f'\n{"="*60}')
        print(f'ITERATION {iteration}  |  ptools: {[p["display_name"] for p in ptools]}')
        print(f'{"="*60}')

        # --- Step 1: Run or load traces ---
        if iteration == 0 and reuse_iter0:
            existing = BASE_DIR / 'traces' / 'react_traces.json'
            if existing.exists():
                with open(existing) as f:
                    traces = json.load(f)
                case_names = {c['name'] for c in cases}
                traces = [t for t in traces if t['case_name'] in case_names]
                print(f'  Loaded {len(traces)} existing traces (iter 0)')
            else:
                print(f'  No existing traces, running ReAct...')
                traces = _run_all_cases(cases, ptools, model, max_steps)
        else:
            print(f'  Running ReAct with {len(ptools)} ptools, max_steps={max_steps}...')
            traces = _run_all_cases(cases, ptools, model, max_steps)

        with open(iter_path / 'traces.json', 'w') as f:
            json.dump(traces, f, indent=2, default=str)

        # Save sample traces for review
        sample = save_sample_traces(traces, iter_path)

        # --- Step 2: Metrics ---
        n_correct = sum(1 for t in traces if t.get('correct', False))
        accuracy = n_correct / len(traces) if traces else 0
        total_cost = sum(t.get('stats', {}).get('cost', 0) for t in traces)
        total_latency = sum(t.get('stats', {}).get('latency', 0) for t in traces)
        avg_cost = total_cost / len(traces) if traces else 0
        avg_latency = total_latency / len(traces) if traces else 0

        ptool_uses = sum(1 for t in traces for s in t['steps'] if s.get('ptool_used'))
        do_uses = sum(1 for t in traces for s in t['steps'] if s.get('action_type') == 'do')
        max_step_hits = sum(1 for t in traces if t.get('termination') == 'max_steps')
        neg1_answers = sum(1 for t in traces if t.get('answer') == -1)

        stats = dict(
            iteration=iteration, n_ptools=len(ptools),
            accuracy=accuracy, n_correct=n_correct, n_total=len(traces),
            avg_cost=avg_cost, avg_latency=avg_latency,
            ptool_uses=ptool_uses, do_uses=do_uses,
            max_step_hits=max_step_hits, neg1_answers=neg1_answers,
        )
        iteration_stats.append(stats)

        print(f'\n  Accuracy: {n_correct}/{len(traces)} ({accuracy:.1%})')
        print(f'  Avg cost: ${avg_cost:.4f}  |  Avg latency: {avg_latency:.1f}s')
        print(f'  Ptool uses: {ptool_uses}  |  Do[] uses: {do_uses}')
        print(f'  Max step hits: {max_step_hits}  |  -1 answers: {neg1_answers}')

        # --- Step 3: Analyze ---
        print(f'\n  Analyzing {source}...')
        items = extract_items(traces, source)
        print(f'  Found {len(items)} {source}')

        if not items:
            print(f'  No {source} found!')
            break

        categories = categorize_items(items, source, model)
        print(f'  Top patterns:')
        for cat, count, _ in categories[:8]:
            print(f'    {count:3d}x  {cat}')

        # --- Step 4: Synthesize with orthogonality ---
        print(f'\n  Synthesizing ptools (batch={batch_size}, sim_threshold={sim_threshold})...')
        new_ptools = select_and_synthesize(
            categories, ptools, model,
            batch_size=batch_size,
            sim_threshold=sim_threshold,
            min_count=min_count,
            next_id=len(ptools) + 1,
        )

        if not new_ptools:
            print('  No new orthogonal ptools found. Converged!')
            break

        for p in new_ptools:
            save_ptool(p, ptools_dir)
            ptools.append(p)
            print(f'  New ptool: {p["display_name"]} — {p["description"]}')

        # Save iteration metadata
        meta = {**stats, 'new_ptools': new_ptools}
        with open(iter_path / 'meta.json', 'w') as f:
            json.dump(meta, f, indent=2, default=str)

    # --- Final summary ---
    print(f'\n\n{"="*60}')
    print(f'FINAL SUMMARY')
    print(f'{"="*60}')
    print(f'{"Iter":>4}  {"Ptools":>6}  {"Accuracy":>8}  {"Avg$":>8}  {"AvgLat":>8}  '
          f'{"Ptool":>5}  {"Do[]":>5}  {"MaxHit":>6}  {"-1ans":>5}')
    for s in iteration_stats:
        print(f'{s["iteration"]:4d}  {s["n_ptools"]:6d}  '
              f'{s["accuracy"]:7.1%}  ${s["avg_cost"]:.4f}  '
              f'{s["avg_latency"]:7.1f}s  {s["ptool_uses"]:5d}  {s["do_uses"]:5d}  '
              f'{s["max_step_hits"]:6d}  {s["neg1_answers"]:5d}')

    print(f'\nInduced ptools:')
    for p in ptools:
        print(f'  {p["id"]}: {p["display_name"]} — {p["description"]}')

    summary = {
        'iteration_stats': iteration_stats,
        'ptools': ptools,
        'config': dict(n_cases=n_cases, n_iters=n_iters, model=model,
                        seed=seed, min_count=min_count, max_steps=max_steps,
                        sim_threshold=sim_threshold, batch_size=batch_size,
                        source=source, output_dir=output_dir),
    }
    with open(out_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f'\nSummary saved to {out_dir / "summary.json"}')


def _run_all_cases(cases: list[dict], ptools: list[dict],
                   model: str, max_steps: int) -> list[dict]:
    results = []
    for idx, case in enumerate(cases):
        print(f'  [{idx+1}/{len(cases)}] {case["name"]}...', end=' ', flush=True)
        try:
            result = run_react_on_case(
                case['narrative'], case['question'], case['choices'],
                ptools, model, max_steps=max_steps)
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
