"""Iterative ptool induction pipeline v3 — real @interface ptools.

Key difference from v1/v2: induced ptools are real @interface functions
bound to `implement_via('simulate')`, not text prompt templates.
When the ReAct agent calls a ptool, the framework's simulate mechanism
handles the LLM call using the ptool's docstring and typed signature.

Usage:
    source .env && export TOGETHER_API_KEY

    # Exp A: orthogonal ptools + max_steps=14 (actions-based)
    uv run python experiments/induction/iterate_v3.py --max-steps 14 --sim-threshold 0.75 --output-dir iterations_v3a

    # Exp B: batch induction (3 from iter 0)
    uv run python experiments/induction/iterate_v3.py --batch-size 3 --max-steps 14 --output-dir iterations_v3b

    # Exp C: thoughts-based induction
    uv run python experiments/induction/iterate_v3.py --source thoughts --max-steps 14 --output-dir iterations_v3c
"""

import json
import re
import sys
import time
import types
from collections import Counter
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / 'src'))
sys.path.insert(0, str(_PROJECT_ROOT / 'benchmarks' / 'musr'))

import typer
from litellm import completion, completion_cost
from secretagent import config
from secretagent.core import interface as make_interface, all_interfaces, Interface
from secretagent.llm_util import llm as llm_cached
from sentence_transformers import SentenceTransformer

from run_react import load_murder_cases

BASE_DIR = Path(__file__).parent


# ═══════════════════════════════════════════════════════════════
# Embedding similarity
# ═══════════════════════════════════════════════════════════════

_EMBED_MODEL = None

def _get_embed_model():
    global _EMBED_MODEL
    if _EMBED_MODEL is None:
        _EMBED_MODEL = SentenceTransformer('all-MiniLM-L6-v2')
    return _EMBED_MODEL


def ptool_similarity(new_desc: str, existing_ptools: list[dict]) -> float:
    if not existing_ptools:
        return 0.0
    model = _get_embed_model()
    texts = [new_desc] + [p['doc'] for p in existing_ptools]
    embeddings = model.encode(texts, normalize_embeddings=True)
    sims = embeddings[0] @ embeddings[1:].T
    return float(np.max(sims))


# ═══════════════════════════════════════════════════════════════
# Dynamic @interface creation
# ═══════════════════════════════════════════════════════════════

# Track induced interfaces so we can find them by name in the ReAct loop
_INDUCED_INTERFACES: dict[str, Interface] = {}


def create_ptool_interface(name: str, doc: str) -> Interface:
    """Create a real @interface ptool and bind it to simulate.

    Every induced ptool has signature:
        def ptool_name(narrative: str, focus: str) -> str

    The docstring drives the simulate implementation — the LLM reads it
    and produces the output based on the narrative and focus.
    """
    # Build a stub function dynamically
    def stub(narrative: str, focus: str) -> str: ...
    stub.__name__ = name
    stub.__qualname__ = name
    stub.__doc__ = doc
    stub.__annotations__ = {'narrative': str, 'focus': str, 'return': str}
    # Wrap in a proper module so secretagent doesn't get confused
    stub.__module__ = 'ptools_induced'

    # Register as @interface
    iface = make_interface(stub)
    # Bind to simulate — LLM will use the docstring to produce output
    iface.implement_via('simulate')

    _INDUCED_INTERFACES[name] = iface
    return iface


def find_induced_ptool(display_name: str) -> Interface | None:
    """Find an induced ptool by display name (case-insensitive)."""
    # display_name is CamelCase, interface name is snake_case
    for name, iface in _INDUCED_INTERFACES.items():
        if name.lower() == display_name.lower():
            return iface
        # Also try matching the display_name directly
        snake = re.sub(r'(?<!^)(?=[A-Z])', '_', display_name).lower()
        if name == snake:
            return iface
    return None


# ═══════════════════════════════════════════════════════════════
# Ptool data management
# ═══════════════════════════════════════════════════════════════

def save_ptool(ptool: dict, ptools_dir: Path):
    ptools_dir.mkdir(parents=True, exist_ok=True)
    path = ptools_dir / f'{ptool["id"]}.json'
    with open(path, 'w') as f:
        json.dump(ptool, f, indent=2)
    print(f'  Saved ptool spec to {path}')


def save_ptools_as_python(ptools: list[dict], output_path: Path):
    """Write all induced ptools as a readable .py file for human review."""
    lines = [
        '"""Auto-induced ptools from iterative induction pipeline.',
        '',
        'These are real @interface stubs. Each is bound to simulate at runtime.',
        '"""',
        '',
        'from secretagent.core import interface',
        '',
    ]
    for p in ptools:
        lines.append(f'@interface')
        lines.append(f'def {p["func_name"]}(narrative: str, focus: str) -> str:')
        # Format docstring
        doc_lines = p['doc'].strip().split('\n')
        lines.append(f'    """{doc_lines[0]}')
        for dl in doc_lines[1:]:
            lines.append(f'    {dl}')
        lines.append(f'    """')
        lines.append(f'')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))
    print(f'  Saved readable ptools to {output_path}')


def format_ptool_actions_for_prompt(ptools: list[dict]) -> str:
    """Format ptools as action descriptions for the ReAct system prompt."""
    lines = []
    for i, p in enumerate(ptools, start=2):
        lines.append(
            f'({i}) {p["display_name"]}[focus], which calls the tool '
            f'`{p["func_name"]}(narrative, focus)` — {p["short_desc"]}'
        )
    return '\n'.join(lines)


# ═══════════════════════════════════════════════════════════════
# ReAct loop — calls real @interface ptools
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


def _execute_do_action(action_arg: str, narrative: str, model: str) -> tuple[str, dict]:
    """Environment for free-form Do[] — same as original run_react.py."""
    env_prompt = (
        f"You are a careful reader assisting a detective. Read the narrative below "
        f"and respond to this request:\n\n"
        f"Request: {action_arg}\n\n"
        f"Narrative:\n{narrative}\n\n"
        f"Respond concisely with only information directly stated in or inferable "
        f"from the narrative. Do not speculate beyond what the text supports."
    )
    response, stats = llm_cached(env_prompt, model)
    if len(response) > 1500:
        response = response[:1500] + '...'
    return response.strip(), stats


def _execute_ptool(iface: Interface, narrative: str, focus: str) -> tuple[str, dict]:
    """Call a real @interface ptool. It's bound to simulate, so the
    framework handles the LLM call using the docstring."""
    start_time = time.time()
    result = iface(narrative=narrative, focus=focus)
    latency = time.time() - start_time
    # Stats are captured by the recording system, approximate here
    text = str(result) if result is not None else ''
    if len(text) > 1500:
        text = text[:1500] + '...'
    return text, {'latency': latency}


def _make_system_prompt(ptools: list[dict]) -> str:
    base = (
        "Solve a murder mystery question with interleaving Thought, Action, Observation steps. "
        "Thought can reason about the current situation. Action can be:\n"
        "(1) Do[description], which performs any reasoning step you describe — "
        "the system will read the narrative and return the relevant information. "
        "You decide what to investigate.\n"
    )
    if ptools:
        base += format_ptool_actions_for_prompt(ptools) + '\n'
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
        return match.group(1), match.group(2).strip()  # preserve case for ptool matching
    return '', text.strip()


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
        action_name_lower = action_name.lower()

        # Finish
        if action_name_lower == 'finish':
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

        # Check if it's a ptool call
        ptool_iface = find_induced_ptool(action_name) if action_name_lower != 'do' else None

        if ptool_iface:
            # Call real @interface ptool
            observation, env_stats = _execute_ptool(ptool_iface, narrative, action_arg)
            accum(env_stats)
            # Find the ptool dict for metadata
            ptool_id = None
            for p in ptools:
                if p['display_name'].lower() == action_name.lower() or p['func_name'] == ptool_iface.name:
                    ptool_id = p['id']
                    break
            steps.append(dict(
                step=i, thought=thought, action=action_str,
                action_type='ptool', action_arg=action_arg,
                observation=observation, ptool_used=ptool_id or ptool_iface.name,
            ))
        elif action_name_lower == 'do' and action_arg:
            observation, env_stats = _execute_do_action(action_arg, narrative, model)
            accum(env_stats)
            steps.append(dict(
                step=i, thought=thought, action=action_str,
                action_type='do', action_arg=action_arg,
                observation=observation, ptool_used=None,
            ))
        elif action_str:
            observation, env_stats = _execute_do_action(action_str, narrative, model)
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
# Pattern analysis (same as v2, supports actions/thoughts)
# ═══════════════════════════════════════════════════════════════

def extract_items(traces: list[dict], source: str) -> list[dict]:
    items = []
    for t in traces:
        for s in t['steps']:
            if source == 'actions':
                if s.get('action_type') == 'do' and s.get('action_arg', '').strip():
                    items.append(dict(case=t['case_name'], step=s['step'],
                                      text=s['action_arg'].strip()))
            elif source == 'thoughts':
                if s.get('thought', '').strip() and s.get('action_type') != 'finish':
                    items.append(dict(case=t['case_name'], step=s['step'],
                                      text=s['thought'].strip()))
    return items


def categorize_items(items: list[dict], source: str, model: str,
                     batch_size: int = 30) -> list[tuple[str, int, list]]:
    if not items:
        return []

    cat_map = {}
    for batch_start in range(0, len(items), batch_size):
        batch = items[batch_start:batch_start + batch_size]
        items_text = ''
        for i, item in enumerate(batch):
            idx = batch_start + i
            items_text += f'\n[{idx}] ({item["case"]}, step {item["step"]}): {item["text"][:300]}\n'

        prompt = f"""You are analyzing {source} from an AI agent solving murder mystery puzzles.
Categorize each into a short, reusable REASONING ACTION TYPE (3-6 words max).

Rules:
- Use consistent, canonical names (merge synonyms)
- Categories must be FUNCTIONALLY DISTINCT
- Focus on WHAT the agent is doing, not case-specific details
- Output ONLY a JSON array with "index" and "category" fields

{source.capitalize()} to categorize:
{items_text}

<answer>
[{{"index": {batch_start}, "category": "your category"}}, ...]
</answer>"""

        response, _ = llm_cached(prompt, model)
        json_str = None
        m = re.search(r'<answer>(.*)</answer>', response, re.DOTALL)
        if m:
            json_str = m.group(1).strip()
        if not json_str:
            m = re.search(r'\[.*\]', response, re.DOTALL)
            if m:
                json_str = m.group(0)
        if json_str:
            try:
                for c in json.loads(json_str):
                    cat_map[c['index']] = c['category']
            except (json.JSONDecodeError, KeyError):
                pass

    for i, item in enumerate(items):
        item['category'] = cat_map.get(i, 'unknown')

    freq = Counter(item['category'] for item in items)
    categories = [cat for cat, _ in freq.most_common()]

    if len(categories) <= 5:
        merged = freq
    else:
        merge_prompt = f"""Below are {len(categories)} category names from analyzing reasoning {source}.
Merge them into 5-10 canonical groups that are FUNCTIONALLY DISTINCT.

Categories:
{json.dumps(categories, indent=2)}

Output a JSON object mapping each original category to its canonical group name.

<answer>
{{"original category": "canonical group", ...}}
</answer>"""
        response, _ = llm_cached(merge_prompt, model)
        merge_map = {}
        m = re.search(r'<answer>(.*)</answer>', response, re.DOTALL)
        if m:
            try:
                merge_map = json.loads(m.group(1).strip())
            except json.JSONDecodeError:
                pass
        if not merge_map:
            m = re.search(r'\{.*\}', response, re.DOTALL)
            if m:
                try:
                    merge_map = json.loads(m.group(0))
                except json.JSONDecodeError:
                    pass

        if merge_map:
            for item in items:
                item['merged_category'] = merge_map.get(item['category'], item['category'])
            merged = Counter(item.get('merged_category', item['category']) for item in items)
        else:
            merged = freq

    result = []
    for cat, count in merged.most_common():
        examples = [a for a in items if a.get('merged_category', a['category']) == cat]
        result.append((cat, count, examples))
    return result


# ═══════════════════════════════════════════════════════════════
# Ptool synthesis — produces real @interface functions
# ═══════════════════════════════════════════════════════════════

def synthesize_ptool(pattern_name: str, examples: list[dict],
                     model: str, ptool_id: str) -> dict:
    """Synthesize a real @interface ptool from a pattern and examples.

    Returns a ptool spec dict AND creates a live @interface bound to simulate.
    """
    examples_text = ''
    for ex in examples[:5]:
        examples_text += f'  - "{ex["text"][:200]}"\n'

    prompt = f"""You are designing a reusable reasoning tool (Python function) for solving murder mystery puzzles.

The tool captures this frequently used reasoning action: "{pattern_name}"

Examples of how agents described this action:
{examples_text}

The tool will be a Python function with this signature:
    def tool_name(narrative: str, focus: str) -> str

Where:
- narrative: the full murder mystery text (always passed)
- focus: what specific aspect to focus on (the agent decides)

Design the tool:
1. func_name: snake_case Python function name
2. display_name: CamelCase version for the agent to call (e.g., VerifyAlibis)
3. short_desc: one sentence for the agent prompt
4. docstring: detailed docstring explaining what to extract/analyze from the narrative

The docstring is critical — it drives the LLM that executes this tool. Be specific about:
- What information to extract from the narrative
- How to structure the response
- What to pay attention to

Output as JSON:
<answer>
{{
  "func_name": "snake_case_name",
  "display_name": "CamelCaseName",
  "short_desc": "one sentence for agent prompt",
  "docstring": "detailed multi-line docstring"
}}
</answer>"""

    response, _ = llm_cached(prompt, model)
    spec = None
    m = re.search(r'<answer>(.*)</answer>', response, re.DOTALL)
    if m:
        try:
            spec = json.loads(m.group(1).strip())
        except json.JSONDecodeError:
            pass
    if not spec:
        m = re.search(r'\{.*\}', response, re.DOTALL)
        if m:
            try:
                spec = json.loads(m.group(0))
            except json.JSONDecodeError:
                pass

    if not spec:
        snake = re.sub(r'[^a-z0-9]+', '_', pattern_name.lower()).strip('_')
        camel = pattern_name.title().replace(' ', '')
        spec = {
            'func_name': snake,
            'display_name': camel,
            'short_desc': pattern_name.lower(),
            'docstring': f'{pattern_name}.\n\nAnalyze the narrative with the given focus.',
        }

    # Create the real @interface
    iface = create_ptool_interface(spec['func_name'], spec['docstring'])

    ptool = {
        'id': ptool_id,
        'func_name': spec['func_name'],
        'display_name': spec['display_name'],
        'short_desc': spec['short_desc'],
        'doc': spec['docstring'],
        'source_pattern': pattern_name,
        'examples': [ex['text'][:200] for ex in examples[:5]],
    }
    return ptool


def select_and_synthesize(categories: list[tuple[str, int, list]],
                          existing_ptools: list[dict],
                          model: str, batch_size: int,
                          sim_threshold: float,
                          min_count: int,
                          next_id: int) -> list[dict]:
    new_ptools = []
    all_ptools = list(existing_ptools)

    for cat, count, examples in categories:
        if len(new_ptools) >= batch_size:
            break
        if count < min_count:
            break

        existing_names = {p['source_pattern'].lower() for p in all_ptools}
        existing_names |= {p.get('display_name', '').lower() for p in all_ptools}
        if cat.lower() in existing_names:
            continue

        ptool_id = f'ptool_{next_id + len(new_ptools):03d}'
        candidate = synthesize_ptool(cat, examples, model, ptool_id)

        sim = ptool_similarity(candidate['doc'], all_ptools)
        if sim > sim_threshold:
            print(f'  SKIP "{cat}" — too similar (cosine={sim:.2f} > {sim_threshold})')
            continue

        print(f'  ACCEPT "{cat}" — {candidate["display_name"]} (cosine={sim:.2f})')
        new_ptools.append(candidate)
        all_ptools.append(candidate)

    return new_ptools


# ═══════════════════════════════════════════════════════════════
# Sample trace output
# ═══════════════════════════════════════════════════════════════

def save_sample_traces(traces: list[dict], iter_path: Path, n: int = 10):
    successes = [t for t in traces if t.get('correct')]
    failures = [t for t in traces if not t.get('correct')]
    sample = {'successes': successes[:n], 'failures': failures[:n]}
    with open(iter_path / 'sample_traces.json', 'w') as f:
        json.dump(sample, f, indent=2, default=str)
    return sample


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
    sim_threshold: float = typer.Option(0.75, help='Cosine sim threshold for orthogonality'),
    batch_size: int = typer.Option(1, help='Ptools to induce per iteration'),
    source: str = typer.Option('actions', help='Induction source: actions or thoughts'),
    output_dir: str = typer.Option('iterations_v3', help='Output directory name'),
):
    """Run iterative ptool induction v3 (real @interface ptools)."""
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

    print(f'Iterative Ptool Induction v3 (real @interface ptools)')
    print(f'  Cases: {n_cases}, Max iters: {n_iters}, Max steps: {max_steps}')
    print(f'  Source: {source}, Batch: {batch_size}, Sim threshold: {sim_threshold}')
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
                traces = _run_all_cases(cases, ptools, model, max_steps)
        else:
            print(f'  Running ReAct with {len(ptools)} ptools (real @interface), max_steps={max_steps}...')
            traces = _run_all_cases(cases, ptools, model, max_steps)

        with open(iter_path / 'traces.json', 'w') as f:
            json.dump(traces, f, indent=2, default=str)
        save_sample_traces(traces, iter_path)

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
        print(f'  Ptool uses: {ptool_uses}  |  Do[]: {do_uses}')
        print(f'  Max step hits: {max_step_hits}  |  -1 answers: {neg1_answers}')

        # --- Step 3: Analyze ---
        print(f'\n  Analyzing {source}...')
        items = extract_items(traces, source)
        print(f'  Found {len(items)} {source}')
        if not items:
            break

        categories = categorize_items(items, source, model)
        print(f'  Top patterns:')
        for cat, count, _ in categories[:8]:
            print(f'    {count:3d}x  {cat}')

        # --- Step 4: Synthesize ---
        print(f'\n  Synthesizing ptools (batch={batch_size}, sim_threshold={sim_threshold})...')
        new_ptools = select_and_synthesize(
            categories, ptools, model,
            batch_size=batch_size, sim_threshold=sim_threshold,
            min_count=min_count, next_id=len(ptools) + 1,
        )

        if not new_ptools:
            print('  No new orthogonal ptools found. Converged!')
            break

        for p in new_ptools:
            save_ptool(p, ptools_dir)
            ptools.append(p)
            print(f'  New ptool: {p["display_name"]} — {p["short_desc"]}')
            print(f'    @interface {p["func_name"]}(narrative, focus) -> str')

        # Save readable ptools file
        save_ptools_as_python(ptools, out_dir / 'ptools_induced.py')

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

    print(f'\nInduced ptools (@interface):')
    for p in ptools:
        print(f'  {p["id"]}: {p["func_name"]}(narrative, focus) -> str')
        print(f'    "{p["short_desc"]}"')

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
        print(f'{status} (pred={result["answer"]}, exp={case["expected"]}, steps={result["n_steps"]})')
    return results


if __name__ == '__main__':
    app()
