"""Iterative ptool induction v6 — RuleArena airline (v2: uses ReActFactory).

Path A: instead of writing a custom ReAct loop with concat-and-stop prompt
construction (the approach in iterate_v6_airline.py that breaks on long
contexts), this version uses William's existing
`secretagent.implement_react.ReActFactory` which:

  - Reformats the full conversation history into a markdown "## Previous
    Steps" section at every LLM call (no concatenation, no stop tokens).
  - Uses XML tags <thought>, <action>, <answer> instead of "Thought N:"
    style markers, which DeepSeek-V3 can't track on long prompts.
  - Each step is a fresh LLM call with explicit history sections.

The v6 induction logic (pattern aggregation, categorize, synthesize,
orthogonality check) sits ON TOP of this — for each case we wrap the
ReAct call in `record.recorder()`, capture the trace, convert to v6's
expected format, then run the rest of the induction pipeline as before.

Usage:
    source .env && export TOGETHER_API_KEY

    # Smoke test n=5
    uv run python experiments/induction/iterate_v6_airline_v2.py baseline --n-cases 5

    # Full n=30 baseline
    uv run python experiments/induction/iterate_v6_airline_v2.py baseline --n-cases 30
"""

import ast
import json
import random
import re
import sys
import time
from pathlib import Path

import typer

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / 'src'))
sys.path.insert(0, str(_PROJECT_ROOT / 'benchmarks' / 'rulearena'))

from calculators.airline import compute_airline_fee  # noqa: E402

from secretagent import config, record  # noqa: E402
from secretagent.core import interface, Interface  # noqa: E402
import secretagent.implement_react  # noqa: F401, E402  (registers 'react' factory)


BASE_DIR = Path(__file__).parent

# ═══════════════════════════════════════════════════════════════
# Reference rules (loaded once, embedded in the interface docstring)
# ═══════════════════════════════════════════════════════════════

_RULES_FILE = _PROJECT_ROOT / 'benchmarks' / 'rulearena' / 'data' / 'airline' / 'reference_rules_textual.txt'
AIRLINE_RULES_TEXT = _RULES_FILE.read_text(encoding='utf-8') if _RULES_FILE.exists() else ''

# Build the interface docstring at module-load time so we can put the rules
# text into the function's __doc__. The ReActFactory uses interface.doc as
# the "## Task" section in its prompt format.
_AIRLINE_TASK_DOC = (
    "Compute the total cost of an airline trip including ticket fare and all "
    "baggage fees (checked bag base fees, oversize fees, overweight fees) "
    "according to American Airlines policy.\n\n"
    "REFERENCE POLICIES (American Airlines):\n"
    "==========================================\n"
    f"{AIRLINE_RULES_TEXT}\n"
    "==========================================\n\n"
    "Given a passenger's natural-language problem statement, return an "
    "INTEGER total cost in dollars (e.g., 2741)."
)


# ═══════════════════════════════════════════════════════════════
# Interface definition: airline cost task
# ═══════════════════════════════════════════════════════════════

def _airline_stub(narrative: str) -> int: ...
_airline_stub.__name__ = 'compute_airline_total_cost'
_airline_stub.__qualname__ = 'compute_airline_total_cost'
_airline_stub.__doc__ = _AIRLINE_TASK_DOC
_airline_stub.__annotations__ = {'narrative': str, 'return': int}
_airline_stub.__module__ = __name__

# Apply the @interface decorator manually so the long doc is captured at
# decoration time (Interface.doc is set from func.__doc__ inside the
# decorator and is NOT a live property).
compute_airline_total_cost = interface(_airline_stub)
compute_airline_total_cost.implement_via('react', max_steps=14, tools=[])


# ═══════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════

def load_airline_cases(n: int, seed: int = 42) -> list[dict]:
    """Stratified sample of airline test cases across difficulty levels L0/L1/L2."""
    data_file = _PROJECT_ROOT / 'benchmarks' / 'rulearena' / 'data' / 'airline' / 'test.jsonl'
    with open(data_file) as f:
        examples = [json.loads(l) for l in f]

    by_level: dict[int, list] = {}
    for ex in examples:
        by_level.setdefault(ex['level'], []).append(ex)

    rng = random.Random(seed)
    sampled = []
    n_levels = len(by_level)
    per_level = max(1, n // n_levels)
    remainder = n - per_level * n_levels
    for level in sorted(by_level.keys()):
        bucket = list(by_level[level])
        rng.shuffle(bucket)
        take = per_level + (1 if remainder > 0 else 0)
        if remainder > 0:
            remainder -= 1
        sampled.extend(bucket[:take])

    rng.shuffle(sampled)
    cases = []
    for ex in sampled:
        try:
            expected = compute_airline_fee(ex['info'])
        except Exception as e:
            print(f"  WARN: ground truth failed for L{ex['level']} idx {ex['orig_idx']}: {e}")
            continue
        cases.append({
            'name': f'airline_L{ex["level"]}_{ex["orig_idx"]}',
            'narrative': ex['prompt'],
            'expected': int(expected),
            'level': ex['level'],
        })
    return cases[:n]


def _within_tolerance_int(predicted, expected: int, tol: float = 0.01) -> bool:
    """1% relative tolerance, or absolute < 0.01 if expected is 0."""
    if predicted is None:
        return False
    try:
        p = float(predicted)
        e = float(expected)
    except (TypeError, ValueError):
        return False
    if abs(e) < 1e-9:
        return abs(p - e) < 0.01
    return abs(p - e) / abs(e) <= tol


# ═══════════════════════════════════════════════════════════════
# Trace conversion: ReActFactory records → v6-style trace
# ═══════════════════════════════════════════════════════════════

def _records_to_v6_trace(records: list[dict], answer, n_steps: int,
                         termination: str) -> dict:
    """Convert the recorder's flat list of records into v6's expected
    trace structure: a dict with `steps` (list of {step, thought, action_type,
    action_arg, observation, ptool_used}), `answer`, `n_steps`, `termination`,
    `stats`.

    The recorder produces something like:
        [
          {func: 'compute_airline_total_cost:react_step_0', output: '<thought>...</thought><action>tool(args)</action>', stats: {...}},
          {func: 'tool', output: <result>, stats: {...}},
          {func: 'compute_airline_total_cost:react_step_1', output: '<thought>...</thought><answer>2741</answer>', stats: {...}},
          {func: 'compute_airline_total_cost', output: 2741, step_info: {...}},
        ]
    """
    steps = []
    total_stats = {'input_tokens': 0, 'output_tokens': 0, 'latency': 0, 'cost': 0}
    react_step_idx = 0
    pending_tool_result = None

    for r in records:
        func = r.get('func', '')
        if ':react_step_' in func:
            # This is a thought+action LLM call
            response = r.get('output', '') or ''
            stats = r.get('stats', {}) or {}
            for k in total_stats:
                total_stats[k] += stats.get(k, 0)

            thought_m = re.search(r'<thought>(.*?)</thought>', response, re.DOTALL)
            action_m = re.search(r'<action>\s*(\w+)\s*\((.*?)\)\s*</action>', response, re.DOTALL)
            answer_m = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL)
            thought = thought_m.group(1).strip() if thought_m else response[:300]

            if answer_m:
                steps.append(dict(
                    step=react_step_idx + 1,
                    thought=thought,
                    action=f'<answer>{answer_m.group(1).strip()}</answer>',
                    action_type='finish',
                    action_arg=answer_m.group(1).strip(),
                    observation=None,
                    ptool_used=None,
                ))
            elif action_m:
                tool_name = action_m.group(1)
                args_str = action_m.group(2)
                steps.append(dict(
                    step=react_step_idx + 1,
                    thought=thought,
                    action=f'{tool_name}({args_str})',
                    action_type='ptool',  # all actions are tool calls in this format
                    action_arg=args_str,
                    observation=None,  # filled in by the next iteration when we see the tool result
                    ptool_used=tool_name,
                ))
            else:
                steps.append(dict(
                    step=react_step_idx + 1,
                    thought=thought,
                    action='',
                    action_type='none',
                    action_arg='',
                    observation='No action or answer found',
                    ptool_used=None,
                ))
            react_step_idx += 1
        elif func and func not in ('compute_airline_total_cost',):
            # This is a tool call result. Attach it as the observation of the
            # most recent step, if that step was an action.
            tool_output = r.get('output', '')
            stats = r.get('stats', {}) or {}
            for k in total_stats:
                total_stats[k] += stats.get(k, 0)
            if steps and steps[-1].get('action_type') == 'ptool' and steps[-1].get('observation') is None:
                obs_str = repr(tool_output) if not isinstance(tool_output, str) else tool_output
                if len(obs_str) > 1500:
                    obs_str = obs_str[:1500] + '...'
                steps[-1]['observation'] = obs_str

    return dict(
        answer=answer,
        steps=steps,
        n_steps=n_steps,
        termination=termination,
        stats=total_stats,
    )


# ═══════════════════════════════════════════════════════════════
# Per-case runner
# ═══════════════════════════════════════════════════════════════

def run_one_case(case: dict, model: str) -> dict:
    """Run the airline interface on one case and return a v6-format trace dict."""
    answer = None
    termination = 'finish'
    n_steps = 0
    error_msg = None
    with record.recorder() as records:
        try:
            answer = compute_airline_total_cost(narrative=case['narrative'])
        except ValueError as ex:
            # ReActFactory raises ValueError on max_steps reached
            termination = 'max_steps'
            error_msg = str(ex)
        except Exception as ex:
            termination = 'error'
            error_msg = f'{type(ex).__name__}: {ex}'

    # Count react steps from records
    n_steps = sum(1 for r in records if ':react_step_' in r.get('func', ''))
    if termination == 'finish' and answer is None:
        termination = 'no_answer'

    trace = _records_to_v6_trace(list(records), answer, n_steps, termination)
    if error_msg:
        trace['error'] = error_msg
    return trace


def _run_all_cases(cases: list[dict], model: str) -> list[dict]:
    results = []
    max_retries = 3
    for idx, case in enumerate(cases):
        print(f'  [{idx+1}/{len(cases)}] {case["name"]}...', end=' ', flush=True)
        result = None
        for attempt in range(max_retries):
            try:
                result = run_one_case(case, model)
                break
            except Exception as ex:
                if attempt < max_retries - 1:
                    wait = 2 ** attempt
                    print(f'RETRY({attempt+1}, wait {wait}s)...', end=' ', flush=True)
                    time.sleep(wait)
                else:
                    print(f'OUTER ERROR: {type(ex).__name__}: {ex}')
                    result = dict(answer=None, steps=[], n_steps=0,
                                  termination='outer_error',
                                  stats={'cost': 0, 'latency': 0},
                                  error=str(ex))
        pred = result.get('answer')
        correct = _within_tolerance_int(pred, case['expected'])
        result['case_name'] = case['name']
        result['expected'] = case['expected']
        result['correct'] = correct
        if 'level' in case:
            result['level'] = case['level']
        results.append(result)
        status = 'OK' if correct else 'WRONG'
        term = result.get('termination', '')
        tag = f' [{term}]' if term not in ('finish',) else ''
        print(f'{status} (pred={pred}, exp={case["expected"]}, steps={result["n_steps"]}{tag})')
        time.sleep(0.5)
    return results


# ═══════════════════════════════════════════════════════════════
# v6 induction: extract paragraph-level items from finish-step thoughts
# ═══════════════════════════════════════════════════════════════

from collections import Counter  # noqa: E402

# Reuse v3/v5 infrastructure for categorize/synthesize/similarity. We
# only override the parts that need to be airline-specific (extract,
# task labels, synthesis prompt).
sys.path.insert(0, str(_PROJECT_ROOT / 'benchmarks' / 'musr'))
from iterate_v3 import (  # noqa: E402
    save_ptool,
    save_ptools_as_python,
    save_sample_traces,
)
from iterate_v5 import _get_embed_model, _docstring_similarity  # noqa: E402
from secretagent.llm_util import llm as llm_cached  # noqa: E402

import numpy as np  # noqa: E402


def extract_thought_paragraphs(traces: list[dict]) -> list[dict]:
    """Split each step's thought into paragraphs and return as items.

    On airline v2 each case has 1 step (finish) with a 5k-25k char thought.
    We split on blank lines into paragraphs of ~100-500 chars each, which
    gives ~10-15 items per case — comparable to the per-case Do[] count
    on calendar/team. Each paragraph captures a discrete reasoning sub-step
    (list bags / determine route / lookup base fee / compute oversize / ...).
    """
    items = []
    for t in traces:
        for s in t['steps']:
            thought = (s.get('thought') or '').strip()
            if not thought:
                continue
            # Strip leading <thought> tag if present
            thought = re.sub(r'^<thought>\s*', '', thought)
            thought = re.sub(r'\s*</thought>\s*$', '', thought)
            # Split on blank lines (one or more newlines with optional whitespace)
            paragraphs = [p.strip() for p in re.split(r'\n\s*\n', thought) if p.strip()]
            for para_idx, para in enumerate(paragraphs):
                # Skip very short fragments (likely just a number or heading)
                if len(para) < 30:
                    continue
                # Cap each item at 800 chars (categorize_items truncates to 300
                # for the LLM call, so anything longer is wasted but kept for
                # audit/synthesis later)
                items.append(dict(
                    case=t['case_name'],
                    step=f'{s["step"]}.p{para_idx}',
                    text=para[:800],
                ))
    return items


# ═══════════════════════════════════════════════════════════════
# Categorize patterns (airline task label)
# ═══════════════════════════════════════════════════════════════

CATEGORIZE_TASK_LABEL = "airline baggage fee calculation tasks"
SYNTHESIZE_TASK_LABEL = "airline baggage fee calculation tasks"
SYNTHESIZE_NARRATIVE_LABEL = (
    "the full problem statement: passenger name, customer class, flight route, "
    "list of bags with sizes (length x width x height in inches) and weights "
    "(lbs), and the ticket price"
)


def categorize_items_air(items: list[dict], source: str, model: str,
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

        prompt = f"""You are analyzing reasoning paragraphs from an AI agent solving {CATEGORIZE_TASK_LABEL}.
Categorize each into a short, reusable REASONING ACTION TYPE (3-6 words max).

Rules:
- Use consistent, canonical names (merge synonyms)
- Categories must be FUNCTIONALLY DISTINCT
- Focus on WHAT the agent is doing, not case-specific details
- Output ONLY a JSON array with "index" and "category" fields

Paragraphs to categorize:
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
        merge_prompt = f"""Below are {len(categories)} category names from analyzing reasoning paragraphs.
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
# Ptool synthesis (airline task label, structured output)
# ═══════════════════════════════════════════════════════════════

def synthesize_ptool_air(pattern_name: str, examples: list[dict],
                         model: str, ptool_id: str,
                         structured_output: bool = True) -> dict:
    examples_text = ''
    for ex in examples[:5]:
        examples_text += f'  - "{ex["text"][:200]}"\n'

    if structured_output:
        output_instruction = """
4. output_format: Specify EXACTLY what the output should look like. Use a structured format like:
   - A dict/JSON of extracted parameters (e.g., {"customer_class": "Premium Economy", "routine": "China", "direction": 1, "bag_list": [{"id": 1, ...}]})
   - A list of per-bag fee components (e.g., [{"bag_id": 2, "base_fee": 150, "oversize_fee": 0, "overweight_fee": 100, "rule_cited": "..."}, ...])
   - A region/class lookup dict (e.g., {"matched_route": "Asia (China)", "matched_class": "Premium Economy", "first_bag_fee": 150, ...})
   - A boolean validity judgment with cited rule (e.g., {"is_oversize": true, "total_inches": 102, "rule_cited": "..."})

The output format MUST be different from other tools — if one returns extracted params, another should return per-bag fee components, etc.
Include the output format specification in the docstring under a "Returns:" section with a concrete example."""
    else:
        output_instruction = ""

    prompt = f"""You are designing a reusable reasoning tool (Python function) for solving {SYNTHESIZE_TASK_LABEL}.

The tool captures this frequently used reasoning action: "{pattern_name}"

Examples of how agents described this action:
{examples_text}

The tool will be a Python function with this signature:
    def tool_name(narrative: str, focus: str) -> str

Where:
- narrative: {SYNTHESIZE_NARRATIVE_LABEL}
- focus: what specific aspect to focus on (e.g., a target bag id, a route region, a customer class, a specific fee component)

Design the tool:
1. func_name: snake_case Python function name
2. display_name: CamelCase version for the agent to call
3. short_desc: one sentence for the agent prompt
{output_instruction}

The docstring is critical — it drives the LLM that executes this tool. Be specific about:
- What information to extract from the narrative (passenger class, route/region, ticket price, bag list with sizes and weights)
- How to structure the response
- What to pay attention to (airline fee calculation: first item is typically a personal item or carry-on (free); checked bag base fees depend on (route region × customer class × bag ordinal); oversize is total dimensions > 62 inches; overweight tiers are 50/70/100 lbs; oversize and overweight surcharges DON'T STACK — apply max(oversize, overweight) per bag)

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

    return {
        'id': ptool_id,
        'func_name': spec['func_name'],
        'display_name': spec['display_name'],
        'short_desc': spec['short_desc'],
        'doc': spec['docstring'],
        'source_pattern': pattern_name,
        'examples': [ex['text'][:200] for ex in examples[:5]],
    }


def select_and_synthesize_air(categories: list[tuple[str, int, list]],
                              existing_ptools: list[dict],
                              model: str, batch_size: int,
                              sim_threshold: float, min_count: int,
                              next_id: int, structured_output: bool) -> list[dict]:
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
        candidate = synthesize_ptool_air(cat, examples, model, ptool_id,
                                         structured_output=structured_output)
        # Use docstring-only similarity (no functional check, since induced
        # ptools aren't actually invoked — they're scaffolding-by-description)
        sim = _docstring_similarity(candidate, all_ptools)
        if sim > sim_threshold:
            print(f'  SKIP "{cat}" — too similar (docstring={sim:.2f} > {sim_threshold})')
            continue
        print(f'  ACCEPT "{cat}" — {candidate["display_name"]} (docstring={sim:.2f})')
        new_ptools.append(candidate)
        all_ptools.append(candidate)
    return new_ptools


# ═══════════════════════════════════════════════════════════════
# Scaffolding-by-description: rebind interface with induced ptool
# descriptions appended to the task docstring
# ═══════════════════════════════════════════════════════════════

def _build_task_doc_with_ptools(induced_ptools: list[dict]) -> str:
    """Construct the airline task docstring with induced ptool descriptions
    appended as suggested reasoning steps.

    Path D from the discussion: induced ptools are NOT given to ReActFactory
    as tools (which would require narrative threading + signature changes).
    Instead, their short_desc + docstring are listed in the task docstring
    as 'suggested reasoning steps' the agent can mentally apply. This
    matches the v6cg 'scaffolding by description' effect from murder mystery
    where the best run had 88% with 0 ptool invocations.
    """
    base = _AIRLINE_TASK_DOC
    if not induced_ptools:
        return base
    suggestion_block = ['', '', 'SUGGESTED REUSABLE REASONING STEPS:',
                         '(These are reasoning patterns that have helped on prior cases.',
                         'Mentally apply them as relevant — you don\'t need to call them as tools.)',
                         '']
    for i, p in enumerate(induced_ptools, start=1):
        suggestion_block.append(f'{i}. **{p["display_name"]}** — {p["short_desc"]}')
        # Indent the docstring under the heading, truncate to ~600 chars to keep
        # total prompt size manageable
        doc_lines = (p['doc'][:600]).strip().split('\n')
        for dl in doc_lines:
            suggestion_block.append(f'   {dl}')
        suggestion_block.append('')
    return base + '\n'.join(suggestion_block)


def _rebind_interface(induced_ptools: list[dict]):
    """Rebuild compute_airline_total_cost.doc and rebind to ReActFactory.

    Since Interface.doc is set at decoration time and not a live property,
    we mutate it directly on the Interface object. ReActFactory reads
    interface.doc fresh at every call, so this propagates immediately.
    """
    new_doc = _build_task_doc_with_ptools(induced_ptools)
    compute_airline_total_cost.doc = new_doc
    # Rebind (no actual tool list changes — induced ptools live in the doc)
    compute_airline_total_cost.implement_via('react', max_steps=14, tools=[])


# ═══════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════

app = typer.Typer()


@app.command()
def baseline(
    n_cases: int = typer.Option(5, help='Number of airline cases'),
    model: str = typer.Option('together_ai/deepseek-ai/DeepSeek-V3', help='LLM model'),
    seed: int = typer.Option(42, help='Random seed'),
):
    """Run baseline ReAct (no tools) on rulearena airline using ReActFactory."""
    config.configure(cfg={
        'llm': {'model': model, 'timeout': 300, 'max_tokens': 8192},
        'cachier': {
            'cache_dir': str(BASE_DIR / 'traces' / 'llm_cache_airline_v2'),
            'enable_caching': True,
        },
    })
    cache_path = BASE_DIR / 'traces' / f'react_traces_airline_v2_n{n_cases}.json'
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    cases = load_airline_cases(n_cases, seed)
    print(f'Running ReActFactory baseline on {len(cases)} airline cases')
    print(f'  Model: {model}, Max steps: 14, Seed: {seed}')
    print(f'  Cache: {cache_path}')
    print()

    traces = _run_all_cases(cases, model)

    with open(cache_path, 'w') as f:
        json.dump(traces, f, indent=2, default=str)

    n_correct = sum(1 for t in traces if t.get('correct'))
    avg_cost = sum(t.get('stats', {}).get('cost', 0) for t in traces) / max(len(traces), 1)
    by_term: dict[str, int] = {}
    for t in traces:
        by_term[t.get('termination', 'unknown')] = by_term.get(t.get('termination', 'unknown'), 0) + 1
    print()
    print(f'Baseline accuracy: {n_correct}/{len(traces)} ({n_correct/len(traces):.1%})')
    print(f'Avg cost: ${avg_cost:.4f}')
    print(f'Termination breakdown: {by_term}')
    print(f'Saved traces to {cache_path}')


@app.command()
def run(
    n_cases: int = typer.Option(30, help='Number of cases'),
    n_iters: int = typer.Option(5, help='Max induction iterations'),
    model: str = typer.Option('together_ai/deepseek-ai/DeepSeek-V3', help='LLM model'),
    seed: int = typer.Option(42, help='Random seed'),
    min_count: int = typer.Option(3, help='Min pattern frequency to induce'),
    sim_threshold: float = typer.Option(0.75, help='Docstring sim threshold'),
    batch_size: int = typer.Option(1, help='Ptools to induce per iteration'),
    structured_output: bool = typer.Option(True, help='Structured output (G fix)'),
    output_dir: str = typer.Option('iterations_v6cg_airline_v2', help='Output directory'),
    reuse_iter0: bool = typer.Option(True, help='Reuse cached baseline as iter 0'),
):
    """Run iterative ptool induction v6cg-style on airline (Path A: ReActFactory)."""
    out_dir = BASE_DIR / output_dir
    config.configure(cfg={
        'llm': {'model': model, 'timeout': 300, 'max_tokens': 8192},
        'cachier': {
            'cache_dir': str(out_dir / 'llm_cache'),
            'enable_caching': True,
        },
    })
    out_dir.mkdir(parents=True, exist_ok=True)
    ptools_dir = out_dir / 'ptools'

    cases = load_airline_cases(n_cases, seed)
    induced: list[dict] = []
    iteration_stats = []

    print(f'Iterative Ptool Induction v6cg — RULEARENA AIRLINE v2')
    print(f'  Source: thoughts (paragraph-split), batch={batch_size}, sim_threshold={sim_threshold}')
    print(f'  Structured output: {structured_output}, min_count={min_count}')
    print(f'  Cases: {n_cases}, Max iters: {n_iters}')
    print(f'  Output: {out_dir}\n')

    for iteration in range(n_iters):
        iter_path = out_dir / f'iter_{iteration}'
        iter_path.mkdir(parents=True, exist_ok=True)

        print(f'\n{"="*60}')
        print(f'ITERATION {iteration}  |  ptools: {[p["display_name"] for p in induced]}')
        print(f'{"="*60}')

        # Rebind interface with current induced ptools in docstring
        _rebind_interface(induced)

        # Step 1: Run all cases (use cached baseline for iter 0 if available)
        cached_baseline = BASE_DIR / 'traces' / f'react_traces_airline_v2_n{n_cases}.json'
        if iteration == 0 and reuse_iter0 and cached_baseline.exists():
            with open(cached_baseline) as f:
                cached = json.load(f)
            case_names = {c['name'] for c in cases}
            cached_names = {t['case_name'] for t in cached}
            if case_names.issubset(cached_names):
                traces = [t for t in cached if t['case_name'] in case_names]
                print(f'  Loaded {len(traces)} cached baseline traces from {cached_baseline.name}')
            else:
                print(f'  Cache missing some cases — running fresh ReAct...')
                traces = _run_all_cases(cases, model)
        else:
            print(f'  Running ReAct with {len(induced)} induced ptools (in docstring), max_steps=14...')
            traces = _run_all_cases(cases, model)

        with open(iter_path / 'traces.json', 'w') as f:
            json.dump(traces, f, indent=2, default=str)
        save_sample_traces(traces, iter_path)

        # Step 2: Stats
        n_correct = sum(1 for t in traces if t.get('correct', False))
        accuracy = n_correct / len(traces) if traces else 0
        total_cost = sum(t.get('stats', {}).get('cost', 0) for t in traces)
        avg_cost = total_cost / len(traces) if traces else 0
        avg_thought_len = sum(
            len(s.get('thought') or '')
            for t in traces for s in t['steps']
        ) / max(sum(len(t['steps']) for t in traces), 1)

        stats = dict(
            iteration=iteration, n_ptools=len(induced),
            accuracy=accuracy, n_correct=n_correct, n_total=len(traces),
            avg_cost=avg_cost, avg_thought_len=avg_thought_len,
        )
        iteration_stats.append(stats)
        print(f'\n  Accuracy: {n_correct}/{len(traces)} ({accuracy:.1%})')
        print(f'  Avg cost: ${avg_cost:.4f}  |  Avg thought len: {avg_thought_len:.0f} chars')

        # Step 3: Extract paragraphs
        items = extract_thought_paragraphs(traces)
        print(f'\n  Extracted {len(items)} thought paragraphs')
        if not items:
            print('  No items to induce from. Stopping.')
            break

        # Step 4: Categorize
        categories = categorize_items_air(items, source='thoughts', model=model)
        print(f'  Top patterns:')
        for cat, count, _ in categories[:8]:
            print(f'    {count:3d}x  {cat}')

        # Step 5: Synthesize
        print(f'\n  Synthesizing ptools (structured={structured_output})...')
        new_ptools = select_and_synthesize_air(
            categories, induced, model,
            batch_size=batch_size, sim_threshold=sim_threshold,
            min_count=min_count, next_id=len(induced) + 1,
            structured_output=structured_output,
        )

        if not new_ptools:
            print('  No new orthogonal ptools found. Converged!')
            break

        for p in new_ptools:
            save_ptool(p, ptools_dir)
            induced.append(p)
            print(f'  New ptool: {p["display_name"]} — {p["short_desc"]}')

        save_ptools_as_python(induced, out_dir / 'ptools_induced.py')
        meta = {**stats, 'new_ptools': new_ptools}
        with open(iter_path / 'meta.json', 'w') as f:
            json.dump(meta, f, indent=2, default=str)

    # Final summary
    print(f'\n\n{"="*60}')
    print(f'FINAL SUMMARY')
    print(f'{"="*60}')
    print(f'{"Iter":>4}  {"Pt":>3}  {"Acc":>8}  {"Avg$":>9}  {"AvgThoughtLen":>15}')
    for s in iteration_stats:
        print(f'{s["iteration"]:4d}  {s["n_ptools"]:3d}  '
              f'{s["accuracy"]:7.1%}  ${s["avg_cost"]:.4f}  '
              f'{s["avg_thought_len"]:15.0f}')

    summary = {
        'iteration_stats': iteration_stats,
        'ptools': induced,
        'config': dict(n_cases=n_cases, n_iters=n_iters, model=model,
                        seed=seed, min_count=min_count,
                        sim_threshold=sim_threshold, batch_size=batch_size,
                        source='thoughts', structured_output=structured_output,
                        output_dir=output_dir, task='rulearena_airline_v2'),
    }
    with open(out_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f'\nSummary saved to {out_dir / "summary.json"}')


if __name__ == '__main__':
    app()
