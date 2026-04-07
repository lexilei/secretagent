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

@interface
def compute_airline_total_cost(narrative: str) -> int:
    ...


# Re-attach the long docstring (the @interface decorator preserves __doc__,
# but we want to set it dynamically based on the loaded rules file).
compute_airline_total_cost.__doc__ = _AIRLINE_TASK_DOC

# Bind to ReAct factory with no tools initially. For induction iterations,
# rebind with tools=[induced_ptool_1, induced_ptool_2, ...].
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
        'llm': {'model': model, 'timeout': 300},
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


if __name__ == '__main__':
    app()
