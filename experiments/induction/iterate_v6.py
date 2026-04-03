"""Iterative ptool induction v6 — LLM-based answer extraction.

Same as v5 but fixes Finish parsing by using extract_index (LLM call)
after each step to check if the agent has reached a conclusion.

Catches: Finish[1] (Alice), Finish[1] in thought, conclusions without Finish.

Usage:
    source .env && export TOGETHER_API_KEY
    uv run python experiments/induction/iterate_v6.py --structured-output --output-dir iterations_v6g
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
from secretagent.core import interface as make_interface, Interface
from secretagent.llm_util import llm as llm_cached
from sentence_transformers import SentenceTransformer

from run_react import load_murder_cases

from iterate_v3 import (
    _INDUCED_INTERFACES,
    create_ptool_interface,
    find_induced_ptool,
    save_ptool,
    save_ptools_as_python,
    format_ptool_actions_for_prompt,
    _execute_do_action,
    _execute_ptool,
    FEW_SHOT,
    extract_items,
    categorize_items,
    save_sample_traces,
)

# Reuse similarity from v5
from iterate_v5 import (
    _get_sample_narratives,
    _get_ptool_outputs,
    _docstring_similarity,
    _functional_similarity,
    compute_similarity,
    select_and_synthesize,
    synthesize_ptool_v5,
)

BASE_DIR = Path(__file__).parent


# ═══════════════════════════════════════════════════════════════
# LLM-based answer extraction (from ptools_common pattern)
# ═══════════════════════════════════════════════════════════════

def extract_answer_index(text: str, choices: list, model: str) -> int | None:
    """Use LLM to extract an answer index from free-form text.

    Returns 0-based index if found, None if the agent hasn't decided yet.
    Based on the extract_index pattern from ptools_common.py.
    """
    choices_str = ', '.join(f'({i}) {c}' for i, c in enumerate(choices))
    prompt = (
        f"The following text is from an agent solving a murder mystery. "
        f"The answer choices are: {choices_str}\n\n"
        f"Text:\n{text}\n\n"
        f"Has the agent decided on a final answer? If yes, return ONLY the "
        f"0-based index number. If the agent is still investigating and has "
        f"not committed to a final answer, return ONLY the word 'none'.\n\n"
        f"<answer>index or none</answer>"
    )
    response, stats = llm_cached(prompt, model)
    match = re.search(r'<answer>\s*(\w+)\s*</answer>', response)
    if match:
        val = match.group(1).strip()
        if val.lower() == 'none':
            return None
        try:
            idx = int(val)
            if 0 <= idx < len(choices):
                return idx
        except ValueError:
            pass
    # Fallback: try to find a bare number
    nums = re.findall(r'\b(\d)\b', response)
    if len(nums) == 1:
        idx = int(nums[0])
        if 0 <= idx < len(choices):
            return idx
    return None


# ═══════════════════════════════════════════════════════════════
# Modified ReAct loop with LLM-based Finish detection
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
    return text, dict(
        input_tokens=response.usage.prompt_tokens,
        output_tokens=response.usage.completion_tokens,
        latency=latency, cost=cost,
    )


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
            f'When you have enough evidence to decide, immediately use Finish[index].\n'
        )
    else:
        base += (
            '(2) Finish[answer], which returns the 0-based index of the correct '
            'answer choice and finishes the task.\n'
        )
    base += 'Here are some examples.\n'
    return base


def _parse_action(text: str) -> tuple[str, str]:
    match = re.match(r'\s*(\w+)\[(.+)\]\s*$', text.strip(), re.DOTALL)
    if match:
        return match.group(1), match.group(2).strip()
    # Relaxed: allow trailing text after ]
    match = re.match(r'\s*(\w+)\[(.+?)\]', text.strip(), re.DOTALL)
    if match:
        return match.group(1), match.group(2).strip()
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

        # Check for Finish (regex-based, now with relaxed parser)
        if action_name_lower == 'finish':
            try:
                answer = int(action_arg.strip().split()[0])
            except ValueError:
                answer = -1
            # If regex couldn't get it, use LLM
            if answer == -1:
                answer = extract_answer_index(action_str, choices, model) or -1
            steps.append(dict(
                step=i, thought=thought, action=action_str,
                action_type='finish', action_arg=action_arg,
                observation=None, ptool_used=None,
            ))
            return dict(
                answer=answer, steps=steps, n_steps=i,
                termination='finish', stats=total_stats,
            )

        # Check if agent concluded in thought (LLM extraction)
        if not action_str or action_name_lower == '':
            # No action — check if thought contains a conclusion
            idx = extract_answer_index(thought, choices, model)
            if idx is not None:
                steps.append(dict(
                    step=i, thought=thought, action='[extracted from thought]',
                    action_type='finish_extracted', action_arg=str(idx),
                    observation=None, ptool_used=None,
                ))
                return dict(
                    answer=idx, steps=steps, n_steps=i,
                    termination='finish_extracted', stats=total_stats,
                )

        # Ptool call
        ptool_iface = find_induced_ptool(action_name) if action_name_lower != 'do' else None

        if ptool_iface:
            observation, env_stats = _execute_ptool(ptool_iface, narrative, action_arg)
            accum(env_stats)
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

    # Max steps reached — last resort: try to extract answer from full history
    full_text = ' '.join(s.get('thought', '') for s in steps[-3:])
    last_idx = extract_answer_index(full_text, choices, model)
    if last_idx is not None:
        return dict(
            answer=last_idx, steps=steps, n_steps=max_steps,
            termination='finish_extracted_at_max', stats=total_stats,
        )

    return dict(
        answer=-1, steps=steps, n_steps=max_steps,
        termination='max_steps', stats=total_stats,
    )


# ═══════════════════════════════════════════════════════════════
# Main (same structure as v5)
# ═══════════════════════════════════════════════════════════════

app = typer.Typer()


@app.command()
def run(
    n_cases: int = typer.Option(75, help='Number of MUSR cases'),
    n_iters: int = typer.Option(5, help='Max iterations'),
    model: str = typer.Option('together_ai/deepseek-ai/DeepSeek-V3', help='LLM model'),
    seed: int = typer.Option(42, help='Random seed'),
    min_count: int = typer.Option(3, help='Min pattern frequency'),
    reuse_iter0: bool = typer.Option(True, help='Use existing traces for iter 0'),
    max_steps: int = typer.Option(14, help='Max ReAct steps'),
    sim_threshold: float = typer.Option(0.75, help='Similarity threshold'),
    batch_size: int = typer.Option(1, help='Ptools per iteration'),
    source: str = typer.Option('actions', help='actions or thoughts'),
    structured_output: bool = typer.Option(False, help='Structured output docstrings'),
    output_dir: str = typer.Option('iterations_v6', help='Output directory'),
):
    """Run iterative ptool induction v6 (LLM-based answer extraction)."""
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

    fixes = ['F: ptool-specific focus', 'H: LLM answer extraction']
    if structured_output:
        fixes.append('G: structured output')

    print(f'Iterative Ptool Induction v6')
    print(f'  Fixes: {", ".join(fixes)}')
    print(f'  Cases: {n_cases}, Max iters: {n_iters}, Max steps: {max_steps}')
    print(f'  Source: {source}, Batch: {batch_size}, Sim threshold: {sim_threshold}')
    print(f'  Output: {out_dir}\n')

    for iteration in range(n_iters):
        iter_path = out_dir / f'iter_{iteration}'
        iter_path.mkdir(parents=True, exist_ok=True)

        print(f'\n{"="*60}')
        print(f'ITERATION {iteration}  |  ptools: {[p["display_name"] for p in ptools]}')
        print(f'{"="*60}')

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
            print(f'  Running ReAct with {len(ptools)} ptools, max_steps={max_steps}...')
            traces = _run_all_cases(cases, ptools, model, max_steps)

        with open(iter_path / 'traces.json', 'w') as f:
            json.dump(traces, f, indent=2, default=str)
        save_sample_traces(traces, iter_path)

        n_correct = sum(1 for t in traces if t.get('correct', False))
        accuracy = n_correct / len(traces) if traces else 0
        total_cost = sum(t.get('stats', {}).get('cost', 0) for t in traces)
        total_latency = sum(t.get('stats', {}).get('latency', 0) for t in traces)
        avg_cost = total_cost / len(traces) if traces else 0
        avg_latency = total_latency / len(traces) if traces else 0
        ptool_uses = sum(1 for t in traces for s in t['steps'] if s.get('ptool_used'))
        do_uses = sum(1 for t in traces for s in t['steps'] if s.get('action_type') == 'do')
        max_step_hits = sum(1 for t in traces if t.get('termination') == 'max_steps')
        extracted = sum(1 for t in traces if 'extracted' in t.get('termination', ''))
        neg1_answers = sum(1 for t in traces if t.get('answer') == -1)

        stats = dict(
            iteration=iteration, n_ptools=len(ptools),
            accuracy=accuracy, n_correct=n_correct, n_total=len(traces),
            avg_cost=avg_cost, avg_latency=avg_latency,
            ptool_uses=ptool_uses, do_uses=do_uses,
            max_step_hits=max_step_hits, neg1_answers=neg1_answers,
            extracted_finishes=extracted,
        )
        iteration_stats.append(stats)

        print(f'\n  Accuracy: {n_correct}/{len(traces)} ({accuracy:.1%})')
        print(f'  Avg cost: ${avg_cost:.4f}  |  Avg latency: {avg_latency:.1f}s')
        print(f'  Ptool uses: {ptool_uses}  |  Do[]: {do_uses}')
        print(f'  Max step hits: {max_step_hits}  |  -1 answers: {neg1_answers}  |  Extracted finishes: {extracted}')

        print(f'\n  Analyzing {source}...')
        items = extract_items(traces, source)
        print(f'  Found {len(items)} {source}')
        if not items:
            break

        categories = categorize_items(items, source, model)
        print(f'  Top patterns:')
        for cat, count, _ in categories[:8]:
            print(f'    {count:3d}x  {cat}')

        print(f'\n  Synthesizing ptools (structured={structured_output})...')
        new_ptools = select_and_synthesize(
            categories, ptools, model,
            batch_size=batch_size, sim_threshold=sim_threshold,
            min_count=min_count, next_id=len(ptools) + 1,
            structured_output=structured_output,
        )

        if not new_ptools:
            print('  No new orthogonal ptools found. Converged!')
            break

        for p in new_ptools:
            save_ptool(p, ptools_dir)
            ptools.append(p)
            print(f'  New ptool: {p["display_name"]} — {p["short_desc"]}')

        save_ptools_as_python(ptools, out_dir / 'ptools_induced.py')

        meta = {**stats, 'new_ptools': new_ptools}
        with open(iter_path / 'meta.json', 'w') as f:
            json.dump(meta, f, indent=2, default=str)

    # Summary
    print(f'\n\n{"="*60}')
    print(f'FINAL SUMMARY')
    print(f'{"="*60}')
    print(f'{"Iter":>4}  {"Pt":>3}  {"Accuracy":>8}  {"Avg$":>8}  '
          f'{"Ptool":>5}  {"Do[]":>5}  {"MxH":>4}  {"-1":>3}  {"Extr":>4}')
    for s in iteration_stats:
        print(f'{s["iteration"]:4d}  {s["n_ptools"]:3d}  '
              f'{s["accuracy"]:7.1%}  ${s["avg_cost"]:.4f}  '
              f'{s["ptool_uses"]:5d}  {s["do_uses"]:5d}  '
              f'{s["max_step_hits"]:4d}  {s["neg1_answers"]:3d}  {s["extracted_finishes"]:4d}')

    print(f'\nInduced ptools:')
    for p in ptools:
        print(f'  {p["id"]}: {p["func_name"]}(narrative, focus)')

    summary = {
        'iteration_stats': iteration_stats,
        'ptools': ptools,
        'config': dict(n_cases=n_cases, n_iters=n_iters, model=model,
                        seed=seed, min_count=min_count, max_steps=max_steps,
                        sim_threshold=sim_threshold, batch_size=batch_size,
                        source=source, structured_output=structured_output,
                        output_dir=output_dir, fixes=fixes),
    }
    with open(out_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f'\nSummary saved to {out_dir / "summary.json"}')


def _run_all_cases(cases: list[dict], ptools: list[dict],
                   model: str, max_steps: int) -> list[dict]:
    results = []
    max_retries = 3
    for idx, case in enumerate(cases):
        print(f'  [{idx+1}/{len(cases)}] {case["name"]}...', end=' ', flush=True)
        result = None
        for attempt in range(max_retries):
            try:
                result = run_react_on_case(
                    case['narrative'], case['question'], case['choices'],
                    ptools, model, max_steps=max_steps)
                break
            except Exception as ex:
                if attempt < max_retries - 1:
                    wait = 2 ** attempt
                    print(f'RETRY({attempt+1}, wait {wait}s)...', end=' ', flush=True)
                    time.sleep(wait)
                else:
                    print(f'ERROR: {ex}')
                    result = dict(answer=-1, steps=[], n_steps=0,
                                  termination='error', stats={})
        correct = (result['answer'] == case['expected'])
        result['case_name'] = case['name']
        result['expected'] = case['expected']
        result['correct'] = correct
        results.append(result)
        term = result.get('termination', '')
        tag = ' [extracted]' if 'extracted' in term else ''
        status = 'OK' if correct else 'WRONG'
        print(f'{status} (pred={result["answer"]}, exp={case["expected"]}, steps={result["n_steps"]}{tag})')
        time.sleep(0.5)
    return results


if __name__ == '__main__':
    app()
