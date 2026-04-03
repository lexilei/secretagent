"""Iterative ptool induction v7 — proper train/val/test splits.

Same as v6 but uses pre-split data:
  - train (75 cases): run ReAct, collect traces, induce ptools
  - val (75 cases): evaluate induced ptools
  - test (100 cases): final held-out evaluation (run manually)

Usage:
    source .env && export TOGETHER_API_KEY

    # Best config from v6: thoughts + structured
    uv run python experiments/induction/iterate_v7.py --structured-output --source thoughts --output-dir iterations_v7cg

    # Actions + structured
    uv run python experiments/induction/iterate_v7.py --structured-output --output-dir iterations_v7g
"""

import json
import re
import sys
import time
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

# Reuse from v3/v5/v6
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
from iterate_v5 import (
    compute_similarity,
    select_and_synthesize,
    synthesize_ptool_v5,
)
from iterate_v6 import (
    extract_answer_index,
    _llm_with_stop,
    _make_system_prompt,
    _parse_action,
    run_react_on_case,
)

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / 'data'


def load_split(split: str) -> list[dict]:
    """Load a pre-split data file (train.json, val.json, test.json)."""
    path = DATA_DIR / f'{split}.json'
    if not path.exists():
        raise FileNotFoundError(f'{path} not found. Run data_split.py first.')
    with open(path) as f:
        return json.load(f)


app = typer.Typer()


@app.command()
def run(
    n_iters: int = typer.Option(5, help='Max iterations'),
    model: str = typer.Option('together_ai/deepseek-ai/DeepSeek-V3', help='LLM model'),
    min_count: int = typer.Option(3, help='Min pattern frequency'),
    max_steps: int = typer.Option(14, help='Max ReAct steps'),
    sim_threshold: float = typer.Option(0.75, help='Similarity threshold'),
    batch_size: int = typer.Option(1, help='Ptools per iteration'),
    source: str = typer.Option('actions', help='actions or thoughts'),
    structured_output: bool = typer.Option(False, help='Structured output docstrings'),
    output_dir: str = typer.Option('iterations_v7', help='Output directory'),
    eval_test: bool = typer.Option(False, help='Also evaluate on test set at end'),
):
    """Run iterative ptool induction v7 (train/val/test splits)."""
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

    train_cases = load_split('train')
    val_cases = load_split('val')
    print(f'Loaded train={len(train_cases)}, val={len(val_cases)}')

    ptools: list[dict] = []
    iteration_stats = []

    fixes = ['F: ptool-specific focus', 'H: LLM answer extraction', 'split: train/val/test']
    if structured_output:
        fixes.append('G: structured output')

    print(f'Iterative Ptool Induction v7 (train/val split)')
    print(f'  Fixes: {", ".join(fixes)}')
    print(f'  Train: {len(train_cases)}, Val: {len(val_cases)}, Max iters: {n_iters}, Max steps: {max_steps}')
    print(f'  Source: {source}, Batch: {batch_size}, Sim threshold: {sim_threshold}')
    print(f'  Output: {out_dir}\n')

    for iteration in range(n_iters):
        iter_path = out_dir / f'iter_{iteration}'
        iter_path.mkdir(parents=True, exist_ok=True)

        print(f'\n{"="*60}')
        print(f'ITERATION {iteration}  |  ptools: {[p["display_name"] for p in ptools]}')
        print(f'{"="*60}')

        # === TRAIN: run ReAct, collect traces, induce ptools ===
        print(f'  [TRAIN] Running ReAct on {len(train_cases)} cases...')
        train_traces = _run_all_cases(train_cases, ptools, model, max_steps)

        with open(iter_path / 'train_traces.json', 'w') as f:
            json.dump(train_traces, f, indent=2, default=str)

        train_correct = sum(1 for t in train_traces if t.get('correct'))
        train_acc = train_correct / len(train_traces)
        print(f'  [TRAIN] Accuracy: {train_correct}/{len(train_traces)} ({train_acc:.1%})')

        # === VAL: evaluate current ptools ===
        print(f'  [VAL] Running ReAct on {len(val_cases)} cases...')
        val_traces = _run_all_cases(val_cases, ptools, model, max_steps)

        with open(iter_path / 'val_traces.json', 'w') as f:
            json.dump(val_traces, f, indent=2, default=str)
        save_sample_traces(val_traces, iter_path)

        val_correct = sum(1 for t in val_traces if t.get('correct'))
        val_acc = val_correct / len(val_traces)
        total_cost = sum(t.get('stats', {}).get('cost', 0) for t in val_traces)
        total_latency = sum(t.get('stats', {}).get('latency', 0) for t in val_traces)
        avg_cost = total_cost / len(val_traces)
        avg_latency = total_latency / len(val_traces)
        ptool_uses = sum(1 for t in val_traces for s in t['steps'] if s.get('ptool_used'))
        do_uses = sum(1 for t in val_traces for s in t['steps'] if s.get('action_type') == 'do')
        neg1 = sum(1 for t in val_traces if t['answer'] == -1)
        extracted = sum(1 for t in val_traces if 'extracted' in t.get('termination', ''))
        avg_steps = sum(t['n_steps'] for t in val_traces) / len(val_traces)

        stats = dict(
            iteration=iteration, n_ptools=len(ptools),
            train_acc=train_acc, train_correct=train_correct,
            val_acc=val_acc, val_correct=val_correct,
            avg_cost=avg_cost, avg_latency=avg_latency,
            ptool_uses=ptool_uses, do_uses=do_uses,
            neg1=neg1, extracted=extracted, avg_steps=avg_steps,
        )
        iteration_stats.append(stats)

        print(f'  [VAL] Accuracy: {val_correct}/{len(val_cases)} ({val_acc:.1%})')
        print(f'  [VAL] Avg cost: ${avg_cost:.4f}  |  Avg latency: {avg_latency:.1f}s')
        print(f'  [VAL] Ptool uses: {ptool_uses}  |  Do[]: {do_uses}  |  -1: {neg1}  |  Extracted: {extracted}')

        # === INDUCE: analyze train traces, synthesize ptools ===
        print(f'\n  [INDUCE] Analyzing {source} from train traces...')
        items = extract_items(train_traces, source)
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

    # === Optional: TEST evaluation ===
    if eval_test:
        print(f'\n{"="*60}')
        print(f'TEST EVALUATION ({len(load_split("test"))} cases)')
        print(f'{"="*60}')
        test_cases = load_split('test')
        test_traces = _run_all_cases(test_cases, ptools, model, max_steps)

        test_path = out_dir / 'test'
        test_path.mkdir(parents=True, exist_ok=True)
        with open(test_path / 'traces.json', 'w') as f:
            json.dump(test_traces, f, indent=2, default=str)

        test_correct = sum(1 for t in test_traces if t.get('correct'))
        test_acc = test_correct / len(test_traces)
        print(f'  TEST Accuracy: {test_correct}/{len(test_traces)} ({test_acc:.1%})')

    # Summary
    print(f'\n\n{"="*60}')
    print(f'FINAL SUMMARY')
    print(f'{"="*60}')
    print(f'{"It":>2} {"#Pt":>3} {"Train":>9} {"Val":>9} {"-1":>3} {"Ex":>3} {"Pt":>4} {"Do":>4} {"Stp":>4} {"$/c":>6} {"Lat":>5}')
    print('-' * 65)
    for s in iteration_stats:
        print(f'{s["iteration"]:>2} {s["n_ptools"]:>3} '
              f'{s["train_correct"]}/75={s["train_acc"]:.0%} '
              f'{s["val_correct"]}/75={s["val_acc"]:.0%} '
              f'{s["neg1"]:>3} {s["extracted"]:>3} '
              f'{s["ptool_uses"]:>4} {s["do_uses"]:>4} '
              f'{s["avg_steps"]:>4.1f} ${s["avg_cost"]:.3f} {s["avg_latency"]:>4.0f}s')

    print(f'\nInduced ptools:')
    for p in ptools:
        print(f'  {p["id"]}: {p["func_name"]}(narrative, focus)')

    summary = {
        'iteration_stats': iteration_stats,
        'ptools': ptools,
        'config': dict(n_iters=n_iters, model=model,
                        min_count=min_count, max_steps=max_steps,
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
        print(f'    [{idx+1}/{len(cases)}] {case["name"]}...', end=' ', flush=True)
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
