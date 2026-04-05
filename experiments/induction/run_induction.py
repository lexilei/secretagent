"""Unified induction pipeline. One script, YAML-configured.

Usage:
    source .env && export TOGETHER_API_KEY
    uv run python experiments/induction/run_induction.py conf/thoughts_structured.yaml
    uv run python experiments/induction/run_induction.py conf/actions_plain.yaml
"""

import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / 'src'))
sys.path.insert(0, str(_PROJECT_ROOT / 'benchmarks' / 'musr'))

import typer
from omegaconf import OmegaConf
from secretagent import config as sa_config

from induction.data import load_cases
from induction.react_loop import run_all_cases
from induction.pattern_analysis import extract_items, categorize_items
from induction.synthesis import select_and_synthesize
from induction.io import save_ptool, save_ptools_as_python, save_sample_traces

BASE_DIR = Path(__file__).parent
CONF_DIR = BASE_DIR / 'conf'

app = typer.Typer()


@app.command()
def run(config_file: Path = typer.Argument(..., help='YAML config file')):
    """Run iterative ptool induction from YAML config."""
    defaults = OmegaConf.load(CONF_DIR / 'defaults.yaml')
    user_cfg = OmegaConf.load(config_file)
    cfg = OmegaConf.merge(defaults, user_cfg)

    out_dir = BASE_DIR / cfg.output_dir
    sa_config.configure(cfg={
        'llm': {'model': cfg.model, 'timeout': 300},
        'cachier': {
            'cache_dir': str(out_dir / 'llm_cache'),
            'enable_caching': True,
        },
    })

    out_dir.mkdir(parents=True, exist_ok=True)
    ptools_dir = out_dir / 'ptools'

    data = load_cases(OmegaConf.to_container(cfg, resolve=True))
    train_cases = data['train']
    val_cases = data.get('val')
    benchmark = cfg.get('data', {}).get('benchmark', 'murder')

    ptools: list[dict] = []
    iteration_stats = []
    ptool_snapshots: list[list[dict]] = []  # ptools at each iteration
    best_train_acc = 0.0
    best_train_iter = 0
    no_improve_count = 0

    print(f'Iterative Ptool Induction')
    print(f'  Config: {config_file}')
    print(f'  Train: {len(train_cases)}, Val: {len(val_cases) if val_cases else "none"}')
    print(f'  Source: {cfg.source}, Batch: {cfg.batch_size}, Sim: {cfg.similarity.mode}>{cfg.similarity.threshold}')
    print(f'  Parsing: {cfg.parsing.mode}, Structured: {cfg.synthesis.structured_output}')
    print(f'  Model: {cfg.model}, Max steps: {cfg.max_steps}, Max iters: {cfg.n_iters}')
    print(f'  Output: {out_dir}\n')

    for iteration in range(cfg.n_iters):
        iter_path = out_dir / f'iter_{iteration}'
        iter_path.mkdir(parents=True, exist_ok=True)

        print(f'\n{"="*60}')
        print(f'ITERATION {iteration}  |  ptools: {[p["display_name"] for p in ptools]}')
        print(f'{"="*60}')

        # Run ReAct on train set
        print(f'  [TRAIN] Running ReAct on {len(train_cases)} cases...')
        traces = run_all_cases(
            train_cases, ptools, cfg.model, cfg.max_steps,
            parsing_mode=cfg.parsing.mode,
            benchmark=benchmark,
        )

        with open(iter_path / 'train_traces.json', 'w') as f:
            json.dump(traces, f, indent=2, default=str)

        # Train metrics
        n_correct = sum(1 for t in traces if t.get('correct'))
        accuracy = n_correct / len(traces)
        avg_cost = sum(t.get('stats', {}).get('cost', 0) for t in traces) / len(traces)
        avg_latency = sum(t.get('stats', {}).get('latency', 0) for t in traces) / len(traces)
        ptool_uses = sum(1 for t in traces for s in t['steps'] if s.get('ptool_used'))
        do_uses = sum(1 for t in traces for s in t['steps'] if s.get('action_type') == 'do')
        neg1 = sum(1 for t in traces if t['answer'] == -1)
        extracted = sum(1 for t in traces if 'extracted' in t.get('termination', ''))
        avg_steps = sum(t['n_steps'] for t in traces) / len(traces)

        stats = dict(
            iteration=iteration, n_ptools=len(ptools),
            train_acc=accuracy, train_correct=n_correct,
            avg_cost=avg_cost, avg_latency=avg_latency,
            ptool_uses=ptool_uses, do_uses=do_uses,
            neg1=neg1, extracted=extracted, avg_steps=avg_steps,
        )
        iteration_stats.append(stats)

        print(f'  [TRAIN] Accuracy: {n_correct}/{len(traces)} ({accuracy:.1%})')
        print(f'  [TRAIN] Steps: {avg_steps:.1f}  |  Cost: ${avg_cost:.4f}  |  Ptool: {ptool_uses}  |  Do: {do_uses}  |  -1: {neg1}  |  Extr: {extracted}')

        # Track best iteration (must beat baseline iter 0 to count)
        ptool_snapshots.append(list(ptools))
        if accuracy >= best_train_acc:
            best_train_acc = accuracy
            best_train_iter = iteration
            no_improve_count = 0
        else:
            no_improve_count += 1

        # Early stopping: stop if accuracy drops below baseline for 2 straight
        # Only start checking after iter 1 (need at least one ptool to evaluate)
        if no_improve_count >= 2 and iteration >= 2:
            print(f'  Early stopping: no improvement for {no_improve_count} iterations')
            break

        # Analyze patterns from train traces
        filter_mode = cfg.get('filter_mode', 'all')
        rank_by = cfg.get('rank_by', 'frequency')
        print(f'\n  [INDUCE] Analyzing {cfg.source} (filter={filter_mode}, rank={rank_by})...')
        items = extract_items(traces, cfg.source, filter_mode=filter_mode)
        print(f'  Found {len(items)} {cfg.source}')
        if not items:
            break

        categories = categorize_items(items, cfg.source, cfg.model, rank_by=rank_by)
        print(f'  Top patterns:')
        for cat, count, _ in categories[:8]:
            print(f'    {count:3d}x  {cat}')

        # Synthesize ptools
        print(f'\n  Synthesizing ptools...')
        new_ptools = select_and_synthesize(
            categories, ptools, cfg.model,
            batch_size=cfg.batch_size,
            sim_threshold=cfg.similarity.threshold,
            sim_mode=cfg.similarity.mode,
            min_count=cfg.min_count,
            next_id=len(ptools) + 1,
            structured_output=cfg.synthesis.structured_output,
            benchmark=benchmark,
        )

        if not new_ptools:
            print('  No new orthogonal ptools found. Converged!')
            break

        # Cap total ptools
        max_ptools = cfg.get('max_ptools', 10)
        remaining_slots = max_ptools - len(ptools)
        if remaining_slots <= 0:
            print(f'  Max ptools ({max_ptools}) reached. Stopping induction.')
            break
        new_ptools = new_ptools[:remaining_slots]

        for p in new_ptools:
            save_ptool(p, ptools_dir)
            ptools.append(p)
            print(f'  New ptool: {p["display_name"]} — {p["short_desc"]}')

        save_ptools_as_python(ptools, out_dir / 'ptools_induced.py')

        meta = {**stats, 'new_ptools': new_ptools}
        with open(iter_path / 'meta.json', 'w') as f:
            json.dump(meta, f, indent=2, default=str)

    # === Validation: use ptools from best training iteration ===
    if val_cases:
        val_ptools = ptool_snapshots[best_train_iter] if ptool_snapshots else []
        print(f'\n{"="*60}')
        print(f'VALIDATION ({len(val_cases)} cases, {len(val_ptools)} ptools from best iter {best_train_iter}, train={best_train_acc:.1%})')
        print(f'{"="*60}')
        val_traces = run_all_cases(
            val_cases, val_ptools, cfg.model, cfg.max_steps,
            parsing_mode=cfg.parsing.mode,
            benchmark=benchmark,
        )

        val_path = out_dir / 'val'
        val_path.mkdir(parents=True, exist_ok=True)
        with open(val_path / 'traces.json', 'w') as f:
            json.dump(val_traces, f, indent=2, default=str)
        save_sample_traces(val_traces, val_path)

        val_correct = sum(1 for t in val_traces if t.get('correct'))
        val_acc = val_correct / len(val_traces)
        val_neg1 = sum(1 for t in val_traces if t['answer'] == -1)
        val_extr = sum(1 for t in val_traces if 'extracted' in t.get('termination', ''))
        val_cost = sum(t.get('stats', {}).get('cost', 0) for t in val_traces) / len(val_traces)
        val_steps = sum(t['n_steps'] for t in val_traces) / len(val_traces)

        print(f'  [VAL] Accuracy: {val_correct}/{len(val_traces)} ({val_acc:.1%})')
        print(f'  [VAL] Steps: {val_steps:.1f}  |  Cost: ${val_cost:.4f}  |  -1: {val_neg1}  |  Extr: {val_extr}')

    # === Optional test ===
    if cfg.get('eval_test', False):
        test_cases = load_cases(OmegaConf.to_container(cfg, resolve=True))['test']
        print(f'\n{"="*60}')
        print(f'TEST ({len(test_cases)} cases)')
        print(f'{"="*60}')
        test_traces = run_all_cases(
            test_cases, ptools, cfg.model, cfg.max_steps,
            parsing_mode=cfg.parsing.mode,
            benchmark=benchmark,
        )
        test_path = out_dir / 'test'
        test_path.mkdir(parents=True, exist_ok=True)
        with open(test_path / 'traces.json', 'w') as f:
            json.dump(test_traces, f, indent=2, default=str)
        test_correct = sum(1 for t in test_traces if t.get('correct'))
        print(f'  [TEST] Accuracy: {test_correct}/{len(test_traces)} ({test_correct/len(test_traces):.1%})')

    # Summary
    print(f'\n\n{"="*60}')
    print(f'FINAL SUMMARY')
    print(f'{"="*60}')
    print(f'{"It":>2} {"#Pt":>3} {"Train":>9} {"-1":>3} {"Ex":>3} {"Pt":>4} {"Do":>4} {"Stp":>4} {"$/c":>6}')
    print('-' * 50)
    for s in iteration_stats:
        print(f'{s["iteration"]:>2} {s["n_ptools"]:>3} '
              f'{s["train_correct"]}/75={s["train_acc"]:.0%} '
              f'{s["neg1"]:>3} {s["extracted"]:>3} '
              f'{s["ptool_uses"]:>4} {s["do_uses"]:>4} '
              f'{s["avg_steps"]:>4.1f} ${s["avg_cost"]:.3f}')

    print(f'\nBest train iter: {best_train_iter} ({best_train_acc:.1%}, {len(ptool_snapshots[best_train_iter]) if ptool_snapshots else 0} ptools)')

    if val_cases:
        print(f'val_acc: {val_acc:.1%}')
        print(f'Val: {val_correct}/{len(val_traces)}={val_acc:.1%}')

    print(f'\nPtools (best iter {best_train_iter}):')
    best_pt = ptool_snapshots[best_train_iter] if ptool_snapshots else []
    for p in best_pt:
        print(f'  {p["id"]}: {p["func_name"]}(narrative, focus) — {p["short_desc"]}')

    summary = {
        'iteration_stats': iteration_stats,
        'ptools': ptools,
        'best_train_iter': best_train_iter,
        'best_train_acc': best_train_acc,
        'best_ptools': ptool_snapshots[best_train_iter] if ptool_snapshots else [],
        'config': OmegaConf.to_container(cfg, resolve=True),
    }
    if val_cases:
        summary['val'] = {'accuracy': val_acc, 'correct': val_correct,
                          'total': len(val_traces), 'neg1': val_neg1}
    with open(out_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f'\nSaved to {out_dir / "summary.json"}')


if __name__ == '__main__':
    app()
