"""Iterative ptool induction v4 — functional similarity checking.

Same as v3 but with configurable similarity modes:
  --sim-mode docstring   : cosine sim of docstring embeddings (v3 default)
  --sim-mode functional  : cosine sim of actual ptool outputs on sample narratives
  --sim-mode combined    : average of docstring + functional similarity

Usage:
    source .env && export TOGETHER_API_KEY

    # Exp D: functional similarity only
    uv run python experiments/induction/iterate_v4.py --sim-mode functional --output-dir iterations_v4d

    # Exp E: combined (docstring + functional)
    uv run python experiments/induction/iterate_v4.py --sim-mode combined --output-dir iterations_v4e
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

# Reuse everything from v3 except similarity
from iterate_v3 import (
    _INDUCED_INTERFACES,
    create_ptool_interface,
    find_induced_ptool,
    save_ptool,
    save_ptools_as_python,
    format_ptool_actions_for_prompt,
    _llm_with_stop,
    _execute_do_action,
    _execute_ptool,
    _make_system_prompt,
    FEW_SHOT,
    _parse_action,
    run_react_on_case,
    extract_items,
    categorize_items,
    synthesize_ptool,
    save_sample_traces,
    _run_all_cases,
)

BASE_DIR = Path(__file__).parent

_EMBED_MODEL = None

def _get_embed_model():
    global _EMBED_MODEL
    if _EMBED_MODEL is None:
        _EMBED_MODEL = SentenceTransformer('all-MiniLM-L6-v2')
    return _EMBED_MODEL


# ═══════════════════════════════════════════════════════════════
# Similarity modes
# ═══════════════════════════════════════════════════════════════

def _docstring_similarity(new_ptool: dict, existing_ptools: list[dict]) -> float:
    """Cosine similarity of docstring embeddings (same as v3)."""
    if not existing_ptools:
        return 0.0
    model = _get_embed_model()
    texts = [new_ptool['doc']] + [p['doc'] for p in existing_ptools]
    embeddings = model.encode(texts, normalize_embeddings=True)
    sims = embeddings[0] @ embeddings[1:].T
    return float(np.max(sims))


# Cache of ptool outputs on sample narratives: {ptool_id: [output1, output2, ...]}
_FUNCTIONAL_CACHE: dict[str, list[str]] = {}
_SAMPLE_NARRATIVES: list[str] = []


def _get_sample_narratives(n: int = 3) -> list[str]:
    """Load a few sample narratives for functional testing."""
    global _SAMPLE_NARRATIVES
    if not _SAMPLE_NARRATIVES:
        cases = load_murder_cases(n=10, seed=99)  # different seed to avoid overlap
        _SAMPLE_NARRATIVES = [c['narrative'] for c in cases[:n]]
    return _SAMPLE_NARRATIVES


def _get_ptool_outputs(ptool: dict) -> list[str]:
    """Run a ptool on sample narratives, cache results."""
    if ptool['id'] in _FUNCTIONAL_CACHE:
        return _FUNCTIONAL_CACHE[ptool['id']]

    iface = _INDUCED_INTERFACES.get(ptool['func_name'])
    if iface is None:
        # ptool was created in a previous run, recreate it
        iface = create_ptool_interface(ptool['func_name'], ptool['doc'])

    narratives = _get_sample_narratives()
    outputs = []
    for narrative in narratives:
        try:
            result = iface(narrative=narrative, focus="analyze all suspects")
            outputs.append(str(result)[:500])
        except Exception:
            outputs.append("")

    _FUNCTIONAL_CACHE[ptool['id']] = outputs
    return outputs


def _functional_similarity(new_ptool: dict, existing_ptools: list[dict]) -> float:
    """Cosine similarity of actual ptool outputs on sample narratives."""
    if not existing_ptools:
        return 0.0

    model = _get_embed_model()
    new_outputs = _get_ptool_outputs(new_ptool)
    new_text = " ".join(new_outputs)

    max_sim = 0.0
    for ep in existing_ptools:
        ep_outputs = _get_ptool_outputs(ep)
        ep_text = " ".join(ep_outputs)

        embeddings = model.encode([new_text, ep_text], normalize_embeddings=True)
        sim = float(embeddings[0] @ embeddings[1])
        max_sim = max(max_sim, sim)

    return max_sim


def _combined_similarity(new_ptool: dict, existing_ptools: list[dict]) -> float:
    """Average of docstring and functional similarity."""
    doc_sim = _docstring_similarity(new_ptool, existing_ptools)
    func_sim = _functional_similarity(new_ptool, existing_ptools)
    return (doc_sim + func_sim) / 2


def compute_similarity(new_ptool: dict, existing_ptools: list[dict],
                       mode: str) -> tuple[float, dict]:
    """Compute similarity with breakdown."""
    if mode == 'docstring':
        score = _docstring_similarity(new_ptool, existing_ptools)
        detail = {'docstring': score}
    elif mode == 'functional':
        score = _functional_similarity(new_ptool, existing_ptools)
        detail = {'functional': score}
    elif mode == 'combined':
        doc = _docstring_similarity(new_ptool, existing_ptools)
        func = _functional_similarity(new_ptool, existing_ptools)
        score = (doc + func) / 2
        detail = {'docstring': doc, 'functional': func, 'combined': score}
    else:
        raise ValueError(f'Unknown sim mode: {mode}')
    return score, detail


def select_and_synthesize(categories: list[tuple[str, int, list]],
                          existing_ptools: list[dict],
                          model: str, batch_size: int,
                          sim_threshold: float,
                          min_count: int,
                          next_id: int,
                          sim_mode: str) -> list[dict]:
    """Select top patterns and synthesize, with configurable similarity."""
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

        sim, detail = compute_similarity(candidate, all_ptools, sim_mode)
        detail_str = ', '.join(f'{k}={v:.2f}' for k, v in detail.items())

        if sim > sim_threshold:
            print(f'  SKIP "{cat}" — too similar ({detail_str} > {sim_threshold})')
            continue

        print(f'  ACCEPT "{cat}" — {candidate["display_name"]} ({detail_str})')
        new_ptools.append(candidate)
        all_ptools.append(candidate)

    return new_ptools


# ═══════════════════════════════════════════════════════════════
# Main (same structure as v3, just uses configurable similarity)
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
    sim_threshold: float = typer.Option(0.75, help='Similarity threshold'),
    batch_size: int = typer.Option(1, help='Ptools per iteration'),
    source: str = typer.Option('actions', help='Induction source: actions or thoughts'),
    sim_mode: str = typer.Option('functional', help='Similarity mode: docstring, functional, combined'),
    output_dir: str = typer.Option('iterations_v4', help='Output directory name'),
):
    """Run iterative ptool induction v4 (configurable similarity)."""
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

    print(f'Iterative Ptool Induction v4 (sim_mode={sim_mode})')
    print(f'  Cases: {n_cases}, Max iters: {n_iters}, Max steps: {max_steps}')
    print(f'  Source: {source}, Batch: {batch_size}, Sim threshold: {sim_threshold}')
    print(f'  Similarity mode: {sim_mode}')
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
            print(f'  Running ReAct with {len(ptools)} ptools, max_steps={max_steps}...')
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

        # --- Step 4: Synthesize with configurable similarity ---
        print(f'\n  Synthesizing ptools (sim_mode={sim_mode}, threshold={sim_threshold})...')
        new_ptools = select_and_synthesize(
            categories, ptools, model,
            batch_size=batch_size, sim_threshold=sim_threshold,
            min_count=min_count, next_id=len(ptools) + 1,
            sim_mode=sim_mode,
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

    # --- Final summary ---
    print(f'\n\n{"="*60}')
    print(f'FINAL SUMMARY (sim_mode={sim_mode})')
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
        print(f'  {p["id"]}: {p["func_name"]}(narrative, focus) -> str')

    summary = {
        'iteration_stats': iteration_stats,
        'ptools': ptools,
        'config': dict(n_cases=n_cases, n_iters=n_iters, model=model,
                        seed=seed, min_count=min_count, max_steps=max_steps,
                        sim_threshold=sim_threshold, batch_size=batch_size,
                        source=source, sim_mode=sim_mode, output_dir=output_dir),
    }
    with open(out_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f'\nSummary saved to {out_dir / "summary.json"}')


if __name__ == '__main__':
    app()
