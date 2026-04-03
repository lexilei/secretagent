"""Iterative ptool induction v5 — fixed functional similarity + structured outputs.

Two fixes over v4:
  Fix F: Use ptool-specific focus args for functional testing (not generic "analyze all suspects")
  Fix G: Ptools specify structured output format in docstring (e.g., {suspect: true/false})

Usage:
    source .env && export TOGETHER_API_KEY

    # Fix F only: ptool-specific focus args
    uv run python experiments/induction/iterate_v5.py --output-dir iterations_v5f

    # Fix G only: structured output docstrings
    uv run python experiments/induction/iterate_v5.py --structured-output --output-dir iterations_v5g

    # Both fixes
    uv run python experiments/induction/iterate_v5.py --structured-output --output-dir iterations_v5fg
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

# Reuse from v3
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
    save_sample_traces,
)

BASE_DIR = Path(__file__).parent

_EMBED_MODEL = None

def _get_embed_model():
    global _EMBED_MODEL
    if _EMBED_MODEL is None:
        _EMBED_MODEL = SentenceTransformer('all-MiniLM-L6-v2')
    return _EMBED_MODEL


# ═══════════════════════════════════════════════════════════════
# Fixed functional similarity — ptool-specific focus args
# ═══════════════════════════════════════════════════════════════

_FUNCTIONAL_CACHE: dict[str, list[str]] = {}
_SAMPLE_NARRATIVES: list[str] = []


def _get_sample_narratives(n: int = 3) -> list[str]:
    global _SAMPLE_NARRATIVES
    if not _SAMPLE_NARRATIVES:
        cases = load_murder_cases(n=10, seed=99)
        _SAMPLE_NARRATIVES = [c['narrative'] for c in cases[:n]]
    return _SAMPLE_NARRATIVES


def _get_ptool_outputs(ptool: dict) -> list[str]:
    """Run ptool on sample narratives with PTOOL-SPECIFIC focus args."""
    cache_key = ptool['id']
    if cache_key in _FUNCTIONAL_CACHE:
        return _FUNCTIONAL_CACHE[cache_key]

    iface = _INDUCED_INTERFACES.get(ptool['func_name'])
    if iface is None:
        iface = create_ptool_interface(ptool['func_name'], ptool['doc'])

    narratives = _get_sample_narratives()

    # FIX F: Use the ptool's own examples as focus args, or derive from short_desc
    focus_args = []
    if ptool.get('examples'):
        # Use actual example texts from the traces as focus
        for ex in ptool['examples'][:3]:
            focus_args.append(ex[:100])
    if len(focus_args) < len(narratives):
        # Fallback: derive focus from the ptool's short description
        focus_args.extend([ptool.get('short_desc', 'analyze')] * (len(narratives) - len(focus_args)))

    outputs = []
    for narrative, focus in zip(narratives, focus_args):
        try:
            result = iface(narrative=narrative, focus=focus)
            outputs.append(str(result)[:500])
        except Exception:
            outputs.append("")

    _FUNCTIONAL_CACHE[cache_key] = outputs
    return outputs


def _docstring_similarity(new_ptool: dict, existing_ptools: list[dict]) -> float:
    if not existing_ptools:
        return 0.0
    model = _get_embed_model()
    texts = [new_ptool['doc']] + [p['doc'] for p in existing_ptools]
    embeddings = model.encode(texts, normalize_embeddings=True)
    sims = embeddings[0] @ embeddings[1:].T
    return float(np.max(sims))


def _functional_similarity(new_ptool: dict, existing_ptools: list[dict]) -> float:
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


def compute_similarity(new_ptool: dict, existing_ptools: list[dict]) -> tuple[float, dict]:
    """Combined similarity: average of docstring + functional."""
    doc = _docstring_similarity(new_ptool, existing_ptools)
    func = _functional_similarity(new_ptool, existing_ptools)
    score = (doc + func) / 2
    return score, {'docstring': doc, 'functional': func, 'combined': score}


# ═══════════════════════════════════════════════════════════════
# Ptool synthesis — with optional structured output format
# ═══════════════════════════════════════════════════════════════

def synthesize_ptool_v5(pattern_name: str, examples: list[dict],
                        model: str, ptool_id: str,
                        structured_output: bool = False) -> dict:
    """Synthesize a ptool. If structured_output=True, docstring specifies output format."""
    examples_text = ''
    for ex in examples[:5]:
        examples_text += f'  - "{ex["text"][:200]}"\n'

    if structured_output:
        output_instruction = """
4. output_format: Specify EXACTLY what the output should look like. Use a structured format like:
   - A dict/JSON with specific keys (e.g., {"suspect_name": {"has_alibi": true/false, "alibi_details": "...", "contradictions": [...]}})
   - A list of specific items
   - A boolean judgment with reasoning

The output format MUST be different from other tools — if one tool returns a per-suspect dict, another should return a timeline list, another should return a boolean verdict, etc. The output format is critical for making tools functionally distinct.

Include the output format specification in the docstring under a "Returns:" section with a concrete example."""
    else:
        output_instruction = ""

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
2. display_name: CamelCase version for the agent to call
3. short_desc: one sentence for the agent prompt
{output_instruction}

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

    iface = create_ptool_interface(spec['func_name'], spec['docstring'])

    return {
        'id': ptool_id,
        'func_name': spec['func_name'],
        'display_name': spec['display_name'],
        'short_desc': spec['short_desc'],
        'doc': spec['docstring'],
        'source_pattern': pattern_name,
        'examples': [ex['text'][:200] for ex in examples[:5]],
    }


def select_and_synthesize(categories: list[tuple[str, int, list]],
                          existing_ptools: list[dict],
                          model: str, batch_size: int,
                          sim_threshold: float,
                          min_count: int,
                          next_id: int,
                          structured_output: bool) -> list[dict]:
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
        candidate = synthesize_ptool_v5(cat, examples, model, ptool_id,
                                         structured_output=structured_output)

        sim, detail = compute_similarity(candidate, all_ptools)
        detail_str = ', '.join(f'{k}={v:.2f}' for k, v in detail.items())

        if sim > sim_threshold:
            print(f'  SKIP "{cat}" — too similar ({detail_str} > {sim_threshold})')
            continue

        print(f'  ACCEPT "{cat}" — {candidate["display_name"]} ({detail_str})')
        new_ptools.append(candidate)
        all_ptools.append(candidate)

    return new_ptools


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
    sim_threshold: float = typer.Option(0.75, help='Similarity threshold'),
    batch_size: int = typer.Option(1, help='Ptools per iteration'),
    source: str = typer.Option('actions', help='Induction source: actions or thoughts'),
    structured_output: bool = typer.Option(False, help='Fix G: structured output format in docstrings'),
    output_dir: str = typer.Option('iterations_v5', help='Output directory name'),
):
    """Run iterative ptool induction v5 (fixed functional sim + structured outputs)."""
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

    fixes = []
    fixes.append('F: ptool-specific focus args')
    if structured_output:
        fixes.append('G: structured output docstrings')

    print(f'Iterative Ptool Induction v5')
    print(f'  Fixes: {", ".join(fixes)}')
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

        print(f'\n  Analyzing {source}...')
        items = extract_items(traces, source)
        print(f'  Found {len(items)} {source}')
        if not items:
            break

        categories = categorize_items(items, source, model)
        print(f'  Top patterns:')
        for cat, count, _ in categories[:8]:
            print(f'    {count:3d}x  {cat}')

        print(f'\n  Synthesizing ptools (structured_output={structured_output})...')
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
            print(f'    Doc preview: {p["doc"][:150]}...')

        save_ptools_as_python(ptools, out_dir / 'ptools_induced.py')

        meta = {**stats, 'new_ptools': new_ptools}
        with open(iter_path / 'meta.json', 'w') as f:
            json.dump(meta, f, indent=2, default=str)

    # Final summary
    print(f'\n\n{"="*60}')
    print(f'FINAL SUMMARY (fixes: {", ".join(fixes)})')
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
        print(f'    "{p["short_desc"]}"')

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
    """Run ReAct on all cases with retry + delay to handle API rate limits."""
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
                    wait = 2 ** attempt  # 1s, 2s, 4s
                    print(f'RETRY({attempt+1}, wait {wait}s)...', end=' ', flush=True)
                    time.sleep(wait)
                else:
                    print(f'ERROR after {max_retries} attempts: {ex}')
                    result = dict(answer=-1, steps=[], n_steps=0,
                                  termination='error', stats={})

        correct = (result['answer'] == case['expected'])
        result['case_name'] = case['name']
        result['expected'] = case['expected']
        result['correct'] = correct
        results.append(result)

        status = 'OK' if correct else 'WRONG'
        print(f'{status} (pred={result["answer"]}, exp={case["expected"]}, steps={result["n_steps"]})')

        # Small delay between cases to avoid rate limit bursts
        time.sleep(0.5)

    return results


if __name__ == '__main__':
    app()
