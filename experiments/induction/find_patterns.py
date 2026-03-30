"""Step 2: Find the most common reasoning patterns in successful ReAct traces.

Algorithmically discovers frequent reasoning steps by:
1. Extracting all "thought" steps from successful traces
2. Using an LLM to categorize each thought into a reasoning action type
3. Counting frequency of each action type across all traces
4. Reporting the most common patterns

Usage:
    uv run python experiments/induction/find_patterns.py
"""

import json
import sys
from collections import Counter
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / 'src'))

import typer
from secretagent import config
from secretagent.llm_util import llm

TRACES_DIR = Path(__file__).parent / 'traces'
OUTPUT_DIR = Path(__file__).parent / 'patterns'


def extract_thoughts(traces: list[dict]) -> list[dict]:
    """Extract all thought steps from successful traces."""
    thoughts = []
    for trace in traces:
        case_name = trace['case_name']
        for i, step in enumerate(trace['steps']):
            if 'thought' in step:
                thoughts.append({
                    'case_name': case_name,
                    'step_index': i,
                    'thought': step['thought'],
                })
    return thoughts


def categorize_thoughts_batch(thoughts: list[dict], model: str) -> list[dict]:
    """Use LLM to categorize each thought into a reasoning action type.

    This is the algorithmic step — we don't manually label, we ask the LLM
    to identify what type of reasoning action each thought represents.
    """
    # Build a prompt with all thoughts for batch categorization
    thoughts_text = ""
    for i, t in enumerate(thoughts):
        # Truncate long thoughts
        text = t['thought'][:500] if len(t['thought']) > 500 else t['thought']
        thoughts_text += f"\n[{i}] (case: {t['case_name']}, step {t['step_index']})\n{text}\n"

    prompt = f"""You are analyzing reasoning traces from an AI agent solving murder mystery puzzles.
Below are individual "thought" steps the agent took. Your job is to categorize each thought
into a short, reusable REASONING ACTION TYPE (3-5 words max).

Examples of good action types:
- "extract suspect list"
- "analyze physical evidence"
- "verify alibi claims"
- "identify contradictions"
- "weigh evidence strength"
- "eliminate unlikely suspects"
- "synthesize final conclusion"

Rules:
- Use consistent, canonical names (don't create synonyms)
- Each category should be a distinct reasoning step, not a sub-step
- Categories should be general enough to appear across different cases
- Output ONLY a JSON array of objects with "index" and "category" fields

Thoughts to categorize:
{thoughts_text}

Output format (JSON array):
[
  {{"index": 0, "category": "extract suspect list"}},
  {{"index": 1, "category": "analyze physical evidence"}},
  ...
]

<answer>
YOUR JSON ARRAY
</answer>"""

    response, stats = llm(prompt, model)

    # Parse the response
    import re
    match = re.search(r'<answer>(.*)</answer>', response, re.DOTALL)
    if not match:
        # Try to find JSON array directly
        match = re.search(r'\[.*\]', response, re.DOTALL)
        if match:
            json_str = match.group(0)
        else:
            raise ValueError(f"Could not parse categories from LLM response:\n{response[:500]}")
    else:
        json_str = match.group(1).strip()

    categories = json.loads(json_str)

    # Merge categories back into thoughts
    cat_map = {c['index']: c['category'] for c in categories}
    for i, t in enumerate(thoughts):
        t['category'] = cat_map.get(i, 'unknown')

    return thoughts


def find_frequent_patterns(categorized_thoughts: list[dict]) -> list[tuple[str, int]]:
    """Count frequency of each reasoning action type."""
    counter = Counter(t['category'] for t in categorized_thoughts)
    return counter.most_common()


def generate_ptool_stub(pattern_name: str, examples: list[dict], model: str) -> str:
    """Generate a ptool stub for the most common reasoning pattern."""
    examples_text = ""
    for ex in examples[:3]:
        text = ex['thought'][:300]
        examples_text += f"\nExample from {ex['case_name']}:\n{text}\n"

    prompt = f"""Based on these examples of the reasoning step "{pattern_name}" from murder mystery solving,
write a Python function stub (interface) that could be used as a reusable reasoning primitive.

The stub should:
1. Have a clear function name (snake_case)
2. Take relevant inputs (narrative, evidence, etc.)
3. Return a string with the analysis
4. Have a detailed docstring explaining what this step does

Examples of this reasoning step:
{examples_text}

Write ONLY the Python function stub with @interface decorator:

<answer>
YOUR PYTHON CODE
</answer>"""

    response, stats = llm(prompt, model)
    import re
    match = re.search(r'<answer>(.*)</answer>', response, re.DOTALL)
    if match:
        return match.group(1).strip()
    return response.strip()


app = typer.Typer()

@app.command()
def main(
    model: str = typer.Option(
        "together_ai/deepseek-ai/DeepSeek-V3",
        help="LLM model for categorization"),
    top_k: int = typer.Option(5, help="Number of top patterns to report"),
):
    """Find frequent reasoning patterns in successful ReAct traces."""
    config.configure(cfg={
        'llm': {'model': model, 'timeout': 300},
        'cachier': {
            'cache_dir': str(TRACES_DIR / 'llm_cache'),
            'enable_caching': True,
        },
    })

    # Load success traces
    success_file = TRACES_DIR / 'success_traces.json'
    if not success_file.exists():
        print(f'No traces found at {success_file}. Run run_react.py first.')
        return

    with open(success_file) as f:
        traces = json.load(f)

    print(f'Loaded {len(traces)} successful traces')

    # Step 1: Extract all thoughts
    thoughts = extract_thoughts(traces)
    print(f'Extracted {len(thoughts)} thought steps')

    if not thoughts:
        print('No thought steps found in traces. Check trace format.')
        return

    # Step 2: Categorize thoughts algorithmically (via LLM)
    print(f'\nCategorizing thoughts via {model}...')
    categorized = categorize_thoughts_batch(thoughts, model)

    # Step 3: Find frequent patterns
    patterns = find_frequent_patterns(categorized)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f'\n{"="*60}')
    print(f'Top {top_k} reasoning patterns:')
    print(f'{"="*60}')
    for i, (pattern, count) in enumerate(patterns[:top_k]):
        pct = count / len(categorized) * 100
        print(f'  {i+1}. "{pattern}" — {count} occurrences ({pct:.0f}%)')

    # Step 4: Generate ptool stubs for top patterns
    print(f'\n{"="*60}')
    print(f'Generating ptool stubs for top {min(top_k, 3)} patterns...')
    print(f'{"="*60}')

    ptool_stubs = []
    for pattern, count in patterns[:min(top_k, 3)]:
        examples = [t for t in categorized if t['category'] == pattern]
        print(f'\n--- Ptool for "{pattern}" ---')
        stub = generate_ptool_stub(pattern, examples, model)
        print(stub)
        ptool_stubs.append({
            'pattern': pattern,
            'count': count,
            'stub': stub,
        })

    # Save results
    results = {
        'n_traces': len(traces),
        'n_thoughts': len(categorized),
        'patterns': [{'pattern': p, 'count': c} for p, c in patterns],
        'categorized_thoughts': categorized,
        'ptool_stubs': ptool_stubs,
    }
    output_file = OUTPUT_DIR / 'pattern_analysis.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f'\nFull analysis saved to {output_file}')


if __name__ == '__main__':
    app()
