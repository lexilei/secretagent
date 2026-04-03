"""Ptool synthesis from patterns + orthogonality-filtered selection."""

import json
import re

from secretagent.llm_util import llm as llm_cached
from .ptool_registry import create_ptool_interface
from .similarity import compute_similarity


def synthesize_ptool(pattern_name: str, examples: list[dict],
                     model: str, ptool_id: str,
                     structured_output: bool = False) -> dict:
    """Synthesize a real @interface ptool from a pattern and examples."""
    examples_text = ''
    for ex in examples[:5]:
        examples_text += f'  - "{ex["text"][:200]}"\n'

    if structured_output:
        output_instruction = """
4. output_format: Specify EXACTLY what the output should look like. Use a structured format like:
   - A dict/JSON with specific keys (e.g., {"suspect_name": {"has_alibi": true/false, "alibi_details": "...", "contradictions": [...]}})
   - A list of specific items
   - A boolean judgment with reasoning

The output format MUST be different from other tools. Include the output format specification in the docstring under a "Returns:" section with a concrete example."""
    else:
        output_instruction = ""

    prompt = f"""You are designing a reusable reasoning tool (Python function) for solving murder mystery puzzles.

The tool captures this frequently used reasoning action: "{pattern_name}"

Examples of how agents described this action:
{examples_text}

The tool will be a Python function with this signature:
    def tool_name(narrative: str, focus: str) -> str

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

    create_ptool_interface(spec['func_name'], spec['docstring'])

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
                          sim_mode: str,
                          min_count: int,
                          next_id: int,
                          structured_output: bool) -> list[dict]:
    """Select top patterns, synthesize ptools, filter by similarity."""
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
        candidate = synthesize_ptool(cat, examples, model, ptool_id,
                                      structured_output=structured_output)

        sim, detail = compute_similarity(candidate, all_ptools, sim_mode)
        detail_str = ', '.join(f'{k}={v:.2f}' for k, v in detail.items())

        if sim > sim_threshold:
            print(f'  SKIP "{cat}" — too similar ({detail_str} > {sim_threshold})')
            continue

        print(f'  ACCEPT "{cat}" — {candidate["display_name"]} ({detail_str})')
        new_ptools.append(candidate)
        all_ptools.append(candidate)

    return new_ptools
