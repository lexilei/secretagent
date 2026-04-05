"""Extract and categorize reasoning patterns from ReAct traces."""

import json
import re
from collections import Counter

from secretagent.llm_util import llm as llm_cached


def extract_items(traces: list[dict], source: str,
                  filter_mode: str = 'all') -> list[dict]:
    """Extract free-form items from traces.

    source: 'actions' or 'thoughts'
    filter_mode: 'all' (default), 'correct_only', 'incorrect_only'
    """
    items = []
    for t in traces:
        if filter_mode == 'correct_only' and not t.get('correct'):
            continue
        if filter_mode == 'incorrect_only' and t.get('correct'):
            continue
        for s in t['steps']:
            if source == 'actions':
                if s.get('action_type') == 'do' and s.get('action_arg', '').strip():
                    items.append(dict(case=t['case_name'], step=s['step'],
                                      text=s['action_arg'].strip(),
                                      correct=t.get('correct', False)))
            elif source == 'thoughts':
                if s.get('thought', '').strip() and s.get('action_type') != 'finish':
                    items.append(dict(case=t['case_name'], step=s['step'],
                                      text=s['thought'].strip(),
                                      correct=t.get('correct', False)))
    return items


def categorize_items(items: list[dict], source: str, model: str,
                     batch_size: int = 30,
                     rank_by: str = 'frequency') -> list[tuple[str, int, list]]:
    """Categorize items via LLM + merge synonyms."""
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

    if rank_by == 'impact' and any(item.get('correct') is not None for item in items):
        # Rank by success correlation: patterns that appear more in correct traces
        def impact_score(cat_tuple):
            cat, count, examples = cat_tuple
            if count < 2:
                return -1  # too few to judge
            n_correct = sum(1 for e in examples if e.get('correct'))
            success_rate = n_correct / count
            # Score = success_rate * log(count) — balance impact with frequency
            import math
            return success_rate * math.log(count + 1)
        result.sort(key=impact_score, reverse=True)

    return result
