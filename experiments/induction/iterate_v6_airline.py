"""Iterative ptool induction v6 — RuleArena airline baggage fee task.

Forked from iterate_v6_calendar.py for the airline domain in rulearena.
Major differences from prior tasks:
  - Numeric answer (integer dollars), not text/MCQ. Eval is tolerance-based
    via _within_tolerance (1% rel error) using compute_airline_fee from
    benchmarks/rulearena/calculators/airline.py.
  - Reference rules text (~5KB) is loaded from
    benchmarks/rulearena/data/airline/reference_rules_textual.txt and
    injected into the agent's system prompt. The agent must reason from
    these rules to compute the answer.
  - The "narrative" is the natural-language problem statement (e.g.,
    "Linda is a Premium Economy passenger flying from Shenzhen to Salt
    Lake City with the following items: ...").
  - `expected` is an integer ground-truth dollar amount.
  - Finish[<integer>] parses an int (or "$1234"); evaluator does
    tolerance compare against ground truth.
  - Cases are stratified across difficulty levels L0/L1/L2 (5 bags / 8
    bags / 11 bags) for a representative sample.

Usage:
    source .env && export TOGETHER_API_KEY

    # Smoke test
    uv run python experiments/induction/iterate_v6_airline.py baseline --n-cases 5
    uv run python experiments/induction/iterate_v6_airline.py run \\
        --n-cases 5 --n-iters 2 --source thoughts --structured-output \\
        --output-dir iterations_v6cg_air_smoke

    # n=30 runs
    uv run python experiments/induction/iterate_v6_airline.py baseline --n-cases 30
    uv run python experiments/induction/iterate_v6_airline.py run \\
        --n-cases 30 --source thoughts --structured-output \\
        --output-dir iterations_v6cg_airline
    uv run python experiments/induction/iterate_v6_airline.py run \\
        --n-cases 30 --batch-size 3 --structured-output \\
        --output-dir iterations_v6bg_airline
"""

import ast
import json
import random
import re
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import typer

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / 'src'))
sys.path.insert(0, str(_PROJECT_ROOT / 'benchmarks' / 'musr'))
sys.path.insert(0, str(_PROJECT_ROOT / 'benchmarks' / 'rulearena'))

from calculators.airline import compute_airline_fee  # noqa: E402

from secretagent import config
from secretagent.core import Interface
from secretagent.llm_util import llm as llm_cached

# Reuse only task-agnostic machinery from existing pipelines
from iterate_v3 import (
    _INDUCED_INTERFACES,
    create_ptool_interface,
    find_induced_ptool,
    save_ptool,
    save_ptools_as_python,
    format_ptool_actions_for_prompt,
    _execute_ptool,
    _parse_action,
    _llm_with_stop,
    extract_items,
    save_sample_traces,
)
from iterate_v5 import _get_embed_model, _docstring_similarity

BASE_DIR = Path(__file__).parent


# ═══════════════════════════════════════════════════════════════
# Task-specific text (RuleArena airline baggage fee)
# ═══════════════════════════════════════════════════════════════

# Reference rules: ~5KB American Airlines fee policies. Loaded once at module
# import time so the agent prompt can include them verbatim. The narrative the
# agent reasons about is the per-case problem text; the rules are shared
# scaffolding in the system prompt.
_RULES_FILE = _PROJECT_ROOT / 'benchmarks' / 'rulearena' / 'data' / 'airline' / 'reference_rules_textual.txt'
AIRLINE_RULES_TEXT = _RULES_FILE.read_text(encoding='utf-8') if _RULES_FILE.exists() else ''


SYSTEM_PROMPT_INTRO = (
    "Solve an airline baggage fee calculation task. You are an American Airlines "
    "fee assistant. Given a passenger's flight class, route, and bag list, you "
    "must compute the TOTAL COST (ticket fare + checked bag fees + oversize fees "
    "+ overweight fees) using the official policy. Reason step by step using "
    "interleaving Thought, Action, Observation steps. Your final answer must be "
    "an integer dollar amount.\n\n"
    "FORMAT RULES (very important):\n"
    "- Each step must emit BOTH a Thought and an Action.\n"
    "- Keep Thought short (1-3 sentences). Move calculation work into Do[...] requests.\n"
    "- Action must be ONE of: Do[<request>] or Finish[<integer>].\n"
    "- The system will respond with Observation N: ...\n"
    "- Never write a multi-line bag-by-bag breakdown inside a single Thought.\n\n"
    "REFERENCE POLICIES (American Airlines):\n"
    "==========================================\n"
    f"{AIRLINE_RULES_TEXT}\n"
    "==========================================\n"
)

FEW_SHOT_AIR = """\
Here are some examples. Keep each Thought brief (1-3 sentences). Always emit an Action on every step.

Task: Sarah is a Basic Economy Class passenger flying from Dallas to Chicago with the following items:
1. A backpack: 18 x 12 x 8 inches, 5 lbs;
2. A luggage box: 28 x 18 x 12 inches, 40 lbs;

Sarah's flight ticket is $250.

Thought 1: Route Dallas-Chicago is within U.S. Class is Basic Economy. Item 1 is a personal item (within 18x14x8 limit) so free. Item 2 is a checked bag.
Action 1: Do[Compute the checked bag fee for Item 2 (28x18x12 inches, 40 lbs) for a Basic Economy passenger flying within the U.S. Include oversize/overweight checks.]
Observation 1: Bag 2 dimensions sum: 28+18+12=58 inches (under 62, no oversize). Weight 40 lbs (under 50, no overweight). Base fee 1st checked bag, Basic Economy, within U.S. = $40. Total bag fee = $40.
Thought 2: Ticket $250 + bag $40 = $290.
Action 2: Finish[290]

Task: Tom is a Main Cabin Class passenger flying from New York to London with the following items:
1. A backpack: 17 x 13 x 7 inches, 6 lbs;
2. A luggage box: 27 x 19 x 11 inches, 48 lbs;
3. A luggage box: 30 x 20 x 13 inches, 55 lbs;

Tom's flight ticket is $600.

Thought 1: Route NYC-London is between Europe and U.S. Class Main Cabin. Item 1 is personal (free). Items 2 and 3 are checked.
Action 1: Do[For Main Cabin passenger between Europe and U.S., compute base fees for the 1st and 2nd checked bags, and check oversize/overweight surcharges for bags weighing 48 lbs (27x19x11) and 55 lbs (30x20x13).]
Observation 1: Main Cabin Europe<->U.S.: 1st checked = $75, 2nd checked = $100. Bag 2: 27+19+11=57 inches (no oversize), 48 lbs (no overweight). Bag 3: 30+20+13=63 inches (over 62, oversize $150 to Europe), 55 lbs (50-70, overweight $100). Per-bag surcharge is max(oversize, overweight): bag 2 = $0, bag 3 = max($150, $100) = $150.
Thought 2: Total = $600 ticket + $75 + $100 (base) + $0 + $150 (surcharges) = $925.
Action 2: Finish[925]

"""

ENV_PROMPT_TEMPLATE = (
    "You are a careful reasoner assisting with an airline baggage fee "
    "calculation. Read the problem below and respond to this request:\n\n"
    "Request: {request}\n\n"
    "Problem:\n{narrative}\n\n"
    "Respond concisely with concrete numbers. Cite the relevant rule when "
    "applying a fee. Be precise about bag dimensions, weights, route region, "
    "and customer class. Do not invent rules not stated in the policies."
)

EXTRACT_PROMPT_TEMPLATE = (
    "The following text is from an agent computing an airline baggage fee. "
    "Has the agent committed to a final TOTAL COST (in dollars)? If yes, "
    "return ONLY the integer dollar amount (no $, no commas). If the agent "
    "has not committed to a final answer, return ONLY the word 'none'.\n\n"
    "Text:\n{text}\n\n"
    "<answer>integer or none</answer>"
)

CATEGORIZE_TASK_LABEL = "airline baggage fee calculation tasks"
SYNTHESIZE_TASK_LABEL = "airline baggage fee calculation tasks"
SYNTHESIZE_NARRATIVE_LABEL = (
    "the full problem statement: passenger name, customer class, flight route, "
    "list of bags with sizes (length x width x height in inches) and weights "
    "(lbs), and the ticket price"
)


# ═══════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════

def load_airline_cases(n: int, seed: int = 42) -> list[dict]:
    """Load airline cases stratified across difficulty levels L0/L1/L2.

    The test split has 60 cases total (20 each at L0, L1, L2). For n=30 we
    take 10 from each level. For n != 30 we distribute proportionally.
    The narrative is the per-case problem text. The expected answer is the
    integer ground truth from compute_airline_fee(info).
    """
    data_file = _PROJECT_ROOT / 'benchmarks' / 'rulearena' / 'data' / 'airline' / 'test.jsonl'
    with open(data_file) as f:
        examples = [json.loads(l) for l in f]

    # Stratify by level
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
            print(f"  WARNING: ground truth failed for level {ex['level']} idx {ex['orig_idx']}: {e}")
            continue
        cases.append({
            'name': f'airline_L{ex["level"]}_{ex["orig_idx"]}',
            'narrative': ex['prompt'],
            'question': '',
            'choices': [],  # numeric answer; no MCQ choices
            'expected': int(expected),
            'level': ex['level'],
        })
    return cases[:n]


def _baseline_cache_path(n_cases: int) -> Path:
    return BASE_DIR / 'traces' / f'react_traces_airline_n{n_cases}.json'


# ═══════════════════════════════════════════════════════════════
# Task-customized core: env action, system prompt, answer extraction
# ═══════════════════════════════════════════════════════════════

def _execute_do_action_air(action_arg: str, narrative: str, model: str) -> tuple[str, dict]:
    env_prompt = ENV_PROMPT_TEMPLATE.format(request=action_arg, narrative=narrative)
    response, stats = llm_cached(env_prompt, model)
    if len(response) > 1500:
        response = response[:1500] + '...'
    return response.strip(), stats


def _llm_with_stop_air(prompt: str, model: str, stop: list[str]) -> tuple[str, dict]:
    """Override of iterate_v3._llm_with_stop with larger max_tokens.

    Airline reasoning involves long bag breakdowns; the default 512-token cap
    truncates Thoughts mid-sentence and the agent fails to emit an Action.
    """
    from litellm import completion, completion_cost
    messages = [dict(role='user', content=prompt)]
    timeout = config.get('llm.timeout', 300)
    start_time = time.time()
    response = completion(
        model=model, messages=messages, stop=stop,
        max_tokens=2048, timeout=timeout, temperature=0,
    )
    latency = time.time() - start_time
    text = response.choices[0].message.content or ''
    try:
        cost = completion_cost(completion_response=response)
    except Exception:
        cost = 0.0
    stats = dict(
        input_tokens=response.usage.prompt_tokens,
        output_tokens=response.usage.completion_tokens,
        latency=latency, cost=cost,
    )
    return text, stats


def _make_system_prompt_air(ptools: list[dict]) -> str:
    base = (
        SYSTEM_PROMPT_INTRO + "\n"
        "Action can be:\n"
        "(1) Do[description], which performs any reasoning step you describe — "
        "the system will read the task and return the relevant information. "
        "You decide what to investigate.\n"
    )
    if ptools:
        base += format_ptool_actions_for_prompt(ptools) + '\n'
        finish_num = len(ptools) + 2
        base += (
            f'({finish_num}) Finish[<integer>], which finishes the task. '
            f'The integer is the TOTAL COST in dollars (no $, no commas, '
            f'just digits, e.g. Finish[2741]).\n\n'
            f'IMPORTANT: Prefer using the specialized tools above over Do[] when '
            f'they match your reasoning need. Use Do[] only for steps not covered '
            f'by a specialized tool.\n'
            f'When you have computed the total cost, immediately use '
            f'Finish[<integer>].\n'
        )
    else:
        base += (
            '(2) Finish[<integer>], which finishes the task. '
            'The integer is the TOTAL COST in dollars (no $, no commas, '
            'just digits, e.g. Finish[2741]).\n'
        )
    base += 'Here are some examples.\n'
    return base


def extract_numeric_answer_air(text: str, model: str) -> int | None:
    """Use LLM to extract an integer dollar amount from free-form text.

    Returns int or None if the agent has not committed to a final answer.
    Tries regex-only extraction first, then falls back to LLM call.
    """
    # Pattern: "total cost is $1,234" or "total = 1234"
    quick = re.search(r'total\s*(?:cost)?\s*(?:is)?\s*[=:]?\s*\$?\s*([\d,]+)',
                      text, re.IGNORECASE)
    if quick:
        try:
            return int(quick.group(1).replace(',', ''))
        except ValueError:
            pass

    prompt = EXTRACT_PROMPT_TEMPLATE.format(text=text)
    response, _ = llm_cached(prompt, model)
    m = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL)
    if m:
        val = m.group(1).strip()
        if val.lower() == 'none':
            return None
        # Strip $, commas, whitespace
        val_clean = re.sub(r'[\$,\s]', '', val)
        try:
            return int(val_clean)
        except ValueError:
            # try float then int
            try:
                return int(float(val_clean))
            except ValueError:
                pass
    # Last resort: find a number near "$"
    m = re.search(r'\$\s*([\d,]+)', response)
    if m:
        try:
            return int(m.group(1).replace(',', ''))
        except ValueError:
            pass
    return None


def _within_tolerance_int(predicted: int, expected: int, tol: float = 0.01) -> bool:
    """Mirror rulearena's tolerance check: 1% relative error, or abs<0.01 if expected is 0."""
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
# ReAct loop (rulearena airline variant — numeric tolerance eval)
# ═══════════════════════════════════════════════════════════════

def run_react_on_case_air(narrative: str, question: str, choices: list,
                          ptools: list[dict], model: str,
                          max_steps: int = 14) -> dict:
    system = _make_system_prompt_air(ptools)
    # Numeric task: question and choices are unused; narrative IS the problem text.
    # Prefix with "Task:" to match few-shot format.
    prompt = system + FEW_SHOT_AIR + f'\nTask: {narrative}\n'
    steps = []
    total_stats = {'input_tokens': 0, 'output_tokens': 0, 'latency': 0, 'cost': 0}

    def accum(s):
        for k in total_stats:
            total_stats[k] += s.get(k, 0)

    for i in range(1, max_steps + 1):
        llm_input = prompt + f'Thought {i}:'
        stop = [f'\nObservation {i}:']
        response, stats = _llm_with_stop_air(llm_input, model, stop=stop)
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

        if action_name_lower == 'finish':
            # action_arg should be an integer dollar amount; try regex first
            answer = None
            arg_clean = re.sub(r'[\$,\s]', '', action_arg.strip())
            try:
                answer = int(arg_clean)
            except ValueError:
                try:
                    answer = int(float(arg_clean))
                except ValueError:
                    pass
            if answer is None:
                # Fall back to LLM extraction over the full action string
                answer = extract_numeric_answer_air(action_str, model)
            steps.append(dict(
                step=i, thought=thought, action=action_str,
                action_type='finish', action_arg=action_arg,
                observation=None, ptool_used=None,
            ))
            return dict(
                answer=answer, steps=steps, n_steps=i,
                termination='finish', stats=total_stats,
            )

        if not action_str or action_name_lower == '':
            sol = extract_numeric_answer_air(thought, model)
            if sol is not None:
                steps.append(dict(
                    step=i, thought=thought, action='[extracted from thought]',
                    action_type='finish_extracted', action_arg=str(sol),
                    observation=None, ptool_used=None,
                ))
                return dict(
                    answer=sol, steps=steps, n_steps=i,
                    termination='finish_extracted', stats=total_stats,
                )

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
            observation, env_stats = _execute_do_action_air(action_arg, narrative, model)
            accum(env_stats)
            steps.append(dict(
                step=i, thought=thought, action=action_str,
                action_type='do', action_arg=action_arg,
                observation=observation, ptool_used=None,
            ))
        elif action_str:
            observation, env_stats = _execute_do_action_air(action_str, narrative, model)
            accum(env_stats)
            steps.append(dict(
                step=i, thought=thought, action=action_str,
                action_type='do', action_arg=action_str,
                observation=observation, ptool_used=None,
            ))
        else:
            observation = 'No action provided. Please use Do[description] or Finish[<solution string>].'
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

    # Max steps reached — last resort: extract from recent thoughts
    full_text = ' '.join(s.get('thought', '') for s in steps[-3:])
    last_sol = extract_numeric_answer_air(full_text, model)
    if last_sol is not None:
        return dict(
            answer=last_sol, steps=steps, n_steps=max_steps,
            termination='finish_extracted_at_max', stats=total_stats,
        )
    return dict(
        answer='', steps=steps, n_steps=max_steps,
        termination='max_steps', stats=total_stats,
    )


def _run_all_cases_air(cases: list[dict], ptools: list[dict],
                       model: str, max_steps: int) -> list[dict]:
    results = []
    max_retries = 3
    for idx, case in enumerate(cases):
        print(f'  [{idx+1}/{len(cases)}] {case["name"]}...', end=' ', flush=True)
        result = None
        for attempt in range(max_retries):
            try:
                result = run_react_on_case_air(
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
                    result = dict(answer=None, steps=[], n_steps=0,
                                  termination='error', stats={})
        # Numeric tolerance evaluation against ground-truth dollar amount
        pred = result.get('answer', None)
        correct = _within_tolerance_int(pred, case['expected'])
        result['case_name'] = case['name']
        result['expected'] = case['expected']
        result['correct'] = correct
        if 'level' in case:
            result['level'] = case['level']
        results.append(result)
        term = result.get('termination', '')
        tag = ' [extracted]' if 'extracted' in term else ''
        status = 'OK' if correct else 'WRONG'
        print(f'{status} (pred={pred}, exp={case["expected"]}, steps={result["n_steps"]}{tag})')
        time.sleep(0.5)
    return results


# ═══════════════════════════════════════════════════════════════
# Pattern categorization (airline task label)
# ═══════════════════════════════════════════════════════════════

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

        prompt = f"""You are analyzing {source} from an AI agent solving {CATEGORIZE_TASK_LABEL}.
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
    return result


# ═══════════════════════════════════════════════════════════════
# Ptool synthesis (airline task label + structured option)
# ═══════════════════════════════════════════════════════════════

def synthesize_ptool_air(pattern_name: str, examples: list[dict],
                         model: str, ptool_id: str,
                         structured_output: bool = False) -> dict:
    examples_text = ''
    for ex in examples[:5]:
        examples_text += f'  - "{ex["text"][:200]}"\n'

    if structured_output:
        output_instruction = """
4. output_format: Specify EXACTLY what the output should look like. Use a structured format like:
   - A dict/JSON of extracted parameters (e.g., {"customer_class": "Premium Economy", "routine": "China", "direction": 1, "base_price": 1141, "bag_list": [{"id": 1, "name": "backpack", "size": [18, 14, 9], "weight": 6}, ...]})
   - A list of per-bag fee components (e.g., [{"bag_id": 2, "base_fee": 150, "oversize_fee": 0, "overweight_fee": 100, "total": 250, "rule_cited": "checked bag #1, China region, Premium Economy"}, ...])
   - A region/class lookup dict (e.g., {"matched_route_category": "Asia (China)", "matched_class": "Premium Economy", "first_bag_fee": 150, "second_bag_fee": 200, ...})
   - A boolean validity judgment with cited rule (e.g., {"is_oversize": true, "total_inches": 102, "rule_cited": "oversize threshold 62 inches", "surcharge": 200})

The output format MUST be different from other tools — if one tool returns extracted params, another should return per-bag fee components, another should return a region lookup, etc. The output format is critical for making tools functionally distinct.

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
- focus: what specific aspect to focus on (the agent decides — e.g., a target bag id, a route region, a customer class, a specific fee component)

Design the tool:
1. func_name: snake_case Python function name
2. display_name: CamelCase version for the agent to call
3. short_desc: one sentence for the agent prompt
{output_instruction}

The docstring is critical — it drives the LLM that executes this tool. Be specific about:
- What information to extract from the narrative (passenger class, route/region, ticket price, bag list with sizes and weights, how many checked bags, oversize/overweight status)
- How to structure the response
- What to pay attention to (airline fee calculation: the first item is typically a personal item or carry-on (free); checked bag base fees depend on (route region × customer class × bag ordinal); oversize is total dimensions > 62 inches; overweight tiers are 50/70/100 lbs; oversize and overweight surcharges DON'T STACK — apply max(oversize, overweight) per bag; complementary bags from Business/First class get free bags; fee tables are in the reference policies in the system prompt)

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


# ═══════════════════════════════════════════════════════════════
# Functional similarity (uses airline narratives as samples)
# ═══════════════════════════════════════════════════════════════

_FUNCTIONAL_CACHE_AIR: dict[str, list[str]] = {}
_SAMPLE_NARRATIVES_AIR: list[str] = []


def _get_sample_narratives_air(n: int = 3) -> list[str]:
    global _SAMPLE_NARRATIVES_AIR
    if not _SAMPLE_NARRATIVES_AIR:
        cases = load_airline_cases(n=10, seed=99)
        _SAMPLE_NARRATIVES_AIR = [c['narrative'] for c in cases[:n]]
    return _SAMPLE_NARRATIVES_AIR


def _get_ptool_outputs_air(ptool: dict) -> list[str]:
    cache_key = ptool['id']
    if cache_key in _FUNCTIONAL_CACHE_AIR:
        return _FUNCTIONAL_CACHE_AIR[cache_key]
    iface = _INDUCED_INTERFACES.get(ptool['func_name'])
    if iface is None:
        iface = create_ptool_interface(ptool['func_name'], ptool['doc'])
    narratives = _get_sample_narratives_air()
    focus_args = []
    if ptool.get('examples'):
        for ex in ptool['examples'][:3]:
            focus_args.append(ex[:100])
    if len(focus_args) < len(narratives):
        focus_args.extend([ptool.get('short_desc', 'analyze')] * (len(narratives) - len(focus_args)))
    outputs = []
    for narrative, focus in zip(narratives, focus_args):
        try:
            result = iface(narrative=narrative, focus=focus)
            outputs.append(str(result)[:500])
        except Exception:
            outputs.append("")
    _FUNCTIONAL_CACHE_AIR[cache_key] = outputs
    return outputs


def _functional_similarity_air(new_ptool: dict, existing_ptools: list[dict]) -> float:
    if not existing_ptools:
        return 0.0
    model = _get_embed_model()
    new_outputs = _get_ptool_outputs_air(new_ptool)
    new_text = " ".join(new_outputs)
    max_sim = 0.0
    for ep in existing_ptools:
        ep_outputs = _get_ptool_outputs_air(ep)
        ep_text = " ".join(ep_outputs)
        embeddings = model.encode([new_text, ep_text], normalize_embeddings=True)
        sim = float(embeddings[0] @ embeddings[1])
        max_sim = max(max_sim, sim)
    return max_sim


def compute_similarity_air(new_ptool: dict, existing_ptools: list[dict]) -> tuple[float, dict]:
    doc = _docstring_similarity(new_ptool, existing_ptools)
    func = _functional_similarity_air(new_ptool, existing_ptools)
    score = (doc + func) / 2
    return score, {'docstring': doc, 'functional': func, 'combined': score}


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
        sim, detail = compute_similarity_air(candidate, all_ptools)
        detail_str = ', '.join(f'{k}={v:.2f}' for k, v in detail.items())
        if sim > sim_threshold:
            print(f'  SKIP "{cat}" — too similar ({detail_str} > {sim_threshold})')
            continue
        print(f'  ACCEPT "{cat}" — {candidate["display_name"]} ({detail_str})')
        new_ptools.append(candidate)
        all_ptools.append(candidate)
    return new_ptools


# ═══════════════════════════════════════════════════════════════
# CLI commands
# ═══════════════════════════════════════════════════════════════

app = typer.Typer()


@app.command()
def baseline(
    n_cases: int = typer.Option(30, help='Number of airline cases'),
    model: str = typer.Option('together_ai/deepseek-ai/DeepSeek-V3', help='LLM model'),
    seed: int = typer.Option(42, help='Random seed'),
    max_steps: int = typer.Option(14, help='Max ReAct steps'),
):
    """Run baseline ReAct (no ptools) on rulearena airline and cache traces."""
    cache_path = _baseline_cache_path(n_cases)
    config.configure(cfg={
        'llm': {'model': model, 'timeout': 300},
        'cachier': {
            'cache_dir': str(BASE_DIR / 'traces' / 'llm_cache_airline'),
            'enable_caching': True,
        },
    })
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    cases = load_airline_cases(n_cases, seed)
    print(f'Running baseline ReAct on {len(cases)} rulearena airline cases')
    print(f'  Model: {model}, Max steps: {max_steps}, Seed: {seed}')
    print(f'  Cache: {cache_path}\n')

    traces = _run_all_cases_air(cases, ptools=[], model=model, max_steps=max_steps)

    with open(cache_path, 'w') as f:
        json.dump(traces, f, indent=2, default=str)

    n_correct = sum(1 for t in traces if t.get('correct'))
    avg_cost = sum(t.get('stats', {}).get('cost', 0) for t in traces) / max(len(traces), 1)
    extracted = sum(1 for t in traces if 'extracted' in t.get('termination', ''))
    neg1 = sum(1 for t in traces if t.get('answer') == -1)
    print(f'\nBaseline accuracy: {n_correct}/{len(traces)} ({n_correct/len(traces):.1%})')
    print(f'Avg cost: ${avg_cost:.4f}  |  Extracted: {extracted}  |  -1 answers: {neg1}')
    print(f'Saved traces to {cache_path}')


@app.command()
def run(
    n_cases: int = typer.Option(30, help='Number of cases'),
    n_iters: int = typer.Option(5, help='Max iterations'),
    model: str = typer.Option('together_ai/deepseek-ai/DeepSeek-V3', help='LLM model'),
    seed: int = typer.Option(42, help='Random seed'),
    min_count: int = typer.Option(3, help='Min pattern frequency'),
    reuse_iter0: bool = typer.Option(True, help='Use cached baseline traces for iter 0'),
    max_steps: int = typer.Option(14, help='Max ReAct steps'),
    sim_threshold: float = typer.Option(0.75, help='Similarity threshold'),
    batch_size: int = typer.Option(1, help='Ptools per iteration'),
    source: str = typer.Option('actions', help='actions or thoughts'),
    structured_output: bool = typer.Option(False, help='Structured output docstrings'),
    output_dir: str = typer.Option('iterations_v6_airline', help='Output directory'),
):
    """Run iterative ptool induction v6 on rulearena airline."""
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
    cases = load_airline_cases(n_cases, seed)
    ptools: list[dict] = []
    iteration_stats = []

    fixes = ['F: ptool-specific focus', 'H: LLM answer extraction']
    if structured_output:
        fixes.append('G: structured output')

    print(f'Iterative Ptool Induction v6 — RULEARENA AIRLINE')
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

        cache_path = _baseline_cache_path(n_cases)
        if iteration == 0 and reuse_iter0 and cache_path.exists():
            with open(cache_path) as f:
                cached = json.load(f)
            case_names = {c['name'] for c in cases}
            cached_names = {t['case_name'] for t in cached}
            if case_names.issubset(cached_names):
                traces = [t for t in cached if t['case_name'] in case_names]
                print(f'  Loaded {len(traces)} cached baseline traces from {cache_path.name}')
            else:
                missing = case_names - cached_names
                print(f'  Cache missing {len(missing)} cases — running fresh ReAct...')
                traces = _run_all_cases_air(cases, ptools, model, max_steps)
        else:
            print(f'  Running ReAct with {len(ptools)} ptools, max_steps={max_steps}...')
            traces = _run_all_cases_air(cases, ptools, model, max_steps)

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
        print(f'  Max step hits: {max_step_hits}  |  -1 answers: {neg1_answers}  |  Extracted: {extracted}')

        print(f'\n  Analyzing {source}...')
        items = extract_items(traces, source)
        print(f'  Found {len(items)} {source}')
        if not items:
            break

        categories = categorize_items_air(items, source, model)
        print(f'  Top patterns:')
        for cat, count, _ in categories[:8]:
            print(f'    {count:3d}x  {cat}')

        print(f'\n  Synthesizing ptools (structured={structured_output})...')
        new_ptools = select_and_synthesize_air(
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

    print(f'\n\n{"="*60}')
    print(f'FINAL SUMMARY')
    print(f'{"="*60}')
    print(f'{"Iter":>4}  {"Pt":>3}  {"Acc":>8}  {"Avg$":>8}  '
          f'{"Pt#":>5}  {"Do[]":>5}  {"MxH":>4}  {"-1":>3}  {"Ext":>4}')
    for s in iteration_stats:
        print(f'{s["iteration"]:4d}  {s["n_ptools"]:3d}  '
              f'{s["accuracy"]:7.1%}  ${s["avg_cost"]:.4f}  '
              f'{s["ptool_uses"]:5d}  {s["do_uses"]:5d}  '
              f'{s["max_step_hits"]:4d}  {s["neg1_answers"]:3d}  {s["extracted_finishes"]:4d}')

    summary = {
        'iteration_stats': iteration_stats,
        'ptools': ptools,
        'config': dict(n_cases=n_cases, n_iters=n_iters, model=model,
                        seed=seed, min_count=min_count, max_steps=max_steps,
                        sim_threshold=sim_threshold, batch_size=batch_size,
                        source=source, structured_output=structured_output,
                        output_dir=output_dir, fixes=fixes, task='rulearena_airline'),
    }
    with open(out_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f'\nSummary saved to {out_dir / "summary.json"}')


if __name__ == '__main__':
    app()
