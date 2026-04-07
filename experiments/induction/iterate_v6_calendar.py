"""Iterative ptool induction v6 — Natural Plan calendar scheduling task.

Forked from iterate_v6_team.py for the calendar_scheduling task in
natural_plan. Major differences from MCQ tasks (object placement, team,
murder mystery):
  - Generation task, not multiple choice. Agent outputs a free text
    solution like "Here is the proposed time: Monday, 14:30 - 15:30".
  - Eval uses eval_calendar_single from natural_plan/eval_utils.py which
    parses (day, start_hour, end_hour) from both response and golden_plan.
  - `expected` field stores the golden_plan string, not an int index.
  - Finish[free_text] instead of Finish[index]. The action arg is the
    full solution string.
  - The "narrative" is the entire prompt_0shot text from the dataset
    (already contains the task description and examples).

Usage:
    source .env && export TOGETHER_API_KEY

    # Smoke test
    uv run python experiments/induction/iterate_v6_calendar.py baseline --n-cases 5
    uv run python experiments/induction/iterate_v6_calendar.py run \\
        --n-cases 5 --n-iters 2 --source thoughts --structured-output \\
        --output-dir iterations_v6cg_cal_smoke

    # n=30 runs
    uv run python experiments/induction/iterate_v6_calendar.py baseline --n-cases 30
    uv run python experiments/induction/iterate_v6_calendar.py run \\
        --n-cases 30 --source thoughts --structured-output \\
        --output-dir iterations_v6cg_cal
    uv run python experiments/induction/iterate_v6_calendar.py run \\
        --n-cases 30 --batch-size 3 --structured-output \\
        --output-dir iterations_v6bg_cal
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
sys.path.insert(0, str(_PROJECT_ROOT / 'benchmarks' / 'natural_plan'))

from eval_utils import eval_calendar_single  # noqa: E402

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
# Task-specific text (Natural Plan calendar scheduling)
# ═══════════════════════════════════════════════════════════════

SYSTEM_PROMPT_INTRO = (
    "Solve a calendar scheduling task. You will be given a description of a "
    "meeting that needs to be scheduled, the participants and their existing "
    "calendars, the desired meeting duration, and possibly preferences. Find a "
    "time slot that works for everyone. Reason step by step using interleaving "
    "Thought, Action, Observation steps. Your final answer must be a string "
    "of the form 'Here is the proposed time: <Day>, <HH:MM> - <HH:MM>'."
)

FEW_SHOT_CAL = """\
Here are some examples.

Task: You need to schedule a meeting for Alice and Bob for one hour between the work hours of 9:00 to 17:00 on Wednesday.
Alice has meetings on Wednesday during 9:00 to 10:00, 13:00 to 14:00;
Bob has blocked their calendar on Wednesday during 11:00 to 12:00, 15:00 to 16:00.
Find a time that works for everyone's schedule and constraints.

Thought 1: I should list everyone's busy periods on Wednesday and find an hour-long gap where no one is busy.
Action 1: Do[List each participant's busy intervals on Wednesday and identify common free hour-long slots between 9:00 and 17:00]
Observation 1: Alice busy: 9:00-10:00, 13:00-14:00. Bob busy: 11:00-12:00, 15:00-16:00. Common free 1-hour slots between 9 and 17: 10:00-11:00, 12:00-13:00, 14:00-15:00, 16:00-17:00. Earliest is 10:00-11:00.
Thought 2: 10:00-11:00 is the earliest hour-long slot that works for both Alice and Bob.
Action 2: Finish[Here is the proposed time: Wednesday, 10:00 - 11:00]

Task: You need to schedule a meeting for Carol, Dan, and Eve for thirty minutes between the work hours of 9:00 to 17:00 on Friday.
Carol has meetings on Friday during 9:00 to 10:30, 14:00 to 15:00;
Dan has blocked their calendar on Friday during 10:30 to 12:00, 13:00 to 13:30;
Eve has meetings on Friday during 11:00 to 11:30, 12:30 to 13:00, 16:00 to 17:00.

Thought 1: I should compute the union of busy intervals from all three participants and find a 30-minute gap.
Action 1: Do[Compute the union of busy intervals for Carol, Dan, and Eve on Friday and identify all common free 30-minute slots between 9:00 and 17:00]
Observation 1: Union of busy: 9:00-10:30 (Carol), 10:30-12:00 (Dan), 11:00-11:30 (Eve, contained), 12:30-13:00 (Eve), 13:00-13:30 (Dan), 14:00-15:00 (Carol), 16:00-17:00 (Eve). Merged: 9:00-12:00, 12:30-13:30, 14:00-15:00, 16:00-17:00. Free 30-min slots: 12:00-12:30, 13:30-14:00, 15:00-15:30, 15:30-16:00.
Thought 2: The earliest free 30-min slot is 12:00-12:30.
Action 2: Finish[Here is the proposed time: Friday, 12:00 - 12:30]

"""

ENV_PROMPT_TEMPLATE = (
    "You are a careful reasoner assisting with a calendar scheduling task. "
    "Read the task below and respond to this request:\n\n"
    "Request: {request}\n\n"
    "Task:\n{narrative}\n\n"
    "Respond concisely. Be precise about times and participants. Compute busy "
    "and free intervals exactly. Do not speculate beyond what the task states."
)

EXTRACT_PROMPT_TEMPLATE = (
    "The following text is from an agent solving a calendar scheduling task. "
    "Has the agent committed to a final scheduled time? If yes, output the "
    "scheduled time in EXACTLY this format on its own line:\n"
    "Here is the proposed time: <Day>, <HH:MM> - <HH:MM>\n\n"
    "If the agent has not committed to a final time, return ONLY the word 'none'.\n\n"
    "Text:\n{text}\n\n"
    "<answer>solution string or none</answer>"
)

CATEGORIZE_TASK_LABEL = "calendar scheduling tasks"
SYNTHESIZE_TASK_LABEL = "calendar scheduling tasks"
SYNTHESIZE_NARRATIVE_LABEL = (
    "the full task description: meeting attendees, the day, work hours, "
    "meeting duration, and each participant's existing busy intervals"
)


# ═══════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════

def _extract_task(prompt_0shot: str) -> str:
    """Extract the TASK: ... block from a natural_plan calendar prompt_0shot.

    The 0-shot prompt is just intro + a single TASK: ... SOLUTION: block.
    We strip the trailing "SOLUTION:" since the agent will produce its own.
    """
    idx = prompt_0shot.find('TASK:')
    if idx == -1:
        return prompt_0shot.strip()
    block = prompt_0shot[idx:]
    sol = block.rfind('SOLUTION:')
    if sol != -1:
        block = block[:sol]
    return block.strip()


def load_calendar_cases(n: int, seed: int = 42) -> list[dict]:
    """Load shuffled calendar scheduling cases from natural_plan."""
    data_file = _PROJECT_ROOT / 'benchmarks' / 'natural_plan' / 'data' / 'calendar_scheduling.json'
    with open(data_file) as f:
        data = json.load(f)
    keys = sorted(data.keys())
    cases = []
    for k in keys:
        ex = data[k]
        cases.append({
            'name': k,
            'narrative': _extract_task(ex['prompt_0shot']),
            'question': '',
            'choices': [],  # generation task; no MCQ choices
            'expected': ex['golden_plan'],  # string, not int
        })
    random.Random(seed).shuffle(cases)
    return cases[:n]


def _baseline_cache_path(n_cases: int) -> Path:
    return BASE_DIR / 'traces' / f'react_traces_calendar_n{n_cases}.json'


# ═══════════════════════════════════════════════════════════════
# Task-customized core: env action, system prompt, answer extraction
# ═══════════════════════════════════════════════════════════════

def _execute_do_action_cal(action_arg: str, narrative: str, model: str) -> tuple[str, dict]:
    env_prompt = ENV_PROMPT_TEMPLATE.format(request=action_arg, narrative=narrative)
    response, stats = llm_cached(env_prompt, model)
    if len(response) > 1500:
        response = response[:1500] + '...'
    return response.strip(), stats


def _make_system_prompt_cal(ptools: list[dict]) -> str:
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
            f'({finish_num}) Finish[<solution string>], which finishes the task. '
            f'The solution string MUST be of the form '
            f'"Here is the proposed time: <Day>, <HH:MM> - <HH:MM>".\n\n'
            f'IMPORTANT: Prefer using the specialized tools above over Do[] when '
            f'they match your reasoning need. Use Do[] only for steps not covered '
            f'by a specialized tool.\n'
            f'When you have enough information to decide, immediately use '
            f'Finish[Here is the proposed time: <Day>, <HH:MM> - <HH:MM>].\n'
        )
    else:
        base += (
            '(2) Finish[<solution string>], which finishes the task. '
            'The solution string MUST be of the form '
            '"Here is the proposed time: <Day>, <HH:MM> - <HH:MM>".\n'
        )
    base += 'Here are some examples.\n'
    return base


def extract_solution_string_cal(text: str, model: str) -> str | None:
    """Use LLM to extract a calendar solution string from free-form text.

    Returns a string like "Here is the proposed time: Monday, 14:30 - 15:30"
    or None if the agent has not committed to a final answer.
    """
    prompt = EXTRACT_PROMPT_TEMPLATE.format(text=text)
    response, _ = llm_cached(prompt, model)
    m = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL)
    if m:
        val = m.group(1).strip()
        if val.lower() == 'none':
            return None
        return val
    # Fallback: regex-search the response itself for the canonical form
    m = re.search(r'Here is the proposed time:[^\n]*', response)
    if m:
        return m.group(0).strip()
    return None


# ═══════════════════════════════════════════════════════════════
# ReAct loop (calendar scheduling variant — generation, not MCQ)
# ═══════════════════════════════════════════════════════════════

def run_react_on_case_cal(narrative: str, question: str, choices: list,
                          ptools: list[dict], model: str,
                          max_steps: int = 14) -> dict:
    system = _make_system_prompt_cal(ptools)
    # Generation task: question and choices are unused; narrative IS the task
    prompt = system + FEW_SHOT_CAL + f'\n{narrative}\n'
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

        if action_name_lower == 'finish':
            # action_arg is the full free-text solution string
            answer = action_arg.strip()
            # If it's not in canonical form, try LLM extraction as fallback
            if 'Here is the proposed time' not in answer:
                extracted = extract_solution_string_cal(action_str, model)
                if extracted:
                    answer = extracted
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
            sol = extract_solution_string_cal(thought, model)
            if sol is not None:
                steps.append(dict(
                    step=i, thought=thought, action='[extracted from thought]',
                    action_type='finish_extracted', action_arg=sol,
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
            observation, env_stats = _execute_do_action_cal(action_arg, narrative, model)
            accum(env_stats)
            steps.append(dict(
                step=i, thought=thought, action=action_str,
                action_type='do', action_arg=action_arg,
                observation=observation, ptool_used=None,
            ))
        elif action_str:
            observation, env_stats = _execute_do_action_cal(action_str, narrative, model)
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
    last_sol = extract_solution_string_cal(full_text, model)
    if last_sol is not None:
        return dict(
            answer=last_sol, steps=steps, n_steps=max_steps,
            termination='finish_extracted_at_max', stats=total_stats,
        )
    return dict(
        answer='', steps=steps, n_steps=max_steps,
        termination='max_steps', stats=total_stats,
    )


def _run_all_cases_cal(cases: list[dict], ptools: list[dict],
                       model: str, max_steps: int) -> list[dict]:
    results = []
    max_retries = 3
    for idx, case in enumerate(cases):
        print(f'  [{idx+1}/{len(cases)}] {case["name"]}...', end=' ', flush=True)
        result = None
        for attempt in range(max_retries):
            try:
                result = run_react_on_case_cal(
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
                    result = dict(answer='', steps=[], n_steps=0,
                                  termination='error', stats={})
        # Use eval_calendar_single for generation task evaluation
        pred = result.get('answer', '') or ''
        correct = eval_calendar_single(str(pred), case['expected'])
        result['case_name'] = case['name']
        result['expected'] = case['expected']
        result['correct'] = correct
        results.append(result)
        term = result.get('termination', '')
        tag = ' [extracted]' if 'extracted' in term else ''
        status = 'OK' if correct else 'WRONG'
        # Truncate long pred/exp strings for log readability
        pred_str = str(pred)[:60].replace('\n', ' ')
        exp_str = str(case['expected'])[:60].replace('\n', ' ')
        print(f'{status} (pred={pred_str!r}, exp={exp_str!r}, steps={result["n_steps"]}{tag})')
        time.sleep(0.5)
    return results


# ═══════════════════════════════════════════════════════════════
# Pattern categorization (calendar scheduling task label)
# ═══════════════════════════════════════════════════════════════

def categorize_items_cal(items: list[dict], source: str, model: str,
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
# Ptool synthesis (calendar scheduling task label + structured option)
# ═══════════════════════════════════════════════════════════════

def synthesize_ptool_cal(pattern_name: str, examples: list[dict],
                         model: str, ptool_id: str,
                         structured_output: bool = False) -> dict:
    examples_text = ''
    for ex in examples[:5]:
        examples_text += f'  - "{ex["text"][:200]}"\n'

    if structured_output:
        output_instruction = """
4. output_format: Specify EXACTLY what the output should look like. Use a structured format like:
   - A dict/JSON keyed by participant with their busy intervals (e.g., {"Alice": [{"day": "Monday", "start": "09:00", "end": "10:00"}, ...]})
   - A list of free intervals satisfying the meeting duration (e.g., [{"day": "Monday", "start": "10:00", "end": "11:00"}, ...])
   - A merged-busy timeline across all participants (e.g., [{"day": "Monday", "start": "09:00", "end": "10:30", "participants": ["Alice", "Bob"]}, ...])
   - A boolean judgment with reasoning (e.g., {"slot_works": true, "violations": [], "reasoning": "..."})

The output format MUST be different from other tools — if one tool returns per-participant busy intervals, another should return free slots, another should return merged busy timeline, etc. The output format is critical for making tools functionally distinct.

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
- focus: what specific aspect to focus on (the agent decides — e.g., a target participant, a target day, a meeting duration, a specific time window)

Design the tool:
1. func_name: snake_case Python function name
2. display_name: CamelCase version for the agent to call
3. short_desc: one sentence for the agent prompt
{output_instruction}

The docstring is critical — it drives the LLM that executes this tool. Be specific about:
- What information to extract from the narrative (busy intervals per participant, work-hour bounds, meeting duration, day, preferences)
- How to structure the response
- What to pay attention to (calendar scheduling: convert all times to a uniform format like minutes-from-midnight; busy intervals are typically half-open [start, end); the meeting must fit ENTIRELY within work hours AND avoid every participant's busy intervals; preferences may further constrain the choice)

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
# Functional similarity (uses calendar scheduling narratives as samples)
# ═══════════════════════════════════════════════════════════════

_FUNCTIONAL_CACHE_CAL: dict[str, list[str]] = {}
_SAMPLE_NARRATIVES_CAL: list[str] = []


def _get_sample_narratives_cal(n: int = 3) -> list[str]:
    global _SAMPLE_NARRATIVES_CAL
    if not _SAMPLE_NARRATIVES_CAL:
        cases = load_calendar_cases(n=10, seed=99)
        _SAMPLE_NARRATIVES_CAL = [c['narrative'] for c in cases[:n]]
    return _SAMPLE_NARRATIVES_CAL


def _get_ptool_outputs_cal(ptool: dict) -> list[str]:
    cache_key = ptool['id']
    if cache_key in _FUNCTIONAL_CACHE_CAL:
        return _FUNCTIONAL_CACHE_CAL[cache_key]
    iface = _INDUCED_INTERFACES.get(ptool['func_name'])
    if iface is None:
        iface = create_ptool_interface(ptool['func_name'], ptool['doc'])
    narratives = _get_sample_narratives_cal()
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
    _FUNCTIONAL_CACHE_CAL[cache_key] = outputs
    return outputs


def _functional_similarity_cal(new_ptool: dict, existing_ptools: list[dict]) -> float:
    if not existing_ptools:
        return 0.0
    model = _get_embed_model()
    new_outputs = _get_ptool_outputs_cal(new_ptool)
    new_text = " ".join(new_outputs)
    max_sim = 0.0
    for ep in existing_ptools:
        ep_outputs = _get_ptool_outputs_cal(ep)
        ep_text = " ".join(ep_outputs)
        embeddings = model.encode([new_text, ep_text], normalize_embeddings=True)
        sim = float(embeddings[0] @ embeddings[1])
        max_sim = max(max_sim, sim)
    return max_sim


def compute_similarity_cal(new_ptool: dict, existing_ptools: list[dict]) -> tuple[float, dict]:
    doc = _docstring_similarity(new_ptool, existing_ptools)
    func = _functional_similarity_cal(new_ptool, existing_ptools)
    score = (doc + func) / 2
    return score, {'docstring': doc, 'functional': func, 'combined': score}


def select_and_synthesize_cal(categories: list[tuple[str, int, list]],
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
        candidate = synthesize_ptool_cal(cat, examples, model, ptool_id,
                                          structured_output=structured_output)
        sim, detail = compute_similarity_cal(candidate, all_ptools)
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
    n_cases: int = typer.Option(30, help='Number of calendar scheduling cases'),
    model: str = typer.Option('together_ai/deepseek-ai/DeepSeek-V3', help='LLM model'),
    seed: int = typer.Option(42, help='Random seed'),
    max_steps: int = typer.Option(14, help='Max ReAct steps'),
):
    """Run baseline ReAct (no ptools) on calendar scheduling and cache traces."""
    cache_path = _baseline_cache_path(n_cases)
    config.configure(cfg={
        'llm': {'model': model, 'timeout': 300},
        'cachier': {
            'cache_dir': str(BASE_DIR / 'traces' / 'llm_cache_calendar'),
            'enable_caching': True,
        },
    })
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    cases = load_calendar_cases(n_cases, seed)
    print(f'Running baseline ReAct on {len(cases)} calendar scheduling cases')
    print(f'  Model: {model}, Max steps: {max_steps}, Seed: {seed}')
    print(f'  Cache: {cache_path}\n')

    traces = _run_all_cases_cal(cases, ptools=[], model=model, max_steps=max_steps)

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
    output_dir: str = typer.Option('iterations_v6_calendar', help='Output directory'),
):
    """Run iterative ptool induction v6 on calendar scheduling."""
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
    cases = load_calendar_cases(n_cases, seed)
    ptools: list[dict] = []
    iteration_stats = []

    fixes = ['F: ptool-specific focus', 'H: LLM answer extraction']
    if structured_output:
        fixes.append('G: structured output')

    print(f'Iterative Ptool Induction v6 — CALENDAR SCHEDULING')
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
                traces = _run_all_cases_cal(cases, ptools, model, max_steps)
        else:
            print(f'  Running ReAct with {len(ptools)} ptools, max_steps={max_steps}...')
            traces = _run_all_cases_cal(cases, ptools, model, max_steps)

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

        categories = categorize_items_cal(items, source, model)
        print(f'  Top patterns:')
        for cat, count, _ in categories[:8]:
            print(f'    {count:3d}x  {cat}')

        print(f'\n  Synthesizing ptools (structured={structured_output})...')
        new_ptools = select_and_synthesize_cal(
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
                        output_dir=output_dir, fixes=fixes, task='calendar_scheduling'),
    }
    with open(out_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f'\nSummary saved to {out_dir / "summary.json"}')


if __name__ == '__main__':
    app()
