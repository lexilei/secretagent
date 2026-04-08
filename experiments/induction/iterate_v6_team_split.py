"""Iterative ptool induction v6 — MUSR team allocation with 75/75/100 split.

Fork of iterate_v6_team.py adding proper train/val/test split (seed=42)
for reproducibility:
  - train  : 75 cases (induction set; ptools synthesized from these traces)
  - val    : 75 cases (per-iter eval; for early stop / convergence check)
  - test   : 100 cases (held out; final accuracy reported once)
  Total: 250 (full team_allocation.json)

The legacy `baseline` and `run` commands (taking --n-cases) are preserved
for backward compatibility with the previous n=30 results. The new
commands `baseline_split` and `run_split` use the train/val/test API.

Usage:
    source .env && export TOGETHER_API_KEY

    # Cache baselines for all three splits
    uv run python experiments/induction/iterate_v6_team_split.py baseline_split --split train
    uv run python experiments/induction/iterate_v6_team_split.py baseline_split --split val
    uv run python experiments/induction/iterate_v6_team_split.py baseline_split --split test

    # Run v6cg induction: train on train split, eval per iter on val,
    # report final test accuracy
    uv run python experiments/induction/iterate_v6_team_split.py run_split \\
        --source thoughts --structured-output \\
        --output-dir iterations_v6cg_team_split
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
# Task-specific text (MUSR team allocation)
# ═══════════════════════════════════════════════════════════════

SYSTEM_PROMPT_INTRO = (
    "Solve a team allocation puzzle. The narrative describes a manager assigning "
    "people to tasks/roles. Each person has different skills, preferences, "
    "experiences, and interpersonal dynamics that affect their suitability for "
    "each role. Some people work well together, others do not. The goal is to "
    "find the unique allocation (one of the multiple-choice options) that best "
    "satisfies all constraints implied by the narrative. Use interleaving "
    "Thought, Action, Observation steps."
)

FEW_SHOT_TEAM = """\
Question: At a small bookshop, the manager must assign Sam, Liz, and Tony to two roles: Cashier (1 needed) and Stock Clerk (2 needed). Sam has worked retail before but is shy with customers and prefers solitary tasks. Liz is outgoing and enjoys conversation but has chronic back problems that make heavy lifting risky. Tony is new and physically strong but tends to avoid social interaction.
Given the story, how would you uniquely allocate each person to make sure both tasks are accomplished efficiently?
The choices are: (0) Cashier: Sam, Stock Clerk: Liz and Tony, (1) Cashier: Liz, Stock Clerk: Sam and Tony, (2) Cashier: Tony, Stock Clerk: Sam and Liz.
Thought 1: I should identify each person's skills, weaknesses, and constraints relevant to Cashier vs Stock Clerk.
Action 1: Do[List each person's strengths, weaknesses, and constraints relevant to Cashier and Stock Clerk roles]
Observation 1: Sam: prior retail experience BUT shy and prefers solitary work (poor cashier fit; ok for stock clerk). Liz: outgoing, customer-friendly (great cashier fit) BUT back problems make heavy lifting unsafe (poor stock clerk fit). Tony: physically strong (good stock clerk fit) BUT avoids social interaction (poor cashier fit).
Thought 2: Liz is the only person whose constraints make Cashier the right fit and Stock Clerk the wrong fit. Tony is clearly Stock Clerk (strength + introvert). Sam fills the second Stock Clerk slot (he can do it, and the cashier role is taken). So: Cashier=Liz, Stock Clerks=Sam and Tony.
Action 2: Finish[1]

Question: At a community center, Diana, Ethan, and Felix need to be assigned to roles: Event Coordinator (1) and Cleanup Crew (2). Diana is a meticulous planner with great organizational skills but is uncomfortable with physical labor. Ethan loves variety and high energy but is impulsive and disorganized. Felix is methodical and works hard but has no leadership experience and prefers to follow directions.
Given the story, how would you uniquely allocate each person to make sure both tasks are accomplished efficiently?
The choices are: (0) Coordinator: Diana, Cleanup: Ethan and Felix, (1) Coordinator: Ethan, Cleanup: Diana and Felix, (2) Coordinator: Felix, Cleanup: Diana and Ethan.
Thought 1: I need to identify each person's strengths and disqualifiers for the Coordinator role (which requires organization and leadership) and for the Cleanup Crew role (which requires physical work).
Action 1: Do[For each person, determine if they fit Event Coordinator and if they fit Cleanup Crew]
Observation 1: Diana: organized planner (Coordinator fit) BUT uncomfortable with physical labor (poor Cleanup fit). Ethan: impulsive and disorganized (poor Coordinator fit), high energy (ok Cleanup fit). Felix: methodical and hard-working (ok Cleanup fit) BUT no leadership and prefers to follow (poor Coordinator fit).
Thought 2: Diana is the only viable Coordinator. Ethan and Felix are both disqualified from Coordinator and both ok for Cleanup. So: Coordinator=Diana, Cleanup=Ethan and Felix.
Action 2: Finish[0]

"""

ENV_PROMPT_TEMPLATE = (
    "You are a careful reader assisting with a team allocation reasoning task. "
    "Read the narrative below and respond to this request:\n\n"
    "Request: {request}\n\n"
    "Narrative:\n{narrative}\n\n"
    "Respond concisely with only information directly stated in or inferable "
    "from the narrative. Be precise about each person's skills, constraints, "
    "and interpersonal dynamics. Do not speculate beyond what the text supports."
)

EXTRACT_PROMPT_TEMPLATE = (
    "The following text is from an agent solving a team allocation puzzle. "
    "The answer choices are: {choices_str}\n\n"
    "Text:\n{text}\n\n"
    "Has the agent decided on a final answer? If yes, return ONLY the 0-based "
    "index number. If the agent is still investigating and has not committed "
    "to a final answer, return ONLY the word 'none'.\n\n"
    "<answer>index or none</answer>"
)

CATEGORIZE_TASK_LABEL = "team allocation puzzles"
SYNTHESIZE_TASK_LABEL = "team allocation puzzles"
SYNTHESIZE_NARRATIVE_LABEL = (
    "the full narrative text describing the people to allocate, their skills, "
    "experiences, preferences, interpersonal dynamics, and the roles to be filled"
)


# ═══════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════

def load_team_cases(n: int, seed: int = 42) -> list[dict]:
    """Load shuffled team allocation cases (legacy single-sample API)."""
    data_file = _PROJECT_ROOT / 'benchmarks' / 'musr' / 'data' / 'team_allocation.json'
    with open(data_file) as f:
        data = json.load(f)
    cases = []
    for i, ex in enumerate(data['examples']):
        choices = ex['choices']
        if isinstance(choices, str):
            choices = ast.literal_eval(choices)
        cases.append({
            'name': f'ex{i:03d}',
            'narrative': ex['narrative'],
            'question': ex['question'],
            'choices': choices,
            'expected': ex['answer_index'],
        })
    random.Random(seed).shuffle(cases)
    return cases[:n]


def load_team_split(split: str, seed: int = 42,
                    n_train: int = 75, n_val: int = 75,
                    n_test: int = 100) -> list[dict]:
    """Load a specific data split (train / val / test) of team allocation cases.

    The deterministic split uses a single shuffle (seed=42 for reproducibility),
    then takes train cases [0:n_train], val [n_train:n_train+n_val], and
    test [n_train+n_val:n_train+n_val+n_test]. Default 75/75/100 = 250 total
    (the full team_allocation.json).

    Args:
        split: 'train' | 'val' | 'test'
        seed: random seed for the shuffle (default 42)
        n_train, n_val, n_test: split sizes
    """
    assert split in ('train', 'val', 'test'), f'unknown split: {split}'
    data_file = _PROJECT_ROOT / 'benchmarks' / 'musr' / 'data' / 'team_allocation.json'
    with open(data_file) as f:
        data = json.load(f)
    cases = []
    for i, ex in enumerate(data['examples']):
        choices = ex['choices']
        if isinstance(choices, str):
            choices = ast.literal_eval(choices)
        cases.append({
            'name': f'ex{i:03d}',
            'narrative': ex['narrative'],
            'question': ex['question'],
            'choices': choices,
            'expected': ex['answer_index'],
        })
    random.Random(seed).shuffle(cases)
    if split == 'train':
        return cases[:n_train]
    elif split == 'val':
        return cases[n_train:n_train + n_val]
    else:  # test
        return cases[n_train + n_val:n_train + n_val + n_test]


def _baseline_cache_path(n_cases: int, split: str = '') -> Path:
    if split:
        return BASE_DIR / 'traces' / f'react_traces_team_split_{split}_n{n_cases}.json'
    return BASE_DIR / 'traces' / f'react_traces_team_n{n_cases}.json'


# ═══════════════════════════════════════════════════════════════
# Task-customized core: env action, system prompt, answer extraction
# ═══════════════════════════════════════════════════════════════

def _execute_do_action_obj(action_arg: str, narrative: str, model: str) -> tuple[str, dict]:
    env_prompt = ENV_PROMPT_TEMPLATE.format(request=action_arg, narrative=narrative)
    response, stats = llm_cached(env_prompt, model)
    if len(response) > 1500:
        response = response[:1500] + '...'
    return response.strip(), stats


def _make_system_prompt_obj(ptools: list[dict]) -> str:
    base = (
        SYSTEM_PROMPT_INTRO + "\n"
        "Action can be:\n"
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


def extract_answer_index_obj(text: str, choices: list, model: str) -> int | None:
    choices_str = ', '.join(f'({i}) {c}' for i, c in enumerate(choices))
    prompt = EXTRACT_PROMPT_TEMPLATE.format(choices_str=choices_str, text=text)
    response, _ = llm_cached(prompt, model)
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
    nums = re.findall(r'\b(\d)\b', response)
    if len(nums) == 1:
        idx = int(nums[0])
        if 0 <= idx < len(choices):
            return idx
    return None


# ═══════════════════════════════════════════════════════════════
# ReAct loop (team allocation variant)
# ═══════════════════════════════════════════════════════════════

def run_react_on_case_obj(narrative: str, question: str, choices: list,
                          ptools: list[dict], model: str,
                          max_steps: int = 14) -> dict:
    system = _make_system_prompt_obj(ptools)
    choices_str = ', '.join(f'({i}) {c}' for i, c in enumerate(choices))
    prompt = (
        system + FEW_SHOT_TEAM
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

        if action_name_lower == 'finish':
            try:
                answer = int(action_arg.strip().split()[0])
            except ValueError:
                answer = -1
            if answer == -1:
                answer = extract_answer_index_obj(action_str, choices, model) or -1
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
            idx = extract_answer_index_obj(thought, choices, model)
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
            observation, env_stats = _execute_do_action_obj(action_arg, narrative, model)
            accum(env_stats)
            steps.append(dict(
                step=i, thought=thought, action=action_str,
                action_type='do', action_arg=action_arg,
                observation=observation, ptool_used=None,
            ))
        elif action_str:
            observation, env_stats = _execute_do_action_obj(action_str, narrative, model)
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

    # Max steps reached — last resort: extract from recent thoughts
    full_text = ' '.join(s.get('thought', '') for s in steps[-3:])
    last_idx = extract_answer_index_obj(full_text, choices, model)
    if last_idx is not None:
        return dict(
            answer=last_idx, steps=steps, n_steps=max_steps,
            termination='finish_extracted_at_max', stats=total_stats,
        )
    return dict(
        answer=-1, steps=steps, n_steps=max_steps,
        termination='max_steps', stats=total_stats,
    )


def _run_all_cases_obj(cases: list[dict], ptools: list[dict],
                       model: str, max_steps: int) -> list[dict]:
    results = []
    max_retries = 3
    for idx, case in enumerate(cases):
        print(f'  [{idx+1}/{len(cases)}] {case["name"]}...', end=' ', flush=True)
        result = None
        for attempt in range(max_retries):
            try:
                result = run_react_on_case_obj(
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


# ═══════════════════════════════════════════════════════════════
# Pattern categorization (team allocation task label)
# ═══════════════════════════════════════════════════════════════

def categorize_items_obj(items: list[dict], source: str, model: str,
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
# Ptool synthesis (team allocation task label + structured option)
# ═══════════════════════════════════════════════════════════════

def synthesize_ptool_obj(pattern_name: str, examples: list[dict],
                         model: str, ptool_id: str,
                         structured_output: bool = False) -> dict:
    examples_text = ''
    for ex in examples[:5]:
        examples_text += f'  - "{ex["text"][:200]}"\n'

    if structured_output:
        output_instruction = """
4. output_format: Specify EXACTLY what the output should look like. Use a structured format like:
   - A dict/JSON keyed by person with their fit per role (e.g., {"Sam": {"role_a_fit": "good/bad/neutral", "role_b_fit": "...", "evidence": ["quote1", "quote2"], "disqualifiers": ["..."]}})
   - A list of constraints extracted from the narrative (e.g., [{"type": "skill", "person": "Liz", "applies_to": "cashier", "polarity": "positive", "quote": "outgoing and customer-friendly"}, ...])
   - A pairwise compatibility matrix (e.g., {"Alex-Mia": "good collaborators", "Alex-Olivia": "tense"})
   - A boolean judgment with reasoning (e.g., {"is_valid_allocation": true, "violations": [], "reasoning": "..."})

The output format MUST be different from other tools — if one tool returns a per-person fit dict, another should return a constraints list, another should return a compatibility matrix, etc. The output format is critical for making tools functionally distinct.

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
- focus: what specific aspect to focus on (the agent decides — e.g., a target person, a specific role, a constraint type, a pair of people)

Design the tool:
1. func_name: snake_case Python function name
2. display_name: CamelCase version for the agent to call
3. short_desc: one sentence for the agent prompt
{output_instruction}

The docstring is critical — it drives the LLM that executes this tool. Be specific about:
- What information to extract from the narrative (skills, weaknesses, preferences, interpersonal dynamics, role requirements, hard disqualifiers)
- How to structure the response
- What to pay attention to (team allocation: hard disqualifiers like "afraid of animals" rule out a role entirely, while soft preferences are tiebreakers)

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
# Functional similarity (uses team allocation narratives as samples)
# ═══════════════════════════════════════════════════════════════

_FUNCTIONAL_CACHE_OBJ: dict[str, list[str]] = {}
_SAMPLE_NARRATIVES_OBJ: list[str] = []


def _get_sample_narratives_obj(n: int = 3) -> list[str]:
    global _SAMPLE_NARRATIVES_OBJ
    if not _SAMPLE_NARRATIVES_OBJ:
        cases = load_team_cases(n=10, seed=99)
        _SAMPLE_NARRATIVES_OBJ = [c['narrative'] for c in cases[:n]]
    return _SAMPLE_NARRATIVES_OBJ


def _get_ptool_outputs_obj(ptool: dict) -> list[str]:
    cache_key = ptool['id']
    if cache_key in _FUNCTIONAL_CACHE_OBJ:
        return _FUNCTIONAL_CACHE_OBJ[cache_key]
    iface = _INDUCED_INTERFACES.get(ptool['func_name'])
    if iface is None:
        iface = create_ptool_interface(ptool['func_name'], ptool['doc'])
    narratives = _get_sample_narratives_obj()
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
    _FUNCTIONAL_CACHE_OBJ[cache_key] = outputs
    return outputs


def _functional_similarity_obj(new_ptool: dict, existing_ptools: list[dict]) -> float:
    if not existing_ptools:
        return 0.0
    model = _get_embed_model()
    new_outputs = _get_ptool_outputs_obj(new_ptool)
    new_text = " ".join(new_outputs)
    max_sim = 0.0
    for ep in existing_ptools:
        ep_outputs = _get_ptool_outputs_obj(ep)
        ep_text = " ".join(ep_outputs)
        embeddings = model.encode([new_text, ep_text], normalize_embeddings=True)
        sim = float(embeddings[0] @ embeddings[1])
        max_sim = max(max_sim, sim)
    return max_sim


def compute_similarity_obj(new_ptool: dict, existing_ptools: list[dict]) -> tuple[float, dict]:
    doc = _docstring_similarity(new_ptool, existing_ptools)
    func = _functional_similarity_obj(new_ptool, existing_ptools)
    score = (doc + func) / 2
    return score, {'docstring': doc, 'functional': func, 'combined': score}


def select_and_synthesize_obj(categories: list[tuple[str, int, list]],
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
        candidate = synthesize_ptool_obj(cat, examples, model, ptool_id,
                                          structured_output=structured_output)
        sim, detail = compute_similarity_obj(candidate, all_ptools)
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
    n_cases: int = typer.Option(30, help='Number of team allocation cases'),
    model: str = typer.Option('together_ai/deepseek-ai/DeepSeek-V3', help='LLM model'),
    seed: int = typer.Option(42, help='Random seed'),
    max_steps: int = typer.Option(14, help='Max ReAct steps'),
):
    """Run baseline ReAct (no ptools) on team allocation and cache traces."""
    cache_path = _baseline_cache_path(n_cases)
    config.configure(cfg={
        'llm': {'model': model, 'timeout': 300},
        'cachier': {
            'cache_dir': str(BASE_DIR / 'traces' / 'llm_cache_team'),
            'enable_caching': True,
        },
    })
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    cases = load_team_cases(n_cases, seed)
    print(f'Running baseline ReAct on {len(cases)} team allocation cases')
    print(f'  Model: {model}, Max steps: {max_steps}, Seed: {seed}')
    print(f'  Cache: {cache_path}\n')

    traces = _run_all_cases_obj(cases, ptools=[], model=model, max_steps=max_steps)

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
    output_dir: str = typer.Option('iterations_v6_team', help='Output directory'),
):
    """Run iterative ptool induction v6 on team allocation."""
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
    cases = load_team_cases(n_cases, seed)
    ptools: list[dict] = []
    iteration_stats = []

    fixes = ['F: ptool-specific focus', 'H: LLM answer extraction']
    if structured_output:
        fixes.append('G: structured output')

    print(f'Iterative Ptool Induction v6 — TEAM ALLOCATION')
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
                traces = _run_all_cases_obj(cases, ptools, model, max_steps)
        else:
            print(f'  Running ReAct with {len(ptools)} ptools, max_steps={max_steps}...')
            traces = _run_all_cases_obj(cases, ptools, model, max_steps)

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

        categories = categorize_items_obj(items, source, model)
        print(f'  Top patterns:')
        for cat, count, _ in categories[:8]:
            print(f'    {count:3d}x  {cat}')

        print(f'\n  Synthesizing ptools (structured={structured_output})...')
        new_ptools = select_and_synthesize_obj(
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
                        output_dir=output_dir, fixes=fixes, task='team_allocation'),
    }
    with open(out_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f'\nSummary saved to {out_dir / "summary.json"}')


@app.command()
def baseline_split(
    split: str = typer.Option('train', help="train | val | test"),
    seed: int = typer.Option(42, help='Random seed (default 42 for reproducibility)'),
    n_train: int = typer.Option(75, help='Train split size'),
    n_val: int = typer.Option(75, help='Val split size'),
    n_test: int = typer.Option(100, help='Test split size'),
    model: str = typer.Option('together_ai/deepseek-ai/DeepSeek-V3', help='LLM model'),
    max_steps: int = typer.Option(14, help='Max ReAct steps'),
):
    """Cache baseline ReAct traces for one split (train/val/test) of team allocation."""
    cases = load_team_split(split, seed=seed, n_train=n_train, n_val=n_val, n_test=n_test)
    cache_path = _baseline_cache_path(len(cases), split=split)

    config.configure(cfg={
        'llm': {'model': model, 'timeout': 300},
        'cachier': {
            'cache_dir': str(BASE_DIR / 'traces' / f'llm_cache_team_split_{split}'),
            'enable_caching': True,
        },
    })
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    print(f'Running baseline ReAct on team allocation [{split}] split')
    print(f'  Model: {model}, Max steps: {max_steps}, Seed: {seed}')
    print(f'  Cases: {len(cases)} (split sizes train={n_train} val={n_val} test={n_test})')
    print(f'  Cache: {cache_path}\n')

    traces = _run_all_cases_obj(cases, ptools=[], model=model, max_steps=max_steps)
    with open(cache_path, 'w') as f:
        json.dump(traces, f, indent=2, default=str)

    n_correct = sum(1 for t in traces if t.get('correct'))
    avg_cost = sum(t.get('stats', {}).get('cost', 0) for t in traces) / max(len(traces), 1)
    extracted = sum(1 for t in traces if 'extracted' in t.get('termination', ''))
    neg1 = sum(1 for t in traces if t.get('answer') == -1)
    print(f'\nBaseline [{split}] accuracy: {n_correct}/{len(traces)} ({n_correct/len(traces):.1%})')
    print(f'Avg cost: ${avg_cost:.4f}  |  Extracted: {extracted}  |  -1 answers: {neg1}')
    print(f'Saved traces to {cache_path}')


@app.command()
def run_split(
    seed: int = typer.Option(42, help='Random seed (default 42)'),
    n_train: int = typer.Option(75, help='Train split size'),
    n_val: int = typer.Option(75, help='Val split size'),
    n_test: int = typer.Option(100, help='Test split size'),
    n_iters: int = typer.Option(5, help='Max induction iterations'),
    model: str = typer.Option('together_ai/deepseek-ai/DeepSeek-V3', help='LLM model'),
    min_count: int = typer.Option(3, help='Min pattern frequency'),
    reuse_iter0: bool = typer.Option(True, help='Use cached baseline_split for iter 0'),
    max_steps: int = typer.Option(14, help='Max ReAct steps'),
    sim_threshold: float = typer.Option(0.75, help='Similarity threshold'),
    batch_size: int = typer.Option(1, help='Ptools per iteration (1=v6cg, 3=v6bg)'),
    source: str = typer.Option('thoughts', help='actions or thoughts'),
    structured_output: bool = typer.Option(True, help='Structured output docstrings'),
    output_dir: str = typer.Option('iterations_v6cg_team_split', help='Output directory'),
):
    """Run iterative ptool induction on team allocation with train/val/test split.

    For each iteration: induce on train traces, eval on val (held out from
    induction). After max iters or convergence, eval the final ptool set on
    test (also held out throughout). All eval numbers reported in summary.json.
    """
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

    train_cases = load_team_split('train', seed=seed, n_train=n_train, n_val=n_val, n_test=n_test)
    val_cases = load_team_split('val', seed=seed, n_train=n_train, n_val=n_val, n_test=n_test)
    test_cases = load_team_split('test', seed=seed, n_train=n_train, n_val=n_val, n_test=n_test)
    print(f'Iterative Ptool Induction v6 — TEAM ALLOCATION (75/75/100 split, seed={seed})')
    print(f'  Train: {len(train_cases)}, Val: {len(val_cases)}, Test: {len(test_cases)}')
    print(f'  Source: {source}, Batch: {batch_size}, Sim threshold: {sim_threshold}')
    print(f'  Structured output: {structured_output}, max_iters: {n_iters}')
    print(f'  Output: {out_dir}\n')

    ptools: list[dict] = []
    iteration_stats = []

    for iteration in range(n_iters):
        iter_path = out_dir / f'iter_{iteration}'
        iter_path.mkdir(parents=True, exist_ok=True)
        print(f'\n{"="*60}')
        print(f'ITERATION {iteration}  |  ptools: {[p["display_name"] for p in ptools]}')
        print(f'{"="*60}')

        # ── train ──
        train_cache = _baseline_cache_path(len(train_cases), split='train')
        if iteration == 0 and reuse_iter0 and train_cache.exists():
            with open(train_cache) as f:
                cached = json.load(f)
            case_names = {c['name'] for c in train_cases}
            if case_names.issubset({t['case_name'] for t in cached}):
                train_traces = [t for t in cached if t['case_name'] in case_names]
                print(f'  [train] Loaded {len(train_traces)} cached baseline traces')
            else:
                print(f'  [train] Cache mismatch — running fresh ReAct...')
                train_traces = _run_all_cases_obj(train_cases, ptools, model, max_steps)
        else:
            print(f'  [train] Running ReAct with {len(ptools)} ptools, max_steps={max_steps}...')
            train_traces = _run_all_cases_obj(train_cases, ptools, model, max_steps)
        with open(iter_path / 'train_traces.json', 'w') as f:
            json.dump(train_traces, f, indent=2, default=str)

        train_correct = sum(1 for t in train_traces if t.get('correct'))
        train_acc = train_correct / len(train_traces) if train_traces else 0

        # ── val ──
        val_cache = _baseline_cache_path(len(val_cases), split='val')
        if iteration == 0 and reuse_iter0 and val_cache.exists():
            with open(val_cache) as f:
                cached_v = json.load(f)
            case_names_v = {c['name'] for c in val_cases}
            if case_names_v.issubset({t['case_name'] for t in cached_v}):
                val_traces = [t for t in cached_v if t['case_name'] in case_names_v]
                print(f'  [val] Loaded {len(val_traces)} cached baseline traces')
            else:
                print(f'  [val] Cache mismatch — running fresh ReAct...')
                val_traces = _run_all_cases_obj(val_cases, ptools, model, max_steps)
        else:
            print(f'  [val] Running ReAct with {len(ptools)} ptools, max_steps={max_steps}...')
            val_traces = _run_all_cases_obj(val_cases, ptools, model, max_steps)
        with open(iter_path / 'val_traces.json', 'w') as f:
            json.dump(val_traces, f, indent=2, default=str)
        val_correct = sum(1 for t in val_traces if t.get('correct'))
        val_acc = val_correct / len(val_traces) if val_traces else 0

        avg_cost = (
            sum(t.get('stats', {}).get('cost', 0) for t in train_traces + val_traces)
            / max(len(train_traces) + len(val_traces), 1)
        )
        stats = dict(
            iteration=iteration, n_ptools=len(ptools),
            train_accuracy=train_acc, train_n_correct=train_correct, train_n_total=len(train_traces),
            val_accuracy=val_acc, val_n_correct=val_correct, val_n_total=len(val_traces),
            avg_cost=avg_cost,
        )
        iteration_stats.append(stats)
        print(f'\n  TRAIN acc: {train_correct}/{len(train_traces)} ({train_acc:.1%})')
        print(f'  VAL   acc: {val_correct}/{len(val_traces)} ({val_acc:.1%})')

        # ── induce on train traces ──
        items = extract_items(train_traces, source)
        print(f'\n  Found {len(items)} {source} from train')
        if not items:
            break
        categories = categorize_items_obj(items, source, model)
        print(f'  Top patterns:')
        for cat, count, _ in categories[:8]:
            print(f'    {count:3d}x  {cat}')

        print(f'\n  Synthesizing ptools (batch={batch_size}, structured={structured_output})...')
        new_ptools = select_and_synthesize_obj(
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

    # ── final test eval ──
    print(f'\n\n{"="*60}')
    print(f'FINAL TEST EVAL  |  ptools: {[p["display_name"] for p in ptools]}')
    print(f'{"="*60}')
    test_cache = _baseline_cache_path(len(test_cases), split='test')
    if reuse_iter0 and not ptools and test_cache.exists():
        # special case: if no ptools were induced, just load cached test baseline
        with open(test_cache) as f:
            cached_t = json.load(f)
        case_names_t = {c['name'] for c in test_cases}
        test_traces = [t for t in cached_t if t['case_name'] in case_names_t]
        print(f'  Loaded {len(test_traces)} cached baseline traces')
    else:
        print(f'  Running ReAct with {len(ptools)} ptools on test...')
        test_traces = _run_all_cases_obj(test_cases, ptools, model, max_steps)
    test_correct = sum(1 for t in test_traces if t.get('correct'))
    test_acc = test_correct / len(test_traces) if test_traces else 0
    test_avg_cost = sum(t.get('stats', {}).get('cost', 0) for t in test_traces) / max(len(test_traces), 1)
    with open(out_dir / 'test_traces.json', 'w') as f:
        json.dump(test_traces, f, indent=2, default=str)
    print(f'\n  TEST acc: {test_correct}/{len(test_traces)} ({test_acc:.1%})')
    print(f'  TEST avg cost: ${test_avg_cost:.4f}')

    # Final summary
    print(f'\n\n{"="*60}')
    print(f'FINAL SUMMARY (75/75/100 split, seed={seed})')
    print(f'{"="*60}')
    print(f'{"Iter":>4}  {"Pt":>3}  {"TrainAcc":>9}  {"ValAcc":>9}  {"Cost":>9}')
    for s in iteration_stats:
        print(f'{s["iteration"]:4d}  {s["n_ptools"]:3d}  '
              f'{s["train_accuracy"]:8.1%}  {s["val_accuracy"]:8.1%}  ${s["avg_cost"]:.4f}')
    print(f'\nFINAL TEST: {test_correct}/{len(test_traces)} ({test_acc:.1%})')

    summary = {
        'split': dict(seed=seed, n_train=n_train, n_val=n_val, n_test=n_test),
        'iteration_stats': iteration_stats,
        'final_test': dict(accuracy=test_acc, n_correct=test_correct,
                           n_total=len(test_traces), avg_cost=test_avg_cost),
        'ptools': ptools,
        'config': dict(model=model, source=source, batch_size=batch_size,
                       structured_output=structured_output, sim_threshold=sim_threshold,
                       min_count=min_count, n_iters=n_iters, max_steps=max_steps,
                       output_dir=output_dir, task='team_allocation_split'),
    }
    with open(out_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f'\nSummary saved to {out_dir / "summary.json"}')


@app.command()
def eval_with_ptools(
    ptools_dir: str = typer.Option(..., help="Path to dir with ptool_*.json files"),
    split: str = typer.Option('val', help="train | val | test"),
    seed: int = typer.Option(42, help='Random seed (default 42)'),
    n_train: int = typer.Option(75, help='Train split size'),
    n_val: int = typer.Option(75, help='Val split size'),
    n_test: int = typer.Option(100, help='Test split size'),
    model: str = typer.Option('together_ai/deepseek-ai/DeepSeek-V3', help='LLM model'),
    max_steps: int = typer.Option(14, help='Max ReAct steps'),
    output_dir: str = typer.Option('eval_n30_ptools_on_val', help='Output dir'),
    only_first_ptool: bool = typer.Option(False, help='Use only ptool_001.json (skip the rest)'),
):
    """Evaluate a SAVED set of induced ptools on one split, without re-inducing.

    Loads ptool_*.json files from ptools_dir, registers them via
    create_ptool_interface, then runs ReAct on the chosen split.
    Useful for taking ptools induced on one sample (e.g., n=30) and
    testing them on a held-out sample (e.g., the n=75 val split).
    """
    out_dir = BASE_DIR / output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    config.configure(cfg={
        'llm': {'model': model, 'timeout': 300},
        'cachier': {
            'cache_dir': str(out_dir / 'llm_cache'),
            'enable_caching': True,
        },
    })

    # Load ptool specs.
    # Resolve relative paths against the project root (parent of experiments/),
    # NOT against BASE_DIR which is `experiments/induction` itself and would
    # double-prefix any "experiments/induction/..." path.
    pdir = Path(ptools_dir)
    if not pdir.is_absolute():
        # Try interpreting relative to current working directory first
        cwd_resolved = Path.cwd() / pdir
        if cwd_resolved.exists():
            pdir = cwd_resolved
        else:
            # Try relative to project root (parent of experiments/)
            project_resolved = _PROJECT_ROOT / pdir
            if project_resolved.exists():
                pdir = project_resolved
            else:
                # Last resort: relative to BASE_DIR (experiments/induction)
                pdir = BASE_DIR / pdir
    pdir = pdir.resolve()
    if not pdir.exists():
        raise SystemExit(f'ptools_dir does not exist: {pdir}')
    ptool_files = sorted(pdir.glob('ptool_*.json'))
    if not ptool_files:
        raise SystemExit(f'no ptool_*.json files found in {pdir}')
    if only_first_ptool and ptool_files:
        ptool_files = ptool_files[:1]
    ptools: list[dict] = []
    for pf in ptool_files:
        with open(pf) as f:
            spec = json.load(f)
        ptools.append(spec)
        # Register the @interface so ReAct loop can find it
        create_ptool_interface(spec['func_name'], spec['doc'])
        print(f'  Loaded ptool: {spec["display_name"]} ({spec["func_name"]})')
    print(f'Loaded {len(ptools)} ptools from {pdir}')

    cases = load_team_split(split, seed=seed,
                             n_train=n_train, n_val=n_val, n_test=n_test)
    print(f'Evaluating on {split} split: {len(cases)} cases')
    print(f'  Model: {model}, Max steps: {max_steps}, Seed: {seed}\n')

    traces = _run_all_cases_obj(cases, ptools=ptools, model=model, max_steps=max_steps)
    with open(out_dir / f'{split}_traces.json', 'w') as f:
        json.dump(traces, f, indent=2, default=str)

    n_correct = sum(1 for t in traces if t.get('correct'))
    avg_cost = sum(t.get('stats', {}).get('cost', 0) for t in traces) / max(len(traces), 1)
    n_ptool_uses = sum(1 for t in traces for s in t['steps'] if s.get('ptool_used'))
    n_extracted = sum(1 for t in traces if 'extracted' in t.get('termination', ''))
    n_neg1 = sum(1 for t in traces if t.get('answer') == -1)

    print(f'\n[{split}] accuracy with {len(ptools)} ptools: {n_correct}/{len(traces)} ({n_correct/len(traces):.1%})')
    print(f'Avg cost: ${avg_cost:.4f}  |  Total ptool invocations: {n_ptool_uses}')
    print(f'Extracted: {n_extracted}  |  -1 answers: {n_neg1}')

    summary = {
        'split': split,
        'n_cases': len(cases),
        'n_ptools': len(ptools),
        'ptools_dir': str(pdir),
        'ptools_loaded': [p['display_name'] for p in ptools],
        'accuracy': n_correct / len(traces) if traces else 0,
        'n_correct': n_correct,
        'n_total': len(traces),
        'avg_cost': avg_cost,
        'n_ptool_invocations': n_ptool_uses,
        'n_extracted_finishes': n_extracted,
        'n_neg1_answers': n_neg1,
    }
    with open(out_dir / f'{split}_summary.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f'\nSummary saved to {out_dir / f"{split}_summary.json"}')


if __name__ == '__main__':
    app()
