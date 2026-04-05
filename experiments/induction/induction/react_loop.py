"""ReAct execution engine with configurable parsing."""

import hashlib
import json
import re
import time
from pathlib import Path

from litellm import completion, completion_cost
from secretagent import config
from secretagent.llm_util import llm as llm_cached

from .ptool_registry import find_induced_ptool
from .parsing import parse_action, extract_answer_index
from .io import format_ptool_actions_for_prompt

# Persistent completion cache (shared across experiments)
_COMPLETION_CACHE_DIR = Path(__file__).parent.parent / 'completion_cache'
_COMPLETION_MEM_CACHE: dict[str, tuple[str, dict]] = {}


FEW_SHOT_MURDER = """\
Question: In a small town, a baker named Tom was found dead in his shop. \
The suspects are: Alice (his business partner), Bob (a disgruntled employee), \
and Carol (his ex-wife). Alice claims she was at home all evening. Bob says he \
was at a bar with friends. Carol says she was visiting her sister in another city. \
A witness saw someone matching Bob's description near the shop at 10pm. \
Tom's financial records show Alice would inherit the business. \
A kitchen knife from the shop was the murder weapon, with no fingerprints. \
The choices are: (0) Alice, (1) Bob, (2) Carol.
Thought 1: I need to understand who the suspects are and what evidence exists.
Action 1: Do[identify all suspects, their relationship to the victim, and any motives mentioned]
Observation 1: Three suspects: Alice is Tom's business partner and would inherit the business (financial motive). Bob is a disgruntled employee (revenge/grudge motive). Carol is Tom's ex-wife (personal motive). Tom's financial records confirm Alice stands to gain the most financially.
Thought 2: Now I know the suspects. I should check their alibis against the physical evidence.
Action 2: Do[check each suspect's alibi and whether any evidence contradicts them]
Observation 2: Alice claims she was at home all evening — no witnesses confirm or deny. Bob claims he was at a bar with friends — but a witness saw someone matching Bob's description near the shop at 10pm, contradicting his alibi. Carol claims she was visiting her sister in another city — unverified but no contradicting evidence.
Thought 3: Bob's alibi is contradicted by a witness. He had access to shop knives as an employee. The wiped fingerprints show premeditation. Bob is the most likely suspect.
Action 3: Finish[1]

"""

# Generic few-shot that works for any narrative reasoning task
FEW_SHOT_GENERIC = """\
Question: A company has three departments. The memo says Department A handles all client requests. \
Department B processes internal reports. Department C manages both external partnerships and budget review. \
A new project requires external partnership management. \
The choices are: (0) Department A, (1) Department B, (2) Department C.
Thought 1: I need to identify which department handles external partnerships.
Action 1: Do[identify what each department is responsible for, focusing on external partnerships]
Observation 1: Department A handles client requests. Department B processes internal reports. Department C manages external partnerships and budget review.
Thought 2: Department C manages external partnerships. The project requires external partnership management. Department C is the answer.
Action 2: Finish[2]

"""

# Map benchmark to task description and few-shot
_BENCHMARK_CONFIG = {
    'murder': {
        'task': 'Solve a murder mystery question',
        'few_shot': FEW_SHOT_MURDER,
        'env_role': 'a careful reader assisting a detective',
    },
    'object': {
        'task': 'Solve an object placement reasoning question',
        'few_shot': FEW_SHOT_GENERIC,
        'env_role': 'a careful reader tracking object locations',
    },
    'team': {
        'task': 'Solve a team allocation reasoning question',
        'few_shot': FEW_SHOT_GENERIC,
        'env_role': 'a careful reader analyzing team assignments',
    },
    'true_detective': {
        'task': 'Solve a mystery reasoning question',
        'few_shot': FEW_SHOT_MURDER,
        'env_role': 'a careful reader assisting a detective',
    },
}

# Default for backward compat
FEW_SHOT = FEW_SHOT_MURDER


def _llm_with_stop(prompt: str, model: str, stop: list[str]) -> tuple[str, dict]:
    # Check persistent cache (temperature=0, deterministic)
    cache_key = hashlib.sha256(
        f"{prompt}|||{model}|||{stop}".encode()).hexdigest()
    if cache_key in _COMPLETION_MEM_CACHE:
        return _COMPLETION_MEM_CACHE[cache_key]
    cache_file = _COMPLETION_CACHE_DIR / f"{cache_key}.json"
    if cache_file.exists():
        cached = json.loads(cache_file.read_text())
        result = (cached[0], cached[1])
        _COMPLETION_MEM_CACHE[cache_key] = result
        return result

    messages = [dict(role='user', content=prompt)]
    timeout = config.get('llm.timeout', 300)
    start_time = time.time()
    response = completion(
        model=model, messages=messages, stop=stop,
        max_tokens=512, timeout=timeout, temperature=0,
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

    # Persist to cache
    _COMPLETION_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(json.dumps([text, stats]))
    _COMPLETION_MEM_CACHE[cache_key] = (text, stats)

    return text, stats


def _execute_do_action(action_arg: str, narrative: str, model: str,
                       benchmark: str = 'murder') -> tuple[str, dict]:
    bcfg = _BENCHMARK_CONFIG.get(benchmark, _BENCHMARK_CONFIG['murder'])
    env_prompt = (
        f"You are {bcfg['env_role']}. Read the narrative below "
        f"and respond to this request:\n\n"
        f"Request: {action_arg}\n\n"
        f"Narrative:\n{narrative}\n\n"
        f"Respond concisely with only information directly stated in or inferable "
        f"from the narrative. Do not speculate beyond what the text supports."
    )
    response, stats = llm_cached(env_prompt, model)
    if len(response) > 1500:
        response = response[:1500] + '...'
    return response.strip(), stats


def _execute_ptool(iface, narrative: str, focus: str) -> tuple[str, dict]:
    start_time = time.time()
    result = iface(narrative=narrative, focus=focus)
    latency = time.time() - start_time
    text = str(result) if result is not None else ''
    if len(text) > 1500:
        text = text[:1500] + '...'
    return text, {'latency': latency}


def _make_system_prompt(ptools: list[dict], parsing_mode: str,
                        benchmark: str = 'murder') -> str:
    bcfg = _BENCHMARK_CONFIG.get(benchmark, _BENCHMARK_CONFIG['murder'])
    base = (
        f"{bcfg['task']} with interleaving Thought, Action, Observation steps. "
        "Thought can reason about the current situation. Action can be:\n"
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
        )
        if parsing_mode == 'llm_extract':
            base += 'When you have enough evidence to decide, immediately use Finish[index].\n'
    else:
        base += (
            '(2) Finish[answer], which returns the 0-based index of the correct '
            'answer choice and finishes the task.\n'
        )
    base += 'Here are some examples.\n'
    return base


def run_react_on_case(narrative: str, question: str, choices: list,
                      ptools: list[dict], model: str,
                      max_steps: int = 14,
                      parsing_mode: str = 'llm_extract',
                      benchmark: str = 'murder') -> dict:
    system = _make_system_prompt(ptools, parsing_mode, benchmark=benchmark)
    bcfg = _BENCHMARK_CONFIG.get(benchmark, _BENCHMARK_CONFIG['murder'])
    choices_str = ', '.join(f'({i}) {c}' for i, c in enumerate(choices))
    prompt = (
        system + bcfg['few_shot']
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

        action_name, action_arg = parse_action(action_str, parsing_mode)
        action_name_lower = action_name.lower()

        # Finish
        if action_name_lower == 'finish':
            try:
                answer = int(action_arg.strip().split()[0])
            except ValueError:
                answer = -1
            if answer == -1 and parsing_mode == 'llm_extract':
                answer = extract_answer_index(action_str, choices, model) or -1
            steps.append(dict(
                step=i, thought=thought, action=action_str,
                action_type='finish', action_arg=action_arg,
                observation=None, ptool_used=None,
            ))
            return dict(answer=answer, steps=steps, n_steps=i,
                        termination='finish', stats=total_stats)

        # LLM extraction: check if thought contains conclusion
        if parsing_mode == 'llm_extract' and (not action_str or action_name_lower == ''):
            idx = extract_answer_index(thought, choices, model)
            if idx is not None:
                steps.append(dict(
                    step=i, thought=thought, action='[extracted from thought]',
                    action_type='finish_extracted', action_arg=str(idx),
                    observation=None, ptool_used=None,
                ))
                return dict(answer=idx, steps=steps, n_steps=i,
                            termination='finish_extracted', stats=total_stats)

        # Ptool call
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
            observation, env_stats = _execute_do_action(action_arg, narrative, model, benchmark=benchmark)
            accum(env_stats)
            steps.append(dict(
                step=i, thought=thought, action=action_str,
                action_type='do', action_arg=action_arg,
                observation=observation, ptool_used=None,
            ))
        elif action_str:
            observation, env_stats = _execute_do_action(action_str, narrative, model, benchmark=benchmark)
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

    # Max steps — last resort extraction
    if parsing_mode == 'llm_extract':
        full_text = ' '.join(s.get('thought', '') for s in steps[-3:])
        last_idx = extract_answer_index(full_text, choices, model)
        if last_idx is not None:
            return dict(answer=last_idx, steps=steps, n_steps=max_steps,
                        termination='finish_extracted_at_max', stats=total_stats)

    return dict(answer=-1, steps=steps, n_steps=max_steps,
                termination='max_steps', stats=total_stats)


def run_all_cases(cases: list[dict], ptools: list[dict],
                  model: str, max_steps: int,
                  parsing_mode: str = 'llm_extract',
                  benchmark: str = 'murder') -> list[dict]:
    """Run ReAct on all cases with retry + delay."""
    results = []
    max_retries = 3
    for idx, case in enumerate(cases):
        print(f'    [{idx+1}/{len(cases)}] {case["name"]}...', end=' ', flush=True)
        result = None
        for attempt in range(max_retries):
            try:
                result = run_react_on_case(
                    case['narrative'], case['question'], case['choices'],
                    ptools, model, max_steps=max_steps,
                    parsing_mode=parsing_mode, benchmark=benchmark)
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
