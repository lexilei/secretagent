"""Step 1: Run open-ended ReAct on MUSR murder mysteries and save traces.

Proper ReAct loop (Yao et al., ICLR 2023):
  - Each turn: LLM generates Thought + Action, stops at Observation
  - Environment returns Observation
  - Full history appended to prompt, repeat

The agent has ONE open-ended action: Do[whatever it wants].
No predefined action taxonomy. The LLM fills in both Thought and Action freely.

Usage:
    source .env && export TOGETHER_API_KEY
    uv run python experiments/induction/run_react.py --n 10
"""

import ast
import json
import re
import sys
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / 'src'))
sys.path.insert(0, str(_PROJECT_ROOT / 'benchmarks' / 'musr'))

import typer
from litellm import completion, completion_cost
from secretagent import config
from secretagent.llm_util import llm as llm_cached

OUTPUT_DIR = Path(__file__).parent / 'traces'

MAX_STEPS = 7

SYSTEM_PROMPT = """\
Solve a murder mystery question with interleaving Thought, Action, Observation steps. \
Thought can reason about the current situation. Action can be two types:
(1) Do[description], which performs any reasoning step you describe — \
the system will read the narrative and return the relevant information. \
You decide what to investigate.
(2) Finish[answer], which returns the 0-based index of the correct answer choice and finishes the task.
Here are some examples.
"""

FEW_SHOT_EXAMPLES = """\
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

Question: Detective Miles investigates the death of Professor Ward, found in his locked study. \
Suspects: his graduate student Lin who needed Ward's recommendation, his colleague Dr. Reeves \
who was competing for the same grant, and his wife Martha who recently learned of his affair. \
Lin was seen in the library until 9pm. Dr. Reeves had a faculty dinner until 8:30pm. \
Martha was home but cannot prove it. The study window was found slightly open. \
Ward's coffee cup contained traces of a rare poison. Only Martha and Dr. Reeves had chemistry backgrounds. \
The choices are: (0) Lin, (1) Dr. Reeves, (2) Martha.
Thought 1: A poisoning case. I need to figure out who had the means to use a rare poison.
Action 1: Do[determine who had the knowledge and ability to obtain and administer a rare poison]
Observation 1: Ward was poisoned via his coffee cup with a rare poison. Only Martha (wife) and Dr. Reeves (colleague) had chemistry backgrounds, giving them the expertise to obtain and use such a poison. Lin has no mentioned chemistry background.
Thought 2: So Lin is unlikely. Between Martha and Reeves, I need to check opportunity.
Action 2: Do[compare Martha and Dr. Reeves on opportunity to access Ward's study and coffee]
Observation 2: Martha, as Ward's wife, had routine access to his study and could easily have poisoned his coffee. She was home but cannot prove it — no alibi. Dr. Reeves had a faculty dinner until 8:30pm. The study window was found slightly open, which could indicate entry from outside.
Thought 3: Martha has chemistry knowledge, no alibi, easy access to his study and coffee, and a fresh motive (just learned of his affair). She is the strongest suspect.
Action 3: Finish[2]

"""


def llm_with_stop(prompt: str, model: str, stop: list[str]) -> tuple[str, dict]:
    """Call LLM with stop tokens. Uses litellm directly."""
    messages = [dict(role='user', content=prompt)]
    timeout = config.get('llm.timeout', 300)
    start_time = time.time()

    response = completion(
        model=model,
        messages=messages,
        stop=stop,
        max_tokens=512,
        timeout=timeout,
        temperature=0,
    )
    latency = time.time() - start_time
    model_output = response.choices[0].message.content or ''

    try:
        cost = completion_cost(completion_response=response)
    except Exception:
        cost = 0.0

    stats = dict(
        input_tokens=response.usage.prompt_tokens,
        output_tokens=response.usage.completion_tokens,
        latency=latency,
        cost=cost,
    )
    return model_output, stats


def make_initial_prompt(narrative: str, question: str, choices: list) -> str:
    choices_str = ", ".join(f"({i}) {c}" for i, c in enumerate(choices))
    return (
        SYSTEM_PROMPT
        + FEW_SHOT_EXAMPLES
        + f"Question: {narrative}\n{question}\nThe choices are: {choices_str}.\n"
    )


def execute_action(action_description: str, narrative: str, model: str) -> tuple[str, dict]:
    """Environment: read the narrative and answer whatever the agent asked."""
    env_prompt = (
        f"You are a careful reader assisting a detective. Read the narrative below "
        f"and respond to this request:\n\n"
        f"Request: {action_description}\n\n"
        f"Narrative:\n{narrative}\n\n"
        f"Respond concisely with only information directly stated in or inferable "
        f"from the narrative. Do not speculate beyond what the text supports."
    )
    response, stats = llm_cached(env_prompt, model)
    if len(response) > 1500:
        response = response[:1500] + "..."
    return response.strip(), stats


def parse_action(text: str) -> tuple[str, str]:
    """Parse 'Do[description]' or 'Finish[answer]' from action text."""
    match = re.match(r'\s*(\w+)\[(.+)\]\s*$', text.strip(), re.DOTALL)
    if match:
        return match.group(1).lower(), match.group(2).strip()
    return "", text.strip()


def run_react_on_case(narrative: str, question: str, choices: list,
                      model: str) -> dict:
    """Run the full open-ended ReAct loop on a single case."""
    prompt = make_initial_prompt(narrative, question, choices)
    steps = []
    total_stats = {'input_tokens': 0, 'output_tokens': 0, 'latency': 0, 'cost': 0}

    def accum(stats):
        for k in total_stats:
            total_stats[k] += stats.get(k, 0)

    for i in range(1, MAX_STEPS + 1):
        # LLM generates Thought i + Action i, stops before Observation i
        llm_input = prompt + f"Thought {i}:"
        stop = [f"\nObservation {i}:"]
        response, stats = llm_with_stop(llm_input, model, stop=stop)
        accum(stats)

        # Parse: response should be "thought text\nAction i: Do[...]\n"
        response = response.strip()
        action_match = re.search(r'Action\s*\d+\s*:\s*(.+)', response, re.DOTALL)
        if action_match:
            thought = response[:action_match.start()].strip()
            action_str = action_match.group(1).strip()
        else:
            # No action found — model might have put Finish inline
            thought = response
            action_str = ""

        action_type, action_arg = parse_action(action_str)

        # Finish
        if action_type == 'finish':
            try:
                answer = int(action_arg.strip())
            except ValueError:
                answer = -1
            steps.append({
                'step': i, 'thought': thought,
                'action': action_str, 'action_type': 'finish',
                'action_arg': action_arg, 'observation': None,
            })
            return {
                'answer': answer, 'steps': steps,
                'n_steps': i, 'termination': 'finish',
                'stats': total_stats,
            }

        # Do[...] or fallback
        if action_type == 'do' and action_arg:
            observation, env_stats = execute_action(action_arg, narrative, model)
            accum(env_stats)
        elif action_str:
            # Model didn't use Do[...] format but wrote something — try it
            observation, env_stats = execute_action(action_str, narrative, model)
            accum(env_stats)
            action_type = 'do'
            action_arg = action_str
        else:
            # Truly empty — force a retry prompt
            observation = "No action provided. Please use Do[description] or Finish[answer_index]."
            action_type = 'none'
            action_arg = ''

        steps.append({
            'step': i, 'thought': thought,
            'action': action_str, 'action_type': action_type,
            'action_arg': action_arg, 'observation': observation,
        })

        # Append full turn to prompt
        prompt += (f"Thought {i}: {thought}\n"
                   f"Action {i}: {action_str}\n"
                   f"Observation {i}: {observation}\n")

    return {
        'answer': -1, 'steps': steps,
        'n_steps': MAX_STEPS, 'termination': 'max_steps',
        'stats': total_stats,
    }


def load_murder_cases(n: int, seed: int = 42) -> list[dict]:
    data_file = _PROJECT_ROOT / 'benchmarks' / 'musr' / 'data' / 'murder_mysteries.json'
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
    import random
    random.Random(seed).shuffle(cases)
    return cases[:n]


app = typer.Typer()

@app.command()
def main(
    n: int = typer.Option(10, help="Number of examples to run"),
    model: str = typer.Option(
        "together_ai/deepseek-ai/DeepSeek-V3",
        help="LLM model"),
    seed: int = typer.Option(42, help="Random seed for shuffling"),
):
    """Run open-ended ReAct on MUSR murder mysteries and save traces."""
    config.configure(cfg={
        'llm': {'model': model, 'timeout': 300},
        'cachier': {
            'cache_dir': str(OUTPUT_DIR / 'llm_cache'),
            'enable_caching': True,
        },
    })

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cases = load_murder_cases(n, seed)
    print(f'Running open-ended ReAct on {len(cases)} murder mystery cases...')
    print(f'Action space: Do[free text], Finish[answer]')
    print(f'Max steps: {MAX_STEPS}, stop tokens enabled\n')

    results = []
    for idx, case in enumerate(cases):
        print(f'[{idx+1}/{n}] {case["name"]}...', end=' ', flush=True)
        try:
            result = run_react_on_case(
                case['narrative'], case['question'], case['choices'], model)
        except Exception as ex:
            print(f'ERROR: {ex}')
            result = {'answer': -1, 'steps': [], 'n_steps': 0,
                      'termination': 'error', 'stats': {}}

        correct = (result['answer'] == case['expected'])
        result['case_name'] = case['name']
        result['expected'] = case['expected']
        result['correct'] = correct
        results.append(result)

        status = 'CORRECT' if correct else 'WRONG'
        print(f'{status} (pred={result["answer"]}, exp={case["expected"]}, '
              f'steps={result["n_steps"]})')
        for s in result['steps']:
            arg = s.get('action_arg', '') or ''
            if len(arg) > 80:
                arg = arg[:77] + '...'
            atype = s.get('action_type', '?')
            print(f'    {s["step"]}. [{atype}] {arg}')

    # Save traces
    output_file = OUTPUT_DIR / 'react_traces.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    successes = [r for r in results if r['correct']]
    success_file = OUTPUT_DIR / 'success_traces.json'
    with open(success_file, 'w') as f:
        json.dump(successes, f, indent=2, default=str)

    correct = sum(1 for r in results if r['correct'])
    total_steps = sum(r['n_steps'] for r in results)
    print(f'\n{"="*60}')
    print(f'Accuracy: {correct}/{n} ({correct/n:.0%})')
    print(f'Avg steps: {total_steps/n:.1f}')
    print(f'Traces saved to {output_file}')
    print(f'Success traces ({len(successes)}) saved to {success_file}')

    # Print all Do[] descriptions for pattern analysis
    print(f'\nAll action descriptions (for induction):')
    for r in results:
        tag = 'OK' if r['correct'] else 'WRONG'
        print(f'\n  {r["case_name"]} ({tag}):')
        for s in r['steps']:
            if s['action_type'] == 'do':
                print(f'    {s["step"]}. {s["action_arg"]}')


if __name__ == '__main__':
    app()
