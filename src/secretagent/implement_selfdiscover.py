"""Self-Discover factory: structured reasoning via self-composed reasoning modules.

Implements the Self-Discover framework (Zhou et al., NeurIPS 2024):
  Stage 1 (SELECT): Pick relevant reasoning modules from a predefined set
  Stage 2 (ADAPT): Adapt selected modules to be task-specific
  Stage 3 (IMPLEMENT): Build a JSON reasoning structure from adapted modules
  Stage 4 (SOLVE): Use the structure to solve the task instance

This is a fixed 3-4 call pipeline (SELECT+ADAPT+IMPLEMENT can share a call
if desired, but we keep them separate for ablation).

Usage via config::

    ptools:
      answer_question:
        method: self_discover
        reasoning_modules: null  # use defaults

Or programmatically::

    interface.implement_via('self_discover')

Reference: "Large Language Models Self-Compose Reasoning Structures"
(Zhou et al., NeurIPS 2024)
"""

import re
from typing import Callable

from secretagent.core import Interface, Implementation, register_factory
from secretagent import config, llm_util, record


# The 39 reasoning modules from the Self-Discover paper (Table 7)
DEFAULT_REASONING_MODULES = [
    "How could I devise an experiment to help solve that problem?",
    "Make a list of ideas for solving this problem, and apply them one by one to the problem to see if any progress can be made.",
    "How could I measure progress on this problem?",
    "How can I simplify the problem so that it is easier to solve?",
    "What are the key assumptions underlying this problem?",
    "What are the potential risks and drawbacks of each solution?",
    "What are the alternative perspectives or viewpoints on this problem?",
    "What are the long-term implications of this solution?",
    "How can I break down this problem into smaller, more manageable parts?",
    "Critical Thinking: This style involves analyzing the problem from different perspectives, questioning assumptions, and evaluating the evidence or information available.",
    "Try creative thinking, currentGenerate innovative and out-of-the-box ideas to solve the problem.",
    "Seek input and collaboration from others to solve the problem.",
    "Use systems thinking: Consider the problem as part of a larger system and understanding the interconnectedness of various elements.",
    "Use Risk Analysis: Evaluate potential risks associated with each solution and try to minimize them.",
    "Use Reflective Thinking: Step back from the problem, take the time for introspection and self-reflection. Examine personal biases, assumptions, and mental models.",
    "What is the core issue or problem that needs to be addressed?",
    "What are the underlying causes or factors contributing to the problem?",
    "Are there any potential solutions or strategies that have been tried before? If yes, what were the outcomes and lessons learned?",
    "What are the potential obstacles or challenges that might arise in solving this problem?",
    "Are there any relevant data or information that can provide insights into the problem?",
    "Are there any stakeholders or parties who are directly affected by the problem?",
    "What resources (time, money, expertise) are available to tackle the problem?",
    "How can progress or success in solving the problem be measured or evaluated?",
    "What indicators or metrics can be used?",
    "Is the problem a technical or practical one that requires a specific expertise or skill set? Or is it more of a conceptual or theoretical problem?",
    "Does the problem involve a physical constraint, such as limited resources, infrastructure, or space?",
    "Is the problem related to human-behavior, such as a social, cultural, or psychological issue?",
    "Does the problem involve decision-making or planning, where choices need to be made under uncertainty or with competing objectives?",
    "Is the problem an analytical one that requires data collection, analysis, and interpretation?",
    "Is the problem a design challenge that requires creative solutions and innovation?",
    "Does the problem require addressing systemic or structural issues rather than just individual instances?",
    "Is the problem time-sensitive or urgent, requiring immediate attention and action?",
    "What kinds of solution typically are produced for this kind of problem specification?",
    "Given the problem specification and the current best solution, have a guess about other possible solutions.",
    "Let's imagine the current best solution is totally wrong, what other ways are there to think about the problem specification?",
    "What is the best way to modify this current best solution, given what you know about these kinds of problem specification?",
    "Ignoring the current best solution, create an entirely new solution to the problem.",
    "Let's think step by step.",
    "Let's make a step by step plan and implement it with good notation and target.",
]


class SelfDiscoverFactory(Implementation.Factory):
    """Self-Discover: compose reasoning structures from a library of modules.

    Fixed 4-call pipeline:
      1. SELECT relevant reasoning modules for the task
      2. ADAPT selected modules to the specific task
      3. IMPLEMENT a reasoning structure (JSON template)
      4. SOLVE using the reasoning structure

    Config keys:
        reasoning_modules: list of module strings (default: paper's 39 modules)
    """

    def build_fn(
        self,
        interface: Interface,
        reasoning_modules: list[str] | None = None,
        **prompt_kw,
    ) -> Callable:
        modules = reasoning_modules or DEFAULT_REASONING_MODULES
        modules_text = "\n".join(f"- {m}" for m in modules)

        def result_fn(*args, **kw):
            with config.configuration(**prompt_kw):
                model = config.require('llm.model')
                input_args = interface.format_args(*args, **kw)
                task_description = f"Task: {interface.doc}\n\nInput:\n{input_args}"

                # Stage 1: SELECT
                select_prompt = (
                    f"Given the task below, select the reasoning modules that are "
                    f"most useful for solving this kind of task.\n\n"
                    f"{task_description}\n\n"
                    f"Reasoning module library:\n{modules_text}\n\n"
                    f"Select the most relevant modules and list them.\n"
                    f"<selected_modules>\nYOUR SELECTION\n</selected_modules>"
                )
                selected, stats1 = llm_util.llm(select_prompt, model)
                record.record(func=f'{interface.name}:select',
                              args=args, kw=kw, output=selected, stats=stats1)

                # Stage 2: ADAPT
                adapt_prompt = (
                    f"Given the task below, adapt the following reasoning modules "
                    f"to be specific to this task.\n\n"
                    f"{task_description}\n\n"
                    f"Selected reasoning modules:\n{selected}\n\n"
                    f"Rephrase and adapt each module to be specific to the task. "
                    f"Make them actionable steps.\n"
                    f"<adapted_modules>\nYOUR ADAPTED MODULES\n</adapted_modules>"
                )
                adapted, stats2 = llm_util.llm(adapt_prompt, model)
                record.record(func=f'{interface.name}:adapt',
                              args=args, kw=kw, output=adapted, stats=stats2)

                # Stage 3: IMPLEMENT reasoning structure
                implement_prompt = (
                    f"Given the task below and the adapted reasoning modules, "
                    f"implement a step-by-step reasoning structure as a JSON template "
                    f"that can be filled in to solve this kind of task.\n\n"
                    f"{task_description}\n\n"
                    f"Adapted reasoning modules:\n{adapted}\n\n"
                    f"Create a JSON reasoning structure where each key is a reasoning "
                    f"step and the value is a placeholder to be filled in.\n"
                    f"<reasoning_structure>\nYOUR JSON STRUCTURE\n</reasoning_structure>"
                )
                structure, stats3 = llm_util.llm(implement_prompt, model)
                record.record(func=f'{interface.name}:implement',
                              args=args, kw=kw, output=structure, stats=stats3)

                # Stage 4: SOLVE using the structure
                return_type = interface.annotations.get('return', str)
                solve_prompt = (
                    f"Follow the reasoning structure below to solve the task. "
                    f"Fill in each step of the structure, then provide your final answer.\n\n"
                    f"{task_description}\n\n"
                    f"Reasoning structure to follow:\n{structure}\n\n"
                    f"Fill in the structure step by step, then give your final answer.\n"
                    f"<answer>\nFINAL ANSWER\n</answer>"
                )
                solution, stats4 = llm_util.llm(solve_prompt, model)

                # Parse answer
                try:
                    match = re.search(r'<answer>(.*)</answer>', solution, re.DOTALL)
                    final_answer = match.group(1).strip()
                    if return_type in [int, str, float]:
                        answer = return_type(final_answer)
                    else:
                        import ast
                        answer = ast.literal_eval(final_answer)
                except Exception as ex:
                    record.record(func=interface.name, args=args, kw=kw,
                                  output=f'**exception**: {ex}', stats=stats4)
                    raise
                record.record(func=interface.name, args=args, kw=kw,
                              output=answer, stats=stats4)
                return answer

        return result_fn


register_factory('self_discover', SelfDiscoverFactory())
