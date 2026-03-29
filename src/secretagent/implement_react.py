"""ReAct factory: Thought-Action-Observation loop with tool use.

Implements ReAct (Yao et al., ICLR 2023) as an Implementation.Factory.
The agent iteratively reasons about the task, selects and calls available
interfaces (tools), observes results, and produces a final answer.

This is the key "open-ended agent loop" baseline in the research plan,
contrasted against fixed-budget structured reasoning (ptools).

Usage via config::

    ptools:
      answer_question:
        method: react
        max_steps: 10

Or programmatically::

    interface.implement_via('react', max_steps=8)

The agent can use all other implemented interfaces as tools.
"""

import ast
import re
from typing import Any, Callable

from secretagent.core import (
    Interface, Implementation, register_factory, all_interfaces,
)
from secretagent import config, llm_util, record


class ReActFactory(Implementation.Factory):
    """ReAct agent: iterative Thought -> Action -> Observation loop.

    Config keys:
        max_steps: maximum reasoning iterations (default 10)
        tools: tool specification (default '__all__')
        budget_cap: if set, stop after this many LLM calls total (for ReAct+budget variant)
    """

    def build_fn(
        self,
        interface: Interface,
        max_steps: int = 10,
        tools: str | list = '__all__',
        budget_cap: int | None = None,
        **prompt_kw,
    ) -> Callable:
        # Resolve which interfaces are available as tools
        tool_interfaces = _resolve_tool_interfaces(interface, tools)

        def result_fn(*args, **kw):
            with config.configuration(**prompt_kw):
                model = config.require('llm.model')
                input_args = interface.format_args(*args, **kw)
                return_type = interface.annotations.get('return', str)

                # Build tool descriptions
                tool_desc = _format_tool_descriptions(tool_interfaces)

                # ReAct loop state
                history: list[dict[str, Any]] = []
                llm_calls = 0
                effective_max = budget_cap if budget_cap else max_steps

                for step in range(effective_max):
                    # Generate thought + action/answer
                    prompt = _format_react_prompt(
                        interface, input_args, tool_desc, history, return_type)
                    response, stats = llm_util.llm(prompt, model)
                    llm_calls += 1
                    record.record(
                        func=f'{interface.name}:react_step_{step}',
                        args=args, kw=kw, output=response, stats=stats)

                    # Check for final answer
                    answer_match = re.search(
                        r'<answer>(.*?)</answer>', response, re.DOTALL)
                    if answer_match:
                        final_answer = answer_match.group(1).strip()
                        try:
                            if return_type in [int, str, float]:
                                answer = return_type(final_answer)
                            else:
                                answer = ast.literal_eval(final_answer)
                        except Exception as ex:
                            record.record(
                                func=interface.name, args=args, kw=kw,
                                output=f'**exception**: {ex}', stats={},
                                step_info={'react_steps': llm_calls})
                            raise
                        record.record(
                            func=interface.name, args=args, kw=kw,
                            output=answer, stats={},
                            step_info={'react_steps': llm_calls,
                                       'termination': 'answer_found'})
                        return answer

                    # Parse and execute action
                    action_match = re.search(
                        r'<action>\s*(\w+)\s*\((.*?)\)\s*</action>',
                        response, re.DOTALL)
                    if action_match:
                        tool_name = action_match.group(1)
                        args_str = action_match.group(2)

                        # Find the tool interface
                        tool_iface = None
                        for t in tool_interfaces:
                            if t.name == tool_name:
                                tool_iface = t
                                break

                        if tool_iface is None:
                            history.append({
                                'thought': response,
                                'action': f'{tool_name}({args_str})',
                                'observation': f'Error: Unknown tool "{tool_name}". '
                                               f'Available: {[t.name for t in tool_interfaces]}',
                            })
                            continue

                        # Parse arguments and call tool
                        try:
                            tool_args = _parse_action_args(args_str)
                            result = tool_iface(**tool_args) if tool_args else tool_iface(*args, **kw)
                            obs = repr(result)
                        except Exception as ex:
                            obs = f'Error: {ex}'

                        history.append({
                            'thought': response,
                            'action': f'{tool_name}({args_str})',
                            'observation': obs,
                        })
                    else:
                        # No action or answer found - record as thinking step
                        history.append({
                            'thought': response,
                            'action': None,
                            'observation': 'No action taken.',
                        })

                # Max steps reached - try to extract any answer from last response
                record.record(
                    func=interface.name, args=args, kw=kw,
                    output='**exception**: max steps reached', stats={},
                    step_info={'react_steps': llm_calls,
                               'termination': 'max_steps'})
                raise ValueError(
                    f'ReAct agent did not produce an answer within {effective_max} steps')

        return result_fn


def _resolve_tool_interfaces(
        interface: Interface, tools: str | list) -> list[Interface]:
    """Resolve tool specification to list of Interface objects."""
    if tools == '__all__':
        return [
            iface for iface in all_interfaces()
            if iface is not interface and iface.implementation is not None
        ]
    if not tools:
        return []
    result = []
    for t in tools:
        if isinstance(t, Interface):
            result.append(t)
        elif isinstance(t, str):
            for iface in all_interfaces():
                if iface.name == t and iface.implementation is not None:
                    result.append(iface)
                    break
    return result


def _format_tool_descriptions(tool_interfaces: list[Interface]) -> str:
    """Format tool descriptions for the prompt."""
    if not tool_interfaces:
        return "No tools available."
    lines = []
    for iface in tool_interfaces:
        # Get parameter names and types
        params = []
        annotations = dict(iface.annotations)
        annotations.pop('return', None)
        for pname, ptype in annotations.items():
            tname = ptype.__name__ if hasattr(ptype, '__name__') else str(ptype)
            params.append(f'{pname}: {tname}')
        param_str = ', '.join(params)
        ret = iface.annotations.get('return', 'Any')
        ret_name = ret.__name__ if hasattr(ret, '__name__') else str(ret)
        lines.append(f"- {iface.name}({param_str}) -> {ret_name}")
        if iface.doc:
            doc_line = iface.doc.strip().split('\n')[0]
            lines.append(f"    {doc_line}")
    return '\n'.join(lines)


def _format_react_prompt(
        interface: Interface,
        input_args: str,
        tool_desc: str,
        history: list[dict],
        return_type: type,
) -> str:
    """Format the ReAct prompt with history."""
    lines = [
        "You are a reasoning agent. Solve the task by thinking step-by-step "
        "and using available tools.",
        "",
        f"## Task",
        f"{interface.doc}",
        "",
        f"## Input",
        f"{input_args}",
        "",
        f"## Available Tools",
        tool_desc,
        "",
    ]

    if history:
        lines.append("## Previous Steps")
        for i, step in enumerate(history):
            lines.append(f"\n**Step {i+1}:**")
            # Truncate thought for context efficiency
            thought = step['thought']
            if len(thought) > 500:
                thought = thought[:500] + '...'
            lines.append(f"Thought: {thought}")
            if step['action']:
                lines.append(f"Action: {step['action']}")
                obs = step['observation']
                if len(str(obs)) > 500:
                    obs = str(obs)[:500] + '...'
                lines.append(f"Observation: {obs}")
        lines.append("")

    ret_name = return_type.__name__ if hasattr(return_type, '__name__') else str(return_type)
    lines.extend([
        "## Instructions",
        "Think about what to do, then either call a tool or give your final answer.",
        "",
        "To call a tool:",
        "<thought>Your reasoning</thought>",
        "<action>tool_name(arg1=value1, arg2=value2)</action>",
        "",
        "To give your final answer:",
        "<thought>Your reasoning</thought>",
        f"<answer>YOUR ANSWER (type: {ret_name})</answer>",
        "",
        "If your answer is an integer, provide just the number.",
    ])

    return '\n'.join(lines)


def _parse_action_args(args_str: str) -> dict[str, Any]:
    """Parse action arguments from string like 'key1=val1, key2=val2'."""
    if not args_str.strip():
        return {}
    # Try Python kwargs parsing
    args = {}
    pattern = r'(\w+)\s*=\s*'
    matches = list(re.finditer(pattern, args_str))
    for i, match in enumerate(matches):
        key = match.group(1)
        start = match.end()
        if i + 1 < len(matches):
            end = matches[i + 1].start()
            value_str = args_str[start:end].rstrip(', \t\n')
        else:
            value_str = args_str[start:].rstrip(', \t\n)')
        try:
            value = ast.literal_eval(value_str)
        except (ValueError, SyntaxError):
            value = value_str.strip('"\'')
        args[key] = value
    return args


register_factory('react', ReActFactory())
