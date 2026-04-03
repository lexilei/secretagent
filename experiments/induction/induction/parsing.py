"""Action parsing and answer extraction."""

import re
from secretagent.llm_util import llm as llm_cached


def parse_action(text: str, mode: str = 'llm_extract') -> tuple[str, str]:
    """Parse 'ToolName[arg]' from action text.

    mode='regex': strict, requires nothing after ]
    mode='llm_extract': relaxed, allows trailing text after ]
    """
    if mode == 'llm_extract':
        # Relaxed: allow trailing text
        match = re.match(r'\s*(\w+)\[(.+?)\]', text.strip(), re.DOTALL)
        if match:
            return match.group(1), match.group(2).strip()
    else:
        match = re.match(r'\s*(\w+)\[(.+)\]\s*$', text.strip(), re.DOTALL)
        if match:
            return match.group(1), match.group(2).strip()
    return '', text.strip()


def extract_answer_index(text: str, choices: list, model: str) -> int | None:
    """Use LLM to extract an answer index from free-form text.

    Returns 0-based index if agent has decided, None if still investigating.
    """
    choices_str = ', '.join(f'({i}) {c}' for i, c in enumerate(choices))
    prompt = (
        f"The following text is from an agent solving a murder mystery. "
        f"The answer choices are: {choices_str}\n\n"
        f"Text:\n{text}\n\n"
        f"Has the agent decided on a final answer? If yes, return ONLY the "
        f"0-based index number. If the agent is still investigating and has "
        f"not committed to a final answer, return ONLY the word 'none'.\n\n"
        f"<answer>index or none</answer>"
    )
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
