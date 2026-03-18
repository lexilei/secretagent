"""Shared interfaces for answer extraction across all MUSR splits."""

import re
from secretagent.core import interface, register_factory, Implementation


def match_choice(text, choices):
    """Match LLM output to a choice index by finding the last-mentioned choice."""
    text_lower = text.strip().lower()
    last_pos = {i: text_lower.rfind(c.lower()) for i, c in enumerate(choices)}
    found = {i: pos for i, pos in last_pos.items() if pos >= 0}
    if found:
        return max(found, key=found.get)
    # Fallback: last bare integer
    for line in reversed(text.strip().splitlines()):
        line = line.strip()
        if re.fullmatch(r'\d+', line):
            return int(line)
    return -1


class ExtractIndexFactory(Implementation.Factory):
    """Direct implementation of extract_index using string matching."""
    def build_fn(self, interface, **_kw):
        return lambda answer_text, choices: match_choice(answer_text, choices)

register_factory('match_choice', ExtractIndexFactory())


@interface
def raw_answer(narrative: str, question: str, choices: list) -> str:
    """Read the narrative and answer the multiple-choice question."""


@interface
def extract_index(answer_text: str, choices: list) -> int:
    """Given an answer and choices, return the 0-based index of the matching choice."""
