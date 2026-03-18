"""Interfaces for MUSR team allocation."""

from secretagent.core import interface
from ptools_common import raw_answer, extract_index


@interface
def extract_profiles(narrative: str) -> str:
    """Extract person profiles and role requirements from the narrative.

    For each person, assess their fit for each role on a 1-5 scale:
    1 = severely unfit (phobia, allergy, physical danger)
    2 = poor fit (discomfort but could manage)
    3 = neutral
    4 = good fit (relevant skills/experience)
    5 = excellent fit

    Also note interpersonal constraints (conflicts, synergies).
    """


@interface
def evaluate_allocations(narrative: str, profiles: str, question: str, choices: list) -> str:
    """Given person profiles and allocation choices, pick the best assignment.

    For each choice, sum the fit scores. Prefer choices with fewer
    score-1 (severely unfit) assignments. When all choices have problems,
    pick the least bad one.
    """


@interface
def answer_question(narrative: str, question: str, choices: list) -> int:
    """Read the narrative and determine the best team allocation.
    Return the 0-based index of the correct choice.
    """
    text = raw_answer(narrative, question, choices)
    return extract_index(text, choices)


@interface
def answer_question_workflow(narrative: str, question: str, choices: list) -> int:
    """Solve by extracting profiles, evaluating allocations, then matching."""
    profiles = extract_profiles(narrative)
    text = evaluate_allocations(narrative, profiles, question, choices)
    return extract_index(text, choices)
