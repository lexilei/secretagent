"""Interfaces for MUSR murder mystery reasoning."""

from secretagent.core import interface
from ptools_common import raw_answer, extract_index


@interface
def extract_evidence(narrative: str) -> str:
    """Extract all suspects and evidence from this murder mystery.

    For each suspect, describe their motive, means, opportunity,
    alibi (and whether it holds up), suspicious behavior, and
    any physical evidence linking them to the crime.
    """


@interface
def deduce_answer(narrative: str, evidence: str, question: str, choices: list) -> str:
    """Given a murder mystery, extracted evidence, and answer choices,
    deduce who committed the murder.

    Weigh physical evidence and alibi contradictions most heavily.
    Consider motive as supporting but not sufficient alone.
    """


@interface
def answer_question(narrative: str, question: str, choices: list) -> int:
    """Read the murder mystery narrative and answer the question.
    Return the 0-based index of the correct choice.
    """
    text = raw_answer(narrative, question, choices)
    return extract_index(text, choices)


@interface
def answer_question_workflow(narrative: str, question: str, choices: list) -> int:
    """Solve by extracting evidence, deducing, then matching the answer."""
    evidence = extract_evidence(narrative)
    text = deduce_answer(narrative, evidence, question, choices)
    return extract_index(text, choices)
