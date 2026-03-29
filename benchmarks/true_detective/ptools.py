"""Interfaces for True Detective abductive mystery reasoning.

Mirrors the MUSR murder mystery design: ptools ANNOTATE and ENHANCE
reasoning on the raw narrative rather than abstracting it away.
The same evidence-centric 3-stage pipeline (extract -> verify -> deduce)
is used to test cross-benchmark generalization.
"""

from secretagent.core import interface


@interface
def extract_clues_and_suspects(narrative: str) -> str:
    """Extract ALL suspects, clues, and evidence from the detective mystery.

    Read the mystery carefully and extract:

    Top level:
    - crime: what happened (theft, murder, fraud, etc.)
    - setting: where and when the events take place
    - key_clues: physical evidence, witness statements, timeline details
    - suspects: for each person who could be guilty

    For each suspect:
    - motive: why they might have done it
    - opportunity: were they present, do they have access
    - alibi: what they claim they were doing
    - suspicious_details: anything that stands out about their behavior or statements
    - contradictions: anything in their story that doesn't add up

    Be thorough - include EVERY suspect and EVERY clue mentioned in the narrative.
    """


@interface
def verify_and_cross_reference(narrative: str, extracted_clues: str) -> str:
    """Re-read the narrative to verify clues and find contradictions.

    You receive the original mystery AND the extracted clues/suspects.
    Cross-reference everything against the narrative to find:

    For each suspect:
    - alibi_verified: does their alibi actually hold up against the facts?
    - timeline_gaps: any unexplained periods
    - contradictions: inconsistencies between their statements and narrative facts
    - physical_evidence_links: evidence that connects or disconnects them from the crime

    Pay special attention to:
    - Small details that contradict alibis
    - Physical impossibilities in their stories
    - Witnesses who say something different from what the suspect claims
    - Timing that doesn't add up
    - Objects or conditions that only the guilty person would know about
    """


@interface
def deduce_guilty(narrative: str, verified_analysis: str, choices: list) -> str:
    """Given the mystery, verified analysis, and answer choices, identify who is guilty.

    You have access to:
    1. The FULL original mystery narrative
    2. Verified analysis with cross-referenced evidence for each suspect
    3. The multiple-choice options

    Synthesize all evidence to determine guilt. Consider:
    - Who has the WEAKEST alibi combined with STRONGEST evidence against them
    - Physical evidence and impossibilities are strongest indicators
    - Contradictions with known facts weight heavily
    - Small, specific details often contain the key to the solution
    - The guilty party often reveals knowledge they shouldn't have
    """


@interface
def answer_question(narrative: str, question: str, choices: list) -> int:
    """Read the detective mystery and answer the question.
    Return the 0-based index of the correct choice.
    """
    text = raw_answer(narrative, question, choices)
    return extract_index(text, choices)


@interface
def answer_question_workflow(narrative: str, question: str, choices: list) -> int:
    """Solve by extracting clues, verifying, deducing, then matching."""
    clues = extract_clues_and_suspects(narrative)
    verified = verify_and_cross_reference(narrative, clues)
    text = deduce_guilty(narrative, verified, choices)
    return extract_index(text, choices)


@interface
def raw_answer(narrative: str, question: str, choices: list) -> str:
    """Read the narrative and answer the multiple-choice question."""


@interface
def extract_index(answer_text: str, choices: list) -> int:
    """Given an answer and choices, return the 0-based index of the matching choice."""
