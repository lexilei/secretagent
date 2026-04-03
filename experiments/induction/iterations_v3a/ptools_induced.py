"""Auto-induced ptools from iterative induction pipeline.

These are real @interface stubs. Each is bound to simulate at runtime.
"""

from secretagent.core import interface

@interface
def evaluate_evidence(narrative: str, focus: str) -> str:
    """"""
    Analyzes evidence from the murder mystery narrative related to a specific focus area (suspect, alibi, weapon, etc.)
    for consistency, contradictions, and connections to other elements of the case.
    
    Args:
        narrative (str): The full murder mystery text containing all evidence, statements, and clues.
        focus (str): The specific aspect to evaluate - typically a suspect's name, an alibi, a piece of evidence,
                     or a specific claim that needs verification.
    
    Returns:
        str: A structured analysis including:
             - All relevant evidence found related to the focus
             - Assessment of consistency/inconsistency within the evidence
             - Any contradictions with other known facts
             - Connections to motives, opportunities, or means
             - Notable gaps or missing information
             - Overall reliability assessment
    
    Pay special attention to:
    - Time-related evidence (alibis, timelines)
    - Statements that conflict with physical evidence
    - Motive, opportunity, and means triangulation
    - Changes in stories or conflicting witness accounts
    - Any forensic or physical evidence connections
    - Whether evidence directly implicates or clears the focus
    """
    """

@interface
def list_evidence_by_person(narrative: str, focus: str) -> str:
    """Extract and organize all evidence related to specific individuals from a murder mystery narrative.
    
    This function parses the narrative to identify and categorize all relevant information about
    specified persons of interest. It extracts:
    - Direct evidence linking individuals to the crime
    - Motives or reasons why someone might commit the murder
    - Alibis or accounts of their whereabouts during the crime
    - Suspicious behaviors or actions
    - Physical evidence associated with each person
    - Relationships with the victim and other characters
    - Any contradictory statements or inconsistencies
    
    Focus parameter format: A string containing one or more names separated by commas or 'and'
    (e.g., 'Nicole and Isabelle', 'Elizabeth, Freya', 'Justin, Frederick').
    
    Response structure:
    - For each person mentioned in the focus parameter, create a separate section
    - Within each section, organize evidence by category (evidence, motives, alibis, etc.)
    - List each piece of evidence clearly and concisely
    - Include direct quotes from the narrative where appropriate
    - Note any conflicts or inconsistencies in the evidence
    - If no evidence exists for a category, state 'No [category] found'
    - End with a summary of the most significant findings for each person
    
    Pay special attention to:
    - Character names (watch for variations or nicknames)
    - Time references and alibi verification
    - Evidence that contradicts other statements
    - Implied motives versus explicitly stated motives
    - Physical evidence descriptions and who had access
    """

@interface
def consistency_and_contradiction_checking(narrative: str, focus: str) -> str:
    """Consistency and Contradiction Checking.
    
    Analyze the narrative with the given focus.
    """
