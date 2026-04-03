"""Auto-induced ptools from iterative induction pipeline.

These are real @interface stubs. Each is bound to simulate at runtime.
"""

from secretagent.core import interface

@interface
def summarize_evidence(narrative: str, focus: str) -> str:
    """Extracts and synthesizes evidence from a murder mystery narrative for specified suspects.
    
    Extracts the following information for each suspect mentioned in the focus parameter:
    - Direct evidence linking them to the crime
    - Potential motives (financial, personal, professional)
    - Opportunities (alibis, access, timing)
    - Relevant expertise or capabilities
    - Contradictions or inconsistencies in their story
    - Relationships to the victim and other suspects
    
    Focus parameter should specify which suspects to analyze (e.g., 'Nicole and Isabelle', 'all suspects', 'Mack and Taylor').
    
    Response structure:
    - Organized by suspect name
    - Bullet points for each category (evidence, motives, opportunities, etc.)
    - Clear distinction between direct evidence and circumstantial evidence
    - Specific quotes or references from the narrative when possible
    - Notes on contradictions or suspicious behavior
    
    Pay special attention to:
    - Timeline inconsistencies
    - Physical evidence mentioned
    - Witness statements
    - Suspicious behaviors or statements
    - Relationships that might indicate motive
    """
