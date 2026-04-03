"""Auto-induced ptools from iterative induction pipeline.

These are real @interface stubs. Each is bound to simulate at runtime.
"""

from secretagent.core import interface

@interface
def evaluate_suspect_mmo(narrative: str, focus: str) -> str:
    """Analyzes the provided murder mystery narrative to evaluate a specific suspect's Motive, Means, and Opportunity (MMO).
    
    Parameters:
        narrative (str): The complete murder mystery text containing all evidence, statements, and clues
        focus (str): The specific suspect to evaluate (e.g., 'Peyton', 'Nicole', 'Isabelle')
    
    Returns:
        str: A structured analysis with three sections:
        - MOTIVE: Evidence suggesting why the suspect might want to kill the victim
        - MEANS: Evidence indicating the suspect's ability/access to commit the murder
        - OPPORTUNITY: Evidence about the suspect's presence/accessibility at time/location
    
    Focus on:
        - Extracting direct evidence from the narrative
        - Identifying both supporting and contradicting evidence
        - Noting specific timings, locations, skills, and relationships mentioned
        - Being objective - report what the evidence shows, not conclusions
        - Using direct quotes from the narrative when possible
        - Separating confirmed facts from speculation
        - Highlighting any gaps or inconsistencies in the evidence
    """
