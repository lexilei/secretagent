"""Auto-induced ptools from iterative induction pipeline."""

from secretagent.core import interface

@interface
def summarize_evidence_and_profiles(narrative: str, focus: str) -> str:
    """Extracts and summarizes all relevant information from a murder mystery narrative for one or more specified individuals.
    
    This function analyzes the provided narrative text to identify:
    - Alibis: Where each person claims to be during the crime
    - Motives: Reasons each person might have committed the murder
    - Physical evidence: Any objects, forensic evidence, or physical traces connecting them to the crime
    - Witness accounts: Statements from others about the person's actions or whereabouts
    - Opportunities: Whether the person had the means and chance to commit the crime
    - Connections to weapons: Any relationship to the murder weapon
    - Suspicious behaviors: Unusual actions or statements
    - Relationships: Connections to the victim or other suspects
    
    Focus on the specific individuals mentioned in the 'focus' parameter, which can be a single name or multiple names separated by commas/and. The response should be structured as a clear, concise summary organized by person and then by evidence category. Pay special attention to both explicit statements and implied connections within the narrative. Present facts objectively without drawing conclusions.
    """
