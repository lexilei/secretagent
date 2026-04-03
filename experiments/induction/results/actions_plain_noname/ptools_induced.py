"""Auto-induced ptools from iterative induction pipeline."""

from secretagent.core import interface

@interface
def summarize_evidence(narrative: str, focus: str) -> str:
    """Extracts and organizes evidence from a murder mystery narrative for specified suspects.
    
    Parameters:
        narrative (str): The full text of the murder mystery containing clues and evidence.
        focus (str): A string specifying which suspects to focus on (e.g., 'Harry and Rosemary', 'all suspects').
    
    Returns:
        str: A structured summary organized by suspect. For each suspect mentioned in the focus parameter:
            - Lists all stated motives (reasons to commit the crime)
            - Details alibi information (time, location, witnesses)
            - Describes physical evidence linking them to the crime (weapons, fingerprints, documents, etc.)
            - Notes any suspicious circumstances or behaviors
    
    Instructions:
        1. Identify all suspects mentioned in the focus parameter
        2. Scan the narrative for any evidence related to these specific suspects
        3. Categorize findings into motives, alibis, and physical evidence
        4. Present findings in clear sections for each suspect
        5. Only include information explicitly stated in the narrative
        6. If no evidence exists for a category, state 'None identified'
        7. Maintain neutral language without drawing conclusions
    """
