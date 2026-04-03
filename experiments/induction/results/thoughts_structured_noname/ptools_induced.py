"""Auto-induced ptools from iterative induction pipeline."""

from secretagent.core import interface

@interface
def summarize_evidence(narrative: str, focus: str) -> str:
    """Analyzes the provided murder mystery narrative to extract and summarize evidence related to the specified suspects.
    
    This function focuses on identifying key details about each suspect's alibi, potential motives,
    evidence linking them to the crime, and any contradictions in their statements or behavior.
    
    Args:
        narrative (str): The full text of the murder mystery story containing clues and character information.
        focus (str): A string indicating which suspects to analyze. Typically contains suspect names
                    (e.g., "Ana and Mackenzie", "Harry and Rosemary", "suspects").
    
    Returns:
        dict: A structured summary organized by suspect with the following format:
        {
            "suspect_name": {
                "has_alibi": bool,          # Whether the suspect has a verifiable alibi
                "alibi_details": str,        # Description of alibi if present
                "motive": str,               # Potential motive for the crime
                "incriminating_evidence": [  # List of evidence pointing to guilt
                    "evidence_item_1",
                    "evidence_item_2"
                ],
                "contradictions": [         # List of inconsistencies in statements/behavior
                    "contradiction_1",
                    "contradiction_2"
                ],
                "suspicious_behavior": str  # Notable behavior during investigation
            }
        }
    
    Example:
        {
            "Mackenzie": {
                "has_alibi": false,
                "alibi_details": "No verifiable alibi provided",
                "motive": "Recent argument with victim about business practices",
                "incriminating_evidence": [
                    "Purchased nunchaku (murder weapon) a week prior",
                    "Seen near crime scene around time of murder"
                ],
                "contradictions": [
                    "Claimed to be at home but no one can confirm",
                    "Changed story about nunchaku purchase"
                ],
                "suspicious_behavior": "Appeared nervous during questioning"
            },
            "Ana": {
                "has_alibi": true,
                "alibi_details": "Was bungee jumping with instructor at time of murder",
                "motive": "None explicitly stated",
                "incriminating_evidence": [
                    "Had access to victim's schedule",
                    "Knew about victim's fear of heights"
                ],
                "contradictions": [],
                "suspicious_behavior": "Displayed nervous demeanor during investigation"
            }
        }
    """
