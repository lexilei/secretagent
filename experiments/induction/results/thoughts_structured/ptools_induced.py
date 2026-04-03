"""Auto-induced ptools from iterative induction pipeline."""

from secretagent.core import interface

@interface
def summarize_suspects(narrative: str, focus: str) -> str:
    """Analyzes a murder mystery narrative to extract and summarize key information about each suspect.
    
    This function identifies all suspects mentioned in the narrative and creates a structured summary
    for each one, focusing on motives, alibis, opportunities to commit the crime, and any contradictions
    in their statements or evidence.
    
    Args:
        narrative (str): The full murder mystery story containing clues, statements, and evidence
        focus (str): Additional context about what to emphasize in the summary (e.g., 'motive', 'opportunity')
    
    Returns:
        dict: A structured summary organized by suspect with the following format:
        {
            "suspects": {
                "suspect_name": {
                    "motives": ["list of potential motives", "..."],
                    "alibi": "description of alibi or lack thereof",
                    "opportunity": "description of opportunity to commit crime",
                    "contradictions": ["list of inconsistencies in statements/evidence", "..."],
                    "key_evidence": ["list of key evidence pointing to this suspect", "..."]
                },
                "another_suspect": {
                    ...
                }
            },
            "focus_analysis": "Additional analysis based on the focus parameter"
        }
    
    Example output:
        {
            "suspects": {
                "Mackenzie": {
                    "motives": ["Professional rivalry with victim", "Jealousy"],
                    "alibi": "Claims to be at bungee jumping site but no witnesses",
                    "opportunity": "Was present at crime scene location",
                    "contradictions": ["Time of departure doesn't match witness statements"],
                    "key_evidence": ["Fingerprints on weapon", "Threatening email"]
                },
                "Ana": {
                    "motives": ["Financial gain from inheritance"],
                    "alibi": "Confirmed at business meeting by multiple colleagues",
                    "opportunity": "No clear opportunity based on timeline",
                    "contradictions": [],
                    "key_evidence": ["Beneficiary in victim's will"]
                }
            },
            "focus_analysis": "Based on the focus on motive, Mackenzie shows stronger motivation"
        }
    """
