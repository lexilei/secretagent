"""Auto-induced ptools from iterative induction pipeline."""

from secretagent.core import interface

@interface
def summarize_evidence(narrative: str, focus: str) -> str:
    """Analyzes a murder mystery narrative to extract and summarize evidence related to specific suspects.
    
    Args:
        narrative (str): The full text of the murder mystery story containing clues, statements, and evidence.
        focus (str): A string specifying which suspects to focus on (e.g., "Ana and Mackenzie", "against Mackenzie", "for Harry and Rosemary").
    
    Returns:
        dict: A structured summary containing:
            {
                "suspects": [list of suspect names mentioned in focus],
                "summary_type": "for/against/both" (based on focus),
                "evidence_summary": {
                    "suspect_name": {
                        "motive": [list of potential motives with supporting details],
                        "opportunity": [list of opportunity evidence with timings/alibis],
                        "physical_evidence": [list of physical clues linking suspect to crime],
                        "contradictions": [list of inconsistencies in suspect's statements/behavior],
                        "witness_statements": [list of relevant testimonies about suspect]
                    }
                },
                "confidence_level": "high/medium/low" (assessment of evidence strength)
            }
    
    Example output:
        {
            "suspects": ["Mackenzie", "Ana"],
            "summary_type": "against",
            "evidence_summary": {
                "Mackenzie": {
                    "motive": ["Financial gain from inheritance", "Personal grudge against victim"],
                    "opportunity": ["Was seen near crime scene at 9 PM", "Alibi provided by friend lacks corroboration"],
                    "physical_evidence": ["Fingerprints found on weapon", "Matching fibers from clothing"],
                    "contradictions": ["Changed story about whereabouts", "Denied knowing victim but phone records show calls"],
                    "witness_statements": ["Colleague reported suspicious behavior", "Neighbor heard argument"]
                },
                "Ana": {
                    "motive": [],
                    "opportunity": ["Has verifiable alibi with multiple witnesses"],
                    "physical_evidence": [],
                    "contradictions": [],
                    "witness_statements": ["Was seen miles away at time of crime"]
                }
            },
            "confidence_level": "high"
        }
    
    Note:
        - Extract only evidence explicitly stated in the narrative
        - Focus on specific categories: motive, opportunity, physical evidence, contradictions, and witness statements
        - If no evidence exists in a category, return empty list
        - Determine summary_type from focus phrase (for/against/both)
        - Assess overall confidence based on quantity and quality of evidence
    """
