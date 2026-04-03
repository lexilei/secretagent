"""Auto-induced ptools from iterative induction pipeline.

These are real @interface stubs. Each is bound to simulate at runtime.
"""

from secretagent.core import interface

@interface
def evaluate_suspect_mmo(narrative: str, focus: str) -> str:
    """Evaluates a specific suspect's motive, means, and opportunity for committing the murder.
    
    Args:
        narrative (str): The full murder mystery text containing all evidence and statements
        focus (str): The specific suspect name to evaluate (e.g., "Peyton", "Nicole")
    
    Returns:
        dict: A structured analysis with three main sections:
        {
            "suspect": "name",
            "motive_analysis": {
                "has_motive": bool,
                "motive_details": "textual description",
                "motive_strength": "strong/moderate/weak/none"
            },
            "means_analysis": {
                "has_means": bool,
                "means_details": "textual description",
                "means_plausibility": "high/moderate/low/unknown"
            },
            "opportunity_analysis": {
                "has_opportunity": bool,
                "opportunity_details": "textual description",
                "time_window_match": bool,
                "location_access": bool
            },
            "overall_assessment": "brief summary of suspect's viability"
        }
    
    Example output:
        {
            "suspect": "Peyton",
            "motive_analysis": {
                "has_motive": true,
                "motive_details": "Guy threatened to expose her secret about financial fraud",
                "motive_strength": "strong"
            },
            "means_analysis": {
                "has_means": false,
                "means_details": "No evidence of crossbow ownership or proficiency",
                "means_plausibility": "low"
            },
            "opportunity_analysis": {
                "has_opportunity": false,
                "opportunity_details": "Works daytime shifts, murder occurred at night",
                "time_window_match": false,
                "location_access": true
            },
            "overall_assessment": "Unlikely suspect due to lack of means and opportunity"
        }
    
    Focus on:
        - Extracting all references to the specified suspect
        - Evaluating motive: reasons, conflicts, or benefits from the victim's death
        - Evaluating means: access to murder weapon, relevant skills or knowledge
        - Evaluating opportunity: presence at crime scene/time, alibi evidence
        - Identifying contradictions or gaps in the evidence
    """
