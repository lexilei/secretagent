"""Auto-induced ptools from iterative induction pipeline.

These are real @interface stubs. Each is bound to simulate at runtime.
"""

from secretagent.core import interface

@interface
def analyze_mom_evidence(narrative: str, focus: str) -> str:
    """Evaluates the means, opportunity, and motive (MOM) for each suspect mentioned in the narrative.
    
    Args:
        narrative: The full murder mystery text containing evidence and character information
        focus: Specific aspect to emphasize - either 'means', 'opportunity', 'motive', or 'all'
    
    Returns:
        dict: A structured analysis with suspect names as keys, each containing:
            {
                "means": {
                    "has_means": bool,  # Whether suspect had access to murder weapon/method
                    "evidence": str,     # Specific evidence supporting means assessment
                    "certainty": str     # Level of certainty (definite, likely, possible, unlikely)
                },
                "opportunity": {
                    "has_opportunity": bool,  # Whether suspect was present/able to commit crime
                    "alibi": str,              # Suspect's alibi if available
                    "evidence": str,           # Specific evidence supporting opportunity assessment
                    "certainty": str           # Level of certainty
                },
                "motive": {
                    "has_motive": bool,    # Whether suspect had reason to commit murder
                    "motive_type": str,    # Type of motive (revenge, financial, protection, etc.)
                    "evidence": str,       # Specific evidence supporting motive assessment
                    "certainty": str       # Level of certainty
                }
            }
    
    Example output:
        {
            "suspect_a": {
                "means": {
                    "has_means": true,
                    "evidence": "Had access to crossbow through hunting club membership",
                    "certainty": "definite"
                },
                "opportunity": {
                    "has_opportunity": false,
                    "alibi": "Was at staff meeting during murder timeframe",
                    "evidence": "Security logs show suspect in different building",
                    "certainty": "definite"
                },
                "motive": {
                    "has_motive": true,
                    "motive_type": "financial",
                    "evidence": "Stood to inherit victim's business",
                    "certainty": "likely"
                }
            }
        }
    """

@interface
def evaluate_alibi_consistency(narrative: str, focus: str) -> str:
    """Evaluates the consistency and reliability of alibis by examining witness statements, timeline events, and potential contradictions.
    
    Args:
        narrative (str): The full murder mystery text containing character statements and timeline details
        focus (str): Specific suspect or timeframe to focus on (e.g., 'suspect_name', 'time_window')
    
    Returns:
        dict: A structured analysis of alibi consistency with the following keys:
            - alibi_strength: str ("strong", "weak", "contradictory", "unverified")
            - supporting_witnesses: list[str] (names of witnesses supporting the alibi)
            - contradicting_witnesses: list[str] (names of witnesses contradicting the alibi)
            - timeline_conflicts: list[str] (specific time conflicts or gaps)
            - confidence_score: int (0-100, indicating reliability)
            - summary: str (brief assessment of overall alibi credibility)
    
    Example:
        {
            "alibi_strength": "contradictory",
            "supporting_witnesses": ["Sarah", "Mike"],
            "contradicting_witnesses": ["John"],
            "timeline_conflicts": ["15-minute gap between witness statements"],
            "confidence_score": 35,
            "summary": "Alibi has multiple witnesses but contains significant contradictions"
        }
    """
