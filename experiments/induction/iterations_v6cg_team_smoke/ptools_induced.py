"""Auto-induced ptools from iterative induction pipeline.

These are real @interface stubs. Each is bound to simulate at runtime.
"""

from secretagent.core import interface

@interface
def extract_role_constraints(narrative: str, focus: str) -> str:
    """Extracts constraints from the narrative that affect role allocation, including skills, weaknesses, preferences, interpersonal dynamics, and disqualifiers.
    
    This function analyzes the narrative to identify all explicit and implicit constraints that determine
    which roles each person is suited for or disqualified from. It captures both hard constraints
    (disqualifiers) that make a person-role pairing impossible, and soft constraints (preferences)
    that indicate better or worse suitability for specific roles.
    
    Args:
        narrative (str): The full narrative text describing people, their attributes, and roles
        focus (str): What specific aspect to focus on (e.g., specific person, role, constraint type)
    
    Returns:
        dict: A dictionary containing constraint lists organized by constraint type
        
        Example output format:
        {
            "hard_constraints": [
                {
                    "type": "disqualifier",
                    "person": "Jessica",
                    "role": "chef",
                    "reason": "afraid of knives",
                    "polarity": "negative",
                    "quote": "Jessica is afraid of knives"
                }
            ],
            "soft_constraints": [
                {
                    "type": "skill",
                    "person": "Samuel",
                    "role": "server",
                    "reason": "outgoing personality",
                    "polarity": "positive",
                    "quote": "Samuel has an outgoing personality"
                },
                {
                    "type": "preference",
                    "person": "Rebecca",
                    "role": "chef",
                    "reason": "prefers cooking",
                    "polarity": "positive",
                    "quote": "Rebecca prefers cooking over serving"
                }
            ],
            "interpersonal_constraints": [
                {
                    "type": "conflict",
                    "persons": ["Mark", "Angela"],
                    "roles_affected": ["all"],
                    "reason": "history of conflict",
                    "polarity": "negative",
                    "quote": "Mark and Angela have a history of conflict"
                }
            ]
        }
    
    Note: Focus parameter can be used to filter constraints (e.g., focus="Jessica" returns only constraints
    related to Jessica, focus="chef" returns constraints related to the chef role, focus="skill" returns
    only skill-based constraints).
    """
