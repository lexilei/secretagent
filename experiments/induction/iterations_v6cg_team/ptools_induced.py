"""Auto-induced ptools from iterative induction pipeline.

These are real @interface stubs. Each is bound to simulate at runtime.
"""

from secretagent.core import interface

@interface
def extract_role_constraints(narrative: str, focus: str) -> str:
    """Extracts constraints from the narrative that affect role assignment for team allocation problems.
    
    Parses the narrative to identify:
    - Required skills and qualifications for each role
    - Person-specific skills, experiences, and strengths
    - Weaknesses, limitations, or disqualifiers for specific roles
    - Personal preferences or stated role interests
    - Interpersonal dynamics that affect team composition
    
    Args:
        narrative (str): The full narrative text describing people, roles, and constraints
        focus (str): Specific aspect to focus on (e.g., role name, person name, constraint type)
    
    Returns:
        dict: A structured constraint dictionary with the following format:
        {
            "constraints": [
                {
                    "type": "skill_match" | "disqualifier" | "preference" | "interpersonal",
                    "person": "Name",
                    "role": "Role Name",
                    "polarity": "positive" | "negative" | "neutral",
                    "constraint": "Description of constraint",
                    "evidence": "Direct quote from narrative",
                    "strength": "hard" | "soft"
                }
            ],
            "role_requirements": {
                "Role Name": {
                    "required_skills": ["skill1", "skill2"],
                    "description": "Role description",
                    "evidence": "Supporting quote"
                }
            }
        }
    
    Example output:
        {
            "constraints": [
                {
                    "type": "skill_match",
                    "person": "Sam",
                    "role": "chef",
                    "polarity": "positive",
                    "constraint": "Professional cooking experience",
                    "evidence": "Sam worked as a chef for 5 years",
                    "strength": "hard"
                },
                {
                    "type": "disqualifier",
                    "person": "Alex",
                    "role": "server",
                    "polarity": "negative",
                    "constraint": "Poor customer service skills",
                    "evidence": "Alex is introverted and dislikes interacting with customers",
                    "strength": "hard"
                }
            ],
            "role_requirements": {
                "chef": {
                    "required_skills": ["cooking", "food preparation", "menu planning"],
                    "description": "Prepares meals and manages kitchen",
                    "evidence": "The chef role involves cooking meals for customers"
                }
            }
        }
    """

@interface
def task_interpretation(narrative: str, focus: str) -> str:
    """Task Interpretation.
    
    Analyze the narrative with the given focus.
    """
