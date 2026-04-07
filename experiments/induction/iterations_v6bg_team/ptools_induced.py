"""Auto-induced ptools from iterative induction pipeline.

These are real @interface stubs. Each is bound to simulate at runtime.
"""

from secretagent.core import interface

@interface
def extract_constraints(narrative: str, focus: str) -> str:
    """Extracts constraints, disqualifiers, and interpersonal dynamics from a narrative for team allocation.
    
    This function parses the provided narrative to identify hard constraints (which disqualify a person from a role),
    soft constraints (preferences or weaknesses), and interpersonal issues (conflicts or synergies between people).
    The focus parameter allows filtering for a specific aspect, such as a target person, role, constraint type, or pair of people.
    
    Args:
        narrative (str): The full narrative text describing people, their attributes, and the roles.
        focus (str): The specific aspect to focus on (e.g., 'chef', 'Sam', 'interpersonal', 'disqualifiers').
    
    Returns:
        list: A list of dictionaries, each representing a constraint or dynamic. Each dictionary has keys:
              - 'type': The constraint type (e.g., 'disqualifier', 'preference', 'interpersonal').
              - 'person': The person to whom the constraint applies (or 'pair' for interpersonal).
              - 'applies_to': The role or person the constraint relates to.
              - 'polarity': 'positive' (e.g., skill, preference) or 'negative' (e.g., weakness, conflict).
              - 'quote': The exact text from the narrative supporting this constraint.
    
        Example:
            [
                {
                    "type": "disqualifier",
                    "person": "Alex",
                    "applies_to": "zookeeper",
                    "polarity": "negative",
                    "quote": "Alex is afraid of animals."
                },
                {
                    "type": "preference",
                    "person": "Mia",
                    "applies_to": "gardener",
                    "polarity": "positive",
                    "quote": "Mia loves plants and has a green thumb."
                },
                {
                    "type": "interpersonal",
                    "person": "pair",
                    "applies_to": "Alex-Mia",
                    "polarity": "negative",
                    "quote": "Alex and Mia had a recent disagreement."
                }
            ]
    
    Note:
        - Hard disqualifiers (e.g., 'afraid of animals' for a zookeeper role) should be flagged with type 'disqualifier'.
        - Pay attention to words like 'cannot', 'refuses', 'afraid of', 'allergic to', which indicate hard constraints.
        - For interpersonal constraints, 'person' is 'pair', and 'applies_to' should list the two people involved.
        - The 'focus' parameter narrows down the extraction; if focus is 'chef', only constraints related to the chef role are returned.
        - If no focus is specified, all constraints are returned.
    """
