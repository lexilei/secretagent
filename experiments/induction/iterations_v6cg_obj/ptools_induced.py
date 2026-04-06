"""Auto-induced ptools from iterative induction pipeline.

These are real @interface stubs. Each is bound to simulate at runtime.
"""

from secretagent.core import interface

@interface
def extract_movement_timeline(narrative: str, focus: str) -> str:
    """Parses the narrative to extract all object movements in chronological order, including details about each movement and who was present/absent.
    
    Args:
        narrative (str): The full narrative text describing characters, object movements, and presence/absence
        focus (str): Specific aspect to focus on (e.g., target object like 'ancient coin' or 'helmets')
    
    Returns:
        list: A chronological list of dictionaries representing each movement event
        
        Format:
        [
            {
                "step": 1,
                "object": "object_name",
                "from": "previous_location",
                "to": "new_location",
                "actor": "character_who_performed_movement",
                "present": ["character1", "character2", ...],
                "absent": ["character3", "character4", ...]
            },
            ...
        ]
        
        Example:
        [
            {
                "step": 1,
                "object": "helmet",
                "from": "storage_closet",
                "to": "bench",
                "actor": "Lisa",
                "present": ["Lisa", "Ellie"],
                "absent": []
            },
            {
                "step": 2,
                "object": "helmet",
                "from": "bench",
                "to": "display_case",
                "actor": "Ellie",
                "present": ["Ellie"],
                "absent": ["Lisa"]
            }
        ]
        
        Notes:
        - Focus on movements of the specified object (if provided) or all relevant object movements
        - Extract presence/absence information from narrative descriptions
        - Maintain strict chronological order of events
        - Record locations exactly as described in the narrative
    """

@interface
def construct_movement_timeline(narrative: str, focus: str) -> str:
    """Constructs a detailed chronological timeline of all movements related to a specified focus object from the narrative.
    
    Extracts each movement event including:
    - Step number (chronological order)
    - Object being moved
    - From location
    - To location
    - Character performing the movement
    - Characters present/witnessing the movement
    - Characters absent/not witnessing the movement
    
    This timeline helps track what each character witnessed regarding the object's movements, which is crucial for determining their beliefs about the object's location (theory of mind).
    
    Args:
        narrative (str): The full narrative text describing characters, object movements, and presence/absence.
        focus (str): The specific object to track movements for (e.g., 'ancient coin', 'laptop').
    
    Returns:
        str: A JSON-formatted string representing a chronological list of movement events with witness information.
        
        Example output format:
        [
            {
                "step": 1,
                "object": "ancient coin",
                "from": "pantry",
                "to": "counter",
                "actor": "Ben",
                "present": ["Anna", "Ben"],
                "absent": ["Cora"]
            },
            {
                "step": 2,
                "object": "ancient coin",
                "from": "counter",
                "to": "drawer",
                "actor": "Anna",
                "present": ["Anna"],
                "absent": ["Ben", "Cora"]
            }
        ]
    """
