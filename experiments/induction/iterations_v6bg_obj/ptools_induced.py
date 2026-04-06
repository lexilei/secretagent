"""Auto-induced ptools from iterative induction pipeline.

These are real @interface stubs. Each is bound to simulate at runtime.
"""

from secretagent.core import interface

@interface
def trace_object_movements(narrative: str, focus: str) -> str:
    """Extracts all movements of a specific object from the narrative and identifies who was present/absent for each movement.
    
    Parameters:
        narrative (str): The full narrative text describing characters, object movements, and presence/absence
        focus (str): The specific object to trace movements for (e.g., 'helmet', 'coin', 'laptop')
    
    Returns:
        list: A chronological list of dictionaries representing each movement event, with keys:
            - step: Integer sequence number (1-indexed)
            - actor: Who performed the movement
            - object: The object moved (should match focus parameter)
            - from_location: Starting location
            - to_location: Destination location
            - present: List of characters who witnessed this movement
            - absent: List of characters who did not witness this movement
            - description: Text description of the movement event
    
    Example output:
        [
            {
                "step": 1,
                "actor": "Ben",
                "object": "helmet",
                "from_location": "pantry",
                "to_location": "counter",
                "present": ["Anna", "Ben"],
                "absent": ["Cora"],
                "description": "Ben moved the helmet from pantry to counter"
            },
            {
                "step": 2,
                "actor": "Anna",
                "object": "helmet",
                "from_location": "counter",
                "to_location": "cabinet",
                "present": ["Anna", "Cora"],
                "absent": ["Ben"],
                "description": "Anna moved the helmet from counter to cabinet"
            }
        ]
    
    Note: Focuses exclusively on movement events (not placements, discoveries, or other actions).
    Identifies witnesses based on explicit presence/absence information in the narrative.
    """

@interface
def verify_presence_during_action(narrative: str, focus: str) -> str:
    """Parses a narrative to extract object movement events and identifies which characters were present or absent during each movement. Focuses on a specific aspect (character, object, or movement) as specified. Returns a chronological list of relevant movements with detailed witness information.
    
    Args:
        narrative (str): The full narrative text describing characters, object movements, and presence/absence.
        focus (str): The specific aspect to focus on (e.g., 'Lisa', 'helmets', 'Ricky moves notebook').
    
    Returns:
        list: A chronological list of dictionaries for each relevant movement event. Each dictionary contains:
            - 'step': Event number (int)
            - 'action': Description of the action (str)
            - 'actor': Character performing the action (str)
            - 'present': List of characters explicitly stated as present (list)
            - 'absent': List of characters explicitly stated as absent (list)
            - 'focus_involved': How the focus relates to this event (str)
            - 'focus_present': Whether the focus character was present (bool or None for object/action focus)
    
    Example:
        [
            {
                'step': 1,
                'action': 'moved the helmets to the storage closet',
                'actor': 'Ellie',
                'present': ['Lisa', 'Ellie'],
                'absent': ['Ricky'],
                'focus_involved': 'Lisa is focus character',
                'focus_present': True
            }
        ]
    """

@interface
def verify_witness_presence(narrative: str, focus: str) -> str:
    """Analyzes a narrative to determine which characters witnessed specific object movements and what they observed.
    
    Extracts all object movement events from the narrative, identifies who was present/absent for each movement,
    and provides detailed witness information including what each character saw and when.
    
    Args:
        narrative (str): The full narrative text describing characters, objects, movements, and presence/absence.
        focus (str): Specific aspect to focus on (e.g., 'Fred witnessing bow movements', 'vase movement witnesses').
    
    Returns:
        dict: A structured dictionary containing:
            - 'movements': List of movement events with details
            - 'witness_analysis': Per-character breakdown of what they witnessed
            - 'focus_verification': Specific analysis related to the focus parameter
    
        Example output format:
        {
            "movements": [
                {
                    "object": "vase",
                    "from_location": "counter",
                    "to_location": "display_window",
                    "actor": "Clara",
                    "time": "afternoon",
                    "present": ["Clara", "Ben"],
                    "absent": ["Anna"],
                    "description": "Clara moved the vase from counter to display window"
                }
            ],
            "witness_analysis": {
                "Clara": {
                    "witnessed_movements": ["vase moved from counter to display window"],
                    "knowledge_level": "full"
                },
                "Ben": {
                    "witnessed_movements": ["vase moved from counter to display window"],
                    "knowledge_level": "partial"
                },
                "Anna": {
                    "witnessed_movements": [],
                    "knowledge_level": "none"
                }
            },
            "focus_verification": {
                "focus": "Clara witnessing vase movement",
                "result": "confirmed",
                "evidence": "Narrative states Clara moved the vase and was present during the movement"
            }
        }
    """

@interface
def trace_object_history(narrative: str, focus: str) -> str:
    """Trace the complete movement history of a specific object throughout the narrative, including who was present or absent for each movement event.
    
    Extracts:
    - Initial location and awareness state
    - All subsequent movements with timestamps/sequence
    - Actor performing each movement
    - Characters present/witnessing each movement
    - Characters absent/missed each movement
    - Any discoveries or revelations about the object
    
    Returns:
        str: A formatted chronological timeline showing object movements and witness information
    
    Format:
    Initial location: [location] (known to: [witnesses])
    [Step/Time] [Actor] moved [object] from [from_location] to [to_location]
      Witnessed by: [present_characters]
      Not present: [absent_characters]
    [Additional movements...]
    [Discovery events...]
    
    Example output:
    Initial location: pantry (known to: Anna, Ben, Cora)
    [Step 1] Ben moved notebook from pantry to counter
      Witnessed by: Anna, Ben
      Not present: Cora
    [Step 2] Anna moved notebook from counter to drawer
      Witnessed by: Anna, Cora
      Not present: Ben
    [Discovery] Cora found notebook in drawer (Step 3)
    """

@interface
def verify_witness_presence(narrative: str, focus: str) -> str:
    """Verifies whether a target character was present during a specific object movement event and returns detailed witness information.
    
    This function analyzes the narrative to determine:
    - The specific movement event being referenced
    - Which characters were present/absent during that movement
    - Whether the target character witnessed the movement
    
    Parameters:
        narrative (str): The full text containing character movements, presence/absence information
        focus (str): The specific focus query (e.g., "Rita during taco shells movement", "Peter during bow movement")
    
    Returns:
        dict: A structured response containing:
            - target_character: The character being checked
            - movement_description: Description of the movement event
            - was_present: Boolean indicating if target witnessed the movement
            - all_present: List of all characters present during the movement
            - all_absent: List of all characters absent during the movement
            - evidence: The specific text from the narrative that supports the determination
    
    Example output:
        {
            "target_character": "Rita",
            "movement_description": "George moved taco shells to front counter",
            "was_present": true,
            "all_present": ["George", "Rita", "Anna"],
            "all_absent": ["Ben", "Cora"],
            "evidence": "When George moved the taco shells to the front counter, Rita, Anna and George were present"
        }
    """

@interface
def extract_movements_and_presence(narrative: str, focus: str) -> str:
    """Extracts all movements of a specified object from the narrative and records who was present and absent for each movement event.
    
    Args:
        narrative (str): The full narrative text describing characters, object movements, and presence/absence.
        focus (str): The specific object to focus on (e.g., 'rare press album', 'pen', 'chisel box').
    
    Returns:
        list: A chronological list of dictionaries, each representing a movement event for the specified object. Each dictionary has the following structure:
            {
                "step": int,           # The chronological step number (starting at 1)
                "from": str,           # The location the object was moved from
                "to": str,             # The location the object was moved to
                "actor": str,          # The character who performed the movement
                "present": list[str],  # List of characters present during the movement
                "absent": list[str]    # List of characters absent during the movement
            }
    
        Example:
            [
                {
                    "step": 1,
                    "from": "pantry",
                    "to": "counter",
                    "actor": "Ben",
                    "present": ["Anna", "Ben"],
                    "absent": ["Cora"]
                },
                {
                    "step": 2,
                    "from": "counter",
                    "to": "drawer",
                    "actor": "Anna",
                    "present": ["Anna", "Cora"],
                    "absent": ["Ben"]
                }
            ]
    
    Notes:
        - Only extracts movements related to the specified focus object
        - Maintains chronological order of events as they appear in the narrative
        - Records all characters mentioned in the narrative (even if not involved in specific movement)
        - Focuses on factual event information (who was present/absent) rather than character beliefs
    """

@interface
def trace_object_movements(narrative: str, focus: str) -> str:
    """Traces the complete movement history of a specific object throughout the narrative, capturing each transfer event chronologically.
    
    Extracts:
    - Initial location of the object (from explicit statements)
    - All movement events involving the object
    - Actor performing each movement
    - Source and destination locations
    - Characters present and absent during each movement
    
    Parameters:
        narrative (str): The full narrative text describing characters, object movements, and presence/absence
        focus (str): The specific object to trace movements for (e.g., "helmets", "necklace")
    
    Returns:
        str: A formatted string showing chronological movement history with witnesses
        
        Format:
        Initial location: [location] ([evidence statement])
        Movement [number]: [actor] moves [object] from [source] to [destination] ([evidence text])
          Present: [comma-separated list of present characters]
          Absent: [comma-separated list of absent characters]
        
        Example output:
        Initial location: storage closet ("The helmets were in the storage closet")
        Movement 1: Lisa moves helmets from storage closet to bench ("Lisa moves the helmets")
          Present: Anna, Ben
          Absent: Cora
    """

@interface
def finalize_solution(narrative: str, focus: str) -> str:
    """Determines the final location of a target object by analyzing the complete chronological history of movements in the narrative. This tool focuses solely on the object's true physical location, ignoring character beliefs or presence/absence. It traces the object's path through all movement events to identify where it was last placed.
    
    Args:
        narrative (str): The full narrative text describing characters, object movements, and presence/absence.
        focus (str): The specific object whose final location is to be determined (e.g., 'violin', 'bow').
    
    Returns:
        str: The final location of the object. Format: "{object_name} is in {final_location}".
        
        Example output: "violin is in music stand"
        
        If the object was never moved or its final location cannot be determined from the narrative, returns "{object_name} location unknown".
    """

@interface
def verify_presence_awareness(narrative: str, focus: str) -> str:
    """Verifies which characters were present or absent during specific object movements or events, and determines what each character could have witnessed.
    
    This function extracts information about character presence/absence during object movements
    and determines what each character would be aware of based on what they witnessed directly.
    
    Args:
        narrative (str): The full narrative text describing characters, objects, movements,
                        and who was present/absent for each event.
        focus (str): The specific focus of inquiry - can be:
                    - A character name (e.g., "Rachel") - checks their presence/awareness
                    - An object (e.g., "chips") - checks who witnessed its movements
                    - A specific movement (e.g., "moving the chips to dining table")
    
    Returns:
        dict: A structured presence matrix with the following format:
        {
            "focus": "[the original focus parameter]",
            "movements": [
                {
                    "step": 1,
                    "object": "chips",
                    "from": "counter",
                    "to": "dining table",
                    "actor": "Sam",
                    "present": ["Rachel", "Sam"],
                    "absent": ["Amy"],
                    "witnessed_by": ["Rachel", "Sam"]
                },
                # ... additional movements
            ],
            "character_awareness": {
                "Rachel": {
                    "witnessed_movements": [1],  # step numbers
                    "aware_of_objects": ["chips"],
                    "last_known_locations": {"chips": "dining table"}
                },
                "Amy": {
                    "witnessed_movements": [],
                    "aware_of_objects": [],
                    "last_known_locations": {}
                }
            }
        }
    
    Example output for focus="chips":
        {
            "focus": "chips",
            "movements": [
                {
                    "step": 1,
                    "object": "chips",
                    "from": "counter",
                    "to": "dining table",
                    "actor": "Sam",
                    "present": ["Rachel", "Sam"],
                    "absent": ["Amy"],
                    "witnessed_by": ["Rachel", "Sam"]
                }
            ],
            "character_awareness": {
                "Rachel": {
                    "witnessed_movements": [1],
                    "aware_of_objects": ["chips"],
                    "last_known_locations": {"chips": "dining table"}
                },
                "Amy": {
                    "witnessed_movements": [],
                    "aware_of_objects": [],
                    "last_known_locations": {}
                },
                "Sam": {
                    "witnessed_movements": [1],
                    "aware_of_objects": ["chips"],
                    "last_known_locations": {"chips": "dining table"}
                }
            }
        }
    """

@interface
def verify_presence_during_events(narrative: str, focus: str) -> str:
    """Analyzes a narrative to identify all object movement events and determine which characters were present or absent during each event. This is crucial for tracking character awareness of object locations based on what they witnessed.
    
    Args:
        narrative (str): The full narrative text describing characters, objects, movements, and presence/absence information.
        focus (str): Specifies the target of the verification. This could be:
                    - A specific character (e.g., 'Lisa'): Returns all events where this character was present/absent
                    - A specific object (e.g., 'water bottle'): Returns all movement events for this object
                    - A specific movement (e.g., 'from pantry to counter'): Returns presence info for this specific event
    
    Returns:
        A chronological list of movement events with detailed presence information. Each event is represented as a dictionary with:
            - 'step': Integer representing the chronological order
            - 'actor': Character who performed the movement
            - 'object': Object being moved
            - 'from': Starting location
            - 'to': Destination location
            - 'present': List of characters explicitly mentioned as present during this event
            - 'absent': List of characters explicitly mentioned as absent during this event
            - 'unknown': List of characters whose presence/absence is not specified
    
        Example output:
        [
            {
                'step': 1,
                'actor': 'Ben',
                'object': 'rare press album',
                'from': 'living room',
                'to': 'bedroom',
                'present': ['Ben', 'Anna'],
                'absent': ['Cora'],
                'unknown': ['David']
            },
            {
                'step': 2,
                'actor': 'Anna',
                'object': 'water bottle',
                'from': 'counter',
                'to': 'backpack',
                'present': ['Anna'],
                'absent': ['Ben', 'Cora'],
                'unknown': ['David']
            }
        ]
    
    Note: Characters not explicitly mentioned as present or absent in the context of a specific event should be listed as 'unknown' rather than assumed absent.
    """
