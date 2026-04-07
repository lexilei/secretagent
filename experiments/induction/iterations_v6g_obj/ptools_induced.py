"""Auto-induced ptools from iterative induction pipeline.

These are real @interface stubs. Each is bound to simulate at runtime.
"""

from secretagent.core import interface

@interface
def trace_object_movement_history(narrative: str, focus: str) -> str:
    """Traces the complete movement history of a specified object throughout the narrative, capturing each transfer event with details about participants and witnesses.
    
    Args:
        narrative (str): The full narrative text describing characters, object movements, and presence/absence
        focus (str): The specific object to trace movements for (e.g., 'helmet', 'ancient coin', 'laptop')
    
    Returns:
        list: A chronological list of dictionaries representing each movement event. Each dictionary contains:
              - step: Sequential movement number (starting at 1)
              - from: Location object moved from
              - to: Location object moved to
              - actor: Character who performed the movement
              - present: List of characters present during this movement
              - absent: List of characters absent during this movement
              - description: Brief description of the movement event
    
    Example:
        [
            {
                "step": 1,
                "from": "pantry",
                "to": "counter",
                "actor": "Ben",
                "present": ["Anna", "Ben"],
                "absent": ["Cora"],
                "description": "Ben moved the object from pantry to counter"
            },
            {
                "step": 2,
                "from": "counter",
                "to": "drawer",
                "actor": "Anna",
                "present": ["Anna"],
                "absent": ["Ben", "Cora"],
                "description": "Anna moved the object from counter to drawer"
            }
        ]
    """

@interface
def trace_object_movement_history(narrative: str, focus: str) -> str:
    """Reconstructs the chronological movement history of a specific object from the narrative,
    including details about who was present and absent during each movement event.
    
    This tool focuses on tracking the physical movements of an object through space and time,
    with particular attention to which characters witnessed each transfer. This is essential
    for determining what each character knows about the object's location history.
    
    Args:
        narrative (str): The full narrative text containing object movement descriptions,
                        character actions, and presence/absence information.
        focus (str): The specific object to track movements for (e.g., 'flight manual', 'notebook').
    
    Returns:
        str: A structured chronological list of all movements for the specified object,
             formatted as a numbered sequence with details about each transfer.
             
             Format:
             """
             Object: [object_name]
             Movement History:
             1. [From location] -> [To location] (Actor: [character])
                Witnesses present: [comma-separated list]
                Witnesses absent: [comma-separated list]
             2. [From location] -> [To location] (Actor: [character])
                Witnesses present: [comma-separated list]
                Witnesses absent: [comma-separated list]
             ...
             """
             
             Example output:
             """
             Object: gradebook
             Movement History:
             1. Madison's desk -> storage cupboard (Actor: Madison)
                Witnesses present: Madison
                Witnesses absent: Ben, Cora
             2. storage cupboard -> Madison's desk (Actor: Madison)
                Witnesses present: Madison
                Witnesses absent: Ben, Cora
             """
    
    Note:
        - Only includes movements explicitly described in the narrative
        - 'Unknown' is used for witnesses when information is not specified
        - Focuses on actual witnessed events, not character beliefs
        - Returns empty history if no movements are found for the specified object
    """

@interface
def trace_movement_history(narrative: str, focus: str) -> str:
    """Analyzes a narrative to trace all movements of a specified object and identify who was present/absent for each movement event.
    
    Extracts:
    - Initial location of the object
    - All movement events (actor, from_location, to_location)
    - Characters present and absent during each movement
    - Maintains chronological order of events
    
    Returns:
    A chronological list of dictionaries representing each movement step. Each dictionary contains:
    - 'step': Integer step number (0 for initial state)
    - 'type': 'initial' or 'movement'
    - 'actor': Who performed the action (None for initial state)
    - 'from_location': Previous location
    - 'to_location': New location
    - 'present': List of characters present during movement
    - 'absent': List of characters absent during movement
    
    Example output:
    [
        {
            'step': 0,
            'type': 'initial',
            'actor': None,
            'from_location': None,
            'to_location': "teacher's desk",
            'present': ['all'],
            'absent': []
        },
        {
            'step': 1,
            'type': 'movement',
            'actor': 'Madison',
            'from_location': 'storage cupboard',
            'to_location': 'secure location',
            'present': ['Madison'],
            'absent': ['teacher', 'other students']
        }
    ]
    
    Note: Focuses on reconstructing the objective movement history rather than character beliefs.
    """

@interface
def verify_presence_matrix(narrative: str, focus: str) -> str:
    """Analyzes the narrative to extract all object movement events and creates a presence matrix showing which characters
    were present or absent during each movement. The matrix format helps identify witness patterns and awareness gaps.
    
    Args:
        narrative: The full narrative text describing characters, object movements, and who was present/absent
        focus: The specific focus for analysis (e.g., target character 'Anna', target object 'flight manual',
               or specific movement 'moved to cockpit')
    
    Returns:
        A dictionary with two keys:
        - 'events': List of movement events in chronological order with details
        - 'matrix': Dictionary mapping (event_id, character) -> presence status ('present'/'absent')
        
        Example output:
        {
            'events': [
                {
                    'id': 'event_1',
                    'step': 1,
                    'object': 'flight manual',
                    'from': 'pantry',
                    'to': 'office',
                    'actor': 'Richard',
                    'description': 'Richard moved the flight manual from pantry to office'
                },
                {
                    'id': 'event_2', 
                    'step': 2,
                    'object': 'flight manual',
                    'from': 'office',
                    'to': 'cockpit',
                    'actor': 'Tom',
                    'description': 'Tom moved the flight manual from office to cockpit'
                }
            ],
            'matrix': {
                ('event_1', 'Richard'): 'present',
                ('event_1', 'Tom'): 'absent',
                ('event_1', 'Ellie'): 'present',
                ('event_2', 'Richard'): 'absent',
                ('event_2', 'Tom'): 'present',
                ('event_2', 'Ellie'): 'present'
            }
        }
        
        Note: This tool focuses on factual presence/absence during movements, not on character beliefs or
        subsequent discoveries. The matrix format helps analyze witness patterns across multiple events.
    """

@interface
def verify_witness_presence(narrative: str, focus: str) -> str:
    """Verifies whether a specific character witnessed a particular object movement event by analyzing the narrative.
    
    This function examines the narrative to determine if the focus character was present when a specific
    object movement occurred. The focus should specify both the character to check and the movement event
    to examine (e.g., 'Ricky when Danny moved the notebook to the producer's desk').
    
    Args:
        narrative (str): The full narrative text describing characters, object movements,
                        and who was present/absent for each event.
        focus (str): A string specifying both the witness to check and the movement event
                    (e.g., 'Ricky when Danny moved the notebook to the producer's desk').
    
    Returns:
        dict: A structured response containing:
            - 'witness_present': Boolean indicating if the character was present
            - 'movement_details': Details about the movement event
            - 'witnesses_present': List of characters confirmed present during the event
            - 'witnesses_absent': List of characters confirmed absent during the event
            - 'evidence': Direct quotes from the narrative supporting the conclusion
    
    Example:
        {
            'witness_present': True,
            'movement_details': {
                'actor': 'Danny',
                'object': 'notebook',
                'from_location': 'office',
                'to_location': "producer's desk"
            },
            'witnesses_present': ['Ricky', 'Danny', 'Anna'],
            'witnesses_absent': ['Cora'],
            'evidence': "Danny moved the notebook from the office to the producer's desk while Ricky, Anna, and Danny were present. Cora was absent."
        }
    
    Note:
        This function only determines factual presence/absence from the narrative, not character beliefs
        about what others may have witnessed. Returns None if the specified movement event is not found.
    """
