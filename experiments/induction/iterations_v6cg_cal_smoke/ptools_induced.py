"""Auto-induced ptools from iterative induction pipeline.

These are real @interface stubs. Each is bound to simulate at runtime.
"""

from secretagent.core import interface

@interface
def parse_and_interpret_input(narrative: str, focus: str) -> str:
    """Parses a calendar scheduling narrative to extract key constraints and structure them for further processing.
    
    Extracts:
    - Meeting participants
    - Target day(s) for scheduling
    - Work hour boundaries (start and end)
    - Meeting duration
    - Each participant's busy intervals
    - Any stated preferences (e.g., "no meetings after X", "prefer not on Y")
    
    Converts all time information to minutes-from-midnight for consistent comparison.
    Busy intervals are stored as half-open [start, end) where end is exclusive.
    
    Args:
        narrative (str): The full task description containing meeting details and constraints
        focus (str): Specific aspect to focus on (e.g., "Harold's constraints", "Monday availability", "duration requirements")
    
    Returns:
        dict: Structured representation of parsed constraints with keys:
            - "participants": list of participant names
            - "days": list of target days
            - "work_hours": {"start": minutes, "end": minutes}
            - "duration": meeting duration in minutes
            - "busy_intervals": {"participant_name": [{"day": str, "start": minutes, "end": minutes}, ...]}
            - "preferences": {"participant_name": ["constraint_description", ...]}
            - "focus_analysis": specific interpretation of the focus parameter
    
    Example:
        {
            "participants": ["Evelyn", "Randy"],
            "days": ["Monday"],
            "work_hours": {"start": 540, "end": 1020},
            "duration": 30,
            "busy_intervals": {
                "Evelyn": [{"day": "Monday", "start": 600, "end": 720}],
                "Randy": [{"day": "Monday", "start": 780, "end": 900}]
            },
            "preferences": {
                "Evelyn": ["no meetings after 13:00"]
            },
            "focus_analysis": "Evelyn cannot meet after 780 minutes (13:00)"
        }
    """
