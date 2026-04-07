"""Auto-induced ptools from iterative induction pipeline.

These are real @interface stubs. Each is bound to simulate at runtime.
"""

from secretagent.core import interface

@interface
def analyze_availability(narrative: str, focus: str) -> str:
    """Extracts and structures participant busy intervals from a calendar scheduling narrative.
    
    Parses the narrative to identify all participants, their scheduled busy time intervals,
    work hour boundaries, and any scheduling constraints. Returns a structured overview
    of availability information for further scheduling analysis.
    
    Args:
        narrative: The full task description containing meeting details, attendees,
                   work hours, meeting duration, and each participant's busy intervals
        focus: Specific aspect to focus on (e.g., particular participant, day, or constraint)
    
    Returns:
        dict: Structured availability analysis with keys:
            - 'participants': List of all participant names
            - 'work_hours': Dict with 'start' and 'end' times (e.g., {'start': '09:00', 'end': '17:00'})
            - 'meeting_duration': Duration in minutes
            - 'days_considered': List of days mentioned for scheduling
            - 'busy_intervals': Dict keyed by participant with their busy time intervals
              (e.g., {'Kathleen': [{'day': 'Thursday', 'start': '10:00', 'end': '11:00'}], ...})
            - 'constraints': List of any special constraints mentioned
            - 'focus_analysis': Additional analysis specific to the focus parameter
    
    Example:
        {
            'participants': ['Kathleen', 'Christian'],
            'work_hours': {'start': '09:00', 'end': '17:00'},
            'meeting_duration': 30,
            'days_considered': ['Thursday'],
            'busy_intervals': {
                'Kathleen': [{'day': 'Thursday', 'start': '10:00', 'end': '11:00'}],
                'Christian': [{'day': 'Thursday', 'start': '14:00', 'end': '15:30'}]
            },
            'constraints': [],
            'focus_analysis': 'Analyzed Kathleen\'s Thursday availability'
        }
    """
