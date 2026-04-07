"""Auto-induced ptools from iterative induction pipeline.

These are real @interface stubs. Each is bound to simulate at runtime.
"""

from secretagent.core import interface

@interface
def identify_busy_periods(narrative: str, focus: str) -> str:
    """Extracts busy intervals for all participants mentioned in a calendar scheduling narrative.
    
    Parameters:
        narrative (str): The full task description containing attendees, days, work hours,
                        meeting duration, and each participant's existing busy intervals.
        focus (str): Specific aspect to focus on (e.g., target day, participant, or time window)
    
    Returns:
        dict: A dictionary keyed by participant names with their busy intervals as lists of dictionaries.
              Each interval dictionary contains 'day', 'start' (HH:MM), and 'end' (HH:MM).
              Example:
              {
                "Nancy": [{"day": "Monday", "start": "09:00", "end": "10:30"},
                          {"day": "Monday", "start": "14:00", "end": "15:00"}],
                "Jose": [{"day": "Monday", "start": "10:00", "end": "11:00"},
                          {"day": "Tuesday", "start": "13:00", "end": "14:30"}]
              }
    
    Notes:
        - Extracts all mentioned participants and their busy intervals from the narrative
        - Normalizes time formats to HH:MM (24-hour format)
        - Focus parameter can be used to filter results (e.g., specific day or participant)
        - Intervals are half-open [start, end) meaning the end time is exclusive
        - Returns empty list for participants with no mentioned busy periods
    """

@interface
def check_specific_time_availability(narrative: str, focus: str) -> str:
    """Evaluates whether a specific time slot is available for scheduling a meeting by checking against participants' busy intervals and work hour constraints.
    
    Extracts from the narrative:
    - Meeting attendees
    - Target day and specific time window to check
    - Meeting duration
    - Each participant's busy intervals
    - Work hour boundaries (if specified)
    
    Parameters:
        narrative (str): Full task description containing participants, day, time window, duration, and busy intervals
        focus (str): Specific time slot to evaluate (e.g., "Tuesday 12:00-12:30")
    
    Returns:
        dict: A boolean judgment with detailed reasoning about slot availability
        
        Example output format:
        {
            "slot_works": false,
            "violations": [
                {
                    "participant": "Daniel",
                    "conflict": "Busy during 11:00-12:00"
                },
                {
                    "participant": "Bradley", 
                    "conflict": "Outside work hours (ends at 17:00)"
                }
            ],
            "reasoning": "The slot conflicts with Daniel's schedule and extends beyond Bradley's work hours"
        }
    
    Notes:
    - Converts all times to minutes-from-midnight for precise comparison
    - Checks if the entire meeting duration fits within the specified time window
    - Verifies the slot is within work hours (if provided)
    - Identifies all participants with conflicting appointments
    - Returns detailed information about why a slot doesn't work if unavailable
    """

@interface
def extract_busy_periods_intervals(narrative: str, focus: str) -> str:
    """Extract Busy Periods/Intervals.
    
    Analyze the narrative with the given focus.
    """

@interface
def extract_free_slots(narrative: str, focus: str) -> str:
    """Extracts free time slots for a specific participant by analyzing their busy periods, work hours, and any specified constraints.
    
    Args:
        narrative (str): The full task description containing participant names, day, work hours, meeting duration, and each participant's busy intervals.
        focus (str): A string specifying the target participant, the time window (e.g., 'before 13:00'), and the minimum slot duration (e.g., 'at least 30 minutes').
    
    Returns:
        list: A list of dictionaries representing free time slots that meet the criteria. Each dictionary has the format:
            {
                'day': str,           # The day of the week (e.g., 'Monday')
                'start': str,         # Start time in HH:MM format
                'end': str,           # End time in HH:MM format
                'duration_minutes': int  # Duration of the free slot in minutes
            }
        Example: [{'day': 'Monday', 'start': '10:30', 'end': '11:00', 'duration_minutes': 30}]
    
    Notes:
        - Extracts the target participant, day, work hours, and minimum duration from the 'focus' string.
        - Parses the 'narrative' to find the participant's busy intervals.
        - All times are converted to minutes from midnight for calculation.
        - Free slots are calculated within the work hour boundaries.
        - Only returns slots with a duration >= the specified minimum.
    """

@interface
def calculate_available_time(narrative: str, focus: str) -> str:
    """"""
    Calculate available time slots for scheduling a meeting that fits within work hours and avoids all participants' busy times.
    
    Extracts from the narrative:
    - List of participants
    - Target day (if specified)
    - Work hour boundaries (start and end time)
    - Required meeting duration
    - Each participant's busy intervals (with day, start, and end times)
    - Any additional constraints (e.g., specific time windows, participant preferences)
    
    Process:
    1. Convert all times to minutes-from-midnight for uniform comparison
    2. Filter busy intervals to focus only on the target day (if specified)
    3. Merge busy intervals across all participants to create a unified busy timeline
    4. Identify free intervals within work hours that are not covered by any busy time
    5. Filter free intervals to only include those that can accommodate the meeting duration
    6. Apply any additional constraints (e.g., specific time windows)
    
    Args:
        narrative (str): The full task description containing meeting attendees, day, work hours,
                         meeting duration, and each participant's existing busy intervals.
        focus (str): Specific aspect to focus on (e.g., target participant, target day, meeting duration,
                     specific time window). Used to extract relevant constraints.
    
    Returns:
        A list of free intervals that satisfy the meeting duration and constraints, formatted as:
        [
            {
                "day": "Monday",
                "start": "10:00",
                "end": "11:00"
            },
            {
                "day": "Monday", 
                "start": "13:30",
                "end": "14:30"
            }
        ]
        
        Example output for a 30-minute meeting on Tuesday between 9:00-17:00:
        [
            {"day": "Tuesday", "start": "09:00", "end": "09:30"},
            {"day": "Tuesday", "start": "10:15", "end": "10:45"},
            {"day": "Tuesday", "start": "16:00", "end": "16:30"}
        ]
    
    Note:
    - Busy intervals are treated as half-open [start, end)
    - Free intervals are returned as time ranges where the entire meeting can be scheduled
    - All times are normalized to the same day/time format
    - The meeting must fit entirely within work hours AND avoid all participants' busy intervals
    """
    """
