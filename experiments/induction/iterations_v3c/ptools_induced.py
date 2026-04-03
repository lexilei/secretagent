"""Auto-induced ptools from iterative induction pipeline.

These are real @interface stubs. Each is bound to simulate at runtime.
"""

from secretagent.core import interface

@interface
def analyze_mmo_framework(narrative: str, focus: str) -> str:
    """Analyzes the provided murder mystery narrative to evaluate Means, Motive, and Opportunity (MMO) for potential suspects.
    
    Extracts and evaluates:
    - Means: Who had access to the murder weapon/mechanism and capability to use it
    - Motive: Who had reason, intention, or benefit from the victim's death
    - Opportunity: Who was present at the crime scene/time and had access to commit the murder
    
    Focus on the specific aspect provided in the 'focus' parameter, which could be:
    - A particular suspect's name
    - A specific MMO component (means/motive/opportunity)
    - A particular weapon or location
    - A specific time period
    
    Structure the response by:
    1. Identifying all relevant suspects mentioned in the narrative
    2. For each suspect, analyze their means, motive, and opportunity based on evidence
    3. Pay special attention to details matching the focus parameter
    4. Compare suspects' MMO profiles to identify the strongest case
    5. Note any contradictions or alibis that weaken the case against a suspect
    
    Return a structured analysis that clearly shows the MMO assessment for each relevant suspect, highlighting the most compelling evidence and any gaps in the case against them.
    """

@interface
def analyze_contradictions(narrative: str, focus: str) -> str:
    """Analyzes the murder mystery narrative to identify contradictions, inconsistencies, or conflicting evidence related to the specified focus area.
    
    This function examines the narrative text to find statements, claims, or evidence that directly
    contradict each other regarding the specified focus (e.g., alibis, weapon ownership, timelines,
    motives, or witness statements). It identifies the conflicting elements, assesses their reliability,
    and provides analysis on how these contradictions impact the investigation.
    
    Parameters:
        narrative (str): The full text of the murder mystery containing all evidence and statements
        focus (str): The specific aspect to analyze for contradictions (e.g., 'weapon ownership', 'timeline', 
                     'witness statements', 'alibi', 'motive')
    
    Returns:
        str: A structured analysis containing:
             - Identified contradictions with specific quotes from the narrative
             - Assessment of which statements appear more reliable/credible
             - Potential explanations for the contradictions
             - Implications for suspect evaluation
             - Recommended follow-up questions or investigations
    
    Focus on:
        - Finding directly conflicting statements about the same fact or event
        - Comparing witness credibility and evidence reliability
        - Identifying gaps or inconsistencies in timelines
        - Noting when physical evidence contradicts testimonial evidence
        - Assessing whether contradictions suggest deception, error, or missing information
    """

@interface
def analyze_timeline(narrative: str, focus: str) -> str:
    """Extract and analyze temporal information from the murder mystery narrative to reconstruct the timeline of events. Focus on:
    
    1. Specific timestamps mentioned in the narrative (exact times, time ranges, relative times)
    2. Sequence of events and character movements throughout the day/night
    3. Character locations at specific times and their proximity to the crime scene
    4. Overlap between suspect movements and the estimated time of death
    5. Alibi verification based on timing and location evidence
    6. Any time-related contradictions or inconsistencies
    
    Pay special attention to:
    - The exact time or time window when the murder occurred
    - Character statements about their whereabouts at specific times
    - Witness observations with timestamps
    - Travel times between locations
    - Events that might have affected timing (e.g., concerts, meetings, appointments)
    - Any temporal constraints mentioned in the narrative
    
    Structure the response as:
    1. Summary of key timeline events with timestamps
    2. Analysis of each suspect's opportunity based on timeline
    3. Identification of any timeline contradictions
    4. Assessment of which suspects could/couldn't have committed the murder based on timing
    5. Any missing timeline information that would be helpful
    
    Focus specifically on: {focus} when analyzing the timeline.
    """

@interface
def analyze_evidence(narrative: str, focus: str) -> str:
    """Analyzes physical evidence mentioned in the murder mystery narrative. Focuses on:
    - Key evidence items (weapons, objects, forensic clues)
    - Their description, condition, and location
    - Ownership or origin of evidence items
    - How evidence might connect to suspects or the crime
    - Any abnormalities or inconsistencies with evidence
    - Potential significance to solving the case
    
    Extract and organize all physical evidence details, paying particular attention to:
    1. Murder weapons and their characteristics
    2. Forensic evidence (fingerprints, DNA, etc.)
    3. Objects found at crime scene
    4. Missing or tampered-with items
    5. Evidence that contradicts statements or alibis
    
    Structure the response with clear sections for each evidence type, noting:
    - What the evidence is
    - Where it was found
    - Its condition/state
    - Who it might belong to
    - How it relates to the crime
    - Any inconsistencies or peculiarities
    """
