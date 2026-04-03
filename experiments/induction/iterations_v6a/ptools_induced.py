"""Auto-induced ptools from iterative induction pipeline.

These are real @interface stubs. Each is bound to simulate at runtime.
"""

from secretagent.core import interface

@interface
def holistic_suspect_evaluation(narrative: str, focus: str) -> str:
    """Analyzes potential suspects in a murder mystery by examining three key criteria:
    
    1. MEANS: Assesses if the suspect had access to the murder weapon/tool and capability to commit the crime
    2. MOTIVE: Evaluates reasons why the suspect would want to harm the victim (financial gain, revenge, etc.)
    3. OPPORTUNITY: Determines if the suspect had the chance to commit the crime (alibi verification, timeline alignment)
    
    Additionally evaluates how well the suspect's profile matches crime scene evidence and witness statements.
    
    Parameters:
        narrative (str): The full murder mystery text containing clues, evidence, and character information
        focus (str): Specific evaluation focus (e.g., 'all suspects', 'Nicole and Isabelle', 'means only', 'opportunity comparison')
    
    Returns:
        str: A structured evaluation including:
        - Suspect-by-suspect breakdown of means, motive, and opportunity
        - Assessment of evidence alignment with each suspect
        - Comparative analysis when multiple suspects are evaluated
        - Any contradictions or gaps in the evidence
        - Final likelihood assessment for each suspect
    
    Pay special attention to:
    - Timeline consistency with alibis
    - Physical capability to commit the crime
    - Psychological plausibility of motives
    - Direct vs circumstantial evidence
    - Witness reliability and corroboration
    """

@interface
def summarize_suspect_evidence(narrative: str, focus: str) -> str:
    """Extracts comprehensive evidence information about a specific suspect from the murder mystery narrative.
    
    Focuses on four key categories:
    1. Motive: Reasons why the suspect might want the victim dead (grudges, conflicts, financial gain)
    2. Opportunity: Evidence placing suspect near crime scene or confirming they could have committed the act
    3. Means: Access to murder weapon or required knowledge/skills
    4. Suspicious Behavior: Any unusual actions, lies, or peculiar behavior related to the crime
    
    Parameters:
        narrative (str): The full murder mystery text
        focus (str): The specific suspect name to analyze (e.g., "Clyde", "Paul", "Lillian")
    
    Returns:
        str: A structured summary organized by category with specific evidence points extracted from the narrative
        
    Notes:
        - Extract only information directly related to the specified suspect
        - Include verbatim quotes from the narrative when possible
        - Note any contradictions or alibis mentioned
        - Pay special attention to timestamps, locations, and relationships mentioned
        - Format response with clear headings for each category
        - Return 'No evidence found' if nothing relevant appears in narrative
    """

@interface
def timeline_opportunity_analysis(narrative: str, focus: str) -> str:
    """Analyzes the narrative to determine timeline, location, and opportunity information for suspects related to a specific focus.
    
    Extracts and examines:
    - Timelines: Specific dates, times, durations, and sequences of events
    - Locations: Where people were positioned and their movements
    - Alibis: Statements about whereabouts or activities
    - Opportunity: Gaps in timelines or access to victim/evidence
    - Contradictions: Inconsistencies in statements about timeline/location
    
    Focus parameter should specify:
    - A person (suspect, victim, witness)
    - An event/time period (murder night, specific date)
    - Location (crime scene, specific place)
    - Or combination (person + time + location)
    
    Response structure:
    1. Timeline reconstruction for focus period
    2. Location/movement analysis for relevant individuals
    3. Opportunity assessment for each suspect
    4. Contradictions or gaps in alibis
    5. Key evidence supporting/refuting opportunity
    
    Pay attention to:
    - Exact time mentions (specific hours, days, durations)
    - Location descriptions and proximity to crime scene
    - Witness statements about others' whereabouts
    - Physical evidence placing people at locations
    - Overlap between suspect/victim timelines
    - Time required to commit crime vs. alibi coverage
    """

@interface
def alibi_analysis(narrative: str, focus: str) -> str:
    """This function analyzes the alibi of a given individual (focus) within the provided narrative. It extracts all mentions of the person's activities, locations, and timeframes related to the murder or specified event. The analysis includes:
    - Identifying explicit alibi statements (e.g., 'was at home', 'with friends').
    - Checking for time gaps or inconsistencies in the alibi.
    - Cross-referencing with other events or testimonies (e.g., overlaps with arrival/departure times of others).
    - Noting any supporting evidence (e.g., witnesses, CCTV) or contradictions.
    The response is structured as a summary of the alibi, followed by key points (evidence, inconsistencies, witnesses), and a conclusion on reliability.
    Parameters:
      narrative (str): The full text of the murder mystery.
      focus (str): The person to analyze (e.g., 'Isolde', 'Mark').
    Returns:
      str: A formatted string with alibi details, including timeframes, activities, supporting evidence, and any gaps or issues.
    """

@interface
def synthesize_evidence(narrative: str, focus: str) -> str:
    """Extracts and synthesizes evidence from the murder mystery narrative related to a specific focus.
    
    Focus should typically be a person's name (e.g., 'Randy', 'Isla') or specific aspect (e.g., 'alibi', 'motive', 'opportunity').
    
    Extracts:
    - All direct statements about the focus from the narrative
    - Actions, behaviors, or characteristics attributed to the focus
    - Relationships and interactions with other characters
    - Physical evidence, alibis, motives, or opportunities
    - Any suspicious or noteworthy details
    
    Structures response as:
    1. Clear heading identifying the focus
    2. Bulleted list of evidence points (each starting with '- ')
    3. Concise summary of key findings
    
    Pay special attention to:
    - Distinguishing between facts and speculation
    - Noting contradictions or inconsistencies
    - Identifying both incriminating and exonerating evidence
    - Contextualizing evidence within the overall narrative
    - Including specific details like timestamps, locations, and relationships
    """
