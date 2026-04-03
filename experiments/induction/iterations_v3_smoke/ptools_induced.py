"""Auto-induced ptools from iterative induction pipeline.

These are real @interface stubs. Each is bound to simulate at runtime.
"""

from secretagent.core import interface

@interface
def evaluate_suspect_mmo(narrative: str, focus: str) -> str:
    """Analyzes the provided narrative to evaluate a suspect's potential for committing the murder by examining three key factors:
    
    1. MOTIVE: Extract and analyze reasons why the suspect would want to harm the victim. Look for evidence of conflicts, grudges, financial gain, personal relationships, secrets, threats, or any stated intentions. Note both explicit statements and implied motivations.
    
    2. MEANS: Evaluate whether the suspect had access to or knowledge of the murder weapon/method. Look for evidence of weapon ownership, relevant skills or knowledge, access to the specific location or tools used, or any special capabilities required for the crime.
    
    3. OPPORTUNITY: Determine if the suspect could have been present at the crime scene during the time of murder. Examine alibis, known whereabouts, access to location, timing constraints, and any evidence placing them at or near the scene.
    
    Structure the response with clear sections for each category. For each category:
    - State what evidence exists (quoting relevant narrative passages)
    - Analyze the strength of the evidence
    - Note any contradictions or weaknesses
    - Conclude with an overall assessment for each category
    
    Pay special attention to:
    - Timeframes mentioned in the narrative
    - Specific locations and access requirements
    - Weapon/method details
    - Relationships between suspect and victim
    - Any alibi evidence or witness statements
    - Contradictions or missing information
    
    Focus specifically on the suspect named in the 'focus' parameter.
    """
