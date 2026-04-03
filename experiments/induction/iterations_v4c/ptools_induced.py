"""Auto-induced ptools from iterative induction pipeline.

These are real @interface stubs. Each is bound to simulate at runtime.
"""

from secretagent.core import interface

@interface
def analyze_mmo(narrative: str, focus: str) -> str:
    """Analyzes the narrative to extract and evaluate evidence related to means, motive, and opportunity for potential suspects.
    
    Extract from the narrative:
    - Means: Who had access to/ability to use the murder weapon or method
    - Motive: Who had reason/desire to harm the victim (financial gain, revenge, etc.)
    - Opportunity: Who was present/could have been at the crime scene during the time window
    
    Structure the response:
    1. Identify all mentioned suspects
    2. For each suspect, analyze:
       - Means: Evidence showing capability with murder weapon/method
       - Motive: Potential reasons to want victim dead
       - Opportunity: Evidence placing them at/near crime scene during relevant time
    3. Compare strength of evidence across suspects
    4. Note any contradictions or alibis that might eliminate suspects
    
    Pay special attention to:
    - Specific details about the murder weapon/method mentioned in the narrative
    - Timeframe and location of the murder
    - Relationships between victim and suspects
    - Any direct evidence placing suspects at the scene
    - Professional skills or access that might provide means
    - Conflicts, debts, or grievances that might provide motive
    """
