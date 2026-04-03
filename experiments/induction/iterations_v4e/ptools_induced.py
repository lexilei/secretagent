"""Auto-induced ptools from iterative induction pipeline.

These are real @interface stubs. Each is bound to simulate at runtime.
"""

from secretagent.core import interface

@interface
def analyze_mmo(narrative: str, focus: str) -> str:
    """Analyzes the narrative to evaluate a suspect's means, motive, and opportunity for committing the murder.
    
    Extracts and evaluates three key components:
    1. MEANS: What weapons, skills, knowledge, or resources the suspect had access to
       that could have been used to commit the murder. Look for mentions of weapons,
       physical capabilities, specialized knowledge, or access to tools.
    
    2. MOTIVE: Why the suspect might have wanted to commit the murder. Look for
       conflicts, threats, financial gain, revenge, jealousy, or any other reasons
       that would provide motivation.
    
    3. OPPORTUNITY: Whether the suspect had the chance to commit the murder based on
       timeline, location, and alibi information. Examine temporal and spatial
       evidence including witness statements, surveillance, and physical presence.
    
    Structure the response with clear sections for Means, Motive, and Opportunity,
    followed by a brief summary assessment. For each category, provide:
    - Supporting evidence found in the narrative
    - Any contradictory evidence or weaknesses
    - Assessment of how strong/weak the case is for that category
    
    Focus specifically on the suspect mentioned in the 'focus' parameter. Return
    'No information available' for categories where no evidence is found.
    """
