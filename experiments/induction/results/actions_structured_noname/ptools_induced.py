"""Auto-induced ptools from iterative induction pipeline."""

from secretagent.core import interface

@interface
def summarize_evidence(narrative: str, focus: str) -> str:
    """Analyzes the murder mystery narrative to extract and summarize evidence pertaining to specific individuals.
    
    This function examines the provided narrative text to identify all evidence (including motives, opportunities,
    alibis, suspicious behaviors, and contradictions) related to the individuals specified in the focus parameter.
    The evidence is categorized and presented in a structured format for clear reasoning analysis.
    
    Args:
        narrative (str): The complete murder mystery story containing characters, events, and clues.
        focus (str): A comma-separated list of names to focus on (e.g., "Harry, Rosemary" or "Amelia").
    
    Returns:
        dict: A structured summary of evidence for each specified individual. The output format is:
        {
            "suspect_name": {
                "motives": [list of strings describing potential motives],
                "opportunities": [list of strings describing potential opportunities],
                "alibis": [list of strings describing alibi claims or evidence],
                "contradictions": [list of strings describing inconsistencies in their story],
                "supporting_evidence": [list of strings describing other relevant clues]
            },
            ...
        }
        
        Example output:
        {
            "Harry": {
                "motives": ["Had financial disputes with victim"],
                "opportunities": ["Was seen near crime scene around time of murder"],
                "alibis": ["Claims to be at home alone"],
                "contradictions": ["Security footage shows him leaving his apartment"],
                "supporting_evidence": ["Fingerprints found on weapon"]
            },
            "Rosemary": {
                "motives": [],
                "opportunities": ["Was in the building at time of murder"],
                "alibis": ["Says she was in meeting with colleagues"],
                "contradictions": ["Colleagues say meeting ended earlier"],
                "supporting_evidence": ["Fibers matching her coat found at scene"]
            }
        }
    """
