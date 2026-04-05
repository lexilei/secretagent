"""Auto-induced ptools from iterative induction pipeline."""

from secretagent.core import interface

@interface
def summarize_evidence(narrative: str, focus: str) -> str:
    """Analyzes the provided murder mystery narrative to summarize evidence against specified suspects.
    
    This function extracts and organizes information about the named suspects, focusing on:
    - Potential motives for the crime
    - Alibi information and its reliability
    - Contradictions in their statements
    - Physical or circumstantial evidence mentioned
    - Any suspicious behavior described
    
    Args:
        narrative (str): The full text of the murder mystery narrative
        focus (str): Comma-separated list of suspect names to analyze (e.g., "Peyton,Isolde")
    
    Returns:
        dict: A structured summary organized by suspect with the following format:
        {
            "suspect_name": {
                "motives": ["list", "of", "potential", "motives"],
                "alibi": "description of alibi if provided",
                "alibi_strength": "strong/weak/none" (based on verification),
                "contradictions": ["list", "of", "statement", "contradictions"],
                "physical_evidence": ["list", "of", "evidence", "items"],
                "suspicious_behavior": ["list", "of", "suspicious", "actions"]
            },
            ...
        }
        
        Example:
        {
            "Peyton": {
                "motives": ["Financial debt to victim", "Recent argument"],
                "alibi": "Was at the library according to security logs",
                "alibi_strength": "strong",
                "contradictions": ["Claimed to be home alone but seen downtown"],
                "physical_evidence": ["Fingerprints on glass"],
                "suspicious_behavior": ["Left party early", "Nervous during questioning"]
            },
            "Isolde": {
                "motives": [],
                "alibi": "With friends at cinema",
                "alibi_strength": "weak",
                "contradictions": [],
                "physical_evidence": [],
                "suspicious_behavior": ["Avoided eye contact with detective"]
            }
        }
    """

@interface
def profile_suspect(narrative: str, focus: str) -> str:
    """Analyzes the provided narrative to extract and structure evidence related to a specific suspect.
    
    This function searches for information about the suspect's relationship to the victim, potential motives,
    access to weapons or means of committing the crime, and details about their alibi or whereabouts at the
    time of the crime. The output is structured to highlight these key investigative categories.
    
    Args:
        narrative (str): The full text of the murder mystery story or case details.
        focus (str): The name of the suspect to profile.
    
    Returns:
        str: A JSON-formatted string with keys 'relationship_to_victim', 'motive', 'access_to_weapon_means',
            and 'alibi_whereabouts'. Each key maps to a string containing the relevant evidence extracted
            from the narrative. If no information is found for a category, its value should be 'None found'.
    
    Example output format:
        {
            "relationship_to_victim": "Suspect is the victim's business partner.",
            "motive": "Suspect stood to gain financially from victim's death.",
            "access_to_weapon_means": "Suspect had access to the murder weapon in the shed.",
            "alibi_whereabouts": "Suspect claims to have been at a conference, but no one can verify."
        }
    """
