"""Auto-induced ptools from iterative induction pipeline.

These are real @interface stubs. Each is bound to simulate at runtime.
"""

from secretagent.core import interface

@interface
def analyze_alibi_consistency(narrative: str, focus: str) -> str:
    """Analyzes alibi consistency for suspects by comparing their stated whereabouts with evidence, timelines, and witness accounts.
    
    Extracts information about suspect alibis, timelines of events, witness testimonies, and physical evidence.
    Evaluates whether alibis are supported or contradicted by available evidence.
    
    Args:
        narrative (str): The full murder mystery text
        focus (str): Specific aspect to analyze (e.g., 'all suspects', 'suspect X', 'time window 8-9pm')
    
    Returns:
        dict: A dictionary with suspect names as keys, each containing:
            - has_alibi: bool (whether an alibi is claimed)
            - alibi_strength: str ('strong', 'weak', 'contradicted', 'none')
            - supporting_evidence: list[str] (evidence that supports the alibi)
            - contradictions: list[str] (evidence that contradicts the alibi)
            - time_gaps: list[str] (unaccounted time periods)
            - reliability_score: int (0-100 based on evidence consistency)
    
    Example:
        {
            "John": {
                "has_alibi": true,
                "alibi_strength": "weak",
                "supporting_evidence": ["Witness saw John at cafe at 8:15pm"],
                "contradictions": ["Security footage shows John near crime scene at 8:30pm"],
                "time_gaps": ["8:20pm-8:40pm"],
                "reliability_score": 30
            },
            "Mary": {
                "has_alibi": false,
                "alibi_strength": "none",
                "supporting_evidence": [],
                "contradictions": ["No alibi provided for murder timeframe"],
                "time_gaps": ["entire murder window"],
                "reliability_score": 0
            }
        }
    """

@interface
def summarize_evidence(narrative: str, focus: str) -> str:
    """Extracts and summarizes evidence for one or more suspects from the murder mystery narrative.
    
    Focus parameter should specify which suspect(s) to analyze. Can be a single name (e.g., 'Harper'), multiple names separated by commas (e.g., 'Addison, Octavia'), or 'each' to analyze all suspects mentioned in the narrative.
    
    Returns:
        A dictionary where each key is a suspect name and the value is another dictionary with structured evidence:
        {
            'suspect_name': {
                'motive': str (description of potential motive),
                'opportunity': str (description of opportunity to commit crime),
                'weapon_access': str (description of access to weapon/tool),
                'sightings': list (locations/times suspect was seen),
                'alibi': str (description of alibi if present),
                'relationship': str (relationship to victim),
                'contradictions': list (any inconsistencies in evidence)
            }
        }
        
    Example output:
        {
            'Harper': {
                'motive': 'Had financial disputes with victim',
                'opportunity': 'Was alone in the study during the murder window',
                'weapon_access': 'Had access to the letter opener used as weapon',
                'sightings': ['Seen entering study at 9:15 PM', 'Leaving study at 9:45 PM'],
                'alibi': 'Claims to be in the library, but no witnesses',
                'relationship': 'Business partner',
                'contradictions': ['Time of exit conflicts with butler testimony']
            }
        }
    """

@interface
def summarize_evidence(narrative: str, focus: str) -> str:
    """Analyzes the murder mystery narrative to extract and summarize evidence related to a specific individual.
    
    Extracts information about:
    - Motive: Reasons why the person might want to harm the victim
    - Opportunity: Whether the person could have been present at the crime scene/time
    - Weapon Access: Whether the person had access to the murder weapon
    - Sightings: Any reported sightings of the person around the crime scene/time
    - Alibi: Any claimed or verified alibi for the time of the murder
    - Relationship: How the person was connected to the victim
    - Suspicious Behavior: Any notable suspicious actions or statements
    
    Args:
        narrative (str): The full murder mystery text
        focus (str): The name of the person to analyze evidence for
    
    Returns:
        dict: A structured summary of evidence with the following keys:
            - has_motive: bool (True if motive evidence exists)
            - motive_details: str (description of motive evidence)
            - has_opportunity: bool (True if opportunity evidence exists)
            - opportunity_details: str (description of opportunity evidence)
            - weapon_access: bool (True if weapon access evidence exists)
            - weapon_details: str (description of weapon access evidence)
            - sightings: list[str] (list of sighting reports)
            - has_alibi: bool (True if alibi evidence exists)
            - alibi_details: str (description of alibi evidence)
            - relationship: str (description of relationship to victim)
            - suspicious_behavior: list[str] (list of suspicious actions/statements)
    
    Example output:
        {
            "has_motive": True,
            "motive_details": "Owed victim $10,000 and had argued about repayment",
            "has_opportunity": True,
            "opportunity_details": "Was seen near victim's office around the time of murder",
            "weapon_access": True,
            "weapon_details": "Had access to the kitchen where the knife was taken",
            "sightings": ["Seen leaving building at 8:05 PM", "Spotted near crime scene at 8:15 PM"],
            "has_alibi": False,
            "alibi_details": "Claims to be home alone but no verification",
            "relationship": "Business partner and creditor",
            "suspicious_behavior": ["Left town abruptly after murder", "Lied about knowing the victim"]
        }
    """

@interface
def check_character_consistency(narrative: str, focus: str) -> str:
    """Examines the narrative for contradictions, inconsistencies, or unexplained gaps in a character's account of events, actions, or statements.
    
    This function focuses on identifying logical gaps, conflicting statements, or actions that don't align with
    established facts or timelines. It looks for discrepancies in what the character claims versus what evidence
    or other accounts suggest.
    
    Args:
        narrative (str): The full murder mystery text containing all clues, statements, and evidence
        focus (str): The specific character to analyze for inconsistencies (e.g., "Gordon", "Letti")
    
    Returns:
        dict: A structured analysis of inconsistencies with the following format:
        {
            "character": "name_of_character",
            "total_inconsistencies": 3,
            "inconsistencies": [
                {
                    "type": "contradiction/timeline_gap/behavior_anomaly",
                    "description": "Specific description of the inconsistency",
                    "evidence": "Text excerpt supporting the finding",
                    "significance": "Why this inconsistency matters"
                }
            ],
            "overall_assessment": "Summary judgment of character's reliability"
        }
    
    Example output:
        {
            "character": "Gordon",
            "total_inconsistencies": 2,
            "inconsistencies": [
                {
                    "type": "contradiction",
                    "description": "Claimed to be in meeting but security logs show him entering building later",
                    "evidence": "Gordon stated: 'I was in my 2pm meeting' vs Security log: 'G. Smith entered at 2:15pm'",
                    "significance": "Creates opportunity for murder committed at 2:10pm"
                }
            ],
            "overall_assessment": "Multiple contradictions in timeline suggest potential deception"
        }
    """

@interface
def evaluate_alibi(narrative: str, focus: str) -> str:
    """Evaluates the alibi of a specific person by examining their claimed whereabouts against witness testimony, physical evidence, and timeline consistency.
    
    Args:
        narrative (str): The full murder mystery text containing all facts, statements, and evidence
        focus (str): The specific person whose alibi should be evaluated (e.g., "Meredith", "Kinsley")
    
    Returns:
        dict: A structured evaluation with the following keys:
            - has_alibi: bool - Whether the person has a verifiable alibi
            - alibi_statement: str - The exact alibi claim from the narrative
            - supporting_evidence: list[str] - Evidence/witnesses that support the alibi
            - contradictions: list[str] - Evidence/witnesses that contradict the alibi
            - timeline_conflict: bool - Whether the alibi conflicts with the murder timeline
            - reliability_score: str - One of 'strong', 'weak', or 'contradicted'
            - final_assessment: str - Brief summary of alibi reliability
    
    Example output:
        {
            "has_alibi": True,
            "alibi_statement": "Was in the rainforest observing animals",
            "supporting_evidence": ["Rainforest access logs show entry at 2:00 PM"],
            "contradictions": ["Jerry saw her near crime scene at 3:15 PM"],
            "timeline_conflict": True,
            "reliability_score": "contradicted",
            "final_assessment": "Alibi is contradicted by witness testimony placing suspect near crime scene"
        }
    """
