"""Auto-induced ptools from iterative induction pipeline.

These are real @interface stubs. Each is bound to simulate at runtime.
"""

from secretagent.core import interface

@interface
def summarize_evidence(narrative: str, focus: str) -> str:
    """Extracts and summarizes evidence related to a specific focus (suspect, location, or object) from the murder mystery narrative.
    
    This function analyzes the provided text to identify and categorize evidence into motive, opportunity,
    means, alibi, and contradictions. It focuses on finding factual statements rather than making judgments.
    
    Args:
        narrative (str): The full text of the murder mystery story.
        focus (str): The specific person, place, or item to analyze (e.g., 'Peyton', 'the knife', 'the mall').
    
    Returns:
        dict: A structured summary with the following keys:
            - 'motive': List of strings describing potential motives
            - 'opportunity': List of strings describing evidence related to being at the scene
            - 'means': List of strings describing access to murder weapons or required knowledge
            - 'alibi': List of strings supporting an alibi or lack of opportunity
            - 'contradictions': List of strings showing inconsistencies in statements or evidence
    
        Each list contains specific, verbatim quotes or close paraphrases from the narrative.
    
    Example:
        {
            'motive': ['Guy threatened to expose her secret'],
            'opportunity': ['Shops at the mall frequently', 'Works daytime shifts'],
            'means': ['Trained in martial arts'],
            'alibi': ['Was at work during the night of the murder'],
            'contradictions': ['Claimed to be home but was seen near crime scene']
        }
    """

@interface
def summarize_evidence(narrative: str, focus: str) -> str:
    """Extracts and summarizes all evidence related to a specific person, relationship, or aspect from the murder mystery narrative.
    
    Args:
        narrative (str): The full text of the murder mystery narrative
        focus (str): The specific person, relationship, or aspect to focus on (e.g., "Chris", "Julia", "Elizabeth", "Gerald's alibi")
    
    Returns:
        dict: A structured summary containing:
            {
                "focus": "[name/aspect from input]",
                "relationships": [list of relationships with other characters],
                "motive": [summary of potential motives],
                "opportunity": [summary of timing/alibi information],
                "contradictions": [list of inconsistencies or suspicious behavior],
                "supporting_evidence": [list of concrete evidence supporting involvement],
                "exonerating_evidence": [list of evidence suggesting innocence]
            }
    
    Example output:
        {
            "focus": "Elizabeth",
            "relationships": ["Married to Thomas", "Director of the play", "Had conflict with Eleanor"],
            "motive": "Eleanor was having an affair with her husband and mocked her during a confrontation",
            "opportunity": "Was present at the theater during the murder timeframe",
            "contradictions": ["Changed story about whereabouts", "Denied argument but witnesses confirm"],
            "supporting_evidence": ["Found with victim's personal item", "Had access to murder weapon"],
            "exonerating_evidence": ["Security footage shows her in different location"]
        }
    
    Focus on extracting specific details from the narrative rather than making judgments. Include both incriminating and exonerating evidence. Pay special attention to relationships, timelines, alibis, and any concrete physical evidence mentioned.
    """

@interface
def finalize_suspect_verdict(narrative: str, focus: str) -> str:
    """Finalizes the murder investigation by identifying the most probable suspect and summarizing the evidence against them.
    
    This function analyzes the narrative to determine which suspect has the strongest evidence pointing
    against them, considering factors like opportunity, motive, contradictory statements, and lack of alibi.
    
    Args:
        narrative (str): The full murder mystery text containing clues, statements, and evidence.
        focus (str): The specific suspect index or name to focus on for final conclusion.
    
    Returns:
        dict: A structured verdict containing:
            {
                "conclusion": "Guilty" or "Not Guilty",
                "suspect_name": "Name of the suspect",
                "suspect_index": 0-3 (based on narrative options),
                "confidence_level": "High", "Medium", or "Low",
                "key_evidence": ["list", "of", "3-5", "most", "critical", "pieces", "of", "evidence"],
                "contradictions": ["list", "of", "inconsistencies", "in", "their", "story"],
                "motive": "Brief description of potential motive",
                "opportunity": "Brief description of when/how they could have committed the crime"
            }
    
    Example output:
        {
            "conclusion": "Guilty",
            "suspect_name": "Taylor",
            "suspect_index": 1,
            "confidence_level": "High",
            "key_evidence": ["Seen near crime scene at time of murder", "Inconsistent alibi timeline", "Financial motive inheritance"],
            "contradictions": ["Claimed to be in meeting but no one can verify", "Changed story about phone call timing"],
            "motive": "$500,000 inheritance from victim",
            "opportunity": "Was alone in the west wing during the 30-minute murder window"
        }
    """

@interface
def extract_evidence(narrative: str, focus: str) -> str:
    """Extracts all evidence from the murder mystery narrative related to a specific focus (person, object, or location).
    
    This function scans the narrative for any mentions, actions, relationships, or circumstances
    related to the focus. It identifies both direct evidence (explicit mentions) and indirect
    evidence (implied connections or contextual information).
    
    Args:
        narrative (str): The full text of the murder mystery story
        focus (str): The specific person, object, or location to find evidence about
    
    Returns:
        str: A structured list of evidence items in the following format:
        
        EVIDENCE RELATED TO [focus]:
        - [Category 1]: [Specific evidence details]
        - [Category 2]: [Specific evidence details]
        - [Category 3]: [Specific evidence details]
        ...
        
        Categories may include but are not limited to:
        - Direct statements/actions
        - Relationships/connections
        - Motive/opportunity
        - Alibi/whereabouts
        - Physical evidence
        - Suspicious behavior
        - Contradictions/inconsistencies
        - Background/history
        - Witness testimony
        
        Example output for focus="Frederick":
        
        EVIDENCE RELATED TO Frederick:
        - Direct statements: Frederick was heard arguing with the victim the night before
        - Relationships: Frederick was the victim's business partner
        - Motive: Frederick stood to inherit the victim's business
        - Alibi: Claims to be at home alone during the murder
        - Physical evidence: Fingerprints found on the murder weapon
        - Contradictions: Neighbor reported seeing Frederick's car leave around the time of the murder
        - Background: Had a criminal record for assault
    """

@interface
def analyze_suspect_consistency(narrative: str, focus: str) -> str:
    """Analyzes a specific suspect's statements and actions for inconsistencies with established facts or contradictions within their own testimony.
    
    This function examines the narrative to identify:
    - Contradictions between the suspect's statements and physical evidence
    - Inconsistencies in the suspect's timeline or alibi
    - Behavioral anomalies or suspicious actions
    - Statements that conflict with other witnesses' accounts
    - Changes in the suspect's story over time
    
    Args:
        narrative (str): The full murder mystery text containing all evidence, statements, and facts
        focus (str): The specific suspect name to analyze (e.g., 'Clyde', 'Aubrey', 'Harry')
    
    Returns:
        dict: A structured analysis of inconsistencies with the following format:
        {
            "suspect": "suspect_name",
            "has_contradictions": true/false,
            "contradiction_details": [
                {
                    "type": "alibi_contradiction/statement_inconsistency/behavioral_anomaly/evidence_conflict",
                    "description": "Detailed description of the specific contradiction",
                    "supporting_evidence": "Quoted text from narrative supporting this finding"
                }
            ],
            "consistency_assessment": "Brief overall assessment of how consistent the suspect's account appears"
        }
        
        Example output:
        {
            "suspect": "Clyde",
            "has_contradictions": true,
            "contradiction_details": [
                {
                    "type": "behavioral_anomaly",
                    "description": "Claimed to be meticulous cleaner but neighbors saw him mowing lawn at unusual hours",
                    "supporting_evidence": "Neighbors saw Clyde mowing his lawn at 2 AM with a heavy lawnmower"
                }
            ],
            "consistency_assessment": "Multiple inconsistencies suggest possible deception"
        }
    """
