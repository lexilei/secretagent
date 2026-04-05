"""Auto-induced ptools from iterative induction pipeline."""

from secretagent.core import interface

@interface
def summarize_evidence(narrative: str, focus: str) -> str:
    """Analyzes a murder mystery narrative to extract and summarize evidence related to a specific focus.
    
    This function examines the provided narrative text to identify all relevant evidence, clues,
    motives, alibis, and contradictions pertaining to the specified focus (typically suspects
    or specific aspects of the case). It organizes this information into a structured summary
    that highlights the most critical elements for investigation.
    
    Args:
        narrative (str): The full text of the murder mystery narrative containing all evidence,
                        statements, and clues.
        focus (str): The specific element to summarize evidence for. This could be:
                    - A suspect name (e.g., 'Peyton', 'Isolde')
                    - A specific aspect (e.g., 'murder weapon', 'timeline', 'motive')
                    - 'all_suspects' to summarize evidence for all mentioned suspects
    
    Returns:
        dict: A structured summary of evidence with the following format:
        {
            "focus": "[original focus parameter]",
            "summary_type": "suspect" | "aspect" | "comparative",
            "key_evidence": [
                {
                    "type": "motive" | "alibi" | "opportunity" | "physical_evidence" | "witness" | "contradiction" | "other",
                    "description": "Detailed description of the evidence",
                    "supporting": true | false | null,  # Does this support guilt? True=supports, False=contradicts, null=neutral
                    "certainty": "definite" | "probable" | "possible" | "speculative"  # Level of certainty about this evidence
                }
            ],
            "overall_assessment": "Brief overall assessment of the evidence strength"
        }
        
        Example output for focus="Peyton":
        {
            "focus": "Peyton",
            "summary_type": "suspect",
            "key_evidence": [
                {
                    "type": "motive",
                    "description": "Victim was blackmailing Peyton about financial fraud",
                    "supporting": true,
                    "certainty": "definite"
                },
                {
                    "type": "alibi",
                    "description": "Claims to be at business meeting but no one can verify",
                    "supporting": false,
                    "certainty": "probable"
                }
            ],
            "overall_assessment": "Strong motive but questionable alibi"
        }
        
        Example output for focus="all_suspects":
        {
            "focus": "all_suspects",
            "summary_type": "comparative",
            "suspects": {
                "Peyton": {
                    "key_evidence": [...],
                    "overall_assessment": "..."
                },
                "Isolde": {
                    "key_evidence": [...],
                    "overall_assessment": "..."
                }
            },
            "comparative_assessment": "Peyton has stronger motive but Isolde had better opportunity"
        }
    """

@interface
def analyze_evidence_components(narrative: str, focus: str) -> str:
    """Analyzes a murder mystery narrative to extract and evaluate evidence components related to a specific focus (suspect, location, weapon, etc.).
    
    This function parses the narrative text to identify and categorize key evidence elements including:
    - Direct evidence linking the focus to the crime
    - Contradictions or inconsistencies in the focus's story
    - Alibi information and its reliability
    - Motive or opportunity evidence
    - Physical evidence connections
    - Witness statements involving the focus
    
    Args:
        narrative (str): The full narrative text of the murder mystery
        focus (str): The specific element to analyze (suspect name, weapon, location, etc.)
    
    Returns:
        dict: A structured analysis containing:
        {
            "focus": "[focus parameter value]",
            "direct_evidence": ["list", "of", "direct", "evidence", "points"],
            "contradictions": ["list", "of", "contradictions", "inconsistencies"],
            "alibi_info": {
                "has_alibi": true/false,
                "alibi_details": "description of alibi",
                "alibi_verified": true/false/null
            },
            "motive_opportunity": {
                "motive": "potential motive description",
                "opportunity": "opportunity description"
            },
            "physical_evidence": ["list", "of", "physical", "evidence", "connections"],
            "witness_statements": ["list", "of", "relevant", "witness", "accounts"]
        }
    
    Example output:
        {
            "focus": "John Doe",
            "direct_evidence": ["Found near crime scene at time of murder", "Fibers matching his coat found on victim"],
            "contradictions": ["Claimed to be at home but phone pinged downtown"],
            "alibi_info": {
                "has_alibi": false,
                "alibi_details": "None provided",
                "alibi_verified": null
            },
            "motive_opportunity": {
                "motive": "Financial dispute with victim",
                "opportunity": "Was seen leaving building around time of murder"
            },
            "physical_evidence": ["Fingerprints on weapon", "DNA under victim's nails"],
            "witness_statements": ["Saw arguing with victim earlier", "Heard loud voices from his apartment"]
        }
    """

@interface
def manual_evidence_extraction_and_evaluation(narrative: str, focus: str) -> str:
    """Manual Evidence Extraction and Evaluation.
    
    Analyze the narrative with the given focus.
    """

@interface
def validate_tool_output(narrative: str, focus: str) -> str:
    """Analyzes tool output to detect potential errors, inconsistencies, or mismatches with the requested focus.
    
    This function examines whether the tool output contains the expected content for the specified
    focus entity (person, location, object) and identifies common tool failure patterns such as:
    - Returning information about the wrong entity
    - Providing generic/placeholder responses
    - Repeating identical output across different queries
    - Missing key information that should be present
    
    Args:
        narrative (str): The text output from a reasoning tool that needs validation
        focus (str): The specific entity (person, location, object) the tool was supposed to analyze
    
    Returns:
        dict: A structured validation report with the following keys:
            - is_valid (bool): Whether the output appears correct for the given focus
            - validation_issues (list): List of specific issues detected (empty if none)
            - focus_appears_in_output (bool): Whether the focus string appears in the output
            - output_consistency_score (int): 0-10 rating of how well output matches expected format
            - suggested_action (str): Recommended next step if validation fails
            - key_phrases_missing (list): Important phrases expected but not found
            - key_phrases_found (list): Important phrases successfully detected
    
    Example:
        {
            "is_valid": false,
            "validation_issues": [
                "Output mentions 'Justin' instead of requested focus 'Frederick'",
                "Output appears to be a generic placeholder response"
            ],
            "focus_appears_in_output": false,
            "output_consistency_score": 2,
            "suggested_action": "Try manual evidence extraction or use a different analysis tool",
            "key_phrases_missing": ["alibi", "motive", "opportunity"],
            "key_phrases_found": ["present at scene"]
        }
    """

@interface
def compare_evidence(narrative: str, focus: str) -> str:
    """Compares evidence between multiple suspects mentioned in a murder mystery narrative.
    
    This function analyzes the provided narrative to extract and compare key evidence elements
    (motive, means, opportunity, alibi, contradictions) for each suspect. It focuses on identifying
    inconsistencies, strengths, and weaknesses in the evidence against each person.
    
    Args:
        narrative (str): The narrative text containing evidence and details about suspects.
        focus (str): A string indicating the primary focus of comparison (e.g., "motive", "opportunity",
                     "weapon_access"). Use "overall" for general comparison.
    
    Returns:
        dict: A structured comparison with the following keys:
            - suspects (list): Names of all suspects mentioned in the narrative
            - comparison (dict): For each suspect, a dictionary containing:
                * motive_strength (str): Assessment of motive strength ("strong", "weak", "none")
                * means_access (str): Assessment of means/weapon access ("yes", "no", "unclear")
                * opportunity (str): Assessment of opportunity ("confirmed", "none", "unclear")
                * alibi_status (str): Alibi status ("confirmed", "busted", "none", "unverified")
                * key_evidence (list): List of key evidence points for this suspect
                * contradictions (list): List of evidence contradictions for this suspect
            - focus_analysis (dict): Detailed analysis of the requested focus area comparing all suspects
            - relative_guilt_assessment (str): Brief assessment of which suspect appears more guilty
    
    Example output:
        {
            "suspects": ["Peyton", "Isolde"],
            "comparison": {
                "Peyton": {
                    "motive_strength": "strong",
                    "means_access": "yes",
                    "opportunity": "unclear",
                    "alibi_status": "none",
                    "key_evidence": ["threatened victim", "had secret victim knew"],
                    "contradictions": ["no witness placed at crime scene"]
                },
                "Isolde": {
                    "motive_strength": "strong",
                    "means_access": "unclear",
                    "opportunity": "confirmed",
                    "alibi_status": "busted",
                    "key_evidence": ["racial animosity", "seen near crime scene"],
                    "contradictions": ["alibi witness unreliable"]
                }
            },
            "focus_analysis": {
                "motive_comparison": "Both have strong motives but Peyton's involves direct threat"
            },
            "relative_guilt_assessment": "Peyton appears more guilty due to direct threat evidence"
        }
    """
