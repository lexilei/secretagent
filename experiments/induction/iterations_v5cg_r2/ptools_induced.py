"""Auto-induced ptools from iterative induction pipeline.

These are real @interface stubs. Each is bound to simulate at runtime.
"""

from secretagent.core import interface

@interface
def compare_suspects(narrative: str, focus: str) -> str:
    """Analyzes the narrative to compare two suspects (provided in the 'focus' parameter) by evaluating their motives, opportunities, evidence against them, and alibis. The focus string should specify the two suspects to compare (e.g., 'Peyton vs Isolde').
    
    Parameters:
        narrative (str): The full murder mystery text.
        focus (str): Two suspect names to compare, separated by 'vs' or 'and' (e.g., 'Peyton vs Isolde').
    
    Returns:
        dict: A structured comparison with keys:
            - 'suspect1': Name of first suspect
            - 'suspect2': Name of second suspect
            - 'comparison_points': List of dicts with keys:
                'aspect': str (e.g., 'Motive', 'Opportunity', 'Weapon Connection', 'Alibi')
                'suspect1_value': str (evaluation for first suspect)
                'suspect2_value': str (evaluation for second suspect)
                'weight': str (how strongly this point favors one suspect)
            - 'conclusion': Dict with keys:
                'most_likely': str (name of more probable culprit)
                'confidence': str (High/Medium/Low)
                'reasoning': str (brief explanation)
    
    Example output:
        {
            'suspect1': 'Peyton',
            'suspect2': 'Isolde',
            'comparison_points': [
                {
                    'aspect': 'Motive',
                    'suspect1_value': 'Weak - no clear motive',
                    'suspect2_value': 'Strong - financial gain',
                    'weight': 'Strongly favors Isolde'
                },
                {
                    'aspect': 'Weapon Connection',
                    'suspect1_value': 'Owns similar weapon',
                    'suspect2_value': 'Trained in weapon use',
                    'weight': 'Slightly favors Peyton'
                }
            ],
            'conclusion': {
                'most_likely': 'Isolde',
                'confidence': 'Medium',
                'reasoning': 'Stronger motive despite weaker weapon connection'
            }
        }
    """

@interface
def compare_suspect_evidence(narrative: str, focus: str) -> str:
    """Analyzes the narrative to compare two suspects across key evidence categories including motive, opportunity, means, alibi, and suspicious behavior. Returns a structured comparison showing which suspect has stronger evidence in each category.
    
    Args:
        narrative (str): The full murder mystery text
        focus (str): Should specify two suspects to compare (e.g., "Taylor vs Gloria" or "suspect_A and suspect_B")
    
    Returns:
        dict: A structured comparison with the following format:
        {
            "suspect1": "name",
            "suspect2": "name", 
            "comparison": {
                "motive": {
                    "winner": "suspect_name",
                    "reasoning": "brief explanation"
                },
                "opportunity": {
                    "winner": "suspect_name", 
                    "reasoning": "brief explanation"
                },
                "means": {
                    "winner": "suspect_name",
                    "reasoning": "brief explanation"
                },
                "alibi": {
                    "winner": "suspect_name",
                    "reasoning": "brief explanation"
                },
                "suspicious_behavior": {
                    "winner": "suspect_name",
                    "reasoning": "brief explanation"
                }
            },
            "overall_stronger_evidence": "suspect_name"
        }
    
    Example output:
        {
            "suspect1": "Taylor",
            "suspect2": "Gloria",
            "comparison": {
                "motive": {
                    "winner": "Taylor",
                    "reasoning": "Taylor had revenge motive from long-term bullying"
                },
                "opportunity": {
                    "winner": "Gloria",
                    "reasoning": "Gloria was often in building late with no alibi"
                },
                "means": {
                    "winner": "Taylor",
                    "reasoning": "Taylor had access to the murder weapon"
                },
                "alibi": {
                    "winner": "Taylor",
                    "reasoning": "Gloria has no alibi while Taylor has partial alibi"
                },
                "suspicious_behavior": {
                    "winner": "Gloria",
                    "reasoning": "Gloria was seen acting nervous after the murder"
                }
            },
            "overall_stronger_evidence": "Taylor"
        }
    """

@interface
def extract_suspect_motives(narrative: str, focus: str) -> str:
    """Analyzes the murder mystery narrative to identify all suspects and extract their potential motives for committing the crime.
    
    Args:
        narrative (str): The full text of the murder mystery story
        focus (str): Specific aspect to focus on (e.g., 'financial motives', 'revenge motives', 'specific suspect')
    
    Returns:
        dict: A structured dictionary with suspect names as keys, each containing:
              - potential_motives: list of identified motives with supporting text evidence
              - motive_strength: qualitative assessment (weak/medium/strong)
              - related_relationships: people connected to this motive
              - narrative_evidence: direct quotes supporting each motive
    
    Example:
        {
            "John Doe": {
                "potential_motives": ["Financial gain", "Revenge"],
                "motive_strength": "strong",
                "related_relationships": ["Victim (business partner)", "Mary Smith (beneficiary)"],
                "narrative_evidence": [
                    "John stood to inherit the business if the victim died",
                    "The victim had recently cheated John out of $50,000"
                ]
            },
            "Mary Smith": {
                "potential_motives": ["Jealousy"],
                "motive_strength": "weak",
                "related_relationships": ["Victim (romantic partner)"],
                "narrative_evidence": [
                    "Mary was seen arguing with the victim about his new assistant"
                ]
            }
        }
    """

@interface
def summarize_evidence_for_conclusion(narrative: str, focus: str) -> str:
    """This function analyzes the provided murder mystery narrative to summarize evidence supporting a conclusion about the most likely murderer. It focuses on key aspects such as motive, opportunity, supporting evidence, and contradictions. The output is a structured string that presents the reasoning in a clear and concise manner.
    
    Args:
        narrative (str): The full text of the murder mystery.
        focus (str): A string indicating the specific aspect to focus on (e.g., 'motive', 'opportunity', 'evidence', 'contradictions'). The agent can use this to emphasize a particular part of the summary.
    
    Returns:
        str: A string containing the conclusion and reasoning. The string should be formatted as follows:
            'Based on the evidence provided in the narrative, [suspect_name] is the more likely murderer. Here\'s the reasoning:\n\n- **Motive**: [summary_of_motive]\n- **Opportunity**: [summary_of_opportunity]\n- **Supporting Evidence**: [summary_of_evidence]\n- **Contradictions/Alternative Explanations**: [summary_of_contradictions]\n'
            Note: Each section should be filled based on the narrative. If a section is not relevant or not found, it should be omitted.
    
    Example:
        >>> narrative = 'Otis was found dead. Andrew was the beneficiary of his insurance policy. He was seen near the crime scene.'
        >>> focus = 'motive'
        >>> summarize_evidence_for_conclusion(narrative, focus)
        "Based on the evidence provided in the narrative, Andrew is the more likely murderer. Here's the reasoning:\n\n- **Motive**: Andrew had a strong financial motive as the beneficiary of Otis's insurance policy.\n- **Opportunity**: Andrew was seen near the crime scene at the time of the murder.\n- **Contradictions/Alternative Explanations**: None mentioned."
    """

@interface
def analyze_evidence_contradictions(narrative: str, focus: str) -> str:
    """Analyzes the murder mystery narrative for evidence contradictions and inconsistencies related to a specific focus.
    
    This function examines the provided narrative text to identify conflicting statements, timeline
    discrepancies, inconsistent accounts, or contradictory evidence specifically related to the
    provided focus (e.g., a suspect, alibi, or event). It looks for statements that conflict with
    each other or with established facts in the case.
    
    Args:
        narrative (str): The full murder mystery text containing all evidence and statements.
        focus (str): The specific aspect to analyze for contradictions (e.g., "Lloyd's alibi",
                     "witness accounts of the scream", "Eddie's fingerprints").
    
    Returns:
        dict: A structured analysis of contradictions with the following keys:
            - "contradictions_found" (bool): Whether any contradictions were identified
            - "contradiction_count" (int): Number of contradictions found
            - "contradiction_details" (list): List of specific contradictions, each as a dict with:
                - "element_1" (str): First conflicting statement or fact
                - "element_2" (str): Second conflicting statement or fact
                - "source_1" (str): Where element_1 appears in the narrative
                - "source_2" (str): Where element_2 appears in the narrative
                - "interpretation" (str): Brief analysis of what this contradiction suggests
    
    Example output:
        {
            "contradictions_found": True,
            "contradiction_count": 2,
            "contradiction_details": [
                {
                    "element_1": "Lloyd claimed he was in the library at 9 PM",
                    "element_2": "Security camera shows Lloyd entering the kitchen at 9:05 PM",
                    "source_1": "Interview with Lloyd, paragraph 3",
                    "source_2": "Security log exhibit B",
                    "interpretation": "Lloyd's alibi appears to be false"
                }
            ]
        }
    """
