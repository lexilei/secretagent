"""Auto-induced ptools from iterative induction pipeline.

These are real @interface stubs. Each is bound to simulate at runtime.
"""

from secretagent.core import interface

@interface
def calculate_individual_bag_fee(narrative: str, focus: str) -> str:
    """Calculates the complete fee breakdown for a specific checked bag based on airline baggage policies.
    
    Extracts from narrative:
    - Passenger customer class (Economy, Premium Economy, Business, First)
    - Flight route region (domestic/international/regional specific)
    - Bag dimensions (length × width × height in inches) and weight (lbs)
    - Ticket price (for potential excess value fee calculation)
    - Bag ordinal position (for base fee determination)
    
    Key calculation rules:
    - Base fee depends on route region × customer class × bag ordinal
    - Oversize: total dimensions > 62 inches ($30 for 63-65", $200 for >65")
    - Overweight tiers: 50-53 lbs ($30), 53-70 lbs ($100), 70-100 lbs ($200), >100 lbs ($400)
    - Oversize and overweight surcharges don't stack - apply max(oversize, overweight) per bag
    - Personal items/carry-ons (typically first items) are free
    
    Args:
        narrative (str): Full problem statement with passenger, flight, and bag details
        focus (str): Specific bag ID to calculate fees for (e.g., "bag 1", "bag 3")
    
    Returns:
        dict: Detailed fee breakdown for the specified bag with format:
        {
            "bag_id": "bag_2",
            "base_fee": 200,
            "oversize_fee": 0,
            "overweight_fee": 100,
            "max_surcharge": 100,
            "total_fee": 300,
            "dimensions": "63 inches",
            "weight": "55 lbs",
            "oversize_reason": "",
            "overweight_reason": "53-70 lbs tier",
            "rule_cited": "Base fee: Premium Economy 2nd bag Asia $200, Max(oversize,overweight)=$100"
        }
    
    Example output:
        {
            "bag_id": "bag_2",
            "base_fee": 200,
            "oversize_fee": 30,
            "overweight_fee": 0,
            "max_surcharge": 30,
            "total_fee": 230,
            "dimensions": "64 inches",
            "weight": "48 lbs",
            "oversize_reason": "63-65 inch oversize",
            "overweight_reason": "",
            "rule_cited": "Base fee: Economy 2nd bag Domestic $200, Max(oversize,overweight)=$30"
        }
    """

@interface
def interpret_policy_ranges(narrative: str, focus: str) -> str:
    """Interprets policy language containing range descriptions (e.g., 'Over 62 in / 158 cm – 65 in / 165 cm') and extracts numeric thresholds with inclusion rules.
    
    Extracts:
    - Range patterns from the narrative (e.g., "Over X to Y")
    - Units (inches, cm, lbs) for context
    - Interprets whether boundaries are inclusive/exclusive (typically 'Over X' means >X, 'to Y' means ≤Y)
    
    Returns:
    A list of dictionaries with interpreted ranges, each containing:
    - 'range_text': The original text pattern matched
    - 'lower_bound': The lower threshold (exclusive if 'Over' is used)
    - 'upper_bound': The upper threshold (inclusive)
    - 'unit': The unit of measurement (e.g., 'in', 'cm', 'lbs')
    - 'interpretation': A human-readable explanation of the range
    
    Example output:
    [
        {
            "range_text": "Over 62 in – 65 in",
            "lower_bound": 62.0,
            "upper_bound": 65.0,
            "unit": "in",
            "interpretation": "Dimensions >62.0 inches and ≤65.0 inches"
        },
        {
            "range_text": "Over 50 lbs to 53 lbs",
            "lower_bound": 50.0,
            "upper_bound": 53.0,
            "unit": "lbs",
            "interpretation": "Weight >50.0 lbs and ≤53.0 lbs"
        }
    ]
    """

@interface
def recall_policy_rules(narrative: str, focus: str) -> str:
    """Extracts passenger class, flight route, ticket price, and bag details from narrative, then identifies applicable policy rules based on the specified focus.
    
    Args:
        narrative: Full problem statement containing passenger info, flight details, and bag specifications
        focus: Specific aspect to focus on (e.g., 'customer_class', 'route_region', 'bag_ordinal_fees', 'size_limits')
    
    Returns:
        dict: Structured policy rule information based on the focus parameter:
        - For 'customer_class': {'customer_class': 'Main Cabin', 'class_features': ['free_carry_on', 'checked_bag_fees'], ...}
        - For 'route_region': {'origin': 'Charlotte', 'destination': 'Washington D.C.', 'region_category': 'US_Domestic', 'route_rules': {...}}
        - For 'bag_ordinal_fees': {'first_bag_fee': 40, 'second_bag_fee': 45, 'third_bag_fee': 150, 'fourth_plus_fee': 200, 'currency': 'USD'}
        - For 'size_limits': {'carry_on_max': '22x14x9', 'personal_item_max': '18x14x8', 'checked_max_inches': 62, 'weight_limits': [50, 70, 100]}
    
    Example output for focus='route_region':
        {
            'origin': 'Charlotte',
            'destination': 'Washington D.C.',
            'region_category': 'US_Domestic',
            'applicable_regions': ['U.S.', 'Puerto Rico', 'U.S. Virgin Islands'],
            'route_rules': {
                'first_bag_fee': 40,
                'second_bag_fee': 45,
                'third_bag_fee': 150,
                'fourth_plus_fee': 200
            }
        }
    
    Note: Extracts passenger class, route information, and identifies applicable fee structures based on region and class combinations.
    """

@interface
def apply_size_constraints(narrative: str, focus: str) -> str:
    """Analyzes baggage dimensions to determine if items can be carried on or must be checked due to oversize constraints.
    
    Args:
        narrative (str): The full problem statement containing passenger details, flight route, baggage dimensions, and weights.
        focus (str): Specific aspect to focus on (e.g., bag ID, oversize check, carry-on eligibility).
    
    Returns:
        dict: A dictionary containing:
            - 'oversize_bags': List of bag IDs that exceed maximum size constraints
            - 'oversize_details': List of dicts with detailed oversize analysis per bag
            - 'carry_on_eligible': List of bag IDs that meet carry-on size requirements
            - 'oversize_rules_cited': Specific size constraint rules applied
            
        Example output format:
            {
                "oversize_bags": [2, 3],
                "oversize_details": [
                    {
                        "bag_id": 2,
                        "dimensions": "44x29x20",
                        "total_inches": 93,
                        "oversize_limit": 62,
                        "oversize_reason": "Total dimensions exceed 62 inches"
                    },
                    {
                        "bag_id": 3, 
                        "dimensions": "45x23x19",
                        "total_inches": 87,
                        "oversize_limit": 62,
                        "oversize_reason": "Total dimensions exceed 62 inches"
                    }
                ],
                "carry_on_eligible": [1],
                "oversize_rules_cited": "Bags exceeding 62 total inches (L+W+H) must be checked"
            }
    
    Note: Uses standard oversize threshold of 62 total inches. Carry-on bags must also meet individual dimension limits (typically 22x14x9 inches).
    """

@interface
def checking_item_dimensions(narrative: str, focus: str) -> str:
    """Checking Item Dimensions.
    
    Analyze the narrative with the given focus.
    """
