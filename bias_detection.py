import re

BIAS_TERMS = {
    "absolutist": ["always", "never", "must", "only", "proven", "guaranteed"],
    "sensational": ["breakthrough", "miracle", "revolutionary", "cure-all"],
    "unsupported claims": ["scientist agree", "studies prove", "experts confirm"]
}

def detect_lexical_bias(response):
    """
    Identifies absolute and sensationalist claims in responses.
    """
    bias_report = {}

    for category, terms in BIAS_TERMS.items():
        found_terms = [term for term in terms if re.search(r'\b' + term + r'\b', response, re.IGNORECASE)]
        if found_terms:
            bias_report[category] = found_terms

    return bias_report if bias_report else "No major lexical bias detected"

