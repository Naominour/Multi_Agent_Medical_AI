import re
from textblob import TextBlob

BIAS_TERMS = {
    "absolutist": ["always", "never", "must", "only", "proven", "guaranteed"],
    "sensational": ["breakthrough", "miracle", "revolutionary", "cure-all"]
}

def detect_lexical_bias(response):
    """Identifies absolute and sensationalist claims."""
    bias_report = {}
    for category, terms in BIAS_TERMS.items():
        found_terms = [term for term in terms if re.search(r'\b' + term + r'\b', response, re.IGNORECASE)]
        if found_terms:
            bias_report[category] = found_terms
    return bias_report if bias_report else "No major lexical bias detected"

def analyse_sentiment(response):
    """Detects sentiment bias in the response."""
    analysis = TextBlob(response)
    sentiment_score = analysis.sentiment.polarity
    if sentiment_score < -0.3:
        return f"Potential negative bias detected (Score: {sentiment_score})"
    elif sentiment_score > 0.3:
        return f"Potential positive bias detected (Score: {sentiment_score})"
    return "Neutral sentiment (No major bias detected)"
