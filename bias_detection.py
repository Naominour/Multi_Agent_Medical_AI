from textblob import TextBlob
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

def analyse_sentiment(response):
    """
    Detects sentiment in the medical response. Overly positive or negative tones can indicate bias.
    """
    analysis = TextBlob(response)
    sentiment_score = analysis.sentiment.polarity # -1 (negative) to 1 (positive)

    if sentiment_score < -0.3:
        return f"Potential negative bias detected (Score: {sentiment_score})"
    elif sentiment_score > 0.3:
        return f"Potential positive bias detected (Score: {sentiment_score})"
    else:
        return "Neutral sentiment (No major bias detected)"
    
    