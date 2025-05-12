# utils/trust_utils.py

import re
from textblob import TextBlob

# Define keyword lists here so they’re shared across the project
BIAS_KEYWORDS = ["shocking", "explosive", "disaster", "hoax", "alert", "cover-up"]
TOXIC_WORDS = ["hate", "kill", "stupid", "traitor"]

def score_article(text: str) -> dict:
    """
    Scores a news article on bias, toxicity, sentiment, and subjectivity.
    Returns a dict with Trust Score out of 100 plus breakdown metrics.
    """
    # Compute sentiment & subjectivity
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity       # [-1, 1]
    subjectivity = blob.sentiment.subjectivity  # [0, 1]

    # Count whole-word hits (use regex word boundaries)
    lowered = text.lower()
    bias_hits = sum(bool(re.search(rf"\b{kw}\b", lowered)) for kw in BIAS_KEYWORDS)
    toxic_hits = sum(bool(re.search(rf"\b{kw}\b", lowered)) for kw in TOXIC_WORDS)

    # Start from 100 and penalize
    score = 100
    score -= subjectivity * 30         # up to –30
    score -= abs(polarity) * 20        # up to –20
    score -= bias_hits * 5             # 5 points per bias keyword
    score -= toxic_hits * 5            # 5 points per toxic word
    score = max(0, min(100, round(score, 1)))

    return {
        "Trust Score": score,
        "Bias Keyword Hits": bias_hits,
        "Toxic Word Hits": toxic_hits,
        "Sentiment Polarity": round(polarity, 2),
        "Subjectivity": round(subjectivity, 2)
    }
