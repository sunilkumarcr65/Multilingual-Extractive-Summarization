from __future__ import annotations

import re
from typing import Dict


TECHNICAL_DOCUMENT_HINTS = {
    "api",
    "model",
    "pipeline",
    "system",
    "architecture",
    "framework",
    "algorithm",
    "resume",
    "education",
    "skills",
}

# ---------------------------------------------------------
# Positive Keywords
# ---------------------------------------------------------

POSITIVE_WORDS = frozenset({
    "good",
    "great",
    "excellent",
    "amazing",
    "outstanding",
    "wonderful",
    "fantastic",
    "positive",
    "happy",
    "success",
    "successful",
    "best",
    "love",
    "perfect",
    "brilliant",
    "superb",
    "well",
    "impressive",
    "effective",
    "helpful",
    "efficient",
    "improved",
    "improvement",
    "beneficial",
})


# ---------------------------------------------------------
# Negative Keywords
# ---------------------------------------------------------

NEGATIVE_WORDS = frozenset({
    "bad",
    "poor",
    "fail",
    "failure",
    "problem",
    "issue",
    "wrong",
    "error",
    "negative",
    "unfortunately",
    "terrible",
    "awful",
    "horrible",
    "worst",
    "useless",
    "broken",
    "disappointing",
    "difficult",
    "slow",
    "expensive",
    "bug",
    "bugs",
    "critical",
    "weak",
})


# ---------------------------------------------------------
# Tokenization
# ---------------------------------------------------------

def tokenize_text(
    text: str,
):
    """
    Lightweight fallback tokenizer.
    """

    cleaned = re.sub(
        r"[^\w\s]",
        " ",
        text.lower(),
    )

    return cleaned.split()


# ---------------------------------------------------------
# Fallback Sentiment
# ---------------------------------------------------------

def fallback_sentiment(
    text: str,
) -> Dict[str, float]:
    """
    Rule-based fallback sentiment.

    Used when:
    - transformers unavailable
    - model loading fails
    - offline fallback needed
    """

    words = tokenize_text(
        text
    )

    if not words:

        return {
            "negative": 0.0,
            "neutral": 1.0,
            "positive": 0.0,
        }

    positive_hits = sum(
        1
        for word in words
        if word in POSITIVE_WORDS
    )

    negative_hits = sum(
        1
        for word in words
        if word in NEGATIVE_WORDS
    )

    technical_hits = sum(
        1
        for word in words
        if word in TECHNICAL_DOCUMENT_HINTS
    )

    if technical_hits >= 3:

        return {
            "negative": 0.05,
            "neutral": 0.85,
            "positive": 0.10,
        }

    total_hits = (
        positive_hits
        + negative_hits
    )

    # ---------------------------------------------------------
    # Neutral fallback
    # ---------------------------------------------------------

    if total_hits == 0:

        return {
            "negative": 0.0,
            "neutral": 1.0,
            "positive": 0.0,
        }

    positive_score = (
        positive_hits
        / total_hits
    )

    negative_score = (
        negative_hits
        / total_hits
    )

    neutral_score = max(
        0.0,
        1.0
        - (
            total_hits
            / max(len(words), 1)
        ),
    )

    # ---------------------------------------------------------
    # Normalize
    # ---------------------------------------------------------

    total = (
        positive_score
        + negative_score
        + neutral_score
    )

    if total == 0:

        return {
            "negative": 0.0,
            "neutral": 1.0,
            "positive": 0.0,
        }

    return {
        "negative":
            round(
                negative_score / total,
                4,
            ),

        "neutral":
            round(
                neutral_score / total,
                4,
            ),

        "positive":
            round(
                positive_score / total,
                4,
            ),
    }