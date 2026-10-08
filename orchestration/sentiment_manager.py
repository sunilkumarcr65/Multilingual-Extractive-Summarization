# orchestration/sentiment_manager.py

from __future__ import annotations

from typing import Any, Dict, List

from sentiment.analyzer import (
    analyze_sentiment,
)

from document_processing.document_classifier import (
    classify_document,
)


# ---------------------------------------------------------
# Default Sentiment
# ---------------------------------------------------------

DEFAULT_SENTIMENT = {
    "positive": 0.0,
    "neutral": 1.0,
    "negative": 0.0,
}


# ---------------------------------------------------------
# Safe Sentiment
# ---------------------------------------------------------

def safe_sentiment_analysis(
    text: str,
) -> Dict[str, float]:
    """
    Safe wrapper around sentiment analysis.

    Expected analyzer output:
    {
        "positive": 0.73,
        "neutral": 0.20,
        "negative": 0.07
    }
    """

    try:

        text = (
            text or ""
        ).strip()

        if not text:

            return dict(
                DEFAULT_SENTIMENT
            )

        # -------------------------------------------------
        # Truncate overly large text
        # -------------------------------------------------

        result = analyze_sentiment(
            " ".join(
                text.split()[:120]
            )
        )

        # -------------------------------------------------
        # Empty result fallback
        # -------------------------------------------------

        if not result:

            return dict(
                DEFAULT_SENTIMENT
            )

        # -------------------------------------------------
        # HuggingFace pipelines often return:
        # [{"label": "POSITIVE", "score": 0.98}]
        # -------------------------------------------------

        if (
            isinstance(result, list)
            and result
        ):

            result = result[0]

        # -------------------------------------------------
        # HF-style format
        # -------------------------------------------------

        if (
            isinstance(result, dict)
            and "label" in result
            and "score" in result
        ):

            label = (
                str(
                    result.get(
                        "label",
                        "neutral",
                    )
                )
                .strip()
                .lower()
            )

            score = float(
                result.get(
                    "score",
                    0.0,
                )
            )

            if label == "positive":

                return {
                    "positive": score,
                    "neutral": 1 - score,
                    "negative": 0.0,
                }

            if label == "negative":

                return {
                    "positive": 0.0,
                    "neutral": 1 - score,
                    "negative": score,
                }

            return {
                "positive": 0.0,
                "neutral": score,
                "negative": 0.0,
            }

        # -------------------------------------------------
        # Standard probability format
        # -------------------------------------------------

        return {

            "positive":
                float(
                    result.get(
                        "positive",
                        0.0,
                    )
                ),

            "neutral":
                float(
                    result.get(
                        "neutral",
                        1.0,
                    )
                ),

            "negative":
                float(
                    result.get(
                        "negative",
                        0.0,
                    )
                ),
        }

    except Exception as error:

        print(
            "[SENTIMENT] "
            f"Analysis failed: "
            f"{error}"
        )

        return dict(
            DEFAULT_SENTIMENT
        )


# ---------------------------------------------------------
# Section Sentiment
# ---------------------------------------------------------

def analyze_section_sentiment(
    summary_text: str,
) -> Dict[str, float]:
    """
    Analyze sentiment of one section.
    """

    summary_text = (
        summary_text or ""
    ).strip()

    if not summary_text:

        return dict(
            DEFAULT_SENTIMENT
        )

    return safe_sentiment_analysis(
        summary_text
    )


# ---------------------------------------------------------
# Aggregation
# ---------------------------------------------------------

def aggregate_sentiments(
    sections: List[Dict[str, Any]],
) -> Dict[str, float]:
    """
    Aggregate section sentiments.
    """

    valid_sentiments = []

    for section in sections:

        sentiment = section.get(
            "sentiment"
        )

        if not sentiment:
            continue

        valid_sentiments.append(
            sentiment
        )

    if not valid_sentiments:

        return dict(
            DEFAULT_SENTIMENT
        )

    aggregated = {}

    for label in (
        "positive",
        "neutral",
        "negative",
    ):

        values = [

            sentiment.get(
                label,
                0.0,
            )

            for sentiment
            in valid_sentiments
        ]

        aggregated[label] = round(

            sum(values)
            / max(len(values), 1),

            4,
        )

    return aggregated


# ---------------------------------------------------------
# Dominant Sentiment
# ---------------------------------------------------------

def dominant_sentiment(
    sentiment: Dict[str, float],
) -> str:
    """
    Get dominant sentiment label.
    """

    if not sentiment:

        return "neutral"

    return max(
        sentiment,
        key=sentiment.get,
    )


# ---------------------------------------------------------
# Sentiment Label Formatter
# ---------------------------------------------------------

def sentiment_label(
    sentiment: Dict[str, float],
) -> str:
    """
    Human-readable sentiment label.
    """

    dominant = dominant_sentiment(
        sentiment
    )

    confidence = round(

        sentiment.get(
            dominant,
            0.0,
        ),

        2,
    )

    return (
        f"{dominant} "
        f"({confidence:.2f})"
    )


# ---------------------------------------------------------
# Main Sentiment Manager
# ---------------------------------------------------------

def analyze_document_sentiment(
    text: str,
) -> dict:
    """
    Main document-level sentiment analysis.

    Returns UI-compatible format:
    {
        "label": "positive",
        "score": 0.98
    }
    """

    try:

        text = (
            text or ""
        ).strip()

        if not text:

            return {
                "label": "neutral",
                "score": 0.0,
            }

        document_type = classify_document(
            text
        )

        # -------------------------------------------------
        # Sentiment is meaningless for
        # resumes/research papers
        # -------------------------------------------------

        if document_type in {

            "resume",

            "research_paper",
        }:

            return {

                "label":
                    "informational",

                "score":
                    1.0,
            }

        # -------------------------------------------------
        # Get normalized probabilities
        # -------------------------------------------------

        sentiment_scores = (
            safe_sentiment_analysis(
                text[:512]
            )
        )

        # -------------------------------------------------
        # Determine dominant label
        # -------------------------------------------------

        label = dominant_sentiment(
            sentiment_scores
        )

        score = float(

            sentiment_scores.get(
                label,
                0.0,
            )
        )

        return {

            "label":
                label,

            "score":
                round(score, 4),
        }

    except Exception as error:

        print(
            "[SENTIMENT ERROR]",
            error,
        )

        return {
            "label": "unknown",
            "score": 0.0,
        }