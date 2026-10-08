from __future__ import annotations

from typing import Dict, List

import torch

from sentiment.model import (
    SENTIMENT_LABELS,
)


# ---------------------------------------------------------
# Weighted Aggregation
# ---------------------------------------------------------

def aggregate_chunk_sentiments(
    chunk_probabilities: List[torch.Tensor],
    chunk_weights: List[int],
) -> Dict[str, float]:
    """
    Aggregate chunk sentiment scores
    using token-length weighting.

    Parameters
    -----------------------------------------
    chunk_probabilities:
        List of probability tensors

    chunk_weights:
        Corresponding token counts
    -----------------------------------------
    """

    if (
        not chunk_probabilities
        or not chunk_weights
    ):

        return {
            "negative": 0.0,
            "neutral": 1.0,
            "positive": 0.0,
        }

    if (
        len(chunk_probabilities)
        != len(chunk_weights)
    ):

        raise ValueError(
            "Mismatch between "
            "probabilities and weights."
        )

    weighted_sum = None

    total_weight = 0

    # ---------------------------------------------------------
    # Weighted accumulation
    # ---------------------------------------------------------

    for probs, weight in zip(
        chunk_probabilities,
        chunk_weights,
    ):

        weighted_probs = (
            probs * weight
        )

        if weighted_sum is None:

            weighted_sum = (
                weighted_probs
            )

        else:

            weighted_sum += (
                weighted_probs
            )

        total_weight += weight

    # ---------------------------------------------------------
    # Safety fallback
    # ---------------------------------------------------------

    if (
        weighted_sum is None
        or total_weight == 0
    ):

        return {
            "negative": 0.0,
            "neutral": 1.0,
            "positive": 0.0,
        }

    # ---------------------------------------------------------
    # Normalize
    # ---------------------------------------------------------

    averaged = (
        weighted_sum
        / total_weight
    ).tolist()

    return {
        label:
        float(score)
        for label, score
        in zip(
            SENTIMENT_LABELS,
            averaged,
        )
    }


# ---------------------------------------------------------
# Dominant Sentiment
# ---------------------------------------------------------

def dominant_sentiment(
    scores: Dict[str, float],
) -> str:
    """
    Return dominant sentiment label.
    """

    if not scores:

        return "neutral"

    return max(
        scores,
        key=scores.get,
    )


# ---------------------------------------------------------
# Confidence Score
# ---------------------------------------------------------

def sentiment_confidence(
    scores: Dict[str, float],
) -> float:
    """
    Return confidence of dominant sentiment.
    """

    if not scores:

        return 0.0

    return float(
        max(scores.values())
    )