from __future__ import annotations

from typing import List, Tuple

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def compute_document_centroid(
    embeddings: np.ndarray,
) -> np.ndarray:
    """
    Compute document centroid embedding.

    The centroid represents the
    overall semantic meaning
    of the document.
    """

    return np.mean(
        embeddings,
        axis=0,
    )


def compute_centrality_scores(
    embeddings: np.ndarray,
) -> np.ndarray:
    """
    Compute sentence centrality.

    Sentences that are semantically
    similar to many other sentences
    receive higher scores.
    """

    similarity_matrix = cosine_similarity(
        embeddings
    )

    centrality_scores = similarity_matrix.mean(
        axis=1
    )

    return centrality_scores


def compute_relevance_scores(
    embeddings: np.ndarray,
    centroid: np.ndarray,
) -> np.ndarray:
    """
    Compute relevance of each sentence
    to overall document meaning.
    """

    relevance_scores = cosine_similarity(
        embeddings,
        centroid.reshape(1, -1),
    ).flatten()

    return relevance_scores


def normalize_scores(
    scores: np.ndarray,
) -> np.ndarray:
    """
    Min-max normalize scores.
    """

    min_score = scores.min()
    max_score = scores.max()

    if max_score - min_score == 0:
        return np.ones_like(scores)

    normalized = (
        scores - min_score
    ) / (
        max_score - min_score
    )

    return normalized


def rank_sentences(
    sentences: List[str],
    embeddings: np.ndarray,
    centrality_weight: float = 0.4,
    relevance_weight: float = 0.6,
    document_type: str = "general",
) -> List[Tuple[int, float]]:
    """
    Rank sentences semantically.

    Combines:
    -----------------------------------------
    - document relevance
    - semantic centrality
    -----------------------------------------

    Higher score =>
    more important sentence.
    """

    if not sentences:
        return []

    if len(sentences) != len(embeddings):
        raise ValueError(
            "Sentence count and embedding "
            "count must match."
        )

    # ---------------------------------------------------------
    # Document centroid
    # ---------------------------------------------------------

    centroid = compute_document_centroid(
        embeddings
    )

    # ---------------------------------------------------------
    # Semantic relevance
    # ---------------------------------------------------------

    relevance_scores = compute_relevance_scores(
        embeddings,
        centroid,
    )

    # ---------------------------------------------------------
    # Sentence centrality
    # ---------------------------------------------------------

    centrality_scores = (
        compute_centrality_scores(
            embeddings
        )
    )

    # ---------------------------------------------------------
    # Normalize
    # ---------------------------------------------------------

    relevance_scores = normalize_scores(
        relevance_scores
    )

    centrality_scores = normalize_scores(
        centrality_scores
    )

    if document_type == "resume":
        centrality_weight = 0.25
        relevance_weight = 0.75
    # ---------------------------------------------------------
    # Final weighted score
    # ---------------------------------------------------------

    final_scores = (
        relevance_weight * relevance_scores
        + centrality_weight
        * centrality_scores
    )

    # ---------------------------------------------------------
    # Rank sentences
    # ---------------------------------------------------------

    ranked = list(
        enumerate(final_scores)
    )

    ranked.sort(
        key=lambda x: x[1],
        reverse=True,
    )

    return ranked