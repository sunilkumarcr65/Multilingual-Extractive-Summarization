# extractive/mmr.py

from __future__ import annotations

from typing import List, Tuple, Optional

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


# ---------------------------------------------------------
# Document-Specific MMR Settings
# ---------------------------------------------------------

# Lower lambda:
# - less aggressive redundancy removal
# - preserves more sections
# - better for resumes

# Higher lambda:
# - more relevance-focused
# - stronger redundancy filtering
# - better for news

MMR_LAMBDA_BY_DOCTYPE = {

    "resume": 0.4,

    "news": 0.7,

    "general": 0.6,

    "research_paper": 0.5,
}


# ---------------------------------------------------------
# Lambda Router
# ---------------------------------------------------------

def get_mmr_lambda(
    document_type: Optional[str],
) -> float:
    """
    Get optimal MMR lambda
    based on document type.
    """

    document_type = (
        document_type
        or "general"
    ).strip().lower()

    return MMR_LAMBDA_BY_DOCTYPE.get(
        document_type,
        0.6,
    )


# ---------------------------------------------------------
# Maximal Marginal Relevance
# ---------------------------------------------------------

def maximal_marginal_relevance(
    embeddings: np.ndarray,
    ranked_sentences: List[
        Tuple[int, float]
    ],
    top_k: int,
    lambda_param: float = 0.7,
    document_type: Optional[str] = None,
) -> List[int]:
    """
    Maximal Marginal Relevance (MMR).

    Balances:
    -----------------------------------------
    - relevance to document
    - diversity among selected sentences
    -----------------------------------------

    Helps reduce:
    - repetitive summaries
    - duplicate information
    - topic domination

    Parameters
    ----------
    embeddings:
        Sentence embeddings matrix.

    ranked_sentences:
        List of:
        (sentence_index, relevance_score)

    top_k:
        Number of sentences to select.

    lambda_param:
        Controls relevance vs diversity.

    document_type:
        Optional document-aware override.

    Notes
    -----
    Lower lambda:
        more diversity

    Higher lambda:
        more relevance
    """

    # -----------------------------------------------------
    # Empty guard
    # -----------------------------------------------------

    if not ranked_sentences:

        return []

    # -----------------------------------------------------
    # Document-aware override
    # -----------------------------------------------------

    if document_type:

        lambda_param = get_mmr_lambda(
            document_type
        )

    # -----------------------------------------------------
    # Safety
    # -----------------------------------------------------

    top_k = min(
        top_k,
        len(ranked_sentences),
    )

    # -----------------------------------------------------
    # Initial selection
    # -----------------------------------------------------

    selected = []

    candidate_indices = [

        idx
        for idx, _
        in ranked_sentences
    ]

    relevance_scores = {

        idx: score
        for idx, score
        in ranked_sentences
    }

    # -----------------------------------------------------
    # Select highest relevance first
    # -----------------------------------------------------

    first_sentence = (
        ranked_sentences[0][0]
    )

    selected.append(
        first_sentence
    )

    candidate_indices.remove(
        first_sentence
    )

    # -----------------------------------------------------
    # Iterative MMR Selection
    # -----------------------------------------------------

    while (

        len(selected) < top_k
        and candidate_indices

    ):

        mmr_scores = []

        for candidate_idx in (
            candidate_indices
        ):

            # -------------------------------------------------
            # Relevance
            # -------------------------------------------------

            relevance = (
                relevance_scores[
                    candidate_idx
                ]
            )

            # -------------------------------------------------
            # Candidate embedding
            # -------------------------------------------------

            candidate_embedding = (
                embeddings[
                    candidate_idx
                ]
                .reshape(1, -1)
            )

            # -------------------------------------------------
            # Similarity against selected
            # -------------------------------------------------

            similarities = []

            for selected_idx in (
                selected
            ):

                selected_embedding = (

                    embeddings[
                        selected_idx
                    ]
                    .reshape(1, -1)
                )

                similarity = (
                    cosine_similarity(

                        candidate_embedding,
                        selected_embedding,

                    )[0][0]
                )

                similarities.append(
                    similarity
                )

            max_similarity = (

                max(similarities)
                if similarities
                else 0.0
            )

            # -------------------------------------------------
            # MMR Formula
            # -------------------------------------------------

            mmr_score = (

                lambda_param
                * relevance

                - (1 - lambda_param)
                * max_similarity
            )

            mmr_scores.append(

                (
                    candidate_idx,
                    mmr_score,
                )
            )

        # -----------------------------------------------------
        # Select best candidate
        # -----------------------------------------------------

        best_candidate = max(

            mmr_scores,

            key=lambda x: x[1],

        )[0]

        selected.append(
            best_candidate
        )

        candidate_indices.remove(
            best_candidate
        )

    # -----------------------------------------------------
    # Preserve original order
    # -----------------------------------------------------

    return selected