from __future__ import annotations

from typing import List, Tuple

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def remove_redundancy(
    selected_sentences: List[Tuple[int, str]],
    embeddings: np.ndarray,
    similarity_threshold: float = 0.85,
    document_type: str = "general",
) -> List[Tuple[int, str]]:
    """
    Remove semantically redundant sentences.

    Helps eliminate:
    -----------------------------------------
    - repeated information
    - near-duplicate sentences
    - overlapping semantic content
    -----------------------------------------

    Parameters
    ----------
    selected_sentences:
        List of:
        (sentence_index, sentence_text)

    embeddings:
        Full sentence embedding matrix.

    similarity_threshold:
        Higher value:
            allows more similarity

        Lower value:
            stricter redundancy removal
    """

    if not selected_sentences:
        return []

    filtered_sentences = []

    kept_indices = []

    if document_type == "resume":
        similarity_threshold = 0.80

    for sentence_idx, sentence_text in selected_sentences:

        current_embedding = embeddings[
            sentence_idx
        ].reshape(1, -1)

        is_redundant = False

        for kept_idx in kept_indices:

            kept_embedding = embeddings[
                kept_idx
            ].reshape(1, -1)

            similarity = cosine_similarity(
                current_embedding,
                kept_embedding,
            )[0][0]

            if similarity >= similarity_threshold:

                is_redundant = True
                break

        if not is_redundant:

            filtered_sentences.append(
                (
                    sentence_idx,
                    sentence_text,
                )
            )

            kept_indices.append(
                sentence_idx
            )

    # ---------------------------------------------------------
    # Preserve original document order
    # ---------------------------------------------------------

    filtered_sentences.sort(
        key=lambda x: x[0]
    )

    return filtered_sentences