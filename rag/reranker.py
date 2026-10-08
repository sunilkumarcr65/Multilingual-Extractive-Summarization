from __future__ import annotations

from functools import lru_cache
from typing import Dict, List

import torch
from sentence_transformers import (
    CrossEncoder,
)


# ---------------------------------------------------------
# Device
# ---------------------------------------------------------

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ---------------------------------------------------------
# Model Config
# ---------------------------------------------------------

DEFAULT_RERANK_MODEL = (
    "cross-encoder/"
    "ms-marco-MiniLM-L-6-v2"
)


# ---------------------------------------------------------
# Cross Encoder Loader
# ---------------------------------------------------------

@lru_cache(maxsize=1)
def get_reranker_model() -> CrossEncoder:
    """
    Load cross-encoder reranker.

    Much more accurate than pure
    embedding similarity retrieval.
    """

    print(
        "[RERANKER] "
        "Loading cross-encoder..."
    )

    return CrossEncoder(
        DEFAULT_RERANK_MODEL,
        device=DEVICE,
    )


# ---------------------------------------------------------
# Reranker
# ---------------------------------------------------------

class CrossEncoderReranker:
    """
    Cross-encoder reranker.

    Pipeline
    -----------------------------------------
    query + chunk
        ↓
    cross-encoder relevance scoring
        ↓
    reranked retrieval results
    -----------------------------------------

    Improves:
    - QA quality
    - retrieval precision
    - semantic grounding
    """

    def __init__(
        self,
        top_k: int = 5,
    ):
        self.top_k = top_k

        self.model = (
            get_reranker_model()
        )

    # ---------------------------------------------------------
    # Main Reranking
    # ---------------------------------------------------------

    def rerank(
        self,
        query: str,
        retrieved_chunks: List[Dict],
    ) -> List[Dict]:
        """
        Rerank retrieved chunks
        using cross-encoder relevance.
        """

        if not retrieved_chunks:
            return []

        if len(retrieved_chunks) <= 2:
            return retrieved_chunks

        # -----------------------------------------------------
        # Query-document pairs
        # -----------------------------------------------------

        pairs = [
            (
                query,
                chunk["text"],
            )
            for chunk in retrieved_chunks
        ]

        # -----------------------------------------------------
        # Predict relevance
        # -----------------------------------------------------

        scores = self.model.predict(
            pairs,
            batch_size=4,
            show_progress_bar=False,
        )

        # -----------------------------------------------------
        # Attach scores
        # -----------------------------------------------------

        reranked = []

        for chunk, score in zip(
            retrieved_chunks,
            scores,
        ):

            updated = dict(chunk)

            updated[
                "reranker_score"
            ] = float(score)

            reranked.append(updated)

        # -----------------------------------------------------
        # Sort descending
        # -----------------------------------------------------

        reranked.sort(
            key=lambda item:
            item["reranker_score"],
            reverse=True,
        )

        return reranked[
            : self.top_k
        ]

    # ---------------------------------------------------------
    # Context Builder
    # ---------------------------------------------------------

    def build_reranked_context(
        self,
        query: str,
        retrieved_chunks: List[Dict],
    ) -> str:
        """
        Build generation context
        from reranked chunks.
        """

        reranked = self.rerank(
            query=query,
            retrieved_chunks=(
                retrieved_chunks
            ),
        )

        if not reranked:
            return ""

        sections = []

        for idx, chunk in enumerate(
            reranked,
            start=1,
        ):

            text = (
                chunk["text"]
                .strip()
            )

            if not text:
                continue

            section = (
                f"[Context {idx}]\n"
                f"{text}"
            )

            sections.append(
                section
            )

        return "\n\n".join(
            sections
        )