from __future__ import annotations

from typing import Dict, List, Optional

from rag.vector_store import (
    FAISSVectorStore,
)


class SemanticRetriever:
    """
    Semantic chunk retriever.

    Features
    -----------------------------------------
    - vector similarity retrieval
    - score filtering
    - deduplication
    - section-aware retrieval
    -----------------------------------------
    """

    def __init__(
        self,
        vector_store: FAISSVectorStore,
    ):
        self.vector_store = (
            vector_store
        )

    # ---------------------------------------------------------
    # Main Retrieval
    # ---------------------------------------------------------

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.18,
        language: Optional[str] = None,
        deduplicate: bool = True,
    ) -> List[Dict]:
        """
        Retrieve semantically relevant chunks.
        """

        results = (
            self.vector_store.search(
                query=query,
                top_k=top_k * 2,
                language=language,
            )
        )

        # -----------------------------------------------------
        # Score filtering
        # -----------------------------------------------------

        filtered = [
            result
            for result in results
            if result["score"]
            >= min_score
        ]

        # Fallback: if strict threshold removes everything,
        # still return best available results instead of empty context.
        if not filtered and results:
            filtered = results[:top_k]

        # -----------------------------------------------------
        # Deduplication
        # -----------------------------------------------------

        if deduplicate:

            filtered = (
                self._deduplicate_results(
                    filtered
                )
            )

        return filtered[:top_k]

    # ---------------------------------------------------------
    # Context Builder
    # ---------------------------------------------------------

    def build_context(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.18,
        language: Optional[str] = None,
    ) -> str:
        """
        Build retrieval context for generation.
        """

        retrieved = self.retrieve(
            query=query,
            top_k=top_k,
            min_score=min_score,
            language=language,
        )

        if not retrieved:
            return ""

        context_chunks = []

        for idx, result in enumerate(
            retrieved,
            start=1,
        ):

            chunk = (
                result["text"]
                .strip()
            )

            if not chunk:
                continue

            context_chunks.append(
                f"[Chunk {idx}]\n"
                f"{chunk}"
            )

        return "\n\n".join(
            context_chunks
        )

    # ---------------------------------------------------------
    # Source Retrieval
    # ---------------------------------------------------------

    def retrieve_with_sources(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.18,
        language: Optional[str] = None,
    ) -> Dict:
        """
        Retrieve chunks + formatted metadata.
        """

        results = self.retrieve(
            query=query,
            top_k=top_k,
            min_score=min_score,
            language=language,
        )

        context = self.build_context(
            query=query,
            top_k=top_k,
            min_score=min_score,
            language=language,
        )

        return {
            "query":
                query,

            "context":
                context,

            "results":
                results,
        }

    # ---------------------------------------------------------
    # Deduplication
    # ---------------------------------------------------------

    @staticmethod
    def _deduplicate_results(
        results: List[Dict],
    ) -> List[Dict]:
        """
        Remove duplicate chunks.
        """

        unique = []

        seen_texts = set()

        for result in results:

            normalized = (
                result["text"]
                .strip()
                .lower()
            )

            if normalized in seen_texts:
                continue

            seen_texts.add(
                normalized
            )

            unique.append(result)

        return unique

    # ---------------------------------------------------------
    # Section Filter
    # ---------------------------------------------------------

    def retrieve_by_section(
        self,
        query: str,
        section: str,
        top_k: int = 5,
        language: Optional[str] = None,
    ) -> List[Dict]:
        """
        Retrieve only from a specific section.
        """

        results = self.retrieve(
            query=query,
            top_k=top_k * 3,
            language=language,
        )

        filtered = []

        target_section = (
            section.strip()
            .lower()
        )

        for result in results:

            result_section = (
                (
                    result.get(
                        "section"
                    )
                    or ""
                )
                .strip()
                .lower()
            )

            if (
                result_section
                == target_section
            ):

                filtered.append(
                    result
                )

        return filtered[:top_k]
