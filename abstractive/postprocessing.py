from __future__ import annotations

import re
from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

try:
    import nltk

    nltk.data.find("tokenizers/punkt")
except LookupError:
    import nltk

    nltk.download("punkt", quiet=True)

from nltk.tokenize import sent_tokenize


class SummaryPostProcessor:
    """
    Handles:
    - duplicate removal
    - text cleanup
    - hallucination filtering
    - factual relevance filtering

    This is VERY important for reducing:
    - hallucinations
    - random unrelated sentences
    - repetitive summaries
    """

    def __init__(
        self,
        similarity_threshold: float = 0.45,
        embedding_model: str = (
            "sentence-transformers/"
            "paraphrase-multilingual-MiniLM-L12-v2"
        ),
    ):
        self.similarity_threshold = similarity_threshold

        print(
            f"[POSTPROCESS] Loading embeddings: "
            f"{embedding_model}"
        )

        self.embedder = SentenceTransformer(
            embedding_model
        )

    # ---------------------------------------------------------
    # Sentence Utilities
    # ---------------------------------------------------------

    @staticmethod
    def split_sentences(text: str) -> List[str]:

        if not text.strip():
            return []

        return [
            s.strip()
            for s in sent_tokenize(text)
            if s.strip()
        ]

    # ---------------------------------------------------------
    # Deduplication
    # ---------------------------------------------------------

    @staticmethod
    def deduplicate(
        text: str,
    ) -> str:
        """
        Remove repeated sentences.
        """

        sentences = SummaryPostProcessor.split_sentences(
            text
        )

        seen = set()
        unique_sentences = []

        for sentence in sentences:

            normalized = re.sub(
                r"\s+",
                " ",
                sentence.lower().strip(),
            )

            if normalized not in seen:
                seen.add(normalized)
                unique_sentences.append(sentence)

        return " ".join(unique_sentences)

    # ---------------------------------------------------------
    # Cleanup
    # ---------------------------------------------------------

    @staticmethod
    def clean(
        text: str,
    ) -> str:
        """
        Basic punctuation and spacing cleanup.
        """

        text = re.sub(
            r"\s+([.,;:!?])",
            r"\1",
            text,
        )

        text = re.sub(
            r"\s{2,}",
            " ",
            text,
        ).strip()

        if text and text[-1] not in ".!?":
            text += "."

        if text:
            text = text[0].upper() + text[1:]

        return text

    # ---------------------------------------------------------
    # Embeddings
    # ---------------------------------------------------------

    def embed(
        self,
        texts: List[str],
    ) -> np.ndarray:

        return self.embedder.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

    # ---------------------------------------------------------
    # Hallucination Filtering
    # ---------------------------------------------------------

    def filter_hallucinations(
        self,
        summary: str,
        source_text: str,
    ) -> str:
        """
        Remove sentences weakly related to source text.

        Helps remove:
        - hallucinated facts
        - random unrelated generations
        - decoder drift
        """

        summary_sentences = self.split_sentences(
            summary
        )

        if not summary_sentences:
            return ""

        source_embedding = self.embed(
            [source_text]
        )[0]

        kept_sentences = []

        sentence_embeddings = self.embed(
            summary_sentences
        )

        for sentence, sentence_embedding in zip(
            summary_sentences,
            sentence_embeddings,
        ):

            similarity = cosine_similarity(
                source_embedding.reshape(1, -1),
                sentence_embedding.reshape(1, -1),
            )[0][0]

            if similarity >= self.similarity_threshold:
                kept_sentences.append(sentence)

        return " ".join(kept_sentences)

    # ---------------------------------------------------------
    # Main Pipeline
    # ---------------------------------------------------------

    def process(
        self,
        summary: str,
        source_text: str,
    ) -> str:
        """
        Full postprocessing pipeline.
        """

        if not summary.strip():
            return ""

        # -----------------------------------------------------
        # Fast path for short summaries
        # Skip expensive embedding similarity checks
        # -----------------------------------------------------

        if len(summary.split()) < 80:

            summary = self.deduplicate(
                summary
            )

            summary = self.clean(
                summary
            )

            return summary

        # -----------------------------------------------------
        # Standard pipeline
        # -----------------------------------------------------

        summary = self.deduplicate(
            summary
        )

        summary = self.filter_hallucinations(
            summary,
            source_text,
        )

        summary = self.clean(
            summary
        )

        return summary