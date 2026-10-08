from __future__ import annotations

from typing import List, Tuple

import numpy as np

from sentence_transformers import (
    SentenceTransformer,
)

from document_processing.document_classifier import (
    classify_document,
)

try:
    import nltk

    nltk.data.find(
        "tokenizers/punkt"
    )

except LookupError:

    import nltk

    nltk.download(
        "punkt",
        quiet=True,
    )

from nltk.tokenize import (
    sent_tokenize,
)

from .ranking import (
    rank_sentences,
)

from .mmr import (
    maximal_marginal_relevance,
)

from .redundancy_filter import (
    remove_redundancy,
)


# ---------------------------------------------------------
# Resume Section Keywords
# ---------------------------------------------------------

RESUME_SECTION_GROUPS = {

    "education": [
        "education",
        "cgpa",
        "percentage",
        "degree",
        "b.tech",
        "bachelor",
        "diploma",
    ],

    "experience": [
        "experience",
        "work experience",
        "internship",
    ],

    "skills": [
        "skills",
        "technical skills",
    ],

    "projects": [
        "projects",
    ],

    "achievements": [
        "achievements",
        "leetcode",
        "codeforces",
        "codechef",
    ],
}


# ---------------------------------------------------------
# Resume Coverage Helper
# ---------------------------------------------------------

def ensure_resume_section_coverage(
    sentences: List[str],
    selected_indices: List[int],
    ranked_sentences: List[Tuple[int, float]],
) -> List[int]:
    """
    Ensure at least 1 sentence from each
    major resume section is included.

    Only adds the HIGHEST-RANKED sentence
    per missing section — not all matches.
    """

    covered = set(
        selected_indices
    )

    # Build rank lookup
    rank_score = {
        idx: score
        for idx, score
        in ranked_sentences
    }

    for section, keywords in (
        RESUME_SECTION_GROUPS.items()
    ):

        # Check if section already covered
        section_covered = any(

            any(
                kw in sentences[idx].lower()
                for kw in keywords
            )

            for idx in covered
            if idx < len(sentences)
        )

        if section_covered:
            continue

        # Find best-ranked sentence
        # for this section
        best_idx = None
        best_score = -1.0

        for idx, sentence in enumerate(
            sentences
        ):

            sentence_lower = (
                sentence.lower()
            )

            if any(
                kw in sentence_lower
                for kw in keywords
            ):

                score = rank_score.get(
                    idx, 0.0
                )

                if (
                    score > best_score
                    and idx not in covered
                ):
                    best_idx = idx
                    best_score = score

        if best_idx is not None:
            covered.add(best_idx)

    return list(covered)


# ---------------------------------------------------------
# Extractive Summarizer
# ---------------------------------------------------------

class ExtractiveSummarizer:
    """
    Modern semantic extractive summarizer.

    Pipeline
    -----------------------------------------------------
    sentence splitting
        ↓
    embeddings
        ↓
    semantic ranking
        ↓
    MMR selection
        ↓
    redundancy filtering
        ↓
    ordered summary
    -----------------------------------------------------
    """

    def __init__(
        self,
        embedding_model: str = (
            "sentence-transformers/"
            "paraphrase-multilingual-MiniLM-L12-v2"
        ),
    ):

        print(
            f"[EXTRACTIVE] "
            f"Loading embeddings: "
            f"{embedding_model}"
        )

        self.embedder = (
            SentenceTransformer(
                embedding_model
            )
        )

    # ---------------------------------------------------------
    # Sentence Splitting
    # ---------------------------------------------------------

    @staticmethod
    def split_sentences(
        text: str,
    ) -> List[str]:
        """
        Split document into sentences.
        """

        text = (
            text or ""
        ).strip()

        if not text:

            return []

        sentences = sent_tokenize(
            text
        )

        return [

            sentence.strip()

            for sentence
            in sentences

            if sentence.strip()
        ]

    # ---------------------------------------------------------
    # Embeddings
    # ---------------------------------------------------------

    def generate_embeddings(
        self,
        sentences: List[str],
    ) -> np.ndarray:
        """
        Generate semantic sentence embeddings.
        """

        if not sentences:

            return np.array([])

        embeddings = (
            self.embedder.encode(

                sentences,

                convert_to_numpy=True,

                normalize_embeddings=True,
            )
        )

        return embeddings

    # ---------------------------------------------------------
    # Sentence Selection
    # ---------------------------------------------------------

    def select_sentences(
        self,
        sentences: List[str],
        embeddings: np.ndarray,
        summary_ratio: float = 0.3,
        document_type: str = "general",
    ) -> List[Tuple[int, str]]:
        """
        Select best summary sentences.
        """

        if not sentences:

            return []

        # -----------------------------------------------------
        # Ranking
        # -----------------------------------------------------

        ranked_sentences = (
            rank_sentences(

                sentences=sentences,

                embeddings=embeddings,

                document_type=document_type,
            )
        )

        # -----------------------------------------------------
        # Summary size
        # -----------------------------------------------------

        total_sentences = len(
            sentences
        )

        # -----------------------------------------------------
        # Adaptive summary sizing
        # -----------------------------------------------------

        if total_sentences <= 15:

            summary_size = max(
                4,
                int(total_sentences * 0.60),
            )

        elif total_sentences <= 40:

            summary_size = max(
                8,
                int(total_sentences * summary_ratio),
            )

        else:

            summary_size = max(
                15,
                int(total_sentences * summary_ratio),
            )

        # -----------------------------------------------------
        # Resume boost
        # -----------------------------------------------------

        if document_type == "resume":

            summary_size = max(
                summary_size,
                min(3, total_sentences),
            )

        # -----------------------------------------------------
        # MMR Selection
        # -----------------------------------------------------

        selected_indices = (
            maximal_marginal_relevance(

                embeddings=embeddings,

                ranked_sentences=(
                    ranked_sentences
                ),

                top_k=summary_size,

                lambda_param=0.78,

                document_type=document_type,
            )
        )

        # -----------------------------------------------------
        # Resume coverage
        # -----------------------------------------------------

        if document_type == "resume":

            selected_indices = (
                ensure_resume_section_coverage(
                    sentences,
                    selected_indices,
                    ranked_sentences,
                )
            )

        # -----------------------------------------------------
        # Preserve MMR ordering
        # -----------------------------------------------------

        selected_indices = list(
            dict.fromkeys(
                selected_indices
            )
        )

        # -----------------------------------------------------
        # Preserve document order
        # -----------------------------------------------------

        selected_indices = sorted(
            selected_indices
        )

        selected_sentences = [

            (
                index,
                sentences[index],
            )

            for index
            in selected_indices
        ]

        # -----------------------------------------------------
        # Redundancy filtering
        # -----------------------------------------------------

        filtered_sentences = (
            remove_redundancy(

                selected_sentences,

                embeddings,

                document_type=document_type,
            )
        )

        return filtered_sentences

    # ---------------------------------------------------------
    # Main Pipeline
    # ---------------------------------------------------------

    def generate_extractive_summary(
        self,
        text: str,
        summary_ratio: float = 0.3,
    ) -> str:
        """
        Main extractive summarization pipeline.
        """

        text = (
            text or ""
        ).strip()

        if not text:

            return ""

        # -----------------------------------------------------
        # Document classification
        # -----------------------------------------------------

        document_type = classify_document(
            text
        )

        # -----------------------------------------------------
        # Sentence splitting
        # -----------------------------------------------------

        sentences = (
            self.split_sentences(
                text
            )
        )

        if not sentences:

            return ""

        # -----------------------------------------------------
        # Fast path for tiny documents
        # -----------------------------------------------------

        if len(sentences) <= 8:

            keep_count = min(
                max(4, len(sentences)),
                len(sentences),
            )

            return " ".join(
                sentences[:keep_count]
            )

        # -----------------------------------------------------
        # Embeddings
        # -----------------------------------------------------

        embeddings = (
            self.generate_embeddings(
                sentences
            )
        )

        if embeddings.size == 0:

            return ""

        # -----------------------------------------------------
        # Selection
        # -----------------------------------------------------

        selected_sentences = (
            self.select_sentences(

                sentences=sentences,

                embeddings=embeddings,

                summary_ratio=summary_ratio,

                document_type=document_type,
            )
        )

        if not selected_sentences:

            return ""

        # -----------------------------------------------------
        # Final summary
        # -----------------------------------------------------

        summary = "\n".join(

            sentence

            for _, sentence
            in selected_sentences
        )

        return summary.strip()

    # ---------------------------------------------------------
    # Batch Summarization
    # ---------------------------------------------------------

    def summarize_batch(
        self,
        texts: List[str],
        summary_ratio: float = 0.3,
    ) -> List[str]:
        """
        Batch summarization support.
        """

        return [

            self.generate_extractive_summary(

                text=text,

                summary_ratio=summary_ratio,
            )

            for text
            in texts
        ]


# ---------------------------------------------------------
# Global Cached Summarizer
# ---------------------------------------------------------

_default_summarizer = None


# ---------------------------------------------------------
# Public API
# ---------------------------------------------------------

def generate_extractive_summary(
    text: str,
    compression_ratio: float = 0.65,
):
    """
    Public extractive summarization API.
    """

    global _default_summarizer

    if _default_summarizer is None:

        _default_summarizer = (
            ExtractiveSummarizer()
        )

    return (
        _default_summarizer
        .generate_extractive_summary(

            text=text,

            summary_ratio=(
                compression_ratio
            ),
        )
    )
