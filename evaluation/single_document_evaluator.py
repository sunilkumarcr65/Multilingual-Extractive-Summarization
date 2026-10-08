from __future__ import annotations

from typing import Any, Dict, Optional

from abstractive.summarizer import (
    AbstractiveSummarizer,
)

from evaluation.hallucination_metrics import (
    evaluate_hallucination,
)

from evaluation.metrics import (
    compute_all_metrics,
)

from extractive.extractor import (
    ExtractiveSummarizer,
)

from document_processing.loaders import (
    load_document_text,
)


class SingleDocumentEvaluator:
    """
    Evaluate summarization quality
    for a single document.

    Supports:
    -----------------------------------------
    - extractive summarization
    - abstractive summarization
    - hybrid summarization
    - hallucination analysis
    - metric comparison
    -----------------------------------------

    Useful for:
    - debugging summaries
    - Streamlit preview
    - qualitative analysis
    - model comparison
    """

    def __init__(
        self,
        abstractive_model_name: str = (
            "csebuetnlp/mT5_multilingual_XLSum"
        ),
    ):
        print(
            "[SINGLE EVAL] Initializing..."
        )

        self.extractive_model = (
            ExtractiveSummarizer()
        )

        self.abstractive_model = (
            AbstractiveSummarizer(
                model_name=abstractive_model_name
            )
        )

    # ---------------------------------------------------------
    # Document Loading
    # ---------------------------------------------------------

    @staticmethod
    def load_document(
        file_path: str,
    ) -> str:
        """
        Extract document text.
        """

        text = load_document_text(
            file_path=file_path
        )

        return text.strip()

    # ---------------------------------------------------------
    # Summary Generation
    # ---------------------------------------------------------

    def generate_summaries(
        self,
        document_text: str,
    ) -> Dict[str, str]:
        """
        Generate all summary variants.
        """

        # -----------------------------------------------------
        # Extractive
        # -----------------------------------------------------

        extractive_summary = (
            self.extractive_model.summarize(
                document_text
            )
        )

        # -----------------------------------------------------
        # Abstractive
        # -----------------------------------------------------

        abstractive_summary = (
            self.abstractive_model.summarize(
                document_text
            )
        )

        # -----------------------------------------------------
        # Hybrid
        # -----------------------------------------------------

        hybrid_summary = (
            self.abstractive_model.summarize(
                extractive_summary
            )
        )

        return {
            "extractive":
                extractive_summary,

            "abstractive":
                abstractive_summary,

            "hybrid":
                hybrid_summary,
        }

    # ---------------------------------------------------------
    # Evaluation
    # ---------------------------------------------------------

    def evaluate(
        self,
        file_path: str,
        reference_summary: Optional[str] = None,
        language: str = "en",
    ) -> Dict[str, Any]:
        """
        Evaluate summaries for one document.

        Returns:
        -----------------------------------------
        summaries
        metrics
        hallucination_analysis
        metadata
        -----------------------------------------
        """

        # -----------------------------------------------------
        # Load document
        # -----------------------------------------------------

        document_text = self.load_document(
            file_path
        )

        if not document_text:

            raise ValueError(
                "Document text is empty."
            )

        # -----------------------------------------------------
        # Generate summaries
        # -----------------------------------------------------

        summaries = (
            self.generate_summaries(
                document_text
            )
        )

        # -----------------------------------------------------
        # Metric Evaluation
        # -----------------------------------------------------

        metrics = {}

        hallucination_results = {}

        for method, summary in (
            summaries.items()
        ):

            # -------------------------------------------------
            # Reference metrics
            # -------------------------------------------------

            if reference_summary:

                metric_scores = (
                    compute_all_metrics(
                        reference=(
                            reference_summary
                        ),
                        generated=summary,
                    )
                )

            else:

                metric_scores = {}

            metrics[method] = metric_scores

            # -------------------------------------------------
            # Hallucination analysis
            # -------------------------------------------------

            hallucination = (
                evaluate_hallucination(
                    source_text=document_text,
                    summary_text=summary,
                )
            )

            hallucination_results[
                method
            ] = {
                "semantic_similarity":
                    hallucination
                    .semantic_similarity,

                "grounded_sentence_ratio":
                    hallucination
                    .grounded_sentence_ratio,

                "hallucinated_sentence_ratio":
                    hallucination
                    .hallucinated_sentence_ratio,

                "entity_preservation_score":
                    hallucination
                    .entity_preservation_score,

                "unsupported_sentences":
                    hallucination
                    .unsupported_sentences,
            }

        # -----------------------------------------------------
        # Metadata
        # -----------------------------------------------------

        metadata = {
            "document_length_chars":
                len(document_text),

            "document_length_words":
                len(
                    document_text.split()
                ),

            "language":
                language,
        }

        return {
            "summaries":
                summaries,

            "metrics":
                metrics,

            "hallucination_analysis":
                hallucination_results,

            "metadata":
                metadata,
        }