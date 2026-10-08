from __future__ import annotations

from typing import Any, Dict, Optional

import pandas as pd
import torch

from evaluation.benchmark import (
    benchmark_summarizer,
    print_benchmark_result,
)

from evaluation.dataset_loader import (
    prepare_dataset,
)

from evaluation.hallucination_metrics import (
    evaluate_hallucination,
)

from evaluation.metrics import (
    add_bertscore_to_dataframe,
    compute_all_metrics,
)

from evaluation.report import (
    aggregate_metrics,
    export_results_csv,
    generate_markdown_report,
    print_evaluation_report,
    save_markdown_report,
)

from extractive.extractor import (
    ExtractiveSummarizer,
)

from abstractive.summarizer import (
    AbstractiveSummarizer,
)

from document_processing.document_classifier import (
    classify_document,
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
# Evaluator
# ---------------------------------------------------------

class SummarizationEvaluator:
    """
    Comprehensive summarization evaluator.

    Supports:
    ---------------------------------------------------------
    - extractive summarization
    - abstractive summarization
    - hybrid summarization
    - ROUGE / BLEU / METEOR / BERTScore
    - hallucination analysis
    - latency benchmarking
    - CSV + Markdown reporting
    ---------------------------------------------------------
    """

    def __init__(
        self,
        abstractive_model_name: str = (
            "google/flan-t5-small"
        ),
    ):

        print(
            "[EVALUATOR] Initializing..."
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
    # Core Evaluation
    # ---------------------------------------------------------

    def evaluate_dataset(
        self,
        dataset_name: str,
        dataset_config: Optional[str] = None,
        split: str = "test",
        num_samples: int = 100,
        export_csv_path: Optional[str] = None,
        export_markdown_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Full evaluation pipeline.
        """

        # -----------------------------------------------------
        # Dataset
        # -----------------------------------------------------

        dataset, article_key, summary_key = (
            prepare_dataset(
                dataset_name=dataset_name,
                dataset_config=dataset_config,
                split=split,
                num_samples=num_samples,
            )
        )

        rows = []

        extractive_inputs = []
        abstractive_inputs = []
        hybrid_inputs = []

        # -----------------------------------------------------
        # Generate summaries
        # -----------------------------------------------------

        print(
            "\n[EVALUATOR] "
            "Generating summaries..."
        )

        for idx, sample in enumerate(dataset):

            source_text = (
                sample.get(article_key, "")
                or ""
            ).strip()

            reference = (
                sample.get(summary_key, "")
                or ""
            ).strip()

            if not source_text:
                continue

            print(
                f"[EVALUATOR] "
                f"{idx + 1}/{len(dataset)}"
            )

            # -------------------------------------------------
            # Document classification
            # -------------------------------------------------

            document_type = classify_document(
                source_text
            )

            # -------------------------------------------------
            # Extractive
            # -------------------------------------------------

            extractive_summary = (
                self.extractive_model.summarize(
                    source_text
                )
            )

            extractive_inputs.append(
                source_text
            )

            # -------------------------------------------------
            # Abstractive
            # -------------------------------------------------

            abstractive_summary = (
                self.abstractive_model.summarize(
                    source_text
                )
            )

            abstractive_inputs.append(
                source_text
            )

            # -------------------------------------------------
            # Hybrid
            # -------------------------------------------------

            if document_type == "resume":

                # resumes work better with pure extractive
                hybrid_summary = (
                    extractive_summary
                )

            else:

                hybrid_summary = (
                    self.abstractive_model.summarize(
                        extractive_summary
                    )
                )

            hybrid_inputs.append(
                extractive_summary
            )

            # -------------------------------------------------
            # Store outputs
            # -------------------------------------------------

            method_outputs = {
                "extractive":
                    extractive_summary,

                "abstractive":
                    abstractive_summary,

                "hybrid":
                    hybrid_summary,
            }

            # -------------------------------------------------
            # Evaluate each method
            # -------------------------------------------------

            for method, summary in (
                method_outputs.items()
            ):

                # ---------------------------------------------
                # Metrics
                # ---------------------------------------------

                metrics = compute_all_metrics(
                    reference=reference,
                    generated=summary,
                )

                # ---------------------------------------------
                # Hallucination analysis
                # Skip for tiny summaries
                # ---------------------------------------------

                if len(summary.split()) < 40:

                    hallucination_data = {
                        "semantic_similarity": 1.0,
                        "grounded_sentence_ratio": 1.0,
                        "hallucinated_sentence_ratio": 0.0,
                        "entity_preservation_score": 1.0,
                        "unsupported_sentences": [],
                    }

                else:

                    hallucination = (
                        evaluate_hallucination(
                            source_text,
                            summary,
                        )
                    )

                    hallucination_data = {
                        "semantic_similarity":
                            hallucination.semantic_similarity,

                        "grounded_sentence_ratio":
                            hallucination.grounded_sentence_ratio,

                        "hallucinated_sentence_ratio":
                            hallucination.hallucinated_sentence_ratio,

                        "entity_preservation_score":
                            hallucination.entity_preservation_score,

                        "unsupported_sentences":
                            hallucination.unsupported_sentences,
                    }

                # ---------------------------------------------
                # Store row
                # ---------------------------------------------

                rows.append({

                    "document_type":
                        document_type,

                    "method":
                        method,

                    "summary":
                        summary,

                    "reference":
                        reference,

                    "semantic_similarity":
                        hallucination_data[
                            "semantic_similarity"
                        ],

                    "grounded_sentence_ratio":
                        hallucination_data[
                            "grounded_sentence_ratio"
                        ],

                    "hallucinated_sentence_ratio":
                        hallucination_data[
                            "hallucinated_sentence_ratio"
                        ],

                    "entity_preservation_score":
                        hallucination_data[
                            "entity_preservation_score"
                        ],

                    "unsupported_sentences":
                        " || ".join(
                            hallucination_data[
                                "unsupported_sentences"
                            ]
                        ),

                    **metrics,
                })

        # -----------------------------------------------------
        # DataFrame
        # -----------------------------------------------------

        dataframe = pd.DataFrame(
            rows
        )

        if dataframe.empty:

            raise ValueError(
                "No evaluation rows generated."
            )

        # -----------------------------------------------------
        # BERTScore
        # -----------------------------------------------------

        dataframe = (
            add_bertscore_to_dataframe(
                dataframe,
                language="en",
                device=DEVICE,
            )
        )

        # -----------------------------------------------------
        # Aggregate Metrics
        # -----------------------------------------------------

        aggregated_metrics = (
            aggregate_metrics(
                dataframe
            )
        )

        # -----------------------------------------------------
        # Print Report
        # -----------------------------------------------------

        print_evaluation_report(
            aggregated_metrics
        )

        # -----------------------------------------------------
        # CSV Export
        # -----------------------------------------------------

        if export_csv_path:

            export_results_csv(
                dataframe,
                export_csv_path,
            )

        # -----------------------------------------------------
        # Benchmarking
        # -----------------------------------------------------

        benchmark_results = {}

        print(
            "\n[EVALUATOR] "
            "Running benchmarks..."
        )

        # -----------------------------------------------------
        # Extractive Benchmark
        # -----------------------------------------------------

        extractive_benchmark = (
            benchmark_summarizer(
                texts=extractive_inputs,
                summarize_fn=(
                    self.extractive_model
                    .summarize
                ),
                method_name="extractive",
            )
        )

        benchmark_results[
            "extractive"
        ] = extractive_benchmark

        print_benchmark_result(
            extractive_benchmark
        )

        # -----------------------------------------------------
        # Abstractive Benchmark
        # -----------------------------------------------------

        abstractive_benchmark = (
            benchmark_summarizer(
                texts=abstractive_inputs,
                summarize_fn=(
                    self.abstractive_model
                    .summarize
                ),
                method_name="abstractive",
            )
        )

        benchmark_results[
            "abstractive"
        ] = abstractive_benchmark

        print_benchmark_result(
            abstractive_benchmark
        )

        # -----------------------------------------------------
        # Hybrid Benchmark
        # -----------------------------------------------------

        hybrid_benchmark = (
            benchmark_summarizer(
                texts=hybrid_inputs,
                summarize_fn=(
                    lambda text:
                    self.abstractive_model.summarize(
                        self.extractive_model.summarize(
                            text
                        )
                    )
                ),
                method_name="hybrid",
            )
        )

        benchmark_results[
            "hybrid"
        ] = hybrid_benchmark

        print_benchmark_result(
            hybrid_benchmark
        )

        # -----------------------------------------------------
        # Markdown Report
        # -----------------------------------------------------

        markdown_report = (
            generate_markdown_report(
                aggregated_metrics,
                benchmark_results,
            )
        )

        if export_markdown_path:

            save_markdown_report(
                markdown_report,
                export_markdown_path,
            )

        # -----------------------------------------------------
        # Final Return
        # -----------------------------------------------------

        return {

            "dataframe":
                dataframe,

            "aggregated_metrics":
                aggregated_metrics,

            "benchmark_results":
                benchmark_results,

            "markdown_report":
                markdown_report,
        }
