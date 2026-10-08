from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd


# ---------------------------------------------------------
# CSV Export
# ---------------------------------------------------------

def export_results_csv(
    dataframe: pd.DataFrame,
    output_path: str,
) -> None:
    """
    Export evaluation results to CSV.
    """

    output = Path(output_path)

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataframe.to_csv(
        output,
        index=False,
    )

    print(
        f"[REPORT] CSV exported → "
        f"{output}"
    )


# ---------------------------------------------------------
# Aggregate Metrics
# ---------------------------------------------------------

def aggregate_metrics(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute average metrics
    grouped by summarization method.
    """

    metric_columns = [
        "rouge1",
        "rouge2",
        "rougeL",
        "bleu",
        "meteor",
        "bertscore",
    ]

    available_metrics = [
        metric
        for metric in metric_columns
        if metric in dataframe.columns
    ]

    aggregated = (
        dataframe
        .groupby("method")[available_metrics]
        .mean()
        .round(4)
    )

    return aggregated


# ---------------------------------------------------------
# Console Report
# ---------------------------------------------------------

def print_evaluation_report(
    aggregated_metrics: pd.DataFrame,
) -> None:
    """
    Pretty-print evaluation metrics.
    """

    print("\n" + "=" * 70)

    print(
        "SUMMARIZATION EVALUATION REPORT"
    )

    print("=" * 70)

    print(
        aggregated_metrics.to_string()
    )

    print("=" * 70)


# ---------------------------------------------------------
# Markdown Report
# ---------------------------------------------------------

def generate_markdown_report(
    aggregated_metrics: pd.DataFrame,
    benchmark_results: Dict | None = None,
) -> str:
    """
    Generate markdown evaluation report.
    """

    report_lines = []

    report_lines.append(
        "# Summarization Evaluation Report\n"
    )

    report_lines.append(
        "## Evaluation Metrics\n"
    )

    report_lines.append(
        aggregated_metrics.to_markdown()
    )

    report_lines.append("\n")

    # ---------------------------------------------------------
    # Benchmark Section
    # ---------------------------------------------------------

    if benchmark_results:

        report_lines.append(
            "## Benchmark Results\n"
        )

        for method, result in benchmark_results.items():

            report_lines.append(
                f"### {method}\n"
            )

            report_lines.append(
                f"- Total Samples: "
                f"{result.total_samples}"
            )

            report_lines.append(
                f"- Average Latency: "
                f"{result.average_latency_seconds:.2f}s"
            )

            report_lines.append(
                f"- Throughput: "
                f"{result.throughput_samples_per_second:.2f} "
                f"samples/sec"
            )

            report_lines.append(
                f"- Peak RAM Usage: "
                f"{result.peak_memory_mb:.2f} MB"
            )

            report_lines.append(
                f"- Peak GPU Memory: "
                f"{result.gpu_memory_mb:.2f} MB"
            )

            report_lines.append("\n")

    return "\n".join(report_lines)


# ---------------------------------------------------------
# Save Markdown Report
# ---------------------------------------------------------

def save_markdown_report(
    markdown_content: str,
    output_path: str,
) -> None:
    """
    Save markdown report to file.
    """

    output = Path(output_path)

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output.write_text(
        markdown_content,
        encoding="utf-8",
    )

    print(
        f"[REPORT] Markdown report saved → "
        f"{output}"
    )