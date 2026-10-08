# ui/metrics_section.py

from __future__ import annotations

import streamlit as st

from metrics.evaluation_metrics import (
    compute_metrics,
)

from visualization.charts import (
    render_metric_radar_chart,
)


# ---------------------------------------------------------
# Metric Cards
# ---------------------------------------------------------

def render_metric_cards(
    metrics: dict,
):
    """
    Render metric cards.
    """

    if not metrics:

        st.warning(
            "No metrics computed."
        )

        return

    metric_emojis = {
        "ROUGE-1": "🟢",
        "ROUGE-2": "🔵",
        "ROUGE-L": "🟣",
        "BLEU": "🟡",
        "METEOR": "🟠",
        "BERTScore": "🔴",
    }

    columns = st.columns(
        len(metrics)
    )

    for column, (
        metric,
        value,
    ) in zip(
        columns,
        metrics.items(),
    ):

        column.metric(
            (
                f"{metric_emojis.get(metric, '⚪')} "
                f"{metric}"
            ),

            (
                f"{value:.4f}"
                if value is not None
                else "N/A"
            ),
        )


# ---------------------------------------------------------
# Main Section
# ---------------------------------------------------------

def render_metrics_section(
    reference_text: str,
    generated_text: str,
    language: str = "en",
):
    """
    Render evaluation metrics section.
    """

    st.subheader(
        "🔬 Evaluate Summary"
    )

    if not reference_text.strip():

        st.info(
            "Paste a reference "
            "summary to compute "
            "evaluation metrics."
        )

        return

    with st.spinner(
        "Computing metrics..."
    ):

        metrics = compute_metrics(
            original_text=reference_text,
            summary_text=generated_text,
            reference_summary=reference_text,
            lang=language,
        )

    st.session_state["latest_metrics"] = metrics

    # ---------------------------------------------------------
    # Metric cards
    # ---------------------------------------------------------

    render_metric_cards(
        metrics
    )

    # ---------------------------------------------------------
    # Radar chart
    # ---------------------------------------------------------

    if metrics:

        render_metric_radar_chart(
            metrics
        )

@st.cache_data(show_spinner=False)
def cached_metrics(
    reference_text: str,
    generated_text: str,
    language: str = "en",
):
    return compute_metrics(
        original_text=reference_text,
        summary_text=generated_text,
        reference_summary=reference_text,
        lang=language,
    )
