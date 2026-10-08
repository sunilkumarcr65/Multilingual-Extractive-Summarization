# metrics/metric_cards.py

from __future__ import annotations

import streamlit as st


# ---------------------------------------------------------
# Metric Styling
# ---------------------------------------------------------

METRIC_CONFIG = {
    "ROUGE-1": {
        "emoji": "🟢",
        "description": (
            "Unigram overlap "
            "between summaries."
        ),
    },

    "ROUGE-2": {
        "emoji": "🔵",
        "description": (
            "Bigram overlap "
            "between summaries."
        ),
    },

    "ROUGE-L": {
        "emoji": "🟣",
        "description": (
            "Longest common "
            "subsequence similarity."
        ),
    },

    "BLEU": {
        "emoji": "🟡",
        "description": (
            "Precision-oriented "
            "translation metric."
        ),
    },

    "METEOR": {
        "emoji": "🟠",
        "description": (
            "Semantic-aware "
            "evaluation metric."
        ),
    },

    "BERTScore": {
        "emoji": "🔴",
        "description": (
            "Embedding similarity "
            "using transformer models."
        ),
    },
}


# ---------------------------------------------------------
# Single Metric Card
# ---------------------------------------------------------

def render_metric_card(
    metric_name: str,
    metric_value: float,
):
    """
    Render one metric card.
    """

    config = METRIC_CONFIG.get(
        metric_name,
        {},
    )

    emoji = config.get(
        "emoji",
        "⚪",
    )

    description = config.get(
        "description",
        "",
    )

    if metric_name == "Compression Ratio":
        display_value = (
            f"{metric_value * 100:.1f}%"
        )
    else:

        display_value = (
            f"{metric_value:.4f}"
        )
        
    st.metric(
        label=(
            f"{emoji} "
            f"{metric_name}"
        ),

        value=display_value
    )

    if description:

        st.caption(
            description
        )


# ---------------------------------------------------------
# Grid Renderer
# ---------------------------------------------------------

def render_metric_grid(
    metrics: dict,
):
    """
    Render metric cards in grid.
    """

    if not metrics:

        st.warning(
            "No metrics available."
        )

        return

    columns = st.columns(
        min(
            len(metrics),
            3,
        )
    )

    for index, (
        metric_name,
        metric_value,
    ) in enumerate(
        metrics.items()
    ):

        with columns[
            index
            % len(columns)
        ]:

            render_metric_card(
                metric_name=(
                    metric_name
                ),

                metric_value=(
                    metric_value
                ),
            )


# ---------------------------------------------------------
# Summary Metric
# ---------------------------------------------------------

def render_summary_score(
    metrics: dict,
):
    """
    Render aggregate quality score.
    """

    if not metrics:
        return

    quality_metrics = {

        key: value

        for key, value
        in metrics.items()

        if key != "Compression Ratio"
    }

    if not quality_metrics:
        return

    average_score = (
        sum(
            quality_metrics.values()
        )
        / len(quality_metrics)
    )

    st.success(

        f"📈 Overall Quality Score: "

        f"{average_score:.4f}"
    )