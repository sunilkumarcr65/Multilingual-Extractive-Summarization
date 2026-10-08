# visualization/charts.py

from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st


def normalize_metric_value(value):

    try:
        value = float(value)
    except Exception:
        return 0.0

    return max(
        0.0,
        min(1.0, value),
    )

def render_plotly_chart(
    figure,
):
    st.plotly_chart(
        figure,
        use_container_width=True,
        config={
            "displayModeBar": False,
        },
    )

# ---------------------------------------------------------
# Radar Chart
# ---------------------------------------------------------

def render_metric_radar_chart(
    metrics: dict,
):
    """
    Render radar chart for evaluation metrics.
    """

    if not metrics:
        return

    labels = list(
        metrics.keys()
    )

    values = [
        normalize_metric_value(value)
        for value in metrics.values()
    ]

    # close polygon
    labels.append(
        labels[0]
    )

    values.append(
        values[0]
    )

    figure = go.Figure()

    figure.add_trace(
        go.Scatterpolar(
            r=values,

            theta=labels,

            fill="toself",

            name="Metrics",
        )
    )

    figure.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1],
            )
        ),

        showlegend=False,

        height=450,

        title=(
            "Evaluation "
            "Metrics Radar"
        ),
    )

    render_plotly_chart(figure)


# ---------------------------------------------------------
# Keyword Score Chart
# ---------------------------------------------------------

def render_keyword_chart(
    keywords,
):
    """
    Render keyword relevance chart.

    Expected:
    -----------------------------------------
    [
        ("keyword", score),
        ...
    ]
    -----------------------------------------
    """

    if not keywords:
        return
    
    keywords = sorted(
        keywords,
        key=lambda x: x[1],
        reverse=True,
    )

    labels = [
        keyword
        for keyword, _
        in keywords
    ]

    scores = [
        score
        for _, score
        in keywords
    ]

    figure = go.Figure(
        data=[
            go.Bar(
                x=scores,

                y=labels,

                orientation="h",

                text=[
                    f"{score:.2f}"
                    for score
                    in scores
                ],

                textposition="auto",
            )
        ]
    )

    figure.update_layout(
        title=(
            "Top Keywords"
        ),

        height=min(
            900,
            max(350, len(labels) * 32),
        ),

        yaxis=dict(
            automargin=True
        ),
    )

    render_plotly_chart(figure)


# ---------------------------------------------------------
# ROUGE Comparison
# ---------------------------------------------------------

def render_rouge_comparison_chart(
    rouge_scores: dict,
):
    """
    Compare ROUGE metrics.
    """

    if not rouge_scores:
        return

    labels = list(
        rouge_scores.keys()
    )

    values = list(
        rouge_scores.values()
    )

    figure = go.Figure(
        data=[
            go.Bar(
                x=labels,

                y=values,

                text=[
                    f"{value:.4f}"
                    for value
                    in values
                ],

                textposition="outside",
            )
        ]
    )

    figure.update_layout(
        title=(
            "ROUGE Score Comparison"
        ),

        yaxis=dict(
            range=[0, 1]
        ),

        height=350,
    )

    render_plotly_chart(figure)


# ---------------------------------------------------------
# Processing Time Chart
# ---------------------------------------------------------

def render_processing_time_chart(
    timings: dict,
):
    """
    Render pipeline timing chart.

    Example:
    -----------------------------------------
    {
        "OCR": 2.1,
        "Chunking": 0.4,
        "Embedding": 1.9,
        "Summarization": 8.2,
    }
    -----------------------------------------
    """

    if not timings:
        return

    labels = list(
        timings.keys()
    )

    values = [
        float(v)
        for v in timings.values()
        if isinstance(v, (int, float))
    ]

    figure = go.Figure(
        data=[
            go.Bar(
                x=labels,

                y=values,

                text=[
                    f"{value:.2f}s"
                    for value
                    in values
                ],

                textposition="outside",
            )
        ]
    )

    figure.update_layout(
        title=(
            "Pipeline Processing Time"
        ),

        yaxis_title="Seconds",

        height=350,
    )

    render_plotly_chart(figure)