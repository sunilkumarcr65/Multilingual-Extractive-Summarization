# visualization/dashboard_plots.py

from __future__ import annotations

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

MAX_VIS_ROWS = 5000

def validate_dataframe(
    dataframe,
):
    return (
        dataframe is not None
        and not dataframe.empty
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

def aggregate_metric(
    dataframe,
    group_column,
    metric,
):

    return (
        dataframe
        .groupby(group_column)[metric]
        .mean()
        .reset_index()
    )

# ---------------------------------------------------------
# Method Comparison
# ---------------------------------------------------------

def render_method_comparison(
    dataframe,
    metric: str = "rouge1",
):
    """
    Compare summarization methods.

    Expected columns:
    -----------------------------------------
    method
    rouge1 / rouge2 / rougeL / etc.
    -----------------------------------------
    """

    required_columns = {
        "method",
        metric,
    }

    
    if len(dataframe) > MAX_VIS_ROWS:

        dataframe = dataframe.sample(
            MAX_VIS_ROWS,
            random_state=42,
        )


    if not required_columns.issubset(
        dataframe.columns
    ):

        st.warning(
            f"Missing columns for "
            f"{metric} comparison."
        )

        return

    grouped = aggregate_metric(
        dataframe,
        "method",
        metric,
    )

    figure = px.bar(
        grouped,

        x="method",

        y=metric,

        text=metric,

        title=(
            f"{metric.upper()} "
            f"by Summarization Method"
        ),

        height=420,
    )

    figure.update_traces(
        texttemplate="%{text:.4f}",
        textposition="outside",
    )

    figure.update_layout(
        yaxis=dict(
            range=[0, 1]
        )
    )

    if not validate_dataframe(dataframe):
        return

    render_plotly_chart(figure)


# ---------------------------------------------------------
# Language Comparison
# ---------------------------------------------------------

def render_language_comparison(
    dataframe,
    metric: str = "rouge1",
):
    """
    Compare performance by language.
    """

    
    if len(dataframe) > MAX_VIS_ROWS:

        dataframe = dataframe.sample(
            MAX_VIS_ROWS,
            random_state=42,
        )

    language_column = next(
        (
            column
            for column in [
                "language",
                "lang",
            ]
            if column
            in dataframe.columns
        ),
        None,
    )

    if (
        not language_column
        or metric
        not in dataframe.columns
    ):

        return

    grouped = aggregate_metric(
        dataframe,
        language_column,
        metric
    )

    figure = px.bar(
        grouped,

        x=language_column,

        y=metric,

        text=metric,

        title=(
            f"{metric.upper()} "
            f"by Language"
        ),

        height=420,
    )

    figure.update_traces(
        texttemplate="%{text:.4f}",
        textposition="outside",
    )

    figure.update_layout(
        yaxis=dict(
            range=[0, 1]
        )
    )

    if not validate_dataframe(dataframe):
        return

    render_plotly_chart(figure)


# ---------------------------------------------------------
# Metric Distribution
# ---------------------------------------------------------

def render_metric_distribution(
    dataframe,
    metric: str = "rouge1",
):
    """
    Histogram of metric distribution.
    """

    
    if metric not in dataframe.columns:

        return

    
    if len(dataframe) > MAX_VIS_ROWS:

        dataframe = dataframe.sample(
            MAX_VIS_ROWS,
            random_state=42,
        )

    dataframe = dataframe.dropna(
        subset=[metric]
    )

    figure = px.histogram(
        dataframe,

        x=metric,

        nbins=20,

        title=(
            f"{metric.upper()} "
            f"Distribution"
        ),

        height=400,
    )

    if not validate_dataframe(dataframe):
        return

    render_plotly_chart(figure)


# ---------------------------------------------------------
# Correlation Heatmap
# ---------------------------------------------------------

def render_metric_correlation(
    dataframe,
):
    """
    Metric correlation heatmap.
    """    
    if len(dataframe) > MAX_VIS_ROWS:

        dataframe = dataframe.sample(
            MAX_VIS_ROWS,
            random_state=42,
        )

    metric_columns = [
        column
        for column in [
            "rouge1",
            "rouge2",
            "rougeL",
            "bleu",
            "meteor",
            "bertscore",
        ]
        if column
        in dataframe.columns
    ]

    if len(metric_columns) < 2:
        return

    correlation = (
        dataframe[
            metric_columns
        ]
        .apply(
            lambda col:
            col.astype(float),
            errors="ignore",
        )
        .corr()
    )

    figure = px.imshow(
        correlation,

        text_auto=True,

        aspect="auto",

        title=(
            "Metric Correlation "
            "Heatmap"
        ),
    )

    if not validate_dataframe(dataframe):
        return

    render_plotly_chart(figure)


# ---------------------------------------------------------
# Processing Time Analysis
# ---------------------------------------------------------

def render_processing_time_analysis(
    dataframe,
):
    """
    Compare processing time.
    """

    if (
        "processing_time"
        not in dataframe.columns
    ):

        return
    
    if len(dataframe) > MAX_VIS_ROWS:

        dataframe = dataframe.sample(
            MAX_VIS_ROWS,
            random_state=42,
        )

    grouping_column = (
        "method"
        if "method"
        in dataframe.columns
        else None
    )

    if grouping_column is None:
        return

    grouped = aggregate_metric(
        dataframe,
        grouping_column,
        "processing_time",
    )

    figure = px.bar(
        grouped,

        x=grouping_column,

        y="processing_time",

        text="processing_time",

        title=(
            "Average Processing Time"
        ),

        height=400,
    )

    figure.update_traces(
        texttemplate="%{text:.2f}s",
        textposition="outside",
    )

    if not validate_dataframe(dataframe):
        return

    render_plotly_chart(figure)