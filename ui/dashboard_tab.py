from __future__ import annotations

import os

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# ---------------------------------------------------------
# CSV Loader
# ---------------------------------------------------------

@st.cache_data
def load_csv(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df.columns = [c.lower().strip() for c in df.columns]
    return df


# ---------------------------------------------------------
# Metric Columns
# ---------------------------------------------------------

def get_metric_columns(df: pd.DataFrame):
    possible = ["rouge1", "rouge2", "rougel", "bleu", "meteor", "bertscore"]
    return [m for m in possible if m in df.columns]


# ---------------------------------------------------------
# Metric Comparison Chart
# ---------------------------------------------------------

def render_metric_comparison_chart(df: pd.DataFrame):
    if "method" not in df.columns:
        return
    metric_cols = get_metric_columns(df)
    if not metric_cols:
        return
    agg = df.groupby("method")[metric_cols].mean().reset_index()
    fig = go.Figure()
    colors = px.colors.qualitative.Plotly
    for i, metric in enumerate(metric_cols):
        fig.add_trace(go.Bar(
            name=metric.upper(),
            x=agg["method"],
            y=agg[metric],
            marker_color=colors[i % len(colors)],
            text=agg[metric].round(4),
            textposition="outside",
        ))
    fig.update_layout(
        title="Summarization Method Comparison",
        barmode="group",
        height=450,
        xaxis_title="Method",
        yaxis_title="Score",
    )
    st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------
# Language Chart
# ---------------------------------------------------------

def render_language_chart(df: pd.DataFrame):
    lang_col = next(
        (c for c in ["language", "lang"] if c in df.columns), None
    )
    if not lang_col or "rouge1" not in df.columns or "method" not in df.columns:
        return
    agg = df.groupby([lang_col, "method"])["rouge1"].mean().reset_index()
    fig = px.bar(
        agg, x=lang_col, y="rouge1", color="method",
        barmode="group", title="ROUGE-1 by Language", height=420,
    )
    st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------
# Model Comparison
# ---------------------------------------------------------

def render_model_comparison(df: pd.DataFrame):
    if "model" not in df.columns:
        return
    metric_cols = get_metric_columns(df)
    if not metric_cols:
        return
    agg = df.groupby("model")[metric_cols].mean().reset_index()
    st.subheader("🔬 Model Comparison")
    st.dataframe(agg.set_index("model").round(4), use_container_width=True)
    fig = go.Figure()
    for _, row in agg.iterrows():
        fig.add_trace(go.Bar(
            name=row["model"],
            x=[m.upper() for m in metric_cols],
            y=[row[m] for m in metric_cols],
            text=[round(row[m], 4) for m in metric_cols],
            textposition="outside",
        ))
    fig.update_layout(
        barmode="group",
        title="Fine-tuned vs Base Models",
        yaxis=dict(range=[0, 1]),
        height=400,
    )
    st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------
# Live Metrics
# ---------------------------------------------------------

def render_live_metrics():
    """
    Render live metrics section.
    Reads from session state keys written by summarizer_tab.
    """

    st.subheader("⚡ Live Summary Metrics")

    pipeline_ran = st.session_state.get(
        "pipeline_ran",
        False,
    )

    live_metrics = st.session_state.get(
        "latest_metrics",
        {},
    )

    if not pipeline_ran:

        st.info(
            "Generate a summary to view live metrics."
        )

        return

    if not live_metrics:

        st.warning(
            "Pipeline ran but no metrics were returned."
        )

        return

    # -----------------------------------------------------
    # Flexible metric getter
    # -----------------------------------------------------

    def _get_metric(
        metrics,
        *keys,
        default=0.0,
    ):

        for key in keys:

            if key in metrics:

                return metrics[key]

        return default

    # -----------------------------------------------------
    # FIXED KEY MAPPING
    # -----------------------------------------------------

    rouge1 = _get_metric(
        live_metrics,

        "ROUGE-1",
        "rouge1",
    )

    rouge2 = _get_metric(
        live_metrics,

        "ROUGE-2",
        "rouge2",
    )

    rougel = _get_metric(
        live_metrics,

        "ROUGE-L",
        "rougeL",
        "rougel",
    )

    bertscore = _get_metric(
        live_metrics,

        "BERTScore",
        "bertscore",
        "bertscore_f1",
    )

    compression = _get_metric(
        live_metrics,
        "Compression Reduction",
        "compression_reduction",
        "actual_reduction_ratio",
        "Compression Ratio",
        "compression_ratio",
    )

    # -----------------------------------------------------
    # Metric cards
    # -----------------------------------------------------

    cols = st.columns(5)

    cols[0].metric(
        "ROUGE-1",
        (
            round(float(rouge1), 4)
            if rouge1 is not None
            else "N/A"
        ),
    )

    cols[1].metric(
        "ROUGE-2",
        (
            round(float(rouge2), 4)
            if rouge2 is not None
            else "N/A"
        ),
    )

    cols[2].metric(
        "ROUGE-L",
        (
            round(float(rougel), 4)
            if rougel is not None
            else "N/A"
        ),
    )

    cols[3].metric(
        "BERTScore",
        (
            round(float(bertscore), 4)
            if bertscore is not None
            else "N/A"
        ),
    )

    # -----------------------------------------------------
    # Compression formatting
    # -----------------------------------------------------

    compression_display = compression * 100 if compression <= 1 else compression

    cols[4].metric(
        "Reduction",
        f"{compression_display:.1f}%"
    )

    # -----------------------------------------------------
    # Full metrics table
    # -----------------------------------------------------
    with st.expander("Research Benchmarks"):
        st.dataframe(
            pd.DataFrame(
                [live_metrics]
            ).round(4),
            use_container_width=True,
        )


# ---------------------------------------------------------
# Live Sentiment + Metadata (new — was fully missing)
# ---------------------------------------------------------

def render_live_document_info():
    """
    Render sentiment and metadata from the last run.
    Both are written to session state by summarizer_tab.
    """

    sentiment = st.session_state.get("latest_sentiment", {})
    metadata  = st.session_state.get("latest_metadata", {})

    if not sentiment and not metadata:
        return

    st.subheader("📋 Last Document Info")

    col1, col2 = st.columns(2)

    with col1:
        if sentiment:
            if "label" in sentiment and "score" in sentiment:
                label = sentiment.get("label", "-")
                score = float(sentiment.get("score", 0.0))
            else:
                label = max(sentiment, key=sentiment.get)
                score = float(sentiment.get(label, 0.0))
            st.metric("Sentiment", label, f"{score:.2f} confidence")

    with col2:
        if metadata:
            fast = metadata.get("fast_path_used")
            if fast is not None:
                st.metric("Processing path", "Fast ⚡" if fast else "Full 🔄")

    # Translated summary indicator
    translated = st.session_state.get("summary_translated")
    target_lang = st.session_state.get("target_language", "en")
    if translated and target_lang != "en":
        st.success(f"Translation present: **{target_lang.upper()}**")


# ---------------------------------------------------------
# Main Dashboard
# ---------------------------------------------------------

def render_dashboard_tab():
    st.title("📊 Research Dashboard")
    st.caption("Evaluation and benchmark analysis.")

    # Live metrics from current session
    render_live_metrics()

    st.divider()

    # Sentiment + metadata from current session
    render_live_document_info()

    st.divider()

    # ---------------------------------------------------------
    # Research Benchmarks (CSV-based — unchanged logic)
    # ---------------------------------------------------------

    st.subheader("🧪 Research Benchmark Metrics")

    base_dir     = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(base_dir)

    csv_files = [
        ("Updated Scoring",  "xlsum_results_updated_scoring.csv"),
        ("Fine-tuned Model", "xlsum_results_finetuned.csv"),
    ]

    available = [
        (label, os.path.join(project_root, fname))
        for label, fname in csv_files
        if os.path.exists(os.path.join(project_root, fname))
    ]

    if not available:
        st.warning("No evaluation CSV files found.")
        return

    for label, path in available:
        st.subheader(f"📁 {label}")
        df = load_csv(path)
        render_metric_comparison_chart(df)
        render_language_chart(df)
        render_model_comparison(df)
        st.divider()
