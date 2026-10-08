from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st


# ---------------------------------------------------------
# Normalizer
# FIX: sentiment_manager.analyze_document_sentiment returns
#      {"label": "positive", "score": 0.98}
#      but this section was calling .get("positive") etc. on
#      that dict — always getting 0.0.
#
#      This function accepts BOTH formats so nothing breaks
#      if the analyzer is swapped in future.
# ---------------------------------------------------------

def normalize_sentiment(sentiment: dict) -> dict:
    """
    Accepts either:
      {"label": "positive", "score": 0.91}   ← what manager returns
      {"positive": 0.91, "neutral": 0.06, "negative": 0.03}  ← full probs

    Always returns the full-probability format the UI needs.
    """

    if not sentiment:
        return {"positive": 0.0, "neutral": 1.0, "negative": 0.0}

    # Full-probability format already — validate and return
    if "positive" in sentiment or "neutral" in sentiment or "negative" in sentiment:
        return {
            "positive": float(sentiment.get("positive", 0.0)),
            "neutral":  float(sentiment.get("neutral",  0.0)),
            "negative": float(sentiment.get("negative", 0.0)),
        }

    # label/score format — expand into full probabilities
    if "label" in sentiment and "score" in sentiment:
        label = str(sentiment["label"]).strip().lower()
        score = float(sentiment["score"])

        # Distribute remaining probability evenly across the other two
        remainder = round((1.0 - score) / 2, 4)

        normalized = {"positive": remainder, "neutral": remainder, "negative": remainder}

        if label in normalized:
            normalized[label] = round(score, 4)
        else:
            # Unknown label — treat as neutral
            normalized["neutral"] = round(score, 4)

        return normalized

    return {"positive": 0.0, "neutral": 1.0, "negative": 0.0}


# ---------------------------------------------------------
# Dominant Sentiment
# ---------------------------------------------------------

def get_dominant_sentiment(sentiment: dict) -> str:
    return max(sentiment, key=sentiment.get)


# ---------------------------------------------------------
# Metric Cards
# ---------------------------------------------------------

def render_sentiment_cards(sentiment: dict):
    cols = st.columns(3)
    cols[0].metric("😊 Positive", f"{sentiment['positive']:.1%}")
    cols[1].metric("😐 Neutral",  f"{sentiment['neutral']:.1%}")
    cols[2].metric("😠 Negative", f"{sentiment['negative']:.1%}")


# ---------------------------------------------------------
# Bar Chart
# ---------------------------------------------------------

def render_sentiment_chart(sentiment: dict):
    labels = ["Positive", "Neutral", "Negative"]
    values = [sentiment["positive"], sentiment["neutral"], sentiment["negative"]]
    colors = ["#2ecc71", "#95a5a6", "#e74c3c"]

    fig = go.Figure(data=[go.Bar(
        x=labels,
        y=values,
        text=[f"{v:.1%}" for v in values],
        textposition="auto",
        marker_color=colors,
    )])

    fig.update_layout(
        title="Sentiment Distribution",
        yaxis=dict(range=[0, 1], tickformat=".0%"),
        height=320,
        showlegend=False,
    )

    st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------
# Main Section
# ---------------------------------------------------------

def render_sentiment_section(sentiment: dict):
    st.subheader("📊 Sentiment Analysis")

    # FIX: normalize before any rendering
    sentiment = normalize_sentiment(sentiment)

    dominant = get_dominant_sentiment(sentiment)
    score    = sentiment[dominant]

    st.caption(f"Dominant sentiment: **{dominant.title()}** ({score:.1%} confidence)")

    render_sentiment_cards(sentiment)
    render_sentiment_chart(sentiment)
