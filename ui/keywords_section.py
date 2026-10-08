# ui/keywords_section.py

from __future__ import annotations

import streamlit as st


# ---------------------------------------------------------
# Optional Dependency
# ---------------------------------------------------------

try:
    from keybert import KeyBERT

    HAS_KEYBERT = True

except ImportError:

    HAS_KEYBERT = False

    KeyBERT = None


# ---------------------------------------------------------
# Cached Model
# ---------------------------------------------------------

@st.cache_resource
def load_keyword_model():
    """
    Load KeyBERT model once.
    """

    if not HAS_KEYBERT:

        return None

    return KeyBERT()


# ---------------------------------------------------------
# Keyword Extraction
# ---------------------------------------------------------

def extract_keywords(
    text: str,
    top_n: int = 12,
):
    """
    Extract keywords/keyphrases.
    """
    if len(text.split()) < 120:
        return []

    text = (
        text or ""
    ).strip()

    if not text:

        return []

    model = load_keyword_model()

    if model is None:

        return []

    try:

        keywords = (
            model.extract_keywords(
                text,

                keyphrase_ngram_range=(
                    1,
                    2,
                ),

                stop_words="english",

                use_mmr=True,

                diversity=0.6,

                top_n=top_n,
            )
        )

        return keywords

    except Exception as error:

        print(
            "[KEYWORDS] "
            f"Extraction failed: "
            f"{error}"
        )

        return []


# ---------------------------------------------------------
# UI Renderer
# ---------------------------------------------------------

def render_keywords_section(
    text: str,
):
    """
    Render keyword section.
    """

    if not HAS_KEYBERT:

        st.caption(
            "KeyBERT not installed."
        )

        return

    keywords = extract_keywords(
        text
    )

    if not keywords:

        return

    with st.expander(
        "🏷️ Top Keywords "
        "& Keyphrases",

        expanded=True,
    ):

        columns = st.columns(4)

        for index, (
            keyword,
            score,
        ) in enumerate(
            keywords
        ):

            columns[
                index % 4
            ].markdown(
                f"`{keyword}` "
                f"**{score:.2f}**"
            )