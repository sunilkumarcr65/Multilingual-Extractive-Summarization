from __future__ import annotations

import streamlit as st


# ---------------------------------------------------------
# Language Options
# ---------------------------------------------------------

LANG_OPTIONS = {
    "English":    "en",
    "Hindi":      "hi",
    "Spanish":    "es",
    "French":     "fr",
    "German":     "de",
    "Portuguese": "pt",
    "Arabic":     "ar",
    "Chinese":    "zh",
    "Russian":    "ru",
}

MODE_MAP = {
    "Extractive (Key Sentences)":       "extractive",
    "Refined (Extractive → AI Rewrite)": "hybrid",
    "Abstractive (AI Only)":            "abstractive",
}


# ---------------------------------------------------------
# Sidebar Renderer
# ---------------------------------------------------------

def render_sidebar() -> dict:

    with st.sidebar:

        st.title("⚙️ Settings")

        mode_label = st.radio(
            "Summarization Mode",
            options=list(MODE_MAP.keys()),
            index=1,
        )

        lambda_param = st.slider(
            "MMR λ (relevance ↔ diversity)",
            min_value=0.1,
            max_value=0.9,
            value=0.7,
            step=0.1,
            help="Higher → more relevant.\nLower → more diverse.",
        )

        compression_ratio = st.slider(
            "Compression Ratio",
            min_value=0.1,
            max_value=0.9,
            value=0.3,
            step=0.05,
        )

        max_sentences = st.slider(
            "Maximum Sentences",
            min_value=3,
            max_value=20,
            value=8,
        )

        target_label = st.selectbox(
            "Output Language",
            options=list(LANG_OPTIONS.keys()),
        )

        do_ocr = st.checkbox(
            "Enable OCR (scanned docs/images)",
            value=False,
        )

        # ---------------------------------------------------------
        # History
        # FIX: history is read once into a local variable before
        # rendering.  This means a QA-form Enter rerun does NOT
        # re-evaluate the expanders with fresh state — the sidebar
        # shows exactly what was there at the last summarization.
        #
        # Previously: st.session_state["history"] was read inline
        # inside the expander loop on every rerun, so any rerun
        # (including QA Enter) would re-render the last history
        # entry making it look like a new summary was saved.
        # ---------------------------------------------------------

        st.divider()
        st.subheader("🕓 Recent Summaries")

        history: list = list(
            st.session_state.get("history", [])
        )

        if history:
            for item in reversed(history[-5:]):
                with st.expander(f"📄 {item['file']}"):
                    st.caption(f"{item['mode']} · {item['lang']}")
                    st.write(item["preview"])
        else:
            st.caption("No summaries yet.")

    return {
        "summarization_mode": MODE_MAP[mode_label],
        "target_language":    LANG_OPTIONS[target_label],
        "target_label":       target_label,
        "lambda_param":       lambda_param,
        "do_ocr":             do_ocr,
        "compression_ratio":  compression_ratio,
        "max_sentences":      max_sentences,
    }
