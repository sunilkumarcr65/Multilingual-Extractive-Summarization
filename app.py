from __future__ import annotations

import streamlit as st


# ---------------------------------------------------------
# Page Config
# MUST be first Streamlit command
# ---------------------------------------------------------

st.set_page_config(
    page_title="GenSumm",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------
# Safe Imports
# ---------------------------------------------------------

try:

    from session.state_manager import (
        initialize_session_state,
    )

    from ui.dashboard_tab import (
        render_dashboard_tab,
    )

    from ui.sidebar import (
        render_sidebar,
    )

    from ui.summarizer_tab import (
        render_summarizer_tab,
    )

except Exception as error:

    st.error(
        "Application failed to initialize."
    )

    st.exception(error)

    st.stop()


# ---------------------------------------------------------
# Session Initialization
# ---------------------------------------------------------

initialize_session_state()


# ---------------------------------------------------------
# App Header
# ---------------------------------------------------------

st.markdown(
    """
# 📄 GenSumm

AI-powered multilingual document summarization,
translation, evaluation, and RAG QA platform.
"""
)


# ---------------------------------------------------------
# Sidebar
# Keep outside tabs so it persists globally
# ---------------------------------------------------------

settings = render_sidebar()


# ---------------------------------------------------------
# Tabs
# ---------------------------------------------------------

tab_summary, tab_dashboard = st.tabs(
    [
        "📄 Summarizer",
        "📊 Dashboard",
    ]
)


# ---------------------------------------------------------
# Summarizer Tab
# ---------------------------------------------------------

with tab_summary:

    try:

        render_summarizer_tab(
            settings=settings
        )

    except Exception as error:

        st.error(
            "Summarizer tab crashed."
        )

        st.exception(error)


# ---------------------------------------------------------
# Dashboard Tab
# IMPORTANT:
# Only render dashboard if results exist.
#
# MASSIVE performance improvement:
# avoids expensive dataframe/chart rerenders
# on every Streamlit interaction.
# ---------------------------------------------------------

with tab_dashboard:
    try:
        render_dashboard_tab()
    except Exception as error:
        st.error(
            "Dashboard failed to render."
        )
        st.exception(error)


# ---------------------------------------------------------
# Footer
# ---------------------------------------------------------

st.divider()

st.caption(
    "GenSumm • Multilingual NLP Platform"
)
