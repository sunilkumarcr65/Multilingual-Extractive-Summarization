# session/state_manager.py

from __future__ import annotations

from typing import Any, Dict, List

import streamlit as st


# ---------------------------------------------------------
# Default State
# ---------------------------------------------------------

DEFAULT_STATE = {
    "history": [],
    "latest_summary": "",
    "latest_sentiment": {},
    "latest_metrics": {},
    "latest_metadata": {},
    "uploaded_file": None,
    "qa_history": [],
    "uploaded_file_hash": None,
    "latest_translated_summary": "",
    "is_processing": False,
    "processing_stage": "",
    "latest_document_type": "general",
}


def set_uploaded_file_hash(
    file_hash: str,
):
    """
    Save uploaded file hash.
    """

    set_state(
        "uploaded_file_hash",
        file_hash,
    )


def get_uploaded_file_hash():
    """
    Get uploaded file hash.
    """

    return get_state(
        "uploaded_file_hash",
        None,
    )

def clear_processing_state():
    """
    Clear processing-related state
    while preserving app history.
    """

    keys_to_clear = [

        "latest_summary",

        "latest_sentiment",

        "latest_metrics",

        "latest_metadata",

        "qa_history",
    ]

    for key in keys_to_clear:

        st.session_state[key] = (
            DEFAULT_STATE[key]
        )

def save_latest_translated_summary(
    summary: str,
):
    set_state(
        "latest_translated_summary",
        summary,
    )


def get_latest_translated_summary():
    return get_state(
        "latest_translated_summary",
        "",
    )

def set_processing(
    is_processing: bool,
    stage: str = "",
):
    """
    Track active processing state.
    """

    set_state(
        "is_processing",
        is_processing,
    )

    set_state(
        "processing_stage",
        stage,
    )

def save_document_type(
    document_type: str,
):
    set_state(
        "latest_document_type",
        document_type,
    )


def get_document_type():
    return get_state(
        "latest_document_type",
        "general",
    )

# ---------------------------------------------------------
# Initialization
# ---------------------------------------------------------

def initialize_session_state():
    """
    Initialize Streamlit session state.
    """

    for key, default_value in (
        DEFAULT_STATE.items()
    ):

        if key not in st.session_state:

            st.session_state[
                key
            ] = default_value


# ---------------------------------------------------------
# Generic Getter
# ---------------------------------------------------------

def get_state(
    key: str,
    default: Any = None,
):
    """
    Get session state value.
    """

    return st.session_state.get(
        key,
        default,
    )


# ---------------------------------------------------------
# Generic Setter
# ---------------------------------------------------------

def set_state(
    key: str,
    value: Any,
):
    """
    Set session state value.
    """

    st.session_state[key] = value


# ---------------------------------------------------------
# Summary State
# ---------------------------------------------------------

def save_latest_summary(
    summary: str,
):
    """
    Save latest generated summary.
    """

    set_state(
        "latest_summary",
        summary,
    )


def get_latest_summary():
    """
    Get latest summary.
    """

    return get_state(
        "latest_summary",
        "",
    )


# ---------------------------------------------------------
# Sentiment State
# ---------------------------------------------------------

def save_latest_sentiment(
    sentiment: Dict[str, float],
):
    """
    Save latest sentiment.
    """

    set_state(
        "latest_sentiment",
        sentiment,
    )


def get_latest_sentiment():
    """
    Get latest sentiment.
    """

    return get_state(
        "latest_sentiment",
        {},
    )


# ---------------------------------------------------------
# Metrics State
# ---------------------------------------------------------

def save_latest_metrics(
    metrics: Dict[str, float],
):
    """
    Save latest evaluation metrics.
    """

    existing = get_latest_metrics()

    updated = {
        **existing,
        **metrics,
    }

    set_state(
        "latest_metrics",
        updated,
    )


def get_latest_metrics():
    """
    Get latest metrics.
    """

    return get_state(
        "latest_metrics",
        {},
    )


# ---------------------------------------------------------
# Metadata State
# ---------------------------------------------------------

def save_latest_metadata(
    metadata: Dict[str, Any],
):
    """
    Save metadata.
    """

    set_state(
        "latest_metadata",
        metadata,
    )


def get_latest_metadata():
    """
    Get latest metadata.
    """

    return get_state(
        "latest_metadata",
        {},
    )


# ---------------------------------------------------------
# Uploaded File
# ---------------------------------------------------------

def set_uploaded_file(
    file_path: str,
):
    """
    Save uploaded file path
    and invalidate stale state.
    """

    previous_file = get_uploaded_file()

    # -----------------------------------------------------
    # New file uploaded
    # -----------------------------------------------------

    if previous_file != file_path:

        clear_processing_state()

    set_state(
        "uploaded_file",
        file_path,
    )


def get_uploaded_file():
    """
    Get uploaded file path.
    """

    return get_state(
        "uploaded_file",
        None,
    )


# ---------------------------------------------------------
# Summary History
# ---------------------------------------------------------

def add_history_item(
    filename: str,
    summary: str,
    mode: str,
    language: str,
):
    """
    Add item to summary history.
    """

    history = get_state(
        "history",
        [],
    )

    preview = (
        summary[:200] + "..."
        if len(summary) > 200
        else summary
    )

    item = {
        "file": filename,
        "mode": mode,
        "lang": language,
        "preview": preview,
    }

    history.append(item)

    MAX_HISTORY_ITEMS = 20

    history = history[
        -MAX_HISTORY_ITEMS:
    ]

    set_state(
        "history",
        history,
    )


def get_history():
    """
    Get summary history.
    """

    return get_state(
        "history",
        [],
    )


def clear_history():
    """
    Clear summary history.
    """

    set_state(
        "history",
        [],
    )


# ---------------------------------------------------------
# QA History
# ---------------------------------------------------------

def add_qa_history(
    question: str,
    answer: str,
):
    """
    Save QA interaction.
    """

    qa_history = get_state(
        "qa_history",
        [],
    )

    qa_history.append(
        {
            "question":
                question,

            "answer":
                answer,
        }
    )

    MAX_HISTORY_ITEMS = 20

    qa_history = qa_history[
        -MAX_HISTORY_ITEMS:
    ]

    set_state(
        "qa_history",
        qa_history,
    )


def get_qa_history():
    """
    Get QA history.
    """

    return get_state(
        "qa_history",
        [],
    )


# ---------------------------------------------------------
# Full Reset
# ---------------------------------------------------------

def clear_all_state():
    """
    Reset application state only.
    """

    for key in DEFAULT_STATE:

        st.session_state[key] = (
            DEFAULT_STATE[key]
        )