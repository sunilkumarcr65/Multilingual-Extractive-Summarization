from __future__ import annotations

import os
import tempfile
from typing import Dict

import streamlit as st

from document_processing.loaders import (
    load_document_text,
)

from main import summarize_document_main

from ui.keywords_section import (
    render_keywords_section,
)

from ui.metrics_section import (
    render_metrics_section,
)

from ui.qa_section import (
    render_qa_section,
)

from ui.sentiment_section import (
    render_sentiment_section,
)


# ---------------------------------------------------------
# Constants
# ---------------------------------------------------------

MAX_HISTORY_ITEMS = 20

LANG_LABELS = {
    "hi": "Hindi",
    "fr": "French",
    "de": "German",
    "es": "Spanish",
    "ar": "Arabic",
    "zh": "Chinese",
    "ja": "Japanese",
    "pt": "Portuguese",
    "ru": "Russian",
    "bn": "Bengali",
    "mr": "Marathi",
    "ta": "Tamil",
    "te": "Telugu",
}


# ---------------------------------------------------------
# File Saver
# ---------------------------------------------------------

def save_uploaded_file(
    uploaded_file,
) -> str:

    suffix = os.path.splitext(
        uploaded_file.name
    )[1]

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=suffix,
    ) as tmp:

        tmp.write(
            uploaded_file.getvalue()
        )

        return tmp.name


# ---------------------------------------------------------
# Reset
# ---------------------------------------------------------

def reset_previous_results():

    defaults = {

        "latest_metrics": {},

        "latest_summary": "",

        "latest_response": None,

        "summary_en": "",

        "summary_translated": None,

        "latest_sentiment": {},

        "latest_metadata": {},

        "pipeline_ran": False,

        "qa_result": None,

        "qa_question": "",

        "document_text": "",
    }

    for key, value in defaults.items():

        st.session_state[key] = value


# ---------------------------------------------------------
# History
# ---------------------------------------------------------

def update_history(
    filename: str,
    summary: str,
    mode: str,
    language: str,
):

    preview = (
        summary[:200] + "..."
        if len(summary) > 200
        else summary
    )

    history = st.session_state.get(
        "history",
        [],
    )

    new_item = {

        "file": filename,

        "mode": mode,

        "lang": language,

        "preview": preview,
    }

    if (
        not history
        or history[-1] != new_item
    ):

        history.append(new_item)

    history = history[-MAX_HISTORY_ITEMS:]

    st.session_state[
        "history"
    ] = history


# ---------------------------------------------------------
# Metadata
# ---------------------------------------------------------

def render_processing_metadata(
    metadata: Dict,
):

    if not metadata:
        return

    cols = st.columns(5)

    items = [

        (
            "⏱️ Time",
            "processing_time_seconds",
            lambda v: f"{float(v):.2f}s",
        ),

        (
            "🤖 Model",
            "model_used",
            str,
        ),

        (
            "📄 Type",
            "document_type",
            str,
        ),

        (
            "🧩 Chunks",
            "chunk_count",
            str,
        ),

        (
            "⚡ Path",
            "fast_path_used",
            lambda v:
            "Fast ⚡"
            if v
            else "Full 🔄",
        ),
    ]

    for col, (
        label,
        key,
        formatter,
    ) in zip(cols, items):

        value = metadata.get(key)

        if value is None:
            continue

        try:

            col.metric(
                label,
                formatter(value),
            )

        except Exception:

            col.metric(
                label,
                str(value),
            )

    target_ratio = metadata.get(
        "target_compression_ratio"
    )
    actual_ratio = metadata.get(
        "actual_retention_ratio"
    )
    if target_ratio is not None or actual_ratio is not None:
        st.caption(
            "Target retention ratio: "
            f"{target_ratio if target_ratio is not None else 'N/A'} | "
            "Actual retention ratio: "
            f"{actual_ratio if actual_ratio is not None else 'N/A'}"
        )


# ---------------------------------------------------------
# Summary
# ---------------------------------------------------------

def render_summary(
    result: Dict,
):

    summary_markdown = result.get(
        "summary",
        "",
    )

    summary_en = result.get(
        "summary_en",
        "",
    )

    summary_translated = result.get(
        "summary_translated"
    )

    target_language = (
        result.get("target_language")
        or st.session_state.get("target_language", "en")
    )

    st.subheader(
        "📝 Summary (English)"
    )

    if not summary_en:

        st.info(
            "No summary generated."
        )

    elif summary_markdown.strip():

        st.markdown(summary_markdown)

    elif len(summary_en.split()) > 120:

        st.text_area(
            "Summary",
            value=summary_en,
            height=320,
        )

    else:

        st.markdown(summary_en)

    # -----------------------------------------------------
    # Translation
    # -----------------------------------------------------

    if (
        summary_translated
        and target_language != "en"
    ):

        language_label = (
            LANG_LABELS.get(
                target_language,
                target_language.upper(),
            )
        )

        st.subheader(
            f"🌐 Summary ({language_label})"
        )

        st.markdown(
            summary_translated
        )


def render_contract_insights(
    insights: Dict,
):
    if not insights:
        return

    clause_map = insights.get("clause_map", [])
    risk_score = insights.get("risk_score", 0)
    risk_level = insights.get("risk_level", "Low")
    risk_flags = insights.get("risk_flags", [])

    if not clause_map and not risk_flags:
        return

    st.divider()
    st.subheader("Contract Insights")

    c1, c2 = st.columns(2)
    c1.metric("Risk Score", f"{risk_score}/100")
    c2.metric("Risk Level", risk_level)

    if risk_flags:
        st.caption("Risk Flags")
        for flag in risk_flags:
            st.markdown(f"- {flag}")

    if clause_map:
        st.caption("Clause Coverage")
        for item in clause_map:
            status = "Found" if item.get("found") else "Not Found"
            st.markdown(f"**{item.get('clause', 'Clause')}**: {status}")
            for evidence in item.get("evidence", [])[:1]:
                st.caption(evidence[:280])


# ---------------------------------------------------------
# Results
# ---------------------------------------------------------

def render_results(
    file_path: str,
):

    result = st.session_state.get(
        "latest_response"
    ) or {}

    sentiment = st.session_state.get(
        "latest_sentiment",
        {},
    )

    metadata = st.session_state.get(
        "latest_metadata",
        {},
    )

    summary_en = st.session_state.get(
        "summary_en",
        "",
    )

    contract_insights = result.get(
        "contract_insights",
        {},
    )

    document_text = st.session_state.get(
        "document_text",
        "",
    )

    # -----------------------------------------------------
    # Metadata
    # -----------------------------------------------------

    render_processing_metadata(
        metadata
    )

    st.divider()

    # -----------------------------------------------------
    # Summary
    # -----------------------------------------------------

    render_summary(result)

    st.divider()

    # -----------------------------------------------------
    # Keywords
    # -----------------------------------------------------

    try:

        keyword_source = (
            summary_en
            if len(summary_en.split()) > 20
            else document_text
        )

        if keyword_source:

            render_keywords_section(
                keyword_source
            )

    except Exception as error:

        print(
            "[KEYWORDS ERROR]",
            error,
        )

    render_contract_insights(
        contract_insights
    )

    st.divider()

    # -----------------------------------------------------
    # Sentiment
    # -----------------------------------------------------

    try:

        render_sentiment_section(
            sentiment
        )

    except Exception as error:

        print(
            "[SENTIMENT ERROR]",
            error,
        )

    st.divider()

    # -----------------------------------------------------
    # Metrics
    # -----------------------------------------------------

    reference_summary = st.text_area(
        "Reference Summary (Optional)",
        placeholder=(
            "Paste reference summary "
            "to compute evaluation metrics..."
        ),
        key="reference_summary_box",
    )

    try:

        if reference_summary.strip():

            render_metrics_section(
                reference_text=reference_summary,
                generated_text=summary_en,
            )

    except Exception as error:

        print(
            "[METRICS ERROR]",
            error,
        )

    st.divider()

    # -----------------------------------------------------
    # QA
    # -----------------------------------------------------

    try:

        render_qa_section(
            file_path=file_path
        )

    except Exception as error:

        print(
            "[QA ERROR]",
            error,
        )


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def render_summarizer_tab(
    settings: Dict,
):

    st.title("📄 GenSumm")

    st.caption(
        "Multilingual AI-powered document summarization platform."
    )

    uploaded_file = st.file_uploader(
        "Upload Document",
        type=[
            "pdf",
            "docx",
            "txt",
            "png",
            "jpg",
            "jpeg",
        ],
    )

    if not uploaded_file:
        return

    current_filename = (
        uploaded_file.name
    )

    # -----------------------------------------------------
    # New upload
    # -----------------------------------------------------

    if (
        st.session_state.get(
            "uploaded_filename"
        )
        != current_filename
    ):

        file_path = save_uploaded_file(
            uploaded_file
        )

        reset_previous_results()

        st.session_state[
            "uploaded_filename"
        ] = current_filename

        st.session_state[
            "uploaded_file_path"
        ] = file_path

    else:

        file_path = st.session_state.get(
            "uploaded_file_path"
        )

    st.success(
        f"Uploaded: {current_filename}"
    )

    # -----------------------------------------------------
    # Existing results
    # -----------------------------------------------------

    if not st.button(
        "Generate Summary",
        type="primary",
        use_container_width=True,
    ):

        if st.session_state.get(
            "pipeline_ran"
        ):

            render_results(file_path)

        return

    # -----------------------------------------------------
    # Fresh summarization
    # -----------------------------------------------------

    reset_previous_results()

    progress = st.progress(0)

    try:

        st.info(
            "Extracting document text..."
        )

        extracted_text = load_document_text(
            file_path=file_path,
            ocr_language="eng",
        )

        st.session_state[
            "document_text"
        ] = extracted_text

        progress.progress(20)

        st.info(
            "Generating summary..."
        )

        result = summarize_document_main(

            file_path=file_path,

            target_language=settings[
                "target_language"
            ],

            summarization_mode=settings[
                "summarization_mode"
            ],

            compression_ratio=(
                settings[
                    "compression_ratio"
                ]
            ),

            max_sentences=(
                settings[
                    "max_sentences"
                ]
            ),

            ocr_language="eng",
        )

        progress.progress(80)

    except Exception as error:

        st.error(
            f"Summarization failed:\n\n{error}"
        )

        return

    if not result.get(
        "success",
        True,
    ):

        st.error(
            result.get(
                "error",
                {},
            ).get(
                "message",
                "Unknown error.",
            )
        )

        return

    # -----------------------------------------------------
    # Outputs
    # -----------------------------------------------------

    summary_en = result.get(
        "summary_en",
        "",
    )

    summary_translated = result.get(
        "summary_translated"
    )

    target_language = settings.get(
        "target_language",
        "en",
    )

    result["target_language"] = target_language

    metadata = result.get(
        "metadata",
        {},
    )

    sentiment = result.get(
        "overall_sentiment",
        {},
    )

    metrics = result.get(
        "metrics",
        {},
    )

    # -----------------------------------------------------
    # Session
    # -----------------------------------------------------

    st.session_state.update({

        "latest_response":
            result,

        "summary_en":
            summary_en,

        "summary_translated":
            summary_translated,

        "target_language":
            target_language,

        "latest_sentiment":
            sentiment,

        "latest_metadata":
            metadata,

        "latest_summary":
            summary_en,

        "latest_metrics":
            metrics,

        "pipeline_ran":
            True,
    })

    progress.progress(100)

    # -----------------------------------------------------
    # History
    # -----------------------------------------------------

    update_history(

        filename=current_filename,

        summary=summary_en,

        mode=settings[
            "summarization_mode"
        ],

        language=target_language,
    )

    # -----------------------------------------------------
    # Render
    # -----------------------------------------------------

    render_results(file_path)
