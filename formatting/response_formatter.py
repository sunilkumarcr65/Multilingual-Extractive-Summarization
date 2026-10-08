# formatting/response_formatter.py

from __future__ import annotations

from typing import Any
from typing import Dict
from typing import List
from typing import Optional


# ---------------------------------------------------------
# Default Sentiment
# ---------------------------------------------------------

DEFAULT_SENTIMENT = {

    "label":
        "NEUTRAL",

    "score":
        0.0,
}


# ---------------------------------------------------------
# Summary Response
# ---------------------------------------------------------

def build_summary_response(
    summary_markdown: str,

    sections: List[Dict[str, Any]],

    overall_sentiment: Optional[
        Dict[str, Any]
    ] = None,

    metadata: Optional[
        Dict[str, Any]
    ] = None,

    translated_summary: str | None = None,

    metrics: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """
    Standardized summarization response.
    """

    return {

        "success":
            True,

        "summary":
            summary_markdown,

        "translated_summary":
            translated_summary or "",

        "sections":
            sections,

        "overall_sentiment":
            (
                overall_sentiment
                or dict(
                    DEFAULT_SENTIMENT
                )
            ),

        "metrics":
            metrics or {},

        "metadata":
            metadata or {},
    }


# ---------------------------------------------------------
# QA Response
# ---------------------------------------------------------

def build_qa_response(
    question: str,
    answer: str,
    context: str,
    retrieved_chunks: Optional[
        List[Dict[str, Any]]
    ] = None,
) -> Dict[str, Any]:
    """
    Standardized QA response.
    """

    return {

        "success":
            True,

        "question":
            question,

        "answer":
            answer,

        "context":
            context,

        "retrieved_chunks":
            retrieved_chunks
            or [],
    }


# ---------------------------------------------------------
# Translation Response
# ---------------------------------------------------------

def build_translation_response(
    original_text: str,
    translated_text: str,
    source_language: str,
    target_language: str,
) -> Dict[str, Any]:
    """
    Standardized translation response.
    """

    return {

        "success":
            True,

        "source_language":
            source_language,

        "target_language":
            target_language,

        "original_text":
            original_text,

        "translated_text":
            translated_text,
    }


# ---------------------------------------------------------
# Error Response
# ---------------------------------------------------------

def build_error_response(
    message: str,
    error_type: str = "runtime_error",
    details: Optional[
        Dict[str, Any]
    ] = None,
) -> Dict[str, Any]:
    """
    Standardized error response.
    """

    return {

        "success":
            False,

        "error": {

            "type":
                error_type,

            "message":
                message,

            "details":
                details or {},
        },
    }


# ---------------------------------------------------------
# Metadata Builder
# ---------------------------------------------------------

def build_metadata(
    file_path: Optional[str] = None,

    source_language: Optional[str] = None,

    target_language: Optional[str] = None,

    model_used: Optional[str] = None,

    processing_time: Optional[float] = None,

    document_type: Optional[str] = None,

    summarization_mode: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Build standardized metadata.
    """

    metadata = {}

    if file_path is not None:

        metadata[
            "file_path"
        ] = file_path

    if source_language is not None:

        metadata[
            "source_language"
        ] = source_language

    if target_language is not None:

        metadata[
            "target_language"
        ] = target_language

    if model_used is not None:

        metadata[
            "model_used"
        ] = model_used

    if document_type is not None:

        metadata[
            "document_type"
        ] = document_type

    if summarization_mode is not None:

        metadata[
            "summarization_mode"
        ] = summarization_mode

    if processing_time is not None:

        metadata[
            "processing_time_seconds"
        ] = round(
            processing_time,
            3,
        )

    return metadata
