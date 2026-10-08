from __future__ import annotations

import time
from typing import Any, Dict, Optional

from formatting.response_formatter import (
    build_error_response,
)

from pipelines.qa_pipeline import (
    answer_question,
)

from pipelines.summarization_pipeline import (
    summarize_document,
)

from pipelines.translation_pipeline import (
    translate_document_text,
)


# ---------------------------------------------------------
# Defaults
# ---------------------------------------------------------

DEFAULT_COMPRESSION_RATIO = 0.25

DEFAULT_MAX_SENTENCES = 10

VALID_SUMMARIZATION_MODES = {
    "extractive",
    "hybrid",
    "abstractive",
}


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------

def normalize_mode(
    mode: str,
) -> str:

    mode = (
        (mode or "")
        .strip()
        .lower()
    )

    if mode not in VALID_SUMMARIZATION_MODES:

        return "hybrid"

    return mode


def validate_compression_ratio(
    ratio: float,
) -> float:

    try:
        ratio = float(ratio)

    except Exception:
        return DEFAULT_COMPRESSION_RATIO

    return max(
        0.05,
        min(0.90, ratio),
    )


def validate_max_sentences(
    value: int,
) -> int:

    try:
        value = int(value)

    except Exception:
        return DEFAULT_MAX_SENTENCES

    return max(
        1,
        min(50, value),
    )


# ---------------------------------------------------------
# Summarization API
# ---------------------------------------------------------

def summarize_document_main(
    file_path: str,
    target_language: str = "en",
    compression_ratio: float = (
        DEFAULT_COMPRESSION_RATIO
    ),
    max_sentences: int = (
        DEFAULT_MAX_SENTENCES
    ),
    summarization_mode: str = "hybrid",
    abstractive_model: Optional[str] = None,
    ocr_language: str = "eng",
) -> Dict[str, Any]:
    """
    Public summarization API.
    """

    start_time = time.time()

    try:

        if not file_path:

            return build_error_response(
                message="File path is required.",
                error_type="validation_error",
            )

        summarization_mode = normalize_mode(
            summarization_mode
        )

        compression_ratio = (
            validate_compression_ratio(
                compression_ratio
            )
        )

        max_sentences = (
            validate_max_sentences(
                max_sentences
            )
        )

        print(
            "[MAIN] Summarization request | "
            f"mode={summarization_mode} | "
            f"target={target_language}"
        )

        result = summarize_document(
            file_path=file_path,

            target_language=(
                target_language
            ),

            compression_ratio=(
                compression_ratio
            ),

            max_sentences=(
                max_sentences
            ),

            summarization_mode=(
                summarization_mode
            ),

            abstractive_model=(
                abstractive_model
            ),

            ocr_language=(
                ocr_language
            ),
        )

        elapsed = (
            time.time()
            - start_time
        )

        result.setdefault(
            "metadata",
            {},
        )

        result["metadata"][
            "api_time_seconds"
        ] = round(elapsed, 3)

        return result

    except Exception as error:

        return build_error_response(
            message=str(error),
            error_type="summarization_api_error",
        )


# ---------------------------------------------------------
# QA API
# ---------------------------------------------------------

def answer_question_main(
    file_path: str,
    question: str,
    ocr_language: str = "eng",
    top_k: int = 5,
) -> Dict[str, Any]:
    """
    Public RAG QA API.
    """

    try:

        if not file_path:

            return build_error_response(
                message="File path is required.",
                error_type="validation_error",
            )

        question = (
            question or ""
        ).strip()

        if not question:

            return build_error_response(
                message="Question cannot be empty.",
                error_type="validation_error",
            )

        top_k = max(
            1,
            min(int(top_k), 15),
        )

        print(
            "[MAIN] QA request"
        )

        return answer_question(
            file_path=file_path,

            question=question,

            ocr_language=(
                ocr_language
            ),

            top_k=top_k,
        )

    except Exception as error:

        return build_error_response(
            message=str(error),
            error_type="qa_api_error",
        )


# ---------------------------------------------------------
# Translation API
# ---------------------------------------------------------

def translate_text_main(
    text: str,
    target_language: str,
    source_language: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Public translation API.
    """

    try:

        text = (
            text or ""
        ).strip()

        if not text:

            return build_error_response(
                message="Text cannot be empty.",
                error_type="validation_error",
            )

        if not target_language:

            return build_error_response(
                message="Target language required.",
                error_type="validation_error",
            )

        print(
            "[MAIN] Translation request"
        )

        return translate_document_text(
            text=text,

            target_language=(
                target_language
            ),

            source_language=(
                source_language
            ),
        )

    except Exception as error:

        return build_error_response(
            message=str(error),
            error_type="translation_api_error",
        )


# ---------------------------------------------------------
# Example Usage
# ---------------------------------------------------------

if __name__ == "__main__":

    print(
        "Multilingual Summarization "
        "System Ready."
    )