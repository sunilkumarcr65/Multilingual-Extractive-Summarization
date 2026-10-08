# pipelines/translation_pipeline.py

from __future__ import annotations

import time
from typing import Any, Dict, Optional

from formatting.response_formatter import (
    build_error_response,
    build_metadata,
    build_translation_response,
)

from preprocessing.language_detection import (
    detect_language,
)

from translation.language_map import (
    is_supported_language,
)

from translation.translator import (
    translate_text,
)

from utils.text.cleaning import (
    clean_text,
)


# ---------------------------------------------------------
# Validation
# ---------------------------------------------------------

def validate_translation_request(
    text: str,
    target_language: str,
):
    """
    Validate translation request.
    """

    text = (
        text or ""
    ).strip()

    if not text:

        raise ValueError(
            "Input text cannot be empty."
        )

    target_language = (
        target_language
        .strip()
        .lower()
    )

    if not is_supported_language(
        target_language
    ):

        raise ValueError(
            f"Unsupported target "
            f"language: "
            f"{target_language}"
        )

    return (
        text,
        target_language,
    )


# ---------------------------------------------------------
# Main Translation Pipeline
# ---------------------------------------------------------

def translate_document_text(
    text: str,
    target_language: str,
    source_language: Optional[
        str
    ] = None,
) -> Dict[str, Any]:
    """
    Translation pipeline.

    Pipeline
    -----------------------------------------
    cleaning
        ↓
    language detection
        ↓
    multilingual translation
    -----------------------------------------
    """

    start_time = time.time()

    try:

        # -----------------------------------------------------
        # Validate
        # -----------------------------------------------------

        (
            text,
            target_language,
        ) = validate_translation_request(
            text=text,
            target_language=(
                target_language
            ),
        )

        # -----------------------------------------------------
        # Cleaning
        # -----------------------------------------------------

        cleaned_text = clean_text(
            text,
            preserve_paragraphs=True,
        )

        # -----------------------------------------------------
        # Detect language
        # -----------------------------------------------------

        if source_language is None:

            source_language = (
                detect_language(
                    cleaned_text
                )
            )

        source_language = (
            source_language
            .strip()
            .lower()
        )

        # -----------------------------------------------------
        # Same language
        # -----------------------------------------------------

        if (
            source_language
            == target_language
        ):

            translated_text = (
                cleaned_text
            )

        else:

            translated_text = (
                translate_text(
                    text=cleaned_text,

                    source_language=(
                        source_language
                    ),

                    target_language=(
                        target_language
                    ),
                )
            )

        # -----------------------------------------------------
        # Timing
        # -----------------------------------------------------

        processing_time = (
            time.time()
            - start_time
        )

        metadata = build_metadata(
            source_language=(
                source_language
            ),

            target_language=(
                target_language
            ),

            processing_time=(
                processing_time
            ),
        )

        # -----------------------------------------------------
        # Response
        # -----------------------------------------------------

        response = (
            build_translation_response(
                original_text=(
                    cleaned_text
                ),

                translated_text=(
                    translated_text
                ),

                source_language=(
                    source_language
                ),

                target_language=(
                    target_language
                ),
            )
        )

        response[
            "metadata"
        ] = metadata

        return response

    except Exception as error:

        return build_error_response(
            message=str(error),

            error_type=(
                "translation_pipeline_error"
            ),
        )