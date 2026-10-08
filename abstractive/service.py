from __future__ import annotations

from functools import lru_cache
from typing import Optional

from abstractive.summarizer import (
    AbstractiveSummarizer,
)

from preprocessing.language_detection import (
    detect_language,
)

from translation.translator import (
    translate_text,
)


# ---------------------------------------------------------
# Models
# ---------------------------------------------------------

ENGLISH_MODEL = (
    "google/flan-t5-base"
)

MULTILINGUAL_MODEL = (
    "csebuetnlp/mT5_multilingual_XLSum"
)

SUPPORTED_SUMMARY_STYLES = {
    "default",
    "brief",
    "detailed",
    "bullet",
}


# ---------------------------------------------------------
# Model Selection
# ---------------------------------------------------------

def select_model(
    source_language: Optional[str],
    model_name: Optional[str] = None,
) -> str:

    if model_name:
        return model_name

    source_language = (
        source_language
        or "unknown"
    ).lower()

    if source_language in {
        "en",
        "unknown",
    }:
        return ENGLISH_MODEL

    return MULTILINGUAL_MODEL


# ---------------------------------------------------------
# Cached Summarizer
# ---------------------------------------------------------

@lru_cache(maxsize=4)
def get_summarizer(
    model_name: str,
) -> AbstractiveSummarizer:

    print(
        f"[ABSTRACTIVE] "
        f"Loading summarizer: "
        f"{model_name}"
    )

    return AbstractiveSummarizer(
        model_name=model_name
    )


# ---------------------------------------------------------
# Style Formatting
# ---------------------------------------------------------

def apply_summary_style(
    summary: str,
    style: str,
) -> str:

    style = (
        style or "default"
    ).lower()

    if style not in (
        SUPPORTED_SUMMARY_STYLES
    ):
        style = "default"

    # -----------------------------------------------------
    # Bullet
    # -----------------------------------------------------

    if style == "bullet":

        sentences = [
            sentence.strip()
            for sentence in summary.split(".")
            if sentence.strip()
        ]

        return "\n".join(
            f"• {sentence}"
            for sentence in sentences
        )

    # -----------------------------------------------------
    # Brief
    # -----------------------------------------------------

    if style == "brief":

        sentences = summary.split(".")

        return (
            ".".join(sentences[:2]).strip()
            + "."
        )

    return summary.strip()


# ---------------------------------------------------------
# Main API
# ---------------------------------------------------------

def generate_abstractive_summary(
    text: str,
    source_language: Optional[str] = None,
    target_language: Optional[str] = None,
    model_name: Optional[str] = None,
    max_new_tokens: int = 180,
    min_new_tokens: int = 60,
    style: str = "default",
    document_type: Optional[str] = None,
    synthesis_mode: bool = False,
) -> str:

    text = (
        text or ""
    ).strip()

    if not text:
        return ""

    # -----------------------------------------------------
    # Detect language
    # -----------------------------------------------------

    if source_language is None:

        source_language = (
            detect_language(text)
        )

    # -----------------------------------------------------
    # Select model
    # -----------------------------------------------------

    chosen_model = select_model(
        source_language=(
            source_language
        ),
        model_name=model_name,
    )

    summarizer = get_summarizer(
        chosen_model
    )

    # -----------------------------------------------------
    # Resume tuning
    # -----------------------------------------------------

    if document_type == "resume":

        max_new_tokens = max(
            max_new_tokens,
            180,
        )

        min_new_tokens = max(
            min_new_tokens,
            80,
        )

    # -----------------------------------------------------
    # Generate
    # -----------------------------------------------------

    summary = summarizer.summarize(

        text=text,

        max_new_tokens=(
            max_new_tokens
        ),

        min_new_tokens=(
            min_new_tokens
        ),

        synthesis_mode=(
            synthesis_mode
        ),

        document_type=(
            document_type
        ),
    )

    if not summary:
        return ""

    # -----------------------------------------------------
    # Translate
    # -----------------------------------------------------

    if (
        target_language
        and source_language
        and target_language
        != source_language
    ):

        try:

            summary = translate_text(
                text=summary,
                target_lang=(
                    target_language
                ),
            )

        except Exception as error:

            print(
                "[ABSTRACTIVE] "
                "Translation failed: "
                f"{error}"
            )

    # -----------------------------------------------------
    # Style
    # -----------------------------------------------------

    summary = apply_summary_style(
        summary=summary,
        style=style,
    )

    return summary.strip()