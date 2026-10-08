# utils/text/pipeline.py

from __future__ import annotations

from typing import List

from utils.text.cleaning import (
    clean_text,
    remove_ocr_artifacts,
    repair_resume_spacing,
)

from utils.text.deduplication import (
    deduplicate_sentences,
)

from utils.text.filtering import (
    filter_sentences,
)

from utils.text.sentence_splitter import (
    split_sentences,
)


# ---------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------

def preprocess_text(
    text: str,
    preserve_paragraphs: bool = False,
    remove_ocr_noise: bool = True,
    deduplicate: bool = True,
    dedupe_threshold: float = 0.85,
) -> List[str]:
    """
    Complete text preprocessing pipeline.

    Pipeline
    -----------------------------------------
    clean
        ↓
    OCR cleanup
        ↓
    sentence splitting
        ↓
    filtering
        ↓
    deduplication
    -----------------------------------------

    Returns:
    -----------------------------------------
    list of clean sentences
    -----------------------------------------
    """

    text = (
        text or ""
    ).strip()

    if not text:

        return []

    # ---------------------------------------------------------
    # Cleaning
    # ---------------------------------------------------------

    cleaned = clean_text(
        text=text,
        preserve_paragraphs=(
            preserve_paragraphs
        ),
    )

    if not cleaned:

        return []

    # ---------------------------------------------------------
    # OCR cleanup
    # ---------------------------------------------------------

    if remove_ocr_noise:

        cleaned = (
            remove_ocr_artifacts(
                cleaned
            )
        )

    # ---------------------------------------------------------
    # Resume spacing repair
    # ---------------------------------------------------------

    text_lower = cleaned.lower()

    if any(
        kw in text_lower
        for kw in (
            "education",
            "experience",
            "skills",
            "cgpa",
        )
    ):

        cleaned = repair_resume_spacing(
            cleaned
        )

    # ---------------------------------------------------------
    # Sentence splitting
    # ---------------------------------------------------------

    sentences = split_sentences(
        cleaned
    )

    if not sentences:

        return []

    # ---------------------------------------------------------
    # Filtering
    # ---------------------------------------------------------

    sentences = filter_sentences(
        sentences
    )

    if not sentences:

        return []

    #----------------------------------------------------------
    # Skip expensive deduplication
    # for small documents
    #----------------------------------------------------------

    if len(sentences) < 25:
        deduplicate = False
    # ---------------------------------------------------------
    # Deduplication
    # ---------------------------------------------------------

    if deduplicate:

        sentences = (
            deduplicate_sentences(
                sentences=sentences,
                similarity_threshold=(
                    dedupe_threshold
                ),
            )
        )

    return sentences


# ---------------------------------------------------------
# Convenience Wrapper
# ---------------------------------------------------------

def preprocess_to_text(
    text: str,
    separator: str = " ",
) -> str:
    """
    Preprocess and rejoin sentences.
    """

    sentences = preprocess_text(
        text=text
    )

    return separator.join(
        sentences
    ).strip()