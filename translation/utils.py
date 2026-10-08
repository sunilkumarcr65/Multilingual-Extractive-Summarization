from __future__ import annotations

import re
from typing import List


# ---------------------------------------------------------
# Sentence Splitting
# ---------------------------------------------------------

def split_sentences(
    text: str,
) -> List[str]:
    """
    Lightweight multilingual sentence splitter.

    Preserves:
    - punctuation
    - paragraph structure
    - multilingual punctuation
    """

    text = (
        text or ""
    ).strip()

    if not text:

        return []

    # ---------------------------------------------------------
    # Normalize whitespace
    # ---------------------------------------------------------

    text = re.sub(
        r"\s+",
        " ",
        text,
    ).strip()

    # ---------------------------------------------------------
    # Sentence splitting
    # ---------------------------------------------------------

    sentences = re.split(
        r"(?<=[.!?।！？•])\s+|\n+",
        text,
    )

    return [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
    ]


# ---------------------------------------------------------
# Paragraph Splitting
# ---------------------------------------------------------

def split_paragraphs(
    text: str,
) -> List[str]:
    """
    Split text into paragraphs.
    """

    text = (
        text or ""
    ).strip()

    if not text:

        return []

    paragraphs = re.split(
        r"\n\s*\n",
        text,
    )

    return [
        paragraph.strip()
        for paragraph in paragraphs
        if paragraph.strip()
    ]


# ---------------------------------------------------------
# Paragraph Preservation
# ---------------------------------------------------------

def preserve_paragraph_structure(
    translated_sentences: List[str],
    sentences_per_paragraph: List[int],
) -> str:
    """
    Restore paragraph layout
    after translation.
    """


    if not translated_sentences:

        return ""

    paragraphs = []

    current_index = 0

    for count in (
        sentences_per_paragraph
    ):

        paragraph_sentences = (
            translated_sentences[
                current_index:
                current_index + count
            ]
        )

        paragraph = " ".join(
            paragraph_sentences
        ).strip()

        if paragraph:

            paragraphs.append(
                paragraph
            )

        current_index += count

    return "\n\n".join(
        paragraphs
    )


# ---------------------------------------------------------
# Flatten Paragraphs
# ---------------------------------------------------------

def flatten_paragraphs(
    paragraphs: List[str],
) -> tuple[
    List[str],
    List[int],
]:
    """
    Convert paragraphs into flat
    sentence list while preserving
    paragraph metadata.
    """

    all_sentences = []

    sentence_counts = []

    for paragraph in paragraphs:

        sentences = split_sentences(
            paragraph
        )

        if not sentences:
            continue

        all_sentences.extend(
            sentences
        )

        sentence_counts.append(
            len(sentences)
        )

    return (
        all_sentences,
        sentence_counts,
    )