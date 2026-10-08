# utils/text/hallucination_filter.py

from __future__ import annotations

import re
from rapidfuzz import fuzz


# ---------------------------------------------------------
# Sentence Support Check
# ---------------------------------------------------------

def sentence_supported(
    sentence: str,
    source_text: str,
    threshold: int = 42,
) -> bool:
    """
    Check whether a generated sentence
    is grounded in source text.
    """

    sentence = (
        sentence or ""
    ).strip()

    source_text = (
        source_text or ""
    ).strip()

    if not sentence:
        return False

    # -----------------------------------------------------
    # Ignore tiny fragments
    # -----------------------------------------------------

    if len(sentence.split()) < 4:
        return True

    score = fuzz.partial_ratio(
        sentence.lower(),
        source_text.lower(),
    )

    return score >= threshold


# ---------------------------------------------------------
# Hallucination Removal
# ---------------------------------------------------------

def remove_hallucinated_sentences(
    generated_text: str,
    source_text: str,
    threshold: int = 58,
) -> str:
    """
    Remove unsupported generated sentences.
    """

    generated_text = (
        generated_text or ""
    ).strip()

    source_text = (
        source_text or ""
    ).strip()

    if not generated_text:
        return ""

    sentences = re.split(
        r"(?<=[.!?])\s+",
        generated_text,
    )

    kept_sentences = []

    for sentence in sentences:

        sentence = sentence.strip()

        if not sentence:
            continue

        if sentence_supported(
            sentence=sentence,
            source_text=source_text,
            threshold=threshold,
        ):

            kept_sentences.append(
                sentence
            )

    return " ".join(
        kept_sentences
    ).strip()