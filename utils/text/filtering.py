# utils/text/filtering.py

from __future__ import annotations

import re
from typing import List


# ---------------------------------------------------------
# Config
# ---------------------------------------------------------

MIN_WORDS = 3

MAX_WORDS = 80

MIN_ALPHA_RATIO = 0.40


# ---------------------------------------------------------
# Regex
# ---------------------------------------------------------

REFERENCE_PATTERN = re.compile(
    r"^\[?\d+\]?\s+\S"
)

DOI_PATTERN = re.compile(
    r"^doi:",
    flags=re.IGNORECASE,
)


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------

def alpha_ratio(
    text: str,
) -> float:

    if not text:
        return 0.0

    alpha_chars = sum(
        char.isalpha()
        for char in text
    )

    return (
        alpha_chars
        / max(len(text), 1)
    )


# ---------------------------------------------------------
# Noise Detection
# ---------------------------------------------------------

def is_noise_sentence(
    sentence: str,
) -> bool:
    """
    Detect noisy/non-natural sentences.
    """

    sentence = (
        sentence or ""
    ).strip()

    if not sentence:

        return True

    words = sentence.split()

    word_count = len(words)

    # ---------------------------------------------------------
    # Length filtering
    # ---------------------------------------------------------

    if (
        word_count < MIN_WORDS
        or word_count > MAX_WORDS
    ):

        return True

    # ---------------------------------------------------------
    # Low alphabetic ratio
    # ---------------------------------------------------------

    if (
        alpha_ratio(sentence)
        < MIN_ALPHA_RATIO
    ):

        return True

    # ---------------------------------------------------------
    # References
    # ---------------------------------------------------------

    if REFERENCE_PATTERN.match(
        sentence
    ):

        return True

    # ---------------------------------------------------------
    # DOI lines
    # ---------------------------------------------------------

    if DOI_PATTERN.match(
        sentence
    ):

        return True

    return False


# ---------------------------------------------------------
# Main Filtering
# ---------------------------------------------------------

def filter_sentences(
    sentences: List[str],
) -> List[str]:
    """
    Remove noisy sentences.
    """

    filtered = []

    for sentence in sentences:

        if is_noise_sentence(
            sentence
        ):

            continue

        filtered.append(
            sentence.strip()
        )

    return filtered