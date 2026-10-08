from __future__ import annotations

from functools import lru_cache

try:
    from langdetect import (
        DetectorFactory,
        detect,
    )

    HAS_LANGDETECT = True

    # Deterministic output
    DetectorFactory.seed = 0

except ImportError:

    detect = None

    HAS_LANGDETECT = False


# ---------------------------------------------------------
# Language Labels
# ---------------------------------------------------------

UNKNOWN_LANGUAGE = "unknown"


# ---------------------------------------------------------
# Detection
# ---------------------------------------------------------

@lru_cache(maxsize=512)
def detect_language(
    text: str,
) -> str:
    """
    Detect document language.

    Uses:
    -----------------------------------------
    langdetect
    -----------------------------------------

    Returns:
    - ISO language code
    - "unknown" on failure
    """

    if not HAS_LANGDETECT:

        return UNKNOWN_LANGUAGE

    text = (text or "").strip()

    if not text:

        return UNKNOWN_LANGUAGE

    # ---------------------------------------------------------
    # Reduce noise for detection
    # ---------------------------------------------------------

    sample = " ".join(
        text.split()[:400]
    )

    try:

        language = detect(sample)

        if not language:

            return UNKNOWN_LANGUAGE

        return language.lower()

    except Exception:

        return UNKNOWN_LANGUAGE


# ---------------------------------------------------------
# Convenience Helpers
# ---------------------------------------------------------

def is_english(
    language: str,
) -> bool:
    """
    Check if language is English.
    """

    return (
        language.lower()
        in {"en", "unknown"}
    )


def is_multilingual(
    language: str,
) -> bool:
    """
    Check whether language is non-English.
    """

    return not is_english(language)