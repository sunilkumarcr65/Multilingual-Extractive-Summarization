# utils/text/cleaning.py

from __future__ import annotations

import re
import unicodedata


# ---------------------------------------------------------
# URL Pattern
# ---------------------------------------------------------

URL_PATTERN = re.compile(
    r"https?://\S+|www\.\S+",
    flags=re.IGNORECASE,
)


# ---------------------------------------------------------
# Control Characters
# ---------------------------------------------------------

CONTROL_CHAR_PATTERN = re.compile(
    r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]"
)


# ---------------------------------------------------------
# Excess Whitespace
# ---------------------------------------------------------

MULTISPACE_PATTERN = re.compile(
    r"\s{2,}"
)

MULTINEWLINE_PATTERN = re.compile(
    r"\n{2,}"
)


# ---------------------------------------------------------
# OCR Hyphen Repair
# ---------------------------------------------------------

HYPHEN_LINEBREAK_PATTERN = re.compile(
    r"(\w)-\n(\w)"
)


# ---------------------------------------------------------
# Cleaning
# ---------------------------------------------------------

def clean_text(
    text: str,
    preserve_paragraphs: bool = False,
) -> str:
    """
    Normalize raw text for NLP processing.

    Features
    -----------------------------------------
    - URL removal
    - Unicode normalization
    - control character removal
    - OCR cleanup
    - whitespace normalization
    -----------------------------------------
    """

    text = (
        text or ""
    )

    if not text.strip():

        return ""

    # ---------------------------------------------------------
    # Unicode normalization
    # ---------------------------------------------------------

    text = unicodedata.normalize(
        "NFKC",
        text,
    )

    text = text.replace("\u00A0", " ")
    text = text.replace("\u2009", " ")
    text = text.replace("\u202F", " ")

    # ---------------------------------------------------------
    # Remove URLs
    # ---------------------------------------------------------

    text = URL_PATTERN.sub(
        "",
        text,
    )

    # ---------------------------------------------------------
    # OCR hyphen repair
    # Example:
    # multi-
    # lingual
    # → multilingual
    # ---------------------------------------------------------

    text = (
        HYPHEN_LINEBREAK_PATTERN
        .sub(r"\1\2", text)
    )

    # ---------------------------------------------------------
    # Remove control chars
    # ---------------------------------------------------------

    text = (
        CONTROL_CHAR_PATTERN
        .sub("", text)
    )

    # ---------------------------------------------------------
    # Normalize whitespace
    # ---------------------------------------------------------

    if preserve_paragraphs:

        text = re.sub(
            r"[^\S\n\t\r]+",
            " ",
            text,
        )

        text = re.sub(
            r"\n{3,}",
            "\n\n",
            text,
        )

    else:

        text = MULTINEWLINE_PATTERN.sub(
            " ",
            text,
        )

        text = re.sub(
            r"[^\S\n\t\r]+",
            " ",
            text,
        )

    text = MULTISPACE_PATTERN.sub(
        " ",
        text,
    )

    return text.strip()


# ---------------------------------------------------------
# OCR Artifact Removal
# ---------------------------------------------------------

def remove_ocr_artifacts(
    text: str,
) -> str:
    """
    Remove common OCR artifacts.
    """

    text = (
        text or ""
    )

    # repeated punctuation
    text = re.sub(
        r"[|]{2,}",
        "|",
        text,
    )

    # repeated symbols
    text = re.sub(
        r"[_]{2,}",
        "_",
        text,
    )

    # broken spaces
    text = re.sub(
        r"\s+([.,!?])",
        r"\1",
        text,
    )

    return text.strip()

# ---------------------------------------------------------
# Resume Text Repair
# ---------------------------------------------------------

def repair_resume_spacing(
    text: str,
) -> str:
    """
    Fix common resume spacing artifacts.

    Examples:
    -----------------------------------------
    RelevantCoursework
    → Relevant Coursework

    DataStructures
    → Data Structures
    -----------------------------------------
    """

    text = (
        text or ""
    )

    # camelCase spacing
    text = re.sub(
        r"([a-z])([A-Z])",
        r"\1 \2",
        text,
    )

    # symbol spacing
    text = re.sub(
        r"([A-Za-z])&([A-Za-z])",
        r"\1 & \2",
        text,
    )

    # colon spacing
    text = re.sub(
        r"([a-zA-Z]):([A-Z])",
        r"\1: \2",
        text,
    )

    return text.strip()