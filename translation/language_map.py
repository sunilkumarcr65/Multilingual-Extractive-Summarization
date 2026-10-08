from __future__ import annotations

from typing import Dict


# ---------------------------------------------------------
# mBART-50 Language Codes
# ---------------------------------------------------------

MBART_LANGUAGE_CODES: Dict[str, str] = {
    # -----------------------------------------------------
    # English
    # -----------------------------------------------------
    "en": "en_XX",

    # -----------------------------------------------------
    # Indian Languages
    # -----------------------------------------------------
    "hi": "hi_IN",
    "bn": "bn_IN",
    "mr": "mr_IN",
    "ta": "ta_IN",
    "te": "te_IN",

    # -----------------------------------------------------
    # European Languages
    # -----------------------------------------------------
    "fr": "fr_XX",
    "de": "de_DE",
    "es": "es_XX",
    "it": "it_IT",
    "pt": "pt_XX",
    "nl": "nl_XX",
    "pl": "pl_PL",
    "tr": "tr_TR",

    # -----------------------------------------------------
    # Asian Languages
    # -----------------------------------------------------
    "zh": "zh_CN",
    "ja": "ja_XX",
    "ko": "ko_KR",

    # -----------------------------------------------------
    # Middle Eastern Languages
    # -----------------------------------------------------
    "ar": "ar_AR",
    "ru": "ru_RU",
}


# ---------------------------------------------------------
# Unsupported But Planned
# ---------------------------------------------------------

UNSUPPORTED_LANGUAGES = {
    # Gujarati unsupported in mBART-50
    "gu",

    # Punjabi
    "pa",

    # Malayalam
    "ml",

    # Kannada
    "kn",

    # Odia
    "or",
}


# ---------------------------------------------------------
# Language Helpers
# ---------------------------------------------------------

def get_language_code(
    language: str,
) -> str | None:
    """
    Get mBART language code.
    """

    if not language:
        return None

    language = (
        language
        .strip()
        .lower()
    )

    return MBART_LANGUAGE_CODES.get(
        language
    )


def is_supported_language(
    language: str,
) -> bool:
    """
    Check whether language is supported.
    """

    return (
        get_language_code(
            language
        )
        is not None
    )


def supported_languages() -> Dict[str, str]:
    """
    Return supported language map.
    """

    return dict(
        MBART_LANGUAGE_CODES
    )


def unsupported_languages():
    """
    Return unsupported language set.
    """

    return set(
        UNSUPPORTED_LANGUAGES
    )