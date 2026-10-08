# orchestration/model_router.py

from __future__ import annotations

from typing import Optional


# ---------------------------------------------------------
# English Languages
# ---------------------------------------------------------

ENGLISH_LANGUAGES = {
    "en",
    "english",
    "unknown",
}


# ---------------------------------------------------------
# Base Models
# ---------------------------------------------------------

ENGLISH_SUMMARIZATION_MODEL = (
    "google/flan-t5-base"
)

MULTILINGUAL_SUMMARIZATION_MODEL = (
    "csebuetnlp/mT5_multilingual_XLSum"
)


# ---------------------------------------------------------
# Document-Specific Models
# ---------------------------------------------------------

DOCUMENT_MODELS = {

    # -----------------------------------------------------
    # Resume / CV
    # -----------------------------------------------------

    "resume":
        "google/flan-t5-base",

    # -----------------------------------------------------
    # Research / academic
    # -----------------------------------------------------

    "research_paper":
        "google/flan-t5-base",

    # -----------------------------------------------------
    # News
    # -----------------------------------------------------

    "news":
        "google/flan-t5-base",

    # -----------------------------------------------------
    # General docs
    # -----------------------------------------------------

    "general":
        "google/flan-t5-base",

    # -----------------------------------------------------
    # Multilingual
    # -----------------------------------------------------

    "multilingual":
        "csebuetnlp/mT5_multilingual_XLSum",
}


# ---------------------------------------------------------
# Translation Models
# ---------------------------------------------------------

DEFAULT_TRANSLATION_MODEL = (
    "facebook/"
    "mbart-large-50-many-to-many-mmt"
)


FORCED_EXTRACTIVE_TYPES = {
    "resume",
}


# ---------------------------------------------------------
# Compression Rules
# ---------------------------------------------------------

# Resume summaries should preserve much more content.
# Research/news/general docs can compress more aggressively.

COMPRESSION_BY_DOCTYPE = {

    "resume": 0.85,
    "research_paper": 0.3,
    "news": 0.4,
    "general": 0.35,
}


# ---------------------------------------------------------
# MMR Diversity Rules
# ---------------------------------------------------------

# Lower lambda = less diversity penalty
# Helps preserve all resume sections.

MMR_LAMBDA_BY_DOCTYPE = {

    "resume": 0.75,
    "news": 0.7,
    "general": 0.6,
    "research_paper": 0.5,
}


# ---------------------------------------------------------
# Fast Path Rules
# ---------------------------------------------------------

# 1-page resumes often exceed 600 tokens
# because of formatting/bullets.
# Raising threshold avoids unnecessary chunking.

SHORT_DOC_TOKEN_THRESHOLD = 900


# ---------------------------------------------------------
# Registry
# ---------------------------------------------------------

MODEL_REGISTRY = {

    "summarization": {

        "english":
            ENGLISH_SUMMARIZATION_MODEL,

        "multilingual":
            MULTILINGUAL_SUMMARIZATION_MODEL,

        "document_models":
            DOCUMENT_MODELS,
    },

    "translation": {

        "default":
            DEFAULT_TRANSLATION_MODEL,
    },
}


# ---------------------------------------------------------
# Language Helpers
# ---------------------------------------------------------

def is_english_language(
    language: Optional[str],
) -> bool:
    """
    Check whether language
    should use English model.
    """

    if not language:

        return True

    return (
        language
        .strip()
        .lower()
        in ENGLISH_LANGUAGES
    )


# ---------------------------------------------------------
# Summarization Mode Router
# ---------------------------------------------------------

def select_summarization_mode(
    document_type: str,
    user_mode: str,
) -> str:
    """
    Force extractive summarization
    for structured documents like resumes.
    """

    document_type = (
        document_type
        or "general"
    ).strip().lower()

    user_mode = (
        user_mode
        or "hybrid"
    ).strip().lower()

    # -----------------------------------------------------
    # Forced routing
    # -----------------------------------------------------

    if document_type in FORCED_EXTRACTIVE_TYPES:
        return "extractive"

    return user_mode


# ---------------------------------------------------------
# Compression Ratio Router
# ---------------------------------------------------------

def get_compression_ratio(
    document_type: str,
    user_ratio: float,
) -> float:
    """
    Dynamically select compression ratio
    based on document type.

    Only overrides when user is using
    default compression value.
    """

    document_type = (
        document_type
        or "general"
    ).strip().lower()

    # -----------------------------------------------------
    # Preserve explicit user setting
    # -----------------------------------------------------

    if user_ratio != 0.25:

        return user_ratio

    return COMPRESSION_BY_DOCTYPE.get(
        document_type,
        user_ratio,
    )


# ---------------------------------------------------------
# MMR Lambda Router
# ---------------------------------------------------------

def get_mmr_lambda(
    document_type: str,
) -> float:
    """
    Return optimal MMR lambda
    for document type.
    """

    document_type = (
        document_type
        or "general"
    ).strip().lower()

    return MMR_LAMBDA_BY_DOCTYPE.get(
        document_type,
        0.6,
    )


# ---------------------------------------------------------
# Fast Path Decision
# ---------------------------------------------------------

def should_use_fast_path(
    estimated_tokens: int,
    document_type: str = "general",
) -> bool:
    """
    Decide whether to bypass
    chunking pipeline.
    """

    document_type = (
        document_type
        or "general"
    ).strip().lower()

    # -----------------------------------------------------
    # Research papers should
    # always use chunking
    # -----------------------------------------------------

    if document_type == "research_paper":

        return False

    return (
        estimated_tokens
        <= SHORT_DOC_TOKEN_THRESHOLD
    )


# ---------------------------------------------------------
# Generic Language Router
# ---------------------------------------------------------

def select_abstractive_model(
    source_language: str,
    override_model: Optional[
        str
    ] = None,
) -> str:
    """
    Select summarization model
    based on language only.
    """

    # -----------------------------------------------------
    # Manual override
    # -----------------------------------------------------

    if override_model:

        return override_model

    source_language = (
        source_language
        or "unknown"
    ).strip().lower()

    # -----------------------------------------------------
    # English
    # -----------------------------------------------------

    if is_english_language(
        source_language
    ):

        return (
            ENGLISH_SUMMARIZATION_MODEL
        )

    # -----------------------------------------------------
    # Multilingual
    # -----------------------------------------------------

    return (
        MULTILINGUAL_SUMMARIZATION_MODEL
    )


# ---------------------------------------------------------
# Document-Aware Router
# ---------------------------------------------------------

def select_document_model(
    document_type: str,
    language: str,
    override_model: Optional[
        str
    ] = None,
) -> str:
    """
    Select optimal summarization model
    based on BOTH:
    - document type
    - language
    """

    # -----------------------------------------------------
    # Manual override
    # -----------------------------------------------------

    if override_model:

        return override_model

    document_type = (
        document_type
        or "general"
    ).strip().lower()

    language = (
        language
        or "unknown"
    ).strip().lower()

    # -----------------------------------------------------
    # Multilingual
    # -----------------------------------------------------

    if not is_english_language(
        language
    ):

        return DOCUMENT_MODELS[
            "multilingual"
        ]

    # -----------------------------------------------------
    # Document-specific routing
    # -----------------------------------------------------

    return DOCUMENT_MODELS.get(

        document_type,

        DOCUMENT_MODELS[
            "general"
        ],
    )


# ---------------------------------------------------------
# Translation Router
# ---------------------------------------------------------

def select_translation_model(
    source_language: str,
    target_language: str,
) -> str:
    """
    Select translation model.
    """

    source_language = (
        source_language
        or ""
    ).strip().lower()

    target_language = (
        target_language
        or ""
    ).strip().lower()

    return (
        DEFAULT_TRANSLATION_MODEL
    )


# ---------------------------------------------------------
# Registry Access
# ---------------------------------------------------------

def get_model_registry():
    """
    Return model registry.
    """

    return dict(
        MODEL_REGISTRY
    )
