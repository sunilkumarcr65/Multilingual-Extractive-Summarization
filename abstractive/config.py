from __future__ import annotations


# ---------------------------------------------------------
# Model Registry
# ---------------------------------------------------------

ENGLISH_SUMMARIZATION_MODEL = (
    "google/flan-t5-base"
)

MULTILINGUAL_SUMMARIZATION_MODEL = (
    "csebuetnlp/mT5_multilingual_XLSum"
)


# ---------------------------------------------------------
# Language Groups
# ---------------------------------------------------------

ENGLISH_LANGUAGES = {
    "en",
    "unknown",
}


# ---------------------------------------------------------
# Chunking
# ---------------------------------------------------------

MAX_INPUT_TOKENS = 1024

CHUNK_OVERLAP = 128


# ---------------------------------------------------------
# Generation Parameters
# ---------------------------------------------------------

DEFAULT_MAX_NEW_TOKENS = 256

DEFAULT_MIN_NEW_TOKENS = 64

DEFAULT_NUM_BEAMS = 4

DEFAULT_LENGTH_PENALTY = 1.0

DEFAULT_REPETITION_PENALTY = 1.2

DEFAULT_NO_REPEAT_NGRAM = 3


# ---------------------------------------------------------
# Sampling
# ---------------------------------------------------------

DEFAULT_TEMPERATURE = 0.7

DEFAULT_TOP_K = 50

DO_SAMPLE = True

DEFAULT_TOP_P = 0.95


# ---------------------------------------------------------
# Summary Styles
# ---------------------------------------------------------

SUPPORTED_SUMMARY_STYLES = {
    "default",
    "brief",
    "detailed",
    "bullet",
}


# ---------------------------------------------------------
# Device
# ---------------------------------------------------------

USE_FP16 = True


# ---------------------------------------------------------
# Translation
# ---------------------------------------------------------

ENABLE_TRANSLATION = True


# ---------------------------------------------------------
# Cache
# ---------------------------------------------------------

MODEL_CACHE_SIZE = 4