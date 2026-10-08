# utils/text/sentence_splitter.py

from __future__ import annotations

import re
from typing import List
from functools import lru_cache

# ---------------------------------------------------------
# spaCy
# ---------------------------------------------------------

_SPACY_NLP = None

HAS_SPACY = False

@lru_cache(maxsize=1)
def load_spacy_model():

    try:
        import spacy

        for model_name in (
            "en_core_web_md",
            "en_core_web_sm",
        ):

            try:

                return spacy.load(
                    model_name,
                    disable=[
                        "ner",
                        "lemmatizer",
                    ],
                )

            except OSError:
                continue

    except ImportError:
        pass

    return None

# ---------------------------------------------------------
# NLTK
# ---------------------------------------------------------

HAS_NLTK = False

_NLTK_SENT_TOKENIZE = None

if not HAS_SPACY:

    try:
        import nltk

        try:

            nltk.data.find(
                "tokenizers/punkt"
            )

        except LookupError:

            nltk.download(
                "punkt",
                quiet=True,
            )

        from nltk.tokenize import (
            sent_tokenize,
        )

        _NLTK_SENT_TOKENIZE = (
            sent_tokenize
        )

        HAS_NLTK = True

    except ImportError:

        HAS_NLTK = False


# ---------------------------------------------------------
# Regex Fallback
# ---------------------------------------------------------

REGEX_SENTENCE_PATTERN = re.compile(
    r'(?<=[.!?।！？])\s+'
)


# ---------------------------------------------------------
# Splitter
# ---------------------------------------------------------

def split_sentences(
    text: str,
) -> List[str]:
    """
    Multilingual sentence splitter.

    Priority:
    -----------------------------------------
    spaCy
        ↓
    NLTK
        ↓
    regex fallback
    -----------------------------------------
    """

    text = (
        text or ""
    ).strip()

    if not text:

        return []

    # ---------------------------------------------------------
    # spaCy
    # ---------------------------------------------------------

    spacy_model = load_spacy_model()

    if spacy_model is not None:

        try:

            document = spacy_model(text)

            sentences = [
                sentence.text.strip()
                for sentence
                in document.sents
                if sentence.text.strip()
            ]

            if sentences:

                return sentences

        except Exception:
            pass

    # ---------------------------------------------------------
    # NLTK
    # ---------------------------------------------------------

    if (
        HAS_NLTK
        and _NLTK_SENT_TOKENIZE
        is not None
    ):

        try:

            sentences = [
                sentence.strip()
                for sentence
                in _NLTK_SENT_TOKENIZE(
                    text
                )
                if sentence.strip()
            ]

            if sentences:

                return sentences

        except Exception:
            pass

    # ---------------------------------------------------------
    # Regex fallback
    # ---------------------------------------------------------

    sentences = (
        REGEX_SENTENCE_PATTERN
        .split(text)
    )

    # ---------------------------------------------------------
    # Return cleaned sentences
    # ---------------------------------------------------------

    return [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
    ]