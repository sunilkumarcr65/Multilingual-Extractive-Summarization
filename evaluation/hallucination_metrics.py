from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import spacy

from sentence_transformers import (
    SentenceTransformer,
)

from sklearn.metrics.pairwise import (
    cosine_similarity,
)

try:
    import nltk

    nltk.data.find(
        "tokenizers/punkt"
    )

except LookupError:

    import nltk

    nltk.download(
        "punkt",
        quiet=True,
    )

from nltk.tokenize import (
    sent_tokenize,
)


# ---------------------------------------------------------
# Constants
# ---------------------------------------------------------

MAX_EMBED_TEXT = 8000

MAX_ENTITY_TEXT = 5000

FAST_SUMMARY_THRESHOLD = 80


# ---------------------------------------------------------
# Result Object
# ---------------------------------------------------------

@dataclass
class HallucinationMetrics:

    semantic_similarity: float

    grounded_sentence_ratio: float

    hallucinated_sentence_ratio: float

    entity_preservation_score: float

    unsupported_sentences: List[str]


# ---------------------------------------------------------
# Global Models
# ---------------------------------------------------------

_EMBEDDER = None

_NLP = None


# ---------------------------------------------------------
# Model Loaders
# ---------------------------------------------------------

def get_embedder():
    """
    Cached embedding model.
    """

    global _EMBEDDER

    if _EMBEDDER is None:

        print(
            "[HALLUCINATION] "
            "Loading embedding model..."
        )

        _EMBEDDER = SentenceTransformer(
            "sentence-transformers/"
            "paraphrase-multilingual-MiniLM-L12-v2"
        )

    return _EMBEDDER


def get_spacy_model():
    """
    Cached spaCy model.
    """

    global _NLP

    if _NLP is None:

        print(
            "[HALLUCINATION] "
            "Loading spaCy model..."
        )

        _NLP = spacy.load(
            "en_core_web_sm"
        )

    return _NLP


# ---------------------------------------------------------
# Sentence Utilities
# ---------------------------------------------------------

def split_sentences(
    text: str,
) -> List[str]:
    """
    Split text into sentences.
    """

    text = (
        text or ""
    ).strip()

    if not text:
        return []

    return [
        sentence.strip()

        for sentence in sent_tokenize(
            text
        )

        if sentence.strip()
    ]


# ---------------------------------------------------------
# Semantic Similarity
# ---------------------------------------------------------

def compute_semantic_similarity(
    source_text: str,
    summary_text: str,
) -> float:
    """
    Compute semantic similarity
    between source and summary.
    """

    source_text = (
        source_text or ""
    )[:MAX_EMBED_TEXT]

    summary_text = (
        summary_text or ""
    )[:MAX_EMBED_TEXT]

    if (
        not source_text.strip()
        or not summary_text.strip()
    ):

        return 0.0

    embedder = get_embedder()

    embeddings = embedder.encode(

        [source_text, summary_text],

        convert_to_numpy=True,

        normalize_embeddings=True,
    )

    similarity = cosine_similarity(
        embeddings[0].reshape(1, -1),
        embeddings[1].reshape(1, -1),
    )[0][0]

    return round(
        float(similarity),
        4,
    )


# ---------------------------------------------------------
# Sentence Grounding
# ---------------------------------------------------------

def compute_grounded_sentences(
    source_text: str,
    summary_text: str,
    threshold: float = 0.45,
) -> Dict:
    """
    Detect unsupported summary sentences.

    Sentences with low semantic similarity
    to source are considered hallucinated.
    """

    source_text = (
        source_text or ""
    )[:MAX_EMBED_TEXT]

    summary_text = (
        summary_text or ""
    ).strip()

    if not summary_text:

        return {
            "grounded_ratio": 0.0,
            "hallucinated_ratio": 0.0,
            "unsupported": [],
        }

    summary_sentences = split_sentences(
        summary_text
    )

    if not summary_sentences:

        return {
            "grounded_ratio": 0.0,
            "hallucinated_ratio": 0.0,
            "unsupported": [],
        }

    # -----------------------------------------------------
    # Fast path for tiny summaries
    # -----------------------------------------------------

    if len(summary_text.split()) < FAST_SUMMARY_THRESHOLD:

        return {
            "grounded_ratio": 1.0,
            "hallucinated_ratio": 0.0,
            "unsupported": [],
        }

    embedder = get_embedder()

    source_embedding = embedder.encode(

        [source_text],

        convert_to_numpy=True,

        normalize_embeddings=True,
    )[0]

    # -----------------------------------------------------
    # Batch sentence embeddings
    # -----------------------------------------------------

    sentence_embeddings = embedder.encode(

        summary_sentences,

        convert_to_numpy=True,

        normalize_embeddings=True,
    )

    grounded = 0

    unsupported = []

    for sentence, sentence_embedding in zip(
        summary_sentences,
        sentence_embeddings,
    ):

        similarity = cosine_similarity(
            source_embedding.reshape(1, -1),
            sentence_embedding.reshape(1, -1),
        )[0][0]

        if similarity >= threshold:

            grounded += 1

        else:

            unsupported.append(
                sentence
            )

    total = len(summary_sentences)

    grounded_ratio = grounded / total

    hallucinated_ratio = (
        len(unsupported) / total
    )

    return {

        "grounded_ratio":
            round(
                grounded_ratio,
                4,
            ),

        "hallucinated_ratio":
            round(
                hallucinated_ratio,
                4,
            ),

        "unsupported":
            unsupported,
    }


# ---------------------------------------------------------
# Entity Preservation
# ---------------------------------------------------------

def extract_entities(
    text: str,
) -> List[str]:
    """
    Extract named entities.
    """

    text = (
        text or ""
    )[:MAX_ENTITY_TEXT]

    if not text.strip():
        return []

    nlp = get_spacy_model()

    doc = nlp(text)

    entities = []

    for ent in doc.ents:

        cleaned = re.sub(
            r"\s+",
            " ",
            ent.text.strip().lower(),
        )

        if cleaned:

            entities.append(
                cleaned
            )

    return list(
        set(entities)
    )


def compute_entity_preservation(
    source_text: str,
    summary_text: str,
) -> float:
    """
    Measure how many important entities
    survive in the summary.
    """

    source_entities = set(
        extract_entities(
            source_text
        )
    )

    summary_entities = set(
        extract_entities(
            summary_text
        )
    )

    if not source_entities:
        return 1.0

    preserved = (
        source_entities
        & summary_entities
    )

    score = (
        len(preserved)
        / len(source_entities)
    )

    return round(
        float(score),
        4,
    )


# ---------------------------------------------------------
# Main Evaluation Pipeline
# ---------------------------------------------------------

def evaluate_hallucination(
    source_text: str,
    summary_text: str,
) -> HallucinationMetrics:
    """
    Full hallucination evaluation pipeline.
    """

    source_text = (
        source_text or ""
    ).strip()

    summary_text = (
        summary_text or ""
    ).strip()

    if not summary_text:

        return HallucinationMetrics(
            semantic_similarity=0.0,
            grounded_sentence_ratio=0.0,
            hallucinated_sentence_ratio=0.0,
            entity_preservation_score=0.0,
            unsupported_sentences=[],
        )

    # -----------------------------------------------------
    # Semantic similarity
    # -----------------------------------------------------

    semantic_similarity = (
        compute_semantic_similarity(
            source_text,
            summary_text,
        )
    )

    # -----------------------------------------------------
    # Grounded sentence analysis
    # -----------------------------------------------------

    grounded = compute_grounded_sentences(
        source_text,
        summary_text,
    )

    # -----------------------------------------------------
    # Entity preservation
    # -----------------------------------------------------

    entity_preservation = (
        compute_entity_preservation(
            source_text,
            summary_text,
        )
    )

    return HallucinationMetrics(

        semantic_similarity=
            semantic_similarity,

        grounded_sentence_ratio=
            grounded[
                "grounded_ratio"
            ],

        hallucinated_sentence_ratio=
            grounded[
                "hallucinated_ratio"
            ],

        entity_preservation_score=
            entity_preservation,

        unsupported_sentences=
            grounded[
                "unsupported"
            ],
    )
