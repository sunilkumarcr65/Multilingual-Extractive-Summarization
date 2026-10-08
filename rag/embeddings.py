from __future__ import annotations

from functools import lru_cache
from typing import List, Optional

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from preprocessing.language_detection import (
    detect_language,
)


# ---------------------------------------------------------
# Device
# ---------------------------------------------------------

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print(
    f"[EMBEDDINGS] Using device: "
    f"{DEVICE}"
)


# ---------------------------------------------------------
# Language Config
# ---------------------------------------------------------

ENGLISH_LANGUAGES = {
    "en",
    "unknown",
}

MINILM_MODEL = (
    "sentence-transformers/"
    "all-MiniLM-L6-v2"
)

LABSE_MODEL = (
    "sentence-transformers/"
    "LaBSE"
)

MULTILINGUAL_MINILM_MODEL = (
    "sentence-transformers/"
    "paraphrase-multilingual-MiniLM-L12-v2"
)


# ---------------------------------------------------------
# Model Loaders
# ---------------------------------------------------------

@lru_cache(maxsize=1)
def get_minilm_model() -> SentenceTransformer:
    """
    English-focused embedding model.
    Faster and lightweight.
    """

    print(
        "[EMBEDDINGS] "
        "Loading MiniLM..."
    )

    return SentenceTransformer(
        MINILM_MODEL,
        device=DEVICE,
    )


@lru_cache(maxsize=1)
def get_labse_model() -> SentenceTransformer:
    """
    Strong multilingual embedding model.
    Better for cross-lingual semantics.
    """

    print(
        "[EMBEDDINGS] "
        "Loading LaBSE..."
    )

    return SentenceTransformer(
        LABSE_MODEL,
        device=DEVICE,
    )


@lru_cache(maxsize=1)
def get_multilingual_minilm_model() -> SentenceTransformer:
    """
    Balanced multilingual model.

    Faster than LaBSE,
    better multilingual support than MiniLM.
    """

    print(
        "[EMBEDDINGS] "
        "Loading multilingual MiniLM..."
    )

    return SentenceTransformer(
        MULTILINGUAL_MINILM_MODEL,
        device=DEVICE,
    )


# ---------------------------------------------------------
# Model Selection
# ---------------------------------------------------------

def select_embedding_model(
    language: str,
    multilingual_strategy: str = "balanced",
) -> SentenceTransformer:
    """
    Dynamically select embedding model.

    Strategies
    -----------------------------------------
    balanced:
        multilingual MiniLM

    accurate:
        LaBSE

    fast:
        MiniLM whenever possible
    -----------------------------------------
    """

    language = (
        language or "unknown"
    ).lower()

    # ---------------------------------------------------------
    # English
    # ---------------------------------------------------------

    if language in ENGLISH_LANGUAGES:

        return get_minilm_model()

    # ---------------------------------------------------------
    # Non-English
    # ---------------------------------------------------------

    if multilingual_strategy == "accurate":

        return get_labse_model()

    if multilingual_strategy == "balanced":

        return (
            get_multilingual_minilm_model()
        )

    return get_minilm_model()


# ---------------------------------------------------------
# Sentence Embeddings
# ---------------------------------------------------------

def embed_sentences(
    sentences: List[str],
    language: Optional[str] = None,
    multilingual_strategy: str = "balanced",
    batch_size: int = 16,
    normalize: bool = True,
) -> np.ndarray:
    """
    Generate sentence embeddings.
    """

    if not sentences:

        return np.zeros(
            (0, 384),
            dtype=np.float32,
        )

    # ---------------------------------------------------------
    # Auto language detection
    # ---------------------------------------------------------

    if language is None:

        sample = " ".join(
            sentences[:10]
        )

        language = detect_language(
            sample
        )

    model = select_embedding_model(
        language=language,
        multilingual_strategy=(
            multilingual_strategy
        ),
    )

    embeddings = model.encode(
        sentences,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=normalize,
        show_progress_bar=False,
    )

    return embeddings.astype(
        np.float32
    )


# ---------------------------------------------------------
# Single Text Embedding
# ---------------------------------------------------------

def embed_text(
    text: str,
    language: Optional[str] = None,
    multilingual_strategy: str = "balanced",
) -> np.ndarray:
    """
    Embed single text/document.
    """

    embeddings = embed_sentences(
        [text],
        language=language,
        multilingual_strategy=(
            multilingual_strategy
        ),
    )

    return embeddings[0]


# ---------------------------------------------------------
# Semantic Similarity
# ---------------------------------------------------------

def embedding_cosine_similarity(
    embedding_a: np.ndarray,
    embedding_b: np.ndarray,
) -> float:
    """
    Compute cosine similarity.
    """

    numerator = np.dot(
        embedding_a,
        embedding_b,
    )

    denominator = (
        np.linalg.norm(embedding_a)
        * np.linalg.norm(embedding_b)
    )

    if denominator == 0:
        return 0.0

    return float(
        numerator / denominator
    )