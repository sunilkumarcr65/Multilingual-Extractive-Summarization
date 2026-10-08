from __future__ import annotations

import threading
from functools import lru_cache
from typing import Tuple

import torch

try:
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
    )

    HAS_TRANSFORMERS = True

except ImportError:

    AutoModelForSequenceClassification = None
    AutoTokenizer = None

    HAS_TRANSFORMERS = False


# ---------------------------------------------------------
# Device
# ---------------------------------------------------------

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ---------------------------------------------------------
# Model Config
# ---------------------------------------------------------

DEFAULT_SENTIMENT_MODEL = (
    "cardiffnlp/twitter-roberta-base-sentiment-latest"
)


# ---------------------------------------------------------
# Thread Safety
# ---------------------------------------------------------

_MODEL_LOCK = threading.Lock()


# ---------------------------------------------------------
# Loader
# ---------------------------------------------------------

@lru_cache(maxsize=2)
def load_sentiment_model(
    model_name: str = (
        DEFAULT_SENTIMENT_MODEL
    ),
) -> Tuple:
    """
    Load multilingual sentiment model.

    Returns:
    -----------------------------------------
    tokenizer, model
    -----------------------------------------
    """

    if not HAS_TRANSFORMERS:

        raise ImportError(
            "transformers is not installed."
        )

    with _MODEL_LOCK:

        print(
            f"[SENTIMENT] "
            f"Loading model: "
            f"{model_name}"
        )

        tokenizer = (
            AutoTokenizer
            .from_pretrained(
                model_name,
                use_fast=True,
            )
        )

        model = (
            AutoModelForSequenceClassification
            .from_pretrained(
                model_name
            )
            .to(DEVICE)
        )

        model.eval()

    return tokenizer, model


# ---------------------------------------------------------
# Labels
# ---------------------------------------------------------

SENTIMENT_LABELS = (
    "negative",
    "neutral",
    "positive",
)