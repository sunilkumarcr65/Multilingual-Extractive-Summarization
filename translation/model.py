from __future__ import annotations

import threading
from functools import lru_cache
from typing import Tuple

import torch

try:
    from transformers import (
        MBart50Tokenizer,
        MBartForConditionalGeneration,
    )

    HAS_TRANSFORMERS = True

except ImportError:

    MBart50Tokenizer = None
    MBartForConditionalGeneration = None

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

DEFAULT_TRANSLATION_MODEL = (
    "facebook/"
    "mbart-large-50-many-to-many-mmt"
)


# ---------------------------------------------------------
# Thread Safety
# ---------------------------------------------------------

_MODEL_LOCK = threading.Lock()


# ---------------------------------------------------------
# Model Loader
# ---------------------------------------------------------

@lru_cache(maxsize=2)
def load_translation_model(
    model_name: str = (
        DEFAULT_TRANSLATION_MODEL
    ),
) -> Tuple:
    """
    Load multilingual translation model.

    Returns:
    -----------------------------------------
    tokenizer, model
    -----------------------------------------
    """

    if not HAS_TRANSFORMERS:

        raise ImportError(
            "transformers is not installed."
        )

    model_name = (
        model_name
        or DEFAULT_TRANSLATION_MODEL
    )

    with _MODEL_LOCK:

        print(
            f"[TRANSLATION] "
            f"Loading model: "
            f"{model_name}"
        )

        tokenizer = (
            MBart50Tokenizer
            .from_pretrained(
                model_name
            )
        )

        model = (
            MBartForConditionalGeneration
            .from_pretrained(
                model_name
            )
            .to(DEVICE)
        )

        model.eval()

    return tokenizer, model
