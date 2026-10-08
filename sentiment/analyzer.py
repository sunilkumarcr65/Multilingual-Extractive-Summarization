from __future__ import annotations

from typing import Dict, List, Optional

import torch

from sentiment.chunking import (
    chunk_text_by_tokens,
)

from sentiment.fallback import (
    fallback_sentiment,
)

from sentiment.model import (
    DEFAULT_SENTIMENT_MODEL,
    SENTIMENT_LABELS,
    load_sentiment_model,
)


# ---------------------------------------------------------
# Config
# ---------------------------------------------------------

MAX_MODEL_TOKENS = 512


# ---------------------------------------------------------
# Inference
# ---------------------------------------------------------

@torch.inference_mode()
def infer_sentiment_chunks(
    chunks: List[str],
    tokenizer,
    model,
) -> Dict[str, float]:
    """
    Run sentiment inference on chunks.

    Uses weighted aggregation
    by token length.
    """

    device = next(
        model.parameters()
    ).device

    weighted_scores = None

    total_weight = 0

    for chunk in chunks:

        inputs = tokenizer(
            chunk,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_MODEL_TOKENS,
            padding=True,
        ).to(device)

        token_count = (
            inputs["input_ids"]
            .shape[1]
        )

        logits = model(
            **inputs
        ).logits

        probabilities = (
            torch.softmax(
                logits / 1.5,
                dim=-1,
            )[0]
            .cpu()
        )

        weighted = (
            probabilities
            * token_count
        )

        if weighted_scores is None:

            weighted_scores = weighted

        else:

            weighted_scores += (
                weighted
            )

        total_weight += (
            token_count
        )

    # ---------------------------------------------------------
    # Safety fallback
    # ---------------------------------------------------------

    if (
        weighted_scores is None
        or total_weight == 0
    ):

        return {
            "negative": 0.0,
            "neutral": 1.0,
            "positive": 0.0,
        }

    averaged = (
        weighted_scores
        / total_weight
    ).tolist()

    return {
        label:
        round(
            float(score),
            4,
        )
        for label, score
        in zip(
            SENTIMENT_LABELS,
            averaged,
        )
    }


# ---------------------------------------------------------
# Public API
# ---------------------------------------------------------

def analyze_sentiment(
    text: str,
    document_type: str = "general",
    model_name: Optional[str] = None,
) -> Optional[Dict[str, float]]:
    """
    Analyze multilingual sentiment.

    Returns:
    -----------------------------------------
    {
        "negative": ...,
        "neutral": ...,
        "positive": ...
    }
    -----------------------------------------
    """

    if document_type in {
        "resume",
        "research_paper",
    }:

        return {
            "negative": 0.05,
            "neutral": 0.90,
            "positive": 0.05,
        }

    text = (
        text or ""
    ).strip()

    if not text:

        return None

    
    LOW_SENTIMENT_DOC_HINTS = {
        "education",
        "skills",
        "experience",
        "methodology",
        "abstract",
        "results",
        "conclusion",
        "project",
        "technical",
        "algorithm",
    }

    lower_text = text.lower()

    hint_count = sum(
        keyword in lower_text
        for keyword in LOW_SENTIMENT_DOC_HINTS
    )

    if hint_count >= 3:

        return {
            "negative": 0.05,
            "neutral": 0.90,
            "positive": 0.05,
        }
    # ---------------------------------------------------------
    # Load model
    # ---------------------------------------------------------

    try:

        tokenizer, model = (
            load_sentiment_model(
                model_name
                or DEFAULT_SENTIMENT_MODEL
            )
        )

    except Exception:
        print("[SENTIMENT] Falling back to rule-based sentiment")

        return fallback_sentiment(
            text
        )

    # ---------------------------------------------------------
    # Chunking
    # ---------------------------------------------------------

    chunks = chunk_text_by_tokens(
        text=text,
        tokenizer=tokenizer,
    )

    if not chunks:

        return {
            "negative": 0.0,
            "neutral": 1.0,
            "positive": 0.0,
        }

    print("[SENTIMENT] Using transformer model")
    # ---------------------------------------------------------
    # Inference
    # ---------------------------------------------------------

    return infer_sentiment_chunks(
        chunks=chunks,
        tokenizer=tokenizer,
        model=model,
    )


# ---------------------------------------------------------
# Batch API
# ---------------------------------------------------------

def analyze_sentiment_batch(
    texts: List[str],
    model_name: Optional[str] = None,
) -> List[
    Optional[
        Dict[str, float]
    ]
]:
    """
    Batch sentiment analysis.
    """

    return [
        analyze_sentiment(
            text=text,
            model_name=model_name,
        )
        for text in texts
    ]


# ---------------------------------------------------------
# Label Helper
# ---------------------------------------------------------

def get_dominant_sentiment(
    scores: Dict[str, float],
) -> str:
    """
    Get dominant sentiment label.
    """

    if not scores:

        return "neutral"

    return max(
        scores,
        key=scores.get,
    )
