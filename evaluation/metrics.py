from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd
import torch

from bert_score import BERTScorer
from nltk.translate.bleu_score import (
    SmoothingFunction,
    sentence_bleu,
)
from nltk.translate.meteor_score import (
    meteor_score,
)
from rouge_score import rouge_scorer


# ---------------------------------------------------------
# Global BERTScore Cache
# ---------------------------------------------------------

_BERT_SCORERS = {}


# ---------------------------------------------------------
# ROUGE
# ---------------------------------------------------------

def compute_rouge(
    reference: str,
    generated: str,
) -> Dict[str, float]:
    """
    Compute ROUGE scores.
    """

    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"],
        use_stemmer=True,
    )

    scores = scorer.score(
        reference,
        generated,
    )

    return {
        "rouge1": round(
            float(scores["rouge1"].fmeasure),
            4,
        ),
        "rouge2": round(
            float(scores["rouge2"].fmeasure),
            4,
        ),
        "rougeL": round(
            float(scores["rougeL"].fmeasure),
            4,
        ),
    }


# ---------------------------------------------------------
# BLEU
# ---------------------------------------------------------

def compute_bleu(
    reference: str,
    generated: str,
) -> float:
    """
    Compute BLEU score.
    """

    smoothing = (
        SmoothingFunction()
        .method4
    )

    score = sentence_bleu(
        [reference.split()],
        generated.split(),
        smoothing_function=smoothing,
    )

    return round(float(score), 4)


# ---------------------------------------------------------
# METEOR
# ---------------------------------------------------------

def compute_meteor(
    reference: str,
    generated: str,
) -> float:
    """
    Compute METEOR score.
    """

    try:

        score = meteor_score(
            [reference.split()],
            generated.split(),
        )

        return round(float(score), 4)

    except Exception:
        return 0.0


# ---------------------------------------------------------
# BERTScore
# ---------------------------------------------------------

def get_bertscorer(
    language: str = "en",
    device: str = "cpu",
) -> BERTScorer:
    """
    Cached BERTScorer loader.
    """

    cache_key = (
        language,
        device,
    )

    if cache_key not in _BERT_SCORERS:

        print(
            f"[METRICS] Loading "
            f"BERTScore model "
            f"({language})"
        )

        _BERT_SCORERS[cache_key] = BERTScorer(
            lang=language,
            rescale_with_baseline=True,
            device=device,
        )

    return _BERT_SCORERS[cache_key]


def compute_bertscore(
    references: List[str],
    generations: List[str],
    language: str = "en",
    device: str = "cpu",
) -> List[float]:
    """
    Compute BERTScore F1.
    """

    scorer = get_bertscorer(
        language=language,
        device=device,
    )

    with torch.no_grad():

        _, _, f1 = scorer.score(
            generations,
            references,
            batch_size=4,
            verbose=False,
        )

    return (
        f1.cpu()
        .numpy()
        .round(4)
        .tolist()
    )


# ---------------------------------------------------------
# Combined Metrics
# ---------------------------------------------------------

def compute_all_metrics(
    reference: str,
    generated: str,
) -> Dict[str, float]:
    """
    Compute all evaluation metrics.
    """

    generated = (
        generated or ""
    ).strip()

    reference = (
        reference or ""
    ).strip()

    if not generated:

        return {
            "rouge1": 0.0,
            "rouge2": 0.0,
            "rougeL": 0.0,
            "bleu": 0.0,
            "meteor": 0.0,
        }

    # -----------------------------------------------------
    # No reference available
    # -----------------------------------------------------

    if not reference:

        return {
            "compression_ratio": round(
                len(generated.split())
                / max(len(generated.split()), 1),
                4,
            )
        }

    rouge_scores = compute_rouge(
        reference,
        generated,
    )

    bleu_score = compute_bleu(
        reference,
        generated,
    )

    meteor = compute_meteor(
        reference,
        generated,
    )

    compression_ratio = round(
        len(generated.split())
        / max(len(reference.split()), 1),
        4,
    )

    return {
        **rouge_scores,
        "bleu": bleu_score,
        "meteor": meteor,
        "compression_ratio":
            compression_ratio,
    }


# ---------------------------------------------------------
# DataFrame Integration
# ---------------------------------------------------------

def add_bertscore_to_dataframe(
    df: pd.DataFrame,
    language: str = "en",
    device: str = "cpu",
) -> pd.DataFrame:
    """
    Add BERTScore column to evaluation DataFrame.

    Expected columns:
    - summary
    - reference
    - method
    """

    df = df.copy()

    df["bertscore"] = np.nan

    methods = df["method"].unique()

    for method in methods:

        mask = df["method"] == method

        generations = df.loc[
            mask,
            "summary",
        ].tolist()

        references = df.loc[
            mask,
            "reference",
        ].tolist()

        if not generations:
            continue

        print(
            f"[METRICS] "
            f"BERTScore "
            f"[{method}] "
            f"({len(generations)} samples)"
        )

        scores = compute_bertscore(
            references=references,
            generations=generations,
            language=language,
            device=device,
        )

        df.loc[
            mask,
            "bertscore",
        ] = scores

    return df