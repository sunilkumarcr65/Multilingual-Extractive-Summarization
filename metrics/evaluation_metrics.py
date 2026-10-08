# metrics/evaluation_metrics.py

from __future__ import annotations

from typing import Dict

import evaluate
import nltk

from nltk.tokenize import word_tokenize


# ---------------------------------------------------------
# NLTK Downloads
# ---------------------------------------------------------

try:

    nltk.data.find(
        "tokenizers/punkt"
    )

except LookupError:

    nltk.download(
        "punkt",
        quiet=True,
    )


# ---------------------------------------------------------
# Metric Loaders
# ---------------------------------------------------------

ROUGE = evaluate.load(
    "rouge"
)

BLEU = evaluate.load(
    "bleu"
)

METEOR = evaluate.load(
    "meteor"
)

try:

    BERTSCORE = evaluate.load(
        "bertscore"
    )

    HAS_BERTSCORE = True

except Exception:

    BERTSCORE = None

    HAS_BERTSCORE = False


# ---------------------------------------------------------
# Text Cleanup
# ---------------------------------------------------------

def normalize_text(
    text: str,
) -> str:
    """
    Normalize evaluation text.
    """

    return (
        (text or "")
        .strip()
        .replace("\n", " ")
    )


# ---------------------------------------------------------
# Compression Ratio
# ---------------------------------------------------------

def compute_compression_ratio(
    original_text: str,
    summary_text: str,
) -> Dict[str, float]:
    """
    Compute compression ratio.

    Lower value = more compression.
    """

    original_words = len(
        normalize_text(
            original_text
        ).split()
    )

    summary_words = len(
        normalize_text(
            summary_text
        ).split()
    )

    if original_words == 0:

        ratio = 0.0

    else:

        ratio = round(
            summary_words
            / original_words,
            4,
        )

    return {
        "Compression Ratio":
            ratio
    }


# ---------------------------------------------------------
# ROUGE
# ---------------------------------------------------------

def compute_rouge(
    reference: str,
    candidate: str,
) -> Dict[str, float]:
    """
    Compute ROUGE scores.
    """

    reference = normalize_text(
        reference
    )

    candidate = normalize_text(
        candidate
    )

    if (
        not reference
        or not candidate
    ):

        return {}

    try:

        scores = ROUGE.compute(

            predictions=[
                candidate
            ],

            references=[
                reference
            ],

            use_stemmer=True,
        )

        return {

            "ROUGE-1":
                round(
                    scores.get(
                        "rouge1",
                        0.0,
                    ),
                    4,
                ),

            "ROUGE-2":
                round(
                    scores.get(
                        "rouge2",
                        0.0,
                    ),
                    4,
                ),

            "ROUGE-L":
                round(
                    scores.get(
                        "rougeL",
                        0.0,
                    ),
                    4,
                ),
        }

    except Exception as error:

        print(
            "[METRICS] "
            f"ROUGE failed: "
            f"{error}"
        )

        return {}


# ---------------------------------------------------------
# BLEU
# ---------------------------------------------------------

def compute_bleu(
    reference: str,
    candidate: str,
) -> Dict[str, float]:
    """
    Compute BLEU score.
    """

    reference = normalize_text(
        reference
    )

    candidate = normalize_text(
        candidate
    )

    if (
        not reference
        or not candidate
    ):

        return {}

    try:

        score = BLEU.compute(

            predictions=[
                candidate
            ],

            references=[
                [reference]
            ],
        )

        return {

            "BLEU":
                round(
                    score.get(
                        "bleu",
                        0.0,
                    ),
                    4,
                )
        }

    except Exception as error:

        print(
            "[METRICS] "
            f"BLEU failed: "
            f"{error}"
        )

        return {}


# ---------------------------------------------------------
# METEOR
# ---------------------------------------------------------

def compute_meteor(
    reference: str,
    candidate: str,
) -> Dict[str, float]:
    """
    Compute METEOR score.
    """

    reference = normalize_text(
        reference
    )

    candidate = normalize_text(
        candidate
    )

    if (
        not reference
        or not candidate
    ):

        return {}

    try:

        reference_tokens = (
            word_tokenize(
                reference
            )
        )

        candidate_tokens = (
            word_tokenize(
                candidate
            )
        )

        score = METEOR.compute(

            predictions=[
                " ".join(
                    candidate_tokens
                )
            ],

            references=[
                " ".join(
                    reference_tokens
                )
            ],
        )

        return {

            "METEOR":
                round(
                    score.get(
                        "meteor",
                        0.0,
                    ),
                    4,
                )
        }

    except Exception as error:

        print(
            "[METRICS] "
            f"METEOR failed: "
            f"{error}"
        )

        return {}


# ---------------------------------------------------------
# BERTScore
# ---------------------------------------------------------

def compute_bertscore(
    reference: str,
    candidate: str,
    lang: str = "en",
) -> Dict[str, float]:
    """
    Compute BERTScore F1.
    """

    if not HAS_BERTSCORE:

        return {}

    reference = normalize_text(
        reference
    )

    candidate = normalize_text(
        candidate
    )

    if (
        not reference
        or not candidate
    ):

        return {}

    try:

        result = (
            BERTSCORE.compute(

                predictions=[
                    candidate
                ],

                references=[
                    reference
                ],

                lang=lang,
            )
        )

        f1 = result[
            "f1"
        ][0]

        return {

            "BERTScore":
                round(
                    float(f1),
                    4,
                )
        }

    except Exception as error:

        print(
            "[METRICS] "
            f"BERTScore failed: "
            f"{error}"
        )

        return {}


# ---------------------------------------------------------
# Full Evaluation
# ---------------------------------------------------------

def compute_metrics(
    original_text: str,
    summary_text: str,
    reference_summary: str | None = None,
    lang: str = "en",
) -> Dict[str, float]:
    """
    Compute all evaluation metrics.

    If no reference summary is provided,
    original document is used as
    pseudo-reference.
    """

    metrics = {}

    # -----------------------------------------------------
    # Normalize
    # -----------------------------------------------------

    original_text = normalize_text(
        original_text
    )

    summary_text = normalize_text(
        summary_text
    )

    reference_summary = normalize_text(
        reference_summary or ""
    )

    # -----------------------------------------------------
    # Empty summary guard
    # -----------------------------------------------------

    if not summary_text:

        return {
            "Compression Ratio": 0.0
        }

    # -----------------------------------------------------
    # Reference selection
    # -----------------------------------------------------

    reference_text = (
        reference_summary
        if reference_summary
        else original_text
    )

    # -----------------------------------------------------
    # ROUGE
    # -----------------------------------------------------

    if reference_text:

        metrics.update(

            compute_rouge(
                reference_text,
                summary_text,
            )
        )

    # -----------------------------------------------------
    # BLEU
    # -----------------------------------------------------

    if reference_text:

        metrics.update(

            compute_bleu(
                reference_text,
                summary_text,
            )
        )

    # -----------------------------------------------------
    # METEOR
    # -----------------------------------------------------

    if reference_text:

        metrics.update(

            compute_meteor(
                reference_text,
                summary_text,
            )
        )

    # -----------------------------------------------------
    # BERTScore
    # -----------------------------------------------------

    if reference_text:

        metrics.update(

            compute_bertscore(
                reference_text,
                summary_text,
                lang=lang,
            )
        )

    # -----------------------------------------------------
    # Compression Ratio
    # -----------------------------------------------------

    metrics.update(

        compute_compression_ratio(
            original_text,
            summary_text,
        )
    )

    return metrics