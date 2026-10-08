# training/evaluation.py

from __future__ import annotations

from typing import Dict, List

import evaluate
import numpy as np
import torch


from training.config import (
    LENGTH_PENALTY,
    MAX_TARGET_LENGTH,
    NO_REPEAT_NGRAM_SIZE,
    NUM_BEAMS,
)


# ---------------------------------------------------------
# Metrics
# ---------------------------------------------------------

ROUGE = evaluate.load("rouge")

BLEU = evaluate.load("bleu")


# ---------------------------------------------------------
# Prediction Processing
# ---------------------------------------------------------

def process_predictions(
    predictions,
):
    """
    Normalize trainer predictions.

    Seq2SeqTrainer may return:
    - tuple
    - logits
    - generated token ids

    This function converts everything into:
    List[List[int]]
    """

    # ---------------------------------------------------------
    # Handle tuple outputs
    # ---------------------------------------------------------

    if isinstance(predictions, tuple):

        predictions = predictions[0]

    # ---------------------------------------------------------
    # Convert tensor -> numpy
    # ---------------------------------------------------------

    if torch.is_tensor(predictions):

        predictions = predictions.detach().cpu().numpy()

    # ---------------------------------------------------------
    # Logits -> token ids
    # Shape:
    # (batch, seq_len, vocab_size)
    # ---------------------------------------------------------

    if len(predictions.shape) == 3:

        predictions = np.argmax(
            predictions,
            axis=-1,
        )

    # ---------------------------------------------------------
    # Convert to python lists
    # ---------------------------------------------------------

    predictions = predictions.tolist()

    return predictions


def process_labels(
    labels,
    tokenizer,
):
    """
    Replace -100 with pad token id.
    """

    if torch.is_tensor(labels):

        labels = labels.detach().cpu().numpy()

    labels = np.where(
        labels != -100,
        labels,
        tokenizer.pad_token_id,
    )

    labels = labels.tolist()

    return labels


# ---------------------------------------------------------
# Decoding
# ---------------------------------------------------------

def decode_predictions(
    predictions,
    tokenizer,
):

    # -----------------------------------------------------
    # Replace invalid values
    # -----------------------------------------------------

    predictions = [
        [
            token
            if token >= 0
            else tokenizer.pad_token_id
            for token in sequence
        ]
        for sequence in predictions
    ]

    decoded = tokenizer.batch_decode(
        predictions,
        skip_special_tokens=True,
    )

    decoded = [
        text.strip()
        for text in decoded
    ]

    return decoded


def decode_labels(
    labels,
    tokenizer,
) -> List[str]:
    """
    Decode label token IDs.
    """

    decoded = tokenizer.batch_decode(
        labels,
        skip_special_tokens=True,
    )

    return [
        label.strip()
        for label in decoded
    ]


# ---------------------------------------------------------
# Metric Computation
# ---------------------------------------------------------

def compute_metrics_builder(
    tokenizer,
):
    """
    Build HuggingFace compute_metrics function.
    """

    def compute_metrics(
        eval_prediction,
    ) -> Dict[str, float]:

        predictions, labels = eval_prediction

        # -----------------------------------------------------
        # Normalize predictions / labels
        # -----------------------------------------------------

        predictions = process_predictions(
            predictions
        )

        labels = process_labels(
            labels,
            tokenizer,
        )

        # -----------------------------------------------------
        # Decode
        # -----------------------------------------------------

        decoded_predictions = decode_predictions(
            predictions,
            tokenizer,
        )

        decoded_labels = decode_labels(
            labels,
            tokenizer,
        )

        # -----------------------------------------------------
        # ROUGE
        # -----------------------------------------------------

        rouge_scores = ROUGE.compute(
            predictions=decoded_predictions,
            references=decoded_labels,
            use_stemmer=True,
        )

        # -----------------------------------------------------
        # BLEU
        # -----------------------------------------------------

        bleu_scores = BLEU.compute(
            predictions=decoded_predictions,
            references=[
                [label]
                for label in decoded_labels
            ],
        )

        metrics = {
            "rouge1": round(
                rouge_scores["rouge1"],
                4,
            ),

            "rouge2": round(
                rouge_scores["rouge2"],
                4,
            ),

            "rougeL": round(
                rouge_scores["rougeL"],
                4,
            ),

            "bleu": round(
                bleu_scores["bleu"],
                4,
            ),
        }

        return metrics

    return compute_metrics


# ---------------------------------------------------------
# Standalone Evaluation
# ---------------------------------------------------------

@torch.inference_mode()
def evaluate_model(
    model,
    tokenizer,
    texts: List[str],
    references: List[str],
) -> Dict[str, float]:
    """
    Manual evaluation pipeline.
    """

    predictions = []

    for text in texts:

        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=1024,
        ).to(model.device)

        generated = model.generate(
            **inputs,

            max_length=MAX_TARGET_LENGTH,

            num_beams=NUM_BEAMS,

            no_repeat_ngram_size=(
                NO_REPEAT_NGRAM_SIZE
            ),

            length_penalty=(
                LENGTH_PENALTY
            ),
        )

        summary = tokenizer.decode(
            generated[0],
            skip_special_tokens=True,
        ).strip()

        predictions.append(summary)

    # ---------------------------------------------------------
    # ROUGE
    # ---------------------------------------------------------

    rouge_scores = ROUGE.compute(
        predictions=predictions,
        references=references,
        use_stemmer=True,
    )

    # ---------------------------------------------------------
    # BLEU
    # ---------------------------------------------------------

    bleu_scores = BLEU.compute(
        predictions=predictions,
        references=[
            [reference]
            for reference in references
        ],
    )

    return {
        "rouge1": round(
            rouge_scores["rouge1"],
            4,
        ),

        "rouge2": round(
            rouge_scores["rouge2"],
            4,
        ),

        "rougeL": round(
            rouge_scores["rougeL"],
            4,
        ),

        "bleu": round(
            bleu_scores["bleu"],
            4,
        ),
    }