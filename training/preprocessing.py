# training/preprocessing.py

from __future__ import annotations

from typing import Dict

from training.config import (
    MAX_SOURCE_LENGTH,
    MAX_TARGET_LENGTH,
    PADDING,
    TRUNCATION,
)


# =========================================================
# LABEL MASKING
# =========================================================

def mask_padding_tokens(
    labels,
    pad_token_id: int,
):

    return [
        [
            token
            if token != pad_token_id
            else -100
            for token in sequence
        ]
        for sequence in labels
    ]


# =========================================================
# DOMAIN PROMPTS
# =========================================================

DOMAIN_PROMPTS = {

    "news":
        "Summarize this news article:\n\n",

    "legal":
        "Summarize this legal document:\n\n",

    "scientific":
        "Summarize this scientific paper:\n\n",

    "dialogue":
        "Summarize this conversation:\n\n",

    "resume":
        "Summarize this resume:\n\n",

    "general":
        "Summarize this document:\n\n",
}


# =========================================================
# BUILD INPUTS
# =========================================================

def build_inputs(
    documents,
    domains,
):

    prompts = []

    for document, domain in zip(
        documents,
        domains,
    ):

        domain = (
            domain or "general"
        ).lower()

        prefix = DOMAIN_PROMPTS.get(
            domain,
            DOMAIN_PROMPTS["general"],
        )

        prompts.append(
            f"{prefix}{document}"
        )

    return prompts


# =========================================================
# MAIN PREPROCESSING
# =========================================================

def preprocess_batch(
    batch: Dict,
    tokenizer,
    max_source_length: int = (
        MAX_SOURCE_LENGTH
    ),
    max_target_length: int = (
        MAX_TARGET_LENGTH
    ),
) -> Dict:

    # -----------------------------------------------------
    # Inputs
    # -----------------------------------------------------

    documents = (
        batch.get("document")
        or batch.get("text")
    )

    domains = batch.get(
        "domain",
        ["general"] * len(documents),
    )

    inputs = build_inputs(
        documents,
        domains,
    )

    # -----------------------------------------------------
    # Targets
    # -----------------------------------------------------

    targets = (
        batch.get("summary")
        or batch.get("highlights")
        or batch.get("target")
    )

    # -----------------------------------------------------
    # Tokenize Inputs
    # -----------------------------------------------------

    model_inputs = tokenizer(
        inputs,
        max_length=max_source_length,
        padding=PADDING,
        truncation=TRUNCATION,
    )

    # -----------------------------------------------------
    # Tokenize Labels
    # -----------------------------------------------------

    labels = tokenizer(
        targets,
        max_length=max_target_length,
        padding=PADDING,
        truncation=TRUNCATION,
    )

    labels["input_ids"] = (
        mask_padding_tokens(
            labels=labels["input_ids"],
            pad_token_id=tokenizer.pad_token_id,
        )
    )

    model_inputs["labels"] = (
        labels["input_ids"]
    )

    return model_inputs


# =========================================================
# MULTILINGUAL PREPROCESSING
# =========================================================

def preprocess_multilingual_batch(
    batch: Dict,
    tokenizer,
) -> Dict:

    documents = (
        batch.get("document")
        or batch.get("text")
    )

    domains = batch.get(
        "domain",
        ["general"] * len(documents),
    )

    languages = batch.get(
        "language",
        ["unknown"] * len(documents),
    )

    inputs = []

    for document, domain, language in zip(
        documents,
        domains,
        languages,
    ):

        domain = (
            domain or "general"
        ).lower()

        domain_prompt = DOMAIN_PROMPTS.get(
            domain,
            DOMAIN_PROMPTS["general"],
        )

        prompt = (
            f"{domain_prompt}"
            f"Language: {language}\n\n"
            f"{document}"
        )

        inputs.append(prompt)

    targets = (
        batch.get("summary")
        or batch.get("highlights")
        or batch.get("target")
    )

    model_inputs = tokenizer(
        inputs,
        max_length=MAX_SOURCE_LENGTH,
        padding=PADDING,
        truncation=TRUNCATION,
    )

    labels = tokenizer(
        targets,
        max_length=MAX_TARGET_LENGTH,
        padding=PADDING,
        truncation=TRUNCATION,
    )

    labels["input_ids"] = (
        mask_padding_tokens(
            labels=labels["input_ids"],
            pad_token_id=tokenizer.pad_token_id,
        )
    )

    model_inputs["labels"] = (
        labels["input_ids"]
    )

    return model_inputs