# training/dataset.py
# Updated: multilingual loading now properly concatenates
# datasets using HuggingFace concatenate_datasets().

from __future__ import annotations

from typing import List, Tuple

from datasets import Dataset, concatenate_datasets, load_dataset

from training.config import (
    DATASET_LANGUAGE,
    DATASET_NAME,
    MULTILINGUAL_LANGUAGES,
    MULTILINGUAL_SAMPLES_PER_LANG,
    TRAIN_SAMPLES,
    VALIDATION_SAMPLES,
)


# ---------------------------------------------------------
# Single Language Dataset
# ---------------------------------------------------------

def load_xlsum_dataset(
    dataset_name: str = DATASET_NAME,
    language: str = DATASET_LANGUAGE,
    train_samples: int = TRAIN_SAMPLES,
    validation_samples: int = VALIDATION_SAMPLES,
    trust_remote_code: bool = True,
) -> Tuple[Dataset, Dataset]:
    """
    Load XL-Sum dataset for a single language.

    XL-Sum is a BBC News multilingual dataset with
    article-summary pairs in 44 languages.

    Dataset columns used:
    -----------------------------------------
    text       — full BBC article body
    summary    — BBC article summary/headline
    -----------------------------------------

    Returns:
    -----------------------------------------
    train_dataset, validation_dataset
    -----------------------------------------
    """

    print(
        f"[DATASET] Loading {dataset_name} ({language})"
    )

    dataset = load_dataset(
        dataset_name,
        language,
        trust_remote_code=trust_remote_code,
    )

    train_dataset = (
        dataset["train"]
        .select(range(min(train_samples, len(dataset["train"]))))
    )

    validation_dataset = (
        dataset["validation"]
        .select(range(min(validation_samples, len(dataset["validation"]))))
    )

    print(f"[DATASET] Train samples:      {len(train_dataset)}")
    print(f"[DATASET] Validation samples: {len(validation_dataset)}")

    return train_dataset, validation_dataset


# ---------------------------------------------------------
# Multilingual Dataset
# ---------------------------------------------------------

def load_multilingual_xlsum(
    languages: List[str] = MULTILINGUAL_LANGUAGES,
    samples_per_language: int = MULTILINGUAL_SAMPLES_PER_LANG,
    trust_remote_code: bool = True,
) -> Tuple[Dataset, Dataset]:
    """
    Load multilingual XL-Sum mixture for training.

    Combines data from multiple languages into a single
    shuffled dataset. Adds a "language" column to each
    sample so the model learns language-aware summarization.

    Default languages: English, Hindi, Gujarati, Bengali
    Each gets `samples_per_language` training samples.

    Returns:
    -----------------------------------------
    train_dataset, validation_dataset
    -----------------------------------------
    """

    train_subsets = []
    val_subsets = []

    for language in languages:

        try:

            print(f"[DATASET] Loading {language}...")

            dataset = load_dataset(DATASET_NAME, language, trust_remote_code=True)

            # -------------------------------------------------
            # Training subset
            # -------------------------------------------------

            train_size = min(
                samples_per_language,
                len(dataset["train"]),
            )

            train_subset = (
                dataset["train"]
                .select(range(train_size))
            )

            # Add language column for language-aware preprocessing
            train_subset = train_subset.add_column(
                "language",
                [language] * len(train_subset),
            )

            train_subsets.append(train_subset)

            # -------------------------------------------------
            # Validation subset (10% of training size)
            # -------------------------------------------------

            val_size = min(
                max(100, train_size // 10),
                len(dataset["validation"]),
            )

            val_subset = (
                dataset["validation"]
                .select(range(val_size))
            )

            val_subset = val_subset.add_column(
                "language",
                [language] * len(val_subset),
            )

            val_subsets.append(val_subset)

            print(
                f"[DATASET] {language}: "
                f"{train_size} train, {val_size} val"
            )

        except Exception as error:

            print(
                f"[DATASET] WARNING: Failed to load "
                f"{language}: {error}"
            )

    if not train_subsets:
        raise ValueError(
            "No multilingual datasets loaded. "
            "Check your language names against: "
            "https://huggingface.co/datasets/csebuetnlp/xlsum"
        )

    # ---------------------------------------------------------
    # Concatenate and shuffle
    # ---------------------------------------------------------

    train_merged = concatenate_datasets(train_subsets).shuffle(seed=42)
    val_merged = concatenate_datasets(val_subsets).shuffle(seed=42)

    print(
        f"[DATASET] Multilingual train total: {len(train_merged)}"
    )
    print(
        f"[DATASET] Multilingual val total:   {len(val_merged)}"
    )
    print(
        f"[DATASET] Languages: {', '.join(languages)}"
    )

    return train_merged, val_merged