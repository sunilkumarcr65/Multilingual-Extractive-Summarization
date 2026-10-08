from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from datasets import Dataset, DatasetDict, load_dataset


# ---------------------------------------------------------
# Default Field Candidates
# ---------------------------------------------------------

ARTICLE_KEYS = (
    "article",
    "document",
    "text",
    "body",
    "content",
    "passage",
)

SUMMARY_KEYS = (
    "summary",
    "highlights",
    "abstract",
    "target",
    "label",
)


# ---------------------------------------------------------
# Field Detection
# ---------------------------------------------------------

def detect_dataset_fields(
    sample: Dict,
) -> Tuple[str, str]:
    """
    Automatically detect:
    - article/document field
    - summary/reference field
    """

    keys = set(sample.keys())

    article_key = next(
        (
            key
            for key in ARTICLE_KEYS
            if key in keys
        ),
        None,
    )

    summary_key = next(
        (
            key
            for key in SUMMARY_KEYS
            if key in keys
        ),
        None,
    )

    if article_key is None:

        raise ValueError(
            "Could not detect article field.\n"
            f"Available keys: {sorted(keys)}"
        )

    if summary_key is None:

        raise ValueError(
            "Could not detect summary field.\n"
            f"Available keys: {sorted(keys)}"
        )

    return article_key, summary_key


# ---------------------------------------------------------
# Dataset Loading
# ---------------------------------------------------------

def load_huggingface_dataset(
    dataset_name: str,
    dataset_config: Optional[str] = None,
) -> DatasetDict:
    """
    Load HuggingFace dataset.
    """

    print(
        f"[DATASET] Loading dataset: "
        f"{dataset_name}"
        + (
            f" / {dataset_config}"
            if dataset_config
            else ""
        )
    )

    dataset = load_dataset(
        dataset_name,
        dataset_config,
    )

    return dataset


# ---------------------------------------------------------
# Split Selection
# ---------------------------------------------------------

def get_dataset_split(
    dataset: DatasetDict,
    split: str = "test",
) -> Dataset:
    """
    Get dataset split safely.

    Falls back if split unavailable.
    """

    if split in dataset:

        print(
            f"[DATASET] Using split: "
            f"{split}"
        )

        return dataset[split]

    available_splits = list(dataset.keys())

    fallback_split = available_splits[0]

    print(
        f"[DATASET] Split '{split}' "
        f"not found."
    )

    print(
        f"[DATASET] Falling back to: "
        f"{fallback_split}"
    )

    return dataset[fallback_split]


# ---------------------------------------------------------
# Sample Selection
# ---------------------------------------------------------

def select_samples(
    dataset: Dataset,
    num_samples: int,
) -> Dataset:
    """
    Select subset of dataset.
    """

    sample_count = min(
        num_samples,
        len(dataset),
    )

    print(
        f"[DATASET] Selecting "
        f"{sample_count} samples"
    )

    return dataset.select(
        range(sample_count)
    )


# ---------------------------------------------------------
# Main Dataset Pipeline
# ---------------------------------------------------------

def prepare_dataset(
    dataset_name: str,
    dataset_config: Optional[str] = None,
    split: str = "test",
    num_samples: int = 100,
    article_key: Optional[str] = None,
    summary_key: Optional[str] = None,
) -> Tuple[Dataset, str, str]:
    """
    Full dataset preparation pipeline.

    Returns:
    -----------------------------------------
    dataset
    article_key
    summary_key
    -----------------------------------------
    """

    dataset_dict = load_huggingface_dataset(
        dataset_name=dataset_name,
        dataset_config=dataset_config,
    )

    dataset = get_dataset_split(
        dataset=dataset_dict,
        split=split,
    )

    dataset = select_samples(
        dataset=dataset,
        num_samples=num_samples,
    )

    # -----------------------------------------------------
    # Auto field detection
    # -----------------------------------------------------

    if article_key is None or summary_key is None:

        if len(dataset) == 0:
            raise ValueError(
                "Dataset split is empty."
            )

        detected_article_key, detected_summary_key = (
            detect_dataset_fields(
                dataset[0]
            )
        )

        article_key = (
            article_key
            or detected_article_key
        )

        summary_key = (
            summary_key
            or detected_summary_key
        )

    print(
        f"[DATASET] "
        f"Article field: '{article_key}'"
    )

    print(
        f"[DATASET] "
        f"Summary field: '{summary_key}'"
    )

    return (
        dataset,
        article_key,
        summary_key,
    )