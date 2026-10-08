from __future__ import annotations

from typing import List


# ---------------------------------------------------------
# Config
# ---------------------------------------------------------

MAX_MODEL_TOKENS = 1024
MAX_BATCH_TOKENS = 900


# ---------------------------------------------------------
# Token Length
# ---------------------------------------------------------

def token_length(
    text: str,
    tokenizer,
) -> int:
    """
    Compute tokenizer-aware token length.
    """

    return len(
        tokenizer(
            text,
            add_special_tokens=False,
        )["input_ids"]
    )


# ---------------------------------------------------------
# Sentence Batching
# ---------------------------------------------------------

def build_translation_batches(
    sentences: List[str],
    tokenizer,
    source_language_code: str,
    max_batch_tokens: int = (
        MAX_BATCH_TOKENS
    ),
    max_sentences_per_batch: int = 8,
) -> List[List[str]]:
    """
    Build token-safe translation batches.

    Ensures:
    - model context never overflows
    - batching remains efficient
    - very long sentences handled safely
    """

    if not sentences:

        return []

    tokenizer.src_lang = (
        source_language_code
    )

    batches = []

    current_batch = []

    current_token_count = 0

    # ---------------------------------------------------------
    # Build batches
    # ---------------------------------------------------------

    for sentence in sentences:

        sentence = (
            sentence or ""
        ).strip()

        if not sentence:
            continue

        sentence_tokens = (
            token_length(
                sentence,
                tokenizer,
            )
        )

        # -----------------------------------------------------
        # Extremely long sentence
        # -----------------------------------------------------

        if (
            sentence_tokens
            >= max_batch_tokens
        ):

            if current_batch:

                batches.append(
                    current_batch
                )

                current_batch = []

                current_token_count = 0

            batches.append(
                [sentence]
            )

            continue

        # -----------------------------------------------------
        # Batch overflow
        # -----------------------------------------------------

        exceeds_token_limit = (
            current_token_count
            + sentence_tokens
            > max_batch_tokens
        )

        exceeds_batch_limit = (
            len(current_batch)
            >= max_sentences_per_batch
        )

        if (
            current_batch
            and (
                exceeds_token_limit
                or exceeds_batch_limit
            )
        ):

            batches.append(
                current_batch
            )

            current_batch = []

            current_token_count = 0

        # -----------------------------------------------------
        # Add sentence
        # -----------------------------------------------------

        current_batch.append(
            sentence
        )

        current_token_count += (
            sentence_tokens
        )

    # ---------------------------------------------------------
    # Remaining batch
    # ---------------------------------------------------------

    if current_batch:

        batches.append(
            current_batch
        )

    return batches