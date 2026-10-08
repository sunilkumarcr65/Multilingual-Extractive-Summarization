from __future__ import annotations

from typing import List


# ---------------------------------------------------------
# Config
# ---------------------------------------------------------

MAX_CHUNK_TOKENS = 384


# ---------------------------------------------------------
# Token-Aware Chunking
# ---------------------------------------------------------

def chunk_text_by_tokens(
    text: str,
    tokenizer,
    max_chunk_tokens: int = (
        MAX_CHUNK_TOKENS
    ),
) -> List[str]:
    """
    Split text into token-safe chunks.

    Uses tokenizer token counts
    instead of word counts.
    """

    text = (
        text or ""
    ).strip()

    if not text:

        return []

    # ---------------------------------------------------------
    # Encode without special tokens
    # ---------------------------------------------------------

    token_ids = tokenizer.encode(
        text,
        add_special_tokens=False,
    )

    # ---------------------------------------------------------
    # Fits within model context
    # ---------------------------------------------------------

    if (
        len(token_ids)
        <= max_chunk_tokens
    ):

        return [text]

    # ---------------------------------------------------------
    # Chunk token IDs
    # ---------------------------------------------------------

    chunks = []

    for start in range(
        0,
        len(token_ids),
        max_chunk_tokens,
    ):

        chunk_ids = token_ids[
            start:
            start + max_chunk_tokens
        ]

        chunk_text = (
            tokenizer.decode(
                chunk_ids,
                skip_special_tokens=True,
            )
            .strip()
        )

        if chunk_text:

            chunks.append(
                chunk_text
            )

    return chunks