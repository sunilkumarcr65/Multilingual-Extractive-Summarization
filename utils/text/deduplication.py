# utils/text/deduplication.py

from __future__ import annotations

import re
from typing import List, Set
from collections import defaultdict

# ---------------------------------------------------------
# Normalization
# ---------------------------------------------------------

def normalize_sentence(
    sentence: str,
) -> str:
    """
    Normalize sentence for deduplication.
    """

    sentence = (
        sentence or ""
    ).strip().lower()

    sentence = re.sub(
        r"\s+",
        " ",
        sentence,
    )

    return sentence


# ---------------------------------------------------------
# Jaccard Similarity
# ---------------------------------------------------------

def jaccard_similarity(
    text_a: str,
    text_b: str,
) -> float:
    """
    Compute word-level Jaccard similarity.
    """

    words_a: Set[str] = set(
        normalize_sentence(
            text_a
        ).split()
    )

    words_b: Set[str] = set(
        normalize_sentence(
            text_b
        ).split()
    )

    union = len(
        words_a | words_b
    )

    if union == 0:

        return 0.0

    intersection = len(
        words_a & words_b
    )

    return (
        intersection
        / union
    )


# ---------------------------------------------------------
# Deduplication
# ---------------------------------------------------------

def deduplicate_sentences(
    sentences: List[str],
    similarity_threshold: float = 0.85,
) -> List[str]:

    if not sentences:
        return []

    unique_sentences = []

    seen_exact = set()

    # Bucket by first 12 chars
    # Massive speedup
    buckets = defaultdict(list)

    for sentence in sentences:

        normalized = normalize_sentence(
            sentence
        )

        if not normalized:
            continue

        if normalized in seen_exact:
            continue

        bucket_key = normalized[:12]

        is_duplicate = False

        for existing in buckets[bucket_key]:

            similarity = jaccard_similarity(
                sentence,
                existing,
            )

            if similarity >= similarity_threshold:

                is_duplicate = True
                break

        if is_duplicate:
            continue

        seen_exact.add(normalized)

        unique_sentences.append(
            sentence.strip()
        )

        buckets[bucket_key].append(
            sentence
        )

    return unique_sentences