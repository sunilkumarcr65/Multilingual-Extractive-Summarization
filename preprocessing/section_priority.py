from __future__ import annotations

from typing import List, Tuple


# ---------------------------------------------------------
# High-value sections
# ---------------------------------------------------------

IMPORTANT_SECTIONS = {

    "abstract": 10,
    "introduction": 9,
    "methodology": 10,
    "proposed methodology": 10,
    "proposed system": 10,
    "algorithms": 10,
    "results": 9,
    "evaluation": 9,
    "discussion": 8,
    "conclusion": 10,
    "future scope": 8,
    "architecture": 9,
    "implementation": 8,
    "pipeline": 9,
    "rag": 9,
    "summarization": 9,
}

# ---------------------------------------------------------
# Low-value sections
# ---------------------------------------------------------

IGNORE_SECTIONS = {

    "certificate",
    "declaration",
    "acknowledgement",
    "acknowledgment",
    "table of contents",
    "list of figures",
    "list of tables",
    "references",
    "bibliography",
    "appendix",
    "index",
}

# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------

def normalize_heading(
    heading: str,
) -> str:

    return (
        heading.lower()
        .replace(":", "")
        .strip()
    )


def should_ignore_section(
    heading: str,
    content: str = "",
) -> bool:

    combined = (
        heading + " " + content[:200]
    ).lower()

    return any(
        ignored in combined
        for ignored in IGNORE_SECTIONS
    )


def score_section(
    heading: str,
    content: str,
) -> int:

    heading = normalize_heading(
        heading
    )

    score = 1

    for keyword, value in IMPORTANT_SECTIONS.items():

        if keyword in heading:

            score += value

    # Longer informative sections
    if len(content.split()) > 200:
        score += 2

    return score


# ---------------------------------------------------------
# Ranking
# ---------------------------------------------------------

def rank_sections(
    sections: List[Tuple[str, str]],
):

    ranked = []

    for heading, content in sections:

        if should_ignore_section(
            heading
        ):
            continue

        score = score_section(
            heading,
            content,
        )

        ranked.append(
            (
                heading,
                content,
                score,
            )
        )

    ranked.sort(
        key=lambda x: x[2],
        reverse=True,
    )

    return ranked