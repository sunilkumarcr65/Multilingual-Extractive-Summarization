# formatting/markdown_formatter.py

from __future__ import annotations

import re
from typing import Dict, List


# ---------------------------------------------------------
# Resume Section Patterns
# ---------------------------------------------------------

RESUME_SECTION_PATTERNS = {
    "Education": [
        r"\beducation\b",
        r"\bacademics\b",
    ],

    "Experience": [
        r"\bexperience\b",
        r"\bwork experience\b",
        r"\binternship\b",
        r"\bemployment\b",
    ],

    "Projects": [
        r"\bprojects?\b",
        r"\bacademic projects?\b",
    ],

    "Skills": [
        r"\bskills?\b",
        r"\btechnologies\b",
        r"\btech stack\b",
    ],

    "Achievements": [
        r"\bachievements?\b",
        r"\bawards?\b",
        r"\brank\b",
        r"\bleetcode\b",
        r"\bcodeforces\b",
        r"\bcodechef\b",
    ],

    "Summary": [
        r"\bsummary\b",
        r"\boverview\b",
        r"\bprofile\b",
    ],
}


# ---------------------------------------------------------
# Cleanup
# ---------------------------------------------------------

def clean_summary_text(
    text: str,
) -> str:
    """
    Clean malformed summary text.
    """

    text = (
        text or ""
    ).strip()

    if not text:
        return ""

    # -----------------------------------------------------
    # Preserve paragraph structure
    # -----------------------------------------------------

    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text,
    )

    # -----------------------------------------------------
    # Normalize spaces
    # -----------------------------------------------------

    text = re.sub(
        r"[ \t]+",
        " ",
        text,
    )

    # -----------------------------------------------------
    # Fix bullet spacing
    # -----------------------------------------------------

    text = re.sub(
        r"\s*•\s*",
        "\n• ",
        text,
    )

    # -----------------------------------------------------
    # Fix markdown bullets
    # -----------------------------------------------------

    text = re.sub(
        r"\s*-\s+",
        "\n- ",
        text,
    )

    # -----------------------------------------------------
    # Remove repeated newlines
    # -----------------------------------------------------

    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text,
    )

    return text.strip()


# ---------------------------------------------------------
# Sentence Splitting
# ---------------------------------------------------------

def split_sentences(
    text: str,
) -> List[str]:
    """
    Robust sentence splitter.
    """

    text = (
        text or ""
    ).strip()

    if not text:
        return []

    sentences = re.split(
        r"(?<=[.!?])\s+",
        text,
    )

    cleaned = []

    for sentence in sentences:

        sentence = (
            sentence.strip()
        )

        if not sentence:
            continue

        # -------------------------------------------------
        # Remove extremely short fragments
        # -------------------------------------------------

        if len(sentence.split()) < 4:
            continue

        # -------------------------------------------------
        # Remove incomplete fragments
        # -------------------------------------------------

        if sentence.endswith(
            (
                ":",
                "|",
                "-",
                "•",
            )
        ):
            continue

        cleaned.append(
            sentence
        )

    return cleaned


# ---------------------------------------------------------
# Bullet Builder
# ---------------------------------------------------------

def build_bullet_list(
    sentences: List[str],
) -> str:
    """
    Convert sentences into markdown bullets.
    """

    bullets = []

    for sentence in sentences:

        sentence = (
            sentence.strip()
        )

        if not sentence:
            continue

        # Remove existing bullets
        sentence = re.sub(
            r"^[-•]+\s*",
            "",
            sentence,
        )

        bullets.append(
            f"- {sentence}"
        )

    return "\n".join(
        bullets
    )


# ---------------------------------------------------------
# Resume Section Detection
# ---------------------------------------------------------

def detect_resume_sections(
    text: str,
) -> Dict[str, List[str]]:
    """
    Detect resume sections using
    heading-aware parsing.
    """

    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip()
    ]

    sections: Dict[str, List[str]] = {}

    current_section = "Summary"

    sections[current_section] = []

    for line in lines:

        normalized = line.lower()

        matched_section = None

        # -------------------------------------------------
        # Heading detection
        # -------------------------------------------------

        for (
            section_name,
            patterns,
        ) in (
            RESUME_SECTION_PATTERNS.items()
        ):

            if any(
                re.search(
                    pattern,
                    normalized,
                )
                for pattern in patterns
            ):

                matched_section = (
                    section_name
                )

                break

        # -------------------------------------------------
        # Switch section
        # -------------------------------------------------

        if matched_section:

            current_section = (
                matched_section
            )

            sections.setdefault(
                current_section,
                [],
            )

            continue

        # -------------------------------------------------
        # Skip noisy fragments
        # -------------------------------------------------

        if len(line.split()) < 3:
            continue

        # -------------------------------------------------
        # Append content
        # -------------------------------------------------

        sections[
            current_section
        ].append(line)

    return sections


# ---------------------------------------------------------
# Resume Formatter
# ---------------------------------------------------------

def format_resume_summary(
    summary: str,
) -> str:
    """
    Structured resume formatter.
    """

    sections = detect_resume_sections(
        summary
    )

    markdown_parts = [
        "# Candidate Profile Summary\n"
    ]

    for (
        section_name,
        content_lines,
    ) in sections.items():

        if not content_lines:
            continue

        markdown_parts.append(
            f"## {section_name}\n"
        )

        # -------------------------------------------------
        # Join section content
        # -------------------------------------------------

        section_text = " ".join(
            content_lines
        )

        sentences = split_sentences(
            section_text
        )

        # -------------------------------------------------
        # Fallback
        # -------------------------------------------------

        if not sentences:

            markdown_parts.append(
                f"- {section_text}"
            )

            continue

        markdown_parts.append(
            build_bullet_list(
                sentences
            )
        )

    return "\n\n".join(
        markdown_parts
    )


# ---------------------------------------------------------
# General Formatter
# ---------------------------------------------------------

def format_general_summary(
    summary: str,
) -> str:
    """
    Format general summaries.
    """

    sentences = split_sentences(
        summary
    )

    if not sentences:

        return (
            "# Document Summary\n\n"
            + summary
        )

    markdown_parts = [
        "# Document Summary\n"
    ]

    # -----------------------------------------------------
    # Introduction
    # -----------------------------------------------------

    intro = sentences[:2]

    if intro:

        markdown_parts.append(
            "## Overview\n"
        )

        markdown_parts.append(
            build_bullet_list(
                intro
            )
        )

    # -----------------------------------------------------
    # Key points
    # -----------------------------------------------------

    remaining = sentences[2:]

    if remaining:

        markdown_parts.append(
            "## Key Points\n"
        )

        markdown_parts.append(
            build_bullet_list(
                remaining
            )
        )

    return "\n\n".join(
        markdown_parts
    )


# ---------------------------------------------------------
# Main Formatter
# ---------------------------------------------------------

def format_summary_markdown(
    summary: str,
    document_type: str = "general",
) -> str:
    """
    Main markdown formatter.
    """

    summary = clean_summary_text(
        summary
    )

    if not summary:

        return (
            "# Document Summary\n\n"
            "No summary generated."
        )

    document_type = (
        document_type
        or "general"
    ).strip().lower()

    # -----------------------------------------------------
    # Resume
    # -----------------------------------------------------

    if document_type == "resume":

        return format_resume_summary(
            summary
        )

    # -----------------------------------------------------
    # General docs
    # -----------------------------------------------------

    return format_general_summary(
        summary
    )