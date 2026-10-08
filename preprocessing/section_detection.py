from __future__ import annotations

import re
from typing import List, Tuple


# ---------------------------------------------------------
# Heading Detection Patterns
# ---------------------------------------------------------

HEADING_PATTERN = re.compile(

    r"^(?:"

    r"[A-Z][A-Z\s:&/-]{2,60}"

    r"|(?:Chapter|Section|Part)\s+\d+[\.\d]*"

    r"|\d+[\.\d]*\s+[A-Z][a-zA-Z\s]{2,60}"

    r"|[A-Z][a-zA-Z\s]{2,40}:"

    r")$"
)


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------

def is_heading(
    text: str,
) -> bool:
    """
    Heuristic heading detection.
    """

    text = text.strip()

    if not text:
        return False

    lines = text.splitlines()

    # ---------------------------------------------------------
    # Headings are usually single-line
    # ---------------------------------------------------------

    if len(lines) != 1:
        return False

    line = lines[0].strip()

    # ---------------------------------------------------------
    # Too long to be heading
    # ---------------------------------------------------------

    if len(line.split()) > 12:
        return False

    return bool(
        HEADING_PATTERN.match(line)
    )


# ---------------------------------------------------------
# Section Detection
# ---------------------------------------------------------

def detect_sections(
    text: str,
    minimum_section_length: int = 180,
) -> List[Tuple[str, str]]:
    """
    Detect logical document sections.

    Returns:
    -----------------------------------------
    [
        (heading, body),
        ...
    ]
    -----------------------------------------

    Falls back to:
    ("General", full_text)
    """

    text = (text or "").strip()

    if not text:

        return [
            ("General", "")
        ]

    # ---------------------------------------------------------
    # Split by paragraph blocks
    # ---------------------------------------------------------

    blocks = re.split(
        r"\n\s*\n",
        text,
    )

    sections = []

    current_heading = "General"

    current_body = []

    # ---------------------------------------------------------
    # Parse blocks
    # ---------------------------------------------------------

    for block in blocks:

        block = block.strip()
        # -----------------------------------------------------
        # Force-detect REFERENCES sections
        # -----------------------------------------------------

        if block.upper().startswith(
            "REFERENCES"
        ):
            sections.append(
                (
                    "REFERENCES",
                    block,
                )
            )
            continue
        if not block:
            continue

        first_line = (
            block.splitlines()[0]
            .strip()
        )

        # -----------------------------------------------------
        # New heading
        # -----------------------------------------------------

        if is_heading(first_line):

            # Flush previous section
            if current_body:

                body_text = " ".join(
                    current_body
                ).strip()

                if (
                    len(body_text)
                    >= minimum_section_length
                ):

                    sections.append(
                        (
                            current_heading,
                            body_text,
                        )
                    )

            current_heading = first_line

            current_body = []

        # -----------------------------------------------------
        # Regular content
        # -----------------------------------------------------

        else:

            if (
                len(block)
                >= minimum_section_length
            ):

                current_body.append(
                    block
                )

    # ---------------------------------------------------------
    # Final flush
    # ---------------------------------------------------------

    if current_body:

        body_text = " ".join(
            current_body
        ).strip()

        if (
            len(body_text)
            >= minimum_section_length
        ):

            sections.append(
                (
                    current_heading,
                    body_text,
                )
            )

    # ---------------------------------------------------------
    # Fallback
    # ---------------------------------------------------------

    if not sections:

        return [
            ("General", text)
        ]

    return sections