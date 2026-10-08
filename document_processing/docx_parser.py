# document_processing/docx_parser.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

try:
    from docx import Document

    HAS_DOCX = True

except ImportError:

    Document = None

    HAS_DOCX = False


# ---------------------------------------------------------
# Page/Section Representation
# ---------------------------------------------------------

@dataclass
class DOCXSection:
    """
    Structured DOCX section.
    """

    section_id: int

    heading: Optional[str]

    text: str


# ---------------------------------------------------------
# Paragraph Cleaning
# ---------------------------------------------------------

def clean_paragraph(
    text: str,
) -> str:
    """
    Normalize paragraph text.
    """

    return (
        text
        .replace("\t", " ")
        .replace("\xa0", " ")
        .strip()
    )


# ---------------------------------------------------------
# Heading Detection
# ---------------------------------------------------------

def is_heading(
    paragraph,
) -> bool:
    """
    Detect heading-style paragraph.
    """

    try:

        style_name = (
            paragraph.style.name
            .lower()
        )

        return (
            "heading"
            in style_name
        )

    except Exception:

        return False


# ---------------------------------------------------------
# DOCX Parsing
# ---------------------------------------------------------

def parse_docx(
    file_path: str,
    preserve_sections: bool = True,
) -> Dict:
    """
    Parse DOCX document.

    Returns:
    -----------------------------------------
    {
        "text": ...,
        "sections": [...],
        "metadata": {...}
    }
    -----------------------------------------
    """

    if not HAS_DOCX:

        raise ImportError(
            "python-docx is not installed."
        )

    document = Document(
        file_path
    )

    paragraphs = []

    sections: List[
        DOCXSection
    ] = []

    current_heading = None

    current_section_text = []

    section_counter = 0

    # ---------------------------------------------------------
    # Iterate paragraphs
    # ---------------------------------------------------------

    for paragraph in (
        document.paragraphs
    ):

        text = clean_paragraph(
            paragraph.text
        )

        if not text:
            continue

        paragraphs.append(text)

        # -----------------------------------------------------
        # Section detection
        # -----------------------------------------------------

        if (
            preserve_sections
            and is_heading(
                paragraph
            )
        ):

            # save previous section
            if current_section_text:

                sections.append(
                    DOCXSection(
                        section_id=(
                            section_counter
                        ),

                        heading=(
                            current_heading
                        ),

                        text="\n".join(
                            current_section_text
                        ).strip(),
                    )
                )

                section_counter += 1

            current_heading = text

            current_section_text = []

            continue

        current_section_text.append(
            text
        )

    # ---------------------------------------------------------
    # Final section
    # ---------------------------------------------------------

    if current_section_text:

        sections.append(
            DOCXSection(
                section_id=(
                    section_counter
                ),

                heading=(
                    current_heading
                ),

                text="\n".join(
                    current_section_text
                ).strip(),
            )
        )

    # ---------------------------------------------------------
    # Final text
    # ---------------------------------------------------------

    if preserve_sections:
        full_text = "\n\n".join(

            section.text

            for section in sections
        ).strip()
    else:

        full_text = "\n".join(
            paragraphs
        ).strip()

    # ---------------------------------------------------------
    # Metadata
    # ---------------------------------------------------------

    metadata = {
        "paragraph_count":
            len(paragraphs),

        "section_count":
            len(sections),
    }

    return {
        "text":
            full_text,

        "sections":
            [
                section.__dict__
                for section
                in sections
            ],

        "metadata":
            metadata,
    }


# ---------------------------------------------------------
# Convenience Wrapper
# ---------------------------------------------------------

def extract_text_from_docx(
    file_path: str,
) -> str:
    """
    Simple DOCX text extraction.
    """

    parsed = parse_docx(
        file_path=file_path,
        preserve_sections=False,
    )

    return parsed[
        "text"
    ]