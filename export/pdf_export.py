from __future__ import annotations

import html
import io
import re

from datetime import datetime

from reportlab.lib.pagesizes import letter

from reportlab.lib.styles import (
    getSampleStyleSheet,
)

from reportlab.platypus import (
    Paragraph,
    SimpleDocTemplate,
    Spacer,
)

from reportlab.platypus.flowables import (
    PageBreak,
)


# ---------------------------------------------------------
# Text Cleanup
# ---------------------------------------------------------

def clean_markdown(
    text: str,
) -> str:
    """
    Cleanup markdown-like formatting
    for ReportLab rendering.
    """

    text = (
        text or ""
    )

    replacements = {

        "# ": "",
        "## ": "",
        "### ": "",

        "**": "",
        "__": "",

        "`": "",

        "- ": "• ",
    }

    cleaned = text

    for old, new in (
        replacements.items()
    ):

        cleaned = cleaned.replace(
            old,
            new,
        )

    # normalize whitespace
    cleaned = re.sub(
        r"\n{3,}",
        "\n\n",
        cleaned,
    )

    # escape unsafe HTML
    cleaned = html.escape(
        cleaned
    )

    # restore line breaks for ReportLab
    cleaned = cleaned.replace(
        "\n",
        "<br/>",
    )

    return cleaned.strip()


# ---------------------------------------------------------
# Header
# ---------------------------------------------------------

def build_header(
    story,
    title: str,
):
    """
    Add PDF title/header.
    """

    styles = (
        getSampleStyleSheet()
    )

    title_style = styles[
        "Title"
    ]

    normal_style = styles[
        "BodyText"
    ]

    story.append(
        Paragraph(
            html.escape(title),
            title_style,
        )
    )

    story.append(
        Spacer(1, 18)
    )

    generated_at = (
        datetime.now()
        .strftime(
            "%Y-%m-%d %H:%M:%S"
        )
    )

    story.append(
        Paragraph(
            (
                f"<b>Generated:</b> "
                f"{generated_at}"
            ),
            normal_style,
        )
    )

    story.append(
        Spacer(1, 20)
    )


# ---------------------------------------------------------
# Generic Section
# ---------------------------------------------------------

def add_text_section(
    story,
    heading: str,
    text: str,
):
    """
    Add text section.
    """

    if not text:
        return

    styles = (
        getSampleStyleSheet()
    )

    heading_style = styles[
        "Heading2"
    ]

    body_style = styles[
        "BodyText"
    ]

    story.append(
        Paragraph(
            html.escape(heading),
            heading_style,
        )
    )

    story.append(
        Spacer(1, 10)
    )

    cleaned_text = clean_markdown(
        text
    )

    story.append(
        Paragraph(
            cleaned_text,
            body_style,
        )
    )

    story.append(
        Spacer(1, 14)
    )


# ---------------------------------------------------------
# Sentiment Section
# ---------------------------------------------------------

def add_sentiment_section(
    story,
    sentiment: dict,
):
    """
    Add sentiment analysis section.
    """

    if not sentiment:
        return

    styles = (
        getSampleStyleSheet()
    )

    heading_style = styles[
        "Heading2"
    ]

    body_style = styles[
        "BodyText"
    ]

    story.append(
        Paragraph(
            "Sentiment Analysis",
            heading_style,
        )
    )

    story.append(
        Spacer(1, 10)
    )

    # -----------------------------------------------------
    # HuggingFace format
    # -----------------------------------------------------

    if (
        "label" in sentiment
        and "score" in sentiment
    ):

        label = sentiment.get(
            "label",
            "UNKNOWN",
        )

        score = sentiment.get(
            "score",
            0.0,
        )

        sentiment_text = (
            f"<b>Label:</b> "
            f"{label}<br/>"

            f"<b>Confidence:</b> "
            f"{score:.2%}"
        )

    # -----------------------------------------------------
    # Legacy probability format
    # -----------------------------------------------------

    else:

        sentiment_text = (
            f"<b>Positive:</b> "
            f"{sentiment.get('positive', 0):.2%}<br/>"

            f"<b>Neutral:</b> "
            f"{sentiment.get('neutral', 0):.2%}<br/>"

            f"<b>Negative:</b> "
            f"{sentiment.get('negative', 0):.2%}"
        )

    story.append(
        Paragraph(
            sentiment_text,
            body_style,
        )
    )

    story.append(
        Spacer(1, 14)
    )


# ---------------------------------------------------------
# Metadata Section
# ---------------------------------------------------------

def add_metadata_section(
    story,
    metadata: dict,
):
    """
    Add metadata block.
    """

    if not metadata:
        return

    styles = (
        getSampleStyleSheet()
    )

    heading_style = styles[
        "Heading2"
    ]

    body_style = styles[
        "BodyText"
    ]

    story.append(
        Paragraph(
            "Metadata",
            heading_style,
        )
    )

    story.append(
        Spacer(1, 10)
    )

    metadata_lines = []

    for key, value in (
        metadata.items()
    ):

        pretty_key = (
            key.replace(
                "_",
                " ",
            )
            .title()
        )

        value = html.escape(
            str(value)
        )

        metadata_lines.append(
            f"<b>{pretty_key}:</b> "
            f"{value}"
        )

    metadata_text = (
        "<br/>".join(
            metadata_lines
        )
    )

    story.append(
        Paragraph(
            metadata_text,
            body_style,
        )
    )

    story.append(
        Spacer(1, 14)
    )


# ---------------------------------------------------------
# Main PDF Export
# ---------------------------------------------------------

def summary_to_pdf(
    summary_text: str,

    translated_summary: str | None = None,

    sentiment: dict | None = None,

    metadata: dict | None = None,

    title: str = (
        "Document Summary"
    ),
):
    """
    Generate summary PDF.

    Returns:
    -----------------------------------------------------
    io.BytesIO
    -----------------------------------------------------
    """

    buffer = io.BytesIO()

    document = SimpleDocTemplate(

        buffer,

        pagesize=letter,

        rightMargin=40,
        leftMargin=40,

        topMargin=40,
        bottomMargin=30,
    )

    story = []

    # -----------------------------------------------------
    # Header
    # -----------------------------------------------------

    build_header(
        story=story,
        title=title,
    )

    # -----------------------------------------------------
    # English Summary
    # -----------------------------------------------------

    add_text_section(

        story=story,

        heading="Summary (English)",

        text=summary_text,
    )

    # -----------------------------------------------------
    # Translated Summary
    # -----------------------------------------------------

    if translated_summary:

        add_text_section(

            story=story,

            heading="Translated Summary",

            text=translated_summary,
        )

    # -----------------------------------------------------
    # Sentiment
    # -----------------------------------------------------

    if sentiment:

        add_sentiment_section(

            story=story,

            sentiment=sentiment,
        )

    # -----------------------------------------------------
    # Metadata
    # -----------------------------------------------------

    if metadata:

        add_metadata_section(

            story=story,

            metadata=metadata,
        )

    # -----------------------------------------------------
    # Build PDF
    # -----------------------------------------------------

    document.build(
        story
    )

    buffer.seek(0)

    return buffer
