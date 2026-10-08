# document_processing/loaders.py

from __future__ import annotations

import os
from typing import Dict

from document_processing.docx_parser import (
    parse_docx,
)

from document_processing.image_parser import (
    parse_image,
)

from document_processing.metadata_extractor import (
    extract_metadata,
)

from document_processing.pdf_parser import (
    parse_pdf,
)


# ---------------------------------------------------------
# Supported Extensions
# ---------------------------------------------------------

PDF_EXTENSIONS = {
    ".pdf",
}

DOCX_EXTENSIONS = {
    ".docx",
    ".doc",
}

IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tiff",
    ".webp",
}


# ---------------------------------------------------------
# Extension
# ---------------------------------------------------------

def get_extension(
    file_path: str,
) -> str:
    """
    Get lowercase file extension.
    """

    return (
        os.path.splitext(
            file_path
        )[1]
        .lower()
    )


# ---------------------------------------------------------
# Type Helpers
# ---------------------------------------------------------

def is_pdf(
    file_path: str,
) -> bool:
    return (
        get_extension(file_path)
        in PDF_EXTENSIONS
    )


def is_docx(
    file_path: str,
) -> bool:
    return (
        get_extension(file_path)
        in DOCX_EXTENSIONS
    )


def is_image(
    file_path: str,
) -> bool:
    return (
        get_extension(file_path)
        in IMAGE_EXTENSIONS
    )


# ---------------------------------------------------------
# Main Loader
# ---------------------------------------------------------

def load_document(
    file_path: str,
    ocr_language: str = "eng",
    enable_ocr: bool = True,
) -> Dict:
    """
    Unified document loader.

    Supports:
    -----------------------------------------
    - PDF
    - DOCX
    - Images
    -----------------------------------------

    Returns:
    -----------------------------------------
    {
        "text": ...,
        "metadata": ...,
        "document_type": ...
    }
    -----------------------------------------
    """

    if not os.path.exists(
        file_path
    ):

        raise FileNotFoundError(
            f"File not found: "
            f"{file_path}"
        )

    extension = get_extension(
        file_path
    )

    # ---------------------------------------------------------
    # PDF
    # ---------------------------------------------------------

    if extension in PDF_EXTENSIONS:

        parsed = parse_pdf(
            file_path=file_path,
            ocr_language=(
                ocr_language
            ),
            enable_ocr_fallback=(
                enable_ocr
            ),
        )

        document_type = "pdf"

    # ---------------------------------------------------------
    # DOCX
    # ---------------------------------------------------------

    elif extension in DOCX_EXTENSIONS:

        parsed = parse_docx(
            file_path=file_path
        )

        document_type = "docx"

    # ---------------------------------------------------------
    # Images
    # ---------------------------------------------------------

    elif extension in IMAGE_EXTENSIONS:

        parsed = parse_image(
            image_path=file_path,
            language=ocr_language,
        )

        document_type = "image"

    # ---------------------------------------------------------
    # Unsupported
    # ---------------------------------------------------------

    else:

        raise ValueError(
            f"Unsupported file type: "
            f"{extension}"
        )

    # ---------------------------------------------------------
    # Metadata
    # ---------------------------------------------------------

    metadata = extract_metadata(
        file_path
    )

    # ---------------------------------------------------------
    # Merge
    # ---------------------------------------------------------

    parsed["metadata"] = {
        **metadata,
        **parsed.get(
            "metadata",
            {},
        ),
    }

    parsed[
        "document_type"
    ] = document_type

    parsed[
        "file_path"
    ] = file_path

    return parsed


# ---------------------------------------------------------
# Convenience Wrapper
# ---------------------------------------------------------

def load_document_text(
    file_path: str,
    ocr_language: str = "eng",
) -> str:
    """
    Load only extracted text.
    """

    document = load_document(
        file_path=file_path,
        ocr_language=ocr_language,
    )

    return document.get(
        "text",
        "",
    )