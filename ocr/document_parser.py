from __future__ import annotations

import os
from pathlib import Path
from typing import List

from ocr.ocr_engine import (
    extract_text_from_image,
    extract_text_from_pil_image,
    is_ocr_available,
)

from utils.text.cleaning import (
    clean_text,
    remove_ocr_artifacts,
)

# ---------------------------------------------------------
# Optional Dependencies
# ---------------------------------------------------------

try:
    import pdfplumber

    HAS_PDF = True

except ImportError:

    pdfplumber = None
    HAS_PDF = False

try:
    from docx import Document

    HAS_DOCX = True

except ImportError:

    Document = None
    HAS_DOCX = False


# ---------------------------------------------------------
# TXT
# ---------------------------------------------------------

def extract_txt(
    file_path: str,
) -> str:
    """
    Extract plain text from TXT file.
    """

    with open(
        file_path,
        "r",
        encoding="utf-8",
        errors="replace",
    ) as file:

        return file.read().strip()


# ---------------------------------------------------------
# DOCX
# ---------------------------------------------------------

def extract_docx(
    file_path: str,
) -> str:
    """
    Extract text from DOCX.
    """

    if not HAS_DOCX:

        raise RuntimeError(
            "python-docx not installed.\n"
            "Install with:\n"
            "pip install python-docx"
        )

    document = Document(file_path)

    paragraphs = [
        paragraph.text.strip()
        for paragraph in document.paragraphs
        if paragraph.text.strip()
    ]

    return "\n\n".join(paragraphs)


# ---------------------------------------------------------
# IMAGE
# ---------------------------------------------------------

def extract_image(
    file_path: str,
    ocr_language: str = "eng",
) -> str:
    """
    Extract text from image using OCR.
    """

    text = extract_text_from_image(

        image_path=file_path,

        language=ocr_language,
    )

    text = clean_text(
        text,
        preserve_paragraphs=True,
    )

    text = remove_ocr_artifacts(
        text
    )

    return text


# ---------------------------------------------------------
# PDF
# ---------------------------------------------------------

def extract_pdf(
    file_path: str,
    do_ocr: bool = False,
    ocr_language: str = "eng",
) -> str:
    """
    Extract text from PDF.

    Uses:
    - direct PDF text extraction
    - OCR fallback for scanned pages
    """

    if not HAS_PDF:

        raise RuntimeError(
            "pdfplumber not installed.\n"
            "Install with:\n"
            "pip install pdfplumber"
        )

    pages: List[str] = []

    with pdfplumber.open(file_path) as pdf:

        for page in pdf.pages:

            text = (
                page.extract_text()
                or ""
            ).strip()

            # -------------------------------------------------
            # Native PDF text
            # -------------------------------------------------

            if text:

                ocr_text = clean_text(
                    ocr_text,
                    preserve_paragraphs=True,
                )

                ocr_text = remove_ocr_artifacts(
                    ocr_text
                )

                pages.append(
                    ocr_text
                )
                continue

            # -------------------------------------------------
            # OCR fallback
            # -------------------------------------------------

            if (
                do_ocr
                and is_ocr_available()
            ):

                try:

                    image = (
                        page.to_image(
                            resolution=180
                        ).original
                    )

                    ocr_text = (
                        extract_text_from_pil_image(
                            image=image,
                            language=ocr_language,
                        )
                    )

                    if ocr_text.strip():

                        pages.append(
                            ocr_text
                        )

                except Exception as error:

                    print(
                        f"[OCR] Failed page OCR: "
                        f"{error}"
                    )

    return "\n\n".join(pages).strip()


# ---------------------------------------------------------
# Main Dispatcher
# ---------------------------------------------------------

SUPPORTED_EXTENSIONS = {
    ".pdf",
    ".docx",
    ".doc",
    ".txt",
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".webp",
    ".tiff",
}


def extract_document_text(
    file_path: str,
    do_ocr: bool = False,
    ocr_language: str = "eng",
) -> str:
    """
    Universal document parser.

    Supported:
    -----------------------------------------
    PDF
    DOCX
    TXT
    PNG/JPG/JPEG/BMP/WEBP
    -----------------------------------------
    """

    if not os.path.isfile(file_path):

        raise FileNotFoundError(
            f"File not found: {file_path}"
        )

    extension = (
        Path(file_path)
        .suffix
        .lower()
    )

    if extension not in SUPPORTED_EXTENSIONS:

        raise ValueError(
            f"Unsupported file type: "
            f"{extension}"
        )

    # ---------------------------------------------------------
    # Dispatch
    # ---------------------------------------------------------

    if extension == ".pdf":

        return extract_pdf(
            file_path=file_path,
            do_ocr=do_ocr,
            ocr_language=ocr_language,
        )

    if extension == ".docx":

        return extract_docx(
            file_path
        )

    if extension == ".txt":

        return extract_txt(
            file_path
        )

    return extract_image(
        file_path=file_path,
        ocr_language=ocr_language,
    )