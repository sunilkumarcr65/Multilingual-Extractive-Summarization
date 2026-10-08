from __future__ import annotations

import re
from typing import Dict

import fitz


# ---------------------------------------------------------
# Cleaning
# ---------------------------------------------------------

def clean_pdf_text(
    text: str,
) -> str:
    """
    Normalize extracted PDF text.
    """

    text = (text or "")

    text = text.replace("\x00", " ")

    text = re.sub(
        r"[ \t]+",
        " ",
        text,
    )

    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text,
    )

    return text.strip()


# ---------------------------------------------------------
# Parse PDF
# ---------------------------------------------------------

def parse_pdf(
    file_path: str,
    ocr_language: str = "eng",
    enable_ocr_fallback: bool = True,
) -> Dict:
    """
    Parse PDF safely using PyMuPDF.
    """

    pages = []

    full_text_parts = []

    # ---------------------------------------------------------
    # IMPORTANT:
    # Use context manager so lifecycle is safe
    # ---------------------------------------------------------

    with fitz.open(file_path) as document:

        total_pages = len(document)

        for page_index in range(total_pages):

            try:

                page = document.load_page(
                    page_index
                )

                extracted_text = (
                    page.get_text("text")
                )

                extracted_text = clean_pdf_text(
                    extracted_text
                )

                pages.append(
                    {
                        "page_number":
                            page_index + 1,

                        "text":
                            extracted_text,

                        "used_ocr":
                            False,
                    }
                )

                if extracted_text:

                    full_text_parts.append(
                        extracted_text
                    )

            except Exception as error:

                print(
                    "[PDF PARSER] "
                    f"Page {page_index + 1} failed: "
                    f"{error}"
                )

    full_text = "\n\n".join(
        full_text_parts
    )

    metadata = {
        "page_count":
            len(pages),

        "ocr_pages":
            0,

        "has_text":
            bool(full_text),
    }

    return {
        "text":
            full_text,

        "pages":
            pages,

        "metadata":
            metadata,
    }


# ---------------------------------------------------------
# Convenience Wrapper
# ---------------------------------------------------------

def extract_text_from_pdf(
    file_path: str,
    ocr_language: str = "eng",
) -> str:
    """
    Simple extraction wrapper.
    """

    parsed = parse_pdf(
        file_path=file_path,
        ocr_language=ocr_language,
    )

    return parsed["text"]
