# document_processing/ocr_fallback.py

from __future__ import annotations

from typing import Optional

try:
    from PIL import Image

    HAS_PIL = True

except ImportError:

    Image = None

    HAS_PIL = False

try:
    import pdfplumber

    HAS_PDFPLUMBER = True

except ImportError:

    pdfplumber = None

    HAS_PDFPLUMBER = False


from ocr.ocr_engine import (
    extract_text_from_image,
)


# ---------------------------------------------------------
# OCR Heuristic
# ---------------------------------------------------------

def requires_ocr(
    extracted_text: Optional[str],
    min_characters: int = 30,
) -> bool:
    """
    Decide whether OCR fallback
    should be triggered.
    """

    extracted_text = (
        extracted_text or ""
    ).strip()

    return (
        len(extracted_text)
        < min_characters
    )


# ---------------------------------------------------------
# PDF Page OCR
# ---------------------------------------------------------

def run_pdf_page_ocr(
    pdf_page,
    language: str = "eng",
    dpi: int = 300,
) -> str:
    """
    OCR a PDF page using image rendering.
    """

    if not HAS_PIL:

        raise ImportError(
            "Pillow is not installed."
        )

    try:

        page_image = (
            pdf_page.to_image(
                resolution=dpi
            )
        )

        pil_image = (
            page_image.original
        )

        text = (
            extract_text_from_image_object(
                pil_image,
                language=language,
            )
        )

        return text.strip()

    except Exception as error:

        print(
            "[OCR FALLBACK] "
            f"PDF OCR failed: "
            f"{error}"
        )

        return ""


# ---------------------------------------------------------
# OCR PIL Image
# ---------------------------------------------------------

def extract_text_from_image_object(
    image,
    language: str = "eng",
) -> str:
    """
    OCR from PIL image object.
    """

    try:

        import pytesseract

    except ImportError:

        raise ImportError(
            "pytesseract is not installed."
        )

    try:

        text = pytesseract.image_to_string(
            image,
            lang=language,
        )

        return (
            text or ""
        ).strip()

    except Exception as error:

        print(
            "[OCR FALLBACK] "
            f"Image OCR failed: "
            f"{error}"
        )

        return ""


# ---------------------------------------------------------
# Image OCR Wrapper
# ---------------------------------------------------------

def run_image_ocr(
    image_path: str,
    language: str = "eng",
) -> str:
    """
    OCR image file.
    """

    try:

        return (
            extract_text_from_image(
                image_path=image_path,
                language=language,
            )
        )

    except Exception as error:

        print(
            "[OCR FALLBACK] "
            f"Image OCR failed: "
            f"{error}"
        )

        return ""