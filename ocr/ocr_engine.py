from __future__ import annotations

from functools import lru_cache
from typing import Optional
from PIL import ImageOps

try:
    import pytesseract
    from PIL import Image

    HAS_TESSERACT = True

except ImportError:

    pytesseract = None
    Image = None

    HAS_TESSERACT = False


# ---------------------------------------------------------
# OCR Availability
# ---------------------------------------------------------

def is_ocr_available() -> bool:
    """
    Check whether OCR dependencies exist.
    """

    return HAS_TESSERACT


# ---------------------------------------------------------
# OCR Model Config
# ---------------------------------------------------------

@lru_cache(maxsize=1)
def get_tesseract_config() -> str:
    """
    Default OCR configuration.

    PSM 3:
        Fully automatic page segmentation.

    OEM 3:
        Default OCR engine.
    """

    return (
        "--oem 3 "
        "--psm 6 "
        "-c preserve_interword_spaces=1"
    )


# ---------------------------------------------------------
# OCR Image Preprocessing
# ---------------------------------------------------------

def preprocess_image(
    image,
):
    """
    Improve OCR quality.

    Steps:
    -----------------------------------------------------
    - grayscale
    - auto contrast
    - binarization
    -----------------------------------------------------
    """

    image = ImageOps.grayscale(
        image
    )

    image = ImageOps.autocontrast(
        image
    )

    image = image.point(
        lambda p:
        255 if p > 140 else 0
    )

    return image

# ---------------------------------------------------------
# OCR Extraction
# ---------------------------------------------------------

def extract_text_from_image(
    image_path: str,
    language: str = "eng",
    config: Optional[str] = None,
) -> str:
    """
    Extract text from image using OCR.
    """

    if not HAS_TESSERACT:

        raise RuntimeError(
            "OCR dependencies not installed.\n"
            "Install with:\n"
            "pip install pytesseract pillow"
        )

    config = config or get_tesseract_config()

    with Image.open(
        image_path
    ) as image:

        image = preprocess_image(
            image
        )

        text = pytesseract.image_to_string(
            image,
            lang=language,
            config=config,
        )

    image = preprocess_image(
        image
    )

    text = pytesseract.image_to_string(
        image,
        lang=language,
        config=config,
    )

    return text.strip()


# ---------------------------------------------------------
# OCR From PIL Image
# ---------------------------------------------------------

def extract_text_from_pil_image(
    image,
    language: str = "eng",
    config: Optional[str] = None,
) -> str:
    """
    OCR directly from PIL image object.
    """

    if not HAS_TESSERACT:

        raise RuntimeError(
            "OCR dependencies not installed."
        )

    config = config or get_tesseract_config()

    image = preprocess_image(image)

    text = pytesseract.image_to_string(
        image,
        lang=language,
        config=config,
    )

    return text.strip()

# ---------------------------------------------------------
# OCR Extraction from PDF Page
# ---------------------------------------------------------

def extract_text_from_pdf_page(
    image,
    language: str = "eng",
) -> str:
    """
    OCR extraction from PDF page image.
    """

    if image is None:
        return ""

    try:

        # PIL image
        text = pytesseract.image_to_string(
            image,
            lang=language,
        )

        return text.strip()

    except Exception as error:

        print(
            "[OCR] "
            f"PDF OCR failed: "
            f"{error}"
        )

        return ""