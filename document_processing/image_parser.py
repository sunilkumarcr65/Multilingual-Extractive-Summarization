# document_processing/image_parser.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional
from utils.text.cleaning import (
    clean_text,
    remove_ocr_artifacts,
)

try:
    from PIL import Image

    HAS_PIL = True

except ImportError:

    Image = None

    HAS_PIL = False


from ocr.ocr_engine import (
    extract_text_from_image,
)


# ---------------------------------------------------------
# OCR Result
# ---------------------------------------------------------

@dataclass
class ImageOCRResult:
    """
    OCR extraction result.
    """

    image_path: str

    extracted_text: str

    width: Optional[int] = None

    height: Optional[int] = None

    format: Optional[str] = None


# ---------------------------------------------------------
# Image Metadata
# ---------------------------------------------------------

def extract_image_metadata(
    image_path: str,
) -> Dict:
    """
    Extract image metadata.
    """

    if not HAS_PIL:

        return {}

    try:

        image = Image.open(
            image_path
        )

        return {
            "width":
                image.width,

            "height":
                image.height,

            "mode":
                image.mode,

            "format":
                image.format,
        }

    except Exception:

        return {}


# ---------------------------------------------------------
# OCR Parsing
# ---------------------------------------------------------

def parse_image(
    image_path: str,
    language: str = "eng",
) -> Dict:
    """
    Parse image document using OCR.

    Returns:
    -----------------------------------------
    {
        "text": ...,
        "metadata": {...}
    }
    -----------------------------------------
    """

    metadata = extract_image_metadata(
        image_path
    )

    try:
        extracted_text = (
            extract_text_from_image(
                image_path=image_path,
                language=language,
            )
        )

        extracted_text = clean_text(
            extracted_text,
            preserve_paragraphs=True,
        )

        extracted_text = remove_ocr_artifacts(
            extracted_text
        )

    except Exception as error:

        print(
            "[IMAGE OCR] "
            f"Extraction failed: "
            f"{error}"
        )

        extracted_text = ""

    result = ImageOCRResult(
        image_path=image_path,

        extracted_text=(
            extracted_text
        ),

        width=metadata.get(
            "width"
        ),

        height=metadata.get(
            "height"
        ),

        format=metadata.get(
            "format"
        ),
    )

    return {
        "text":
            result.extracted_text,

        "metadata":
            {
                "image_path":
                    result.image_path,

                "width":
                    result.width,

                "height":
                    result.height,

                "format":
                    result.format,
            },
    }


# ---------------------------------------------------------
# Convenience Wrapper
# ---------------------------------------------------------

def extract_text_from_image_file(
    image_path: str,
    language: str = "eng",
) -> str:
    """
    Simple image OCR extraction.
    """

    parsed = parse_image(
        image_path=image_path,
        language=language,
    )

    return parsed[
        "text"
    ]