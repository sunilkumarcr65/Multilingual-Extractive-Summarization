# document_processing/metadata_extractor.py

from __future__ import annotations

import os
from datetime import datetime
from typing import Dict, Optional

try:
    import pdfplumber

    HAS_PDFPLUMBER = True

except ImportError:

    pdfplumber = None

    HAS_PDFPLUMBER = False

try:
    from docx import Document

    HAS_DOCX = True

except ImportError:

    Document = None

    HAS_DOCX = False


# ---------------------------------------------------------
# File Metadata
# ---------------------------------------------------------

def extract_file_metadata(
    file_path: str,
) -> Dict:
    """
    Extract generic filesystem metadata.
    """

    if not os.path.exists(
        file_path
    ):

        return {}

    stat = os.stat(
        file_path
    )

    return {
        "file_name":
            os.path.basename(
                file_path
            ),

        "file_extension":
            os.path.splitext(
                file_path
            )[1].lower(),

        "file_size_bytes":
            stat.st_size,

        "created_at":
            datetime.fromtimestamp(
                stat.st_ctime
            ).isoformat(),

        "modified_at":
            datetime.fromtimestamp(
                stat.st_mtime
            ).isoformat(),
    }


# ---------------------------------------------------------
# PDF Metadata
# ---------------------------------------------------------

def extract_pdf_metadata(
    file_path: str,
) -> Dict:
    """
    Extract PDF metadata.
    """

    if not HAS_PDFPLUMBER:

        return {}

    try:

        with pdfplumber.open(
            file_path
        ) as pdf:

            raw_metadata = (
                pdf.metadata
                or {}
            )

            return {
                "title":
                    raw_metadata.get(
                        "Title"
                    ),

                "author":
                    raw_metadata.get(
                        "Author"
                    ),

                "subject":
                    raw_metadata.get(
                        "Subject"
                    ),

                "creator":
                    raw_metadata.get(
                        "Creator"
                    ),

                "producer":
                    raw_metadata.get(
                        "Producer"
                    ),

                "keywords":
                    raw_metadata.get(
                        "Keywords"
                    ),

                "page_count":
                    len(pdf.pages),
            }

    except Exception as error:

        print(
            "[METADATA] "
            f"PDF metadata failed: "
            f"{error}"
        )

        return {}


# ---------------------------------------------------------
# DOCX Metadata
# ---------------------------------------------------------

def extract_docx_metadata(
    file_path: str,
) -> Dict:
    """
    Extract DOCX metadata.
    """

    if not HAS_DOCX:

        return {}

    try:

        document = Document(
            file_path
        )

        properties = (
            document.core_properties
        )

        return {
            "title":
                properties.title,

            "author":
                properties.author,

            "subject":
                properties.subject,

            "keywords":
                properties.keywords,

            "category":
                properties.category,

            "comments":
                properties.comments,

            "created":
                str(
                    properties.created
                )
                if properties.created
                else None,

            "modified":
                str(
                    properties.modified
                )
                if properties.modified
                else None,
        }

    except Exception as error:

        print(
            "[METADATA] "
            f"DOCX metadata failed: "
            f"{error}"
        )

        return {}


# ---------------------------------------------------------
# Unified Metadata
# ---------------------------------------------------------

def extract_metadata(
    file_path: str,
) -> Dict:
    """
    Unified metadata extraction.

    Supports:
    - PDF
    - DOCX
    - generic file metadata
    """

    extension = (
        os.path.splitext(
            file_path
        )[1]
        .lower()
    )

    metadata = {
        "file":
            extract_file_metadata(
                file_path
            )
    }

    # ---------------------------------------------------------
    # PDF
    # ---------------------------------------------------------

    if extension == ".pdf":

        metadata[
            "document"
        ] = extract_pdf_metadata(
            file_path
        )

    # ---------------------------------------------------------
    # DOCX
    # ---------------------------------------------------------

    elif extension in {
        ".docx",
        ".doc",
    }:

        metadata[
            "document"
        ] = extract_docx_metadata(
            file_path
        )

    else:

        metadata[
            "document"
        ] = {}

    return {
        **metadata["file"],
        **metadata["document"],
    }