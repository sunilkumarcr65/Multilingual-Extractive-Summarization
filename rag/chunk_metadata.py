from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional


# ---------------------------------------------------------
# Chunk Metadata
# ---------------------------------------------------------

@dataclass
class ChunkMetadata:
    """
    Metadata associated with a semantic chunk.

    Useful for:
    -----------------------------------------
    - source tracking
    - page attribution
    - citations
    - retrieval debugging
    - hallucination tracing
    -----------------------------------------
    """

    chunk_id: int

    text: str

    source: Optional[str] = None

    section: Optional[str] = None

    page: Optional[int] = None

    language: Optional[str] = None

    token_count: Optional[int] = None

    embedding_model: Optional[str] = None

    document_id: Optional[str] = None


# ---------------------------------------------------------
# Serialization
# ---------------------------------------------------------

def metadata_to_dict(
    metadata: ChunkMetadata,
) -> Dict:
    """
    Convert metadata object to dictionary.
    """

    return asdict(metadata)


def metadata_from_dict(
    data: Dict,
) -> ChunkMetadata:
    """
    Restore metadata object from dictionary.
    """

    return ChunkMetadata(**data)


# ---------------------------------------------------------
# Utilities
# ---------------------------------------------------------

def build_chunk_metadata(
    chunk_id: int,
    text: str,
    source: Optional[str] = None,
    section: Optional[str] = None,
    page: Optional[int] = None,
    language: Optional[str] = None,
    token_count: Optional[int] = None,
    embedding_model: Optional[str] = None,
    document_id: Optional[str] = None,
) -> ChunkMetadata:
    """
    Convenience metadata builder.
    """

    return ChunkMetadata(
        chunk_id=chunk_id,

        text=text,

        source=source,

        section=section,

        page=page,

        language=language,

        token_count=token_count,

        embedding_model=embedding_model,

        document_id=document_id,
    )


# ---------------------------------------------------------
# Metadata Formatter
# ---------------------------------------------------------

def format_metadata(
    metadata: ChunkMetadata,
) -> str:
    """
    Human-readable metadata summary.
    """

    components: List[str] = []

    if metadata.source:

        components.append(
            f"Source: {metadata.source}"
        )

    if metadata.section:

        components.append(
            f"Section: {metadata.section}"
        )

    if metadata.page is not None:

        components.append(
            f"Page: {metadata.page}"
        )

    if metadata.language:

        components.append(
            f"Language: {metadata.language}"
        )

    if metadata.token_count:

        components.append(
            f"Tokens: {metadata.token_count}"
        )

    return " | ".join(
        components
    )