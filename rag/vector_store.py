from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional

import faiss
import numpy as np

from rag.embeddings import (
    embed_sentences,
)


# ---------------------------------------------------------
# Chunk Metadata
# ---------------------------------------------------------

@dataclass
class ChunkMetadata:
    """
    Metadata associated with each chunk.
    """

    chunk_id: int
    text: str

    source: Optional[str] = None
    section: Optional[str] = None
    page: Optional[int] = None
    language: Optional[str] = None


# ---------------------------------------------------------
# Vector Store
# ---------------------------------------------------------

class FAISSVectorStore:
    """
    FAISS-backed semantic vector store.

    Features
    -----------------------------------------
    - semantic chunk indexing
    - multilingual embeddings
    - metadata storage
    - persistence
    - fast retrieval
    -----------------------------------------
    """

    def __init__(
        self,
        embedding_dimension: int = 384, #embed_sentences(["test"])
    ):
        self.embedding_dimension = (
            embedding_dimension
        )

        # Cosine similarity via inner product
        self.index = faiss.IndexFlatIP(
            embedding_dimension
        )

        self.metadata: List[
            ChunkMetadata
        ] = []

        self.is_trained = True

    # ---------------------------------------------------------
    # Add Chunks
    # ---------------------------------------------------------

    def add_documents(
        self,
        chunks: List[str],
        source: Optional[str] = None,
        language: Optional[str] = None,
        section: Optional[str] = None,
    ) -> None:
        """
        Add semantic chunks to vector store.
        """

        if not chunks:
            return

        embeddings = embed_sentences(
            chunks,
            language=language,
            normalize=True,
        )

        embeddings = embeddings.astype(
            np.float32
        )

        self.index.add(embeddings)

        # -----------------------------------------------------
        # Metadata
        # -----------------------------------------------------

        start_id = len(self.metadata)

        for idx, chunk in enumerate(chunks):

            metadata = ChunkMetadata(
                chunk_id=start_id + idx,

                text=chunk,

                source=source,

                section=section,

                language=language,
            )

            self.metadata.append(
                metadata
            )

    # ---------------------------------------------------------
    # Search
    # ---------------------------------------------------------

    def search(
        self,
        query: str,
        top_k: int = 5,
        language: Optional[str] = None,
    ) -> List[Dict]:
        """
        Semantic vector search.
        """

        if self.index.ntotal == 0:

            return []

        query_embedding = embed_sentences(
            [query],
            language=language,
            normalize=True,
        )

        query_embedding = (
            query_embedding.astype(
                np.float32
            )
        )

        similarities, indices = (
            self.index.search(
                query_embedding,
                top_k,
            )
        )

        results = []

        for similarity, idx in zip(
            similarities[0],
            indices[0],
        ):

            if idx < 0:
                continue

            metadata = self.metadata[idx]

            results.append({
                "score":
                    float(similarity),

                "text":
                    metadata.text,

                "chunk_id":
                    metadata.chunk_id,

                "source":
                    metadata.source,

                "section":
                    metadata.section,

                "page":
                    metadata.page,

                "language":
                    metadata.language,
            })

        return results

    # ---------------------------------------------------------
    # Persistence
    # ---------------------------------------------------------

    def save(
        self,
        index_path: str,
        metadata_path: str,
    ) -> None:
        """
        Save FAISS index + metadata.
        """

        os.makedirs(
            os.path.dirname(index_path),
            exist_ok=True,
        )

        os.makedirs(
            os.path.dirname(
                metadata_path
            ),
            exist_ok=True,
        )

        # -----------------------------------------------------
        # Save FAISS
        # -----------------------------------------------------

        faiss.write_index(
            self.index,
            index_path,
        )

        # -----------------------------------------------------
        # Save metadata
        # -----------------------------------------------------

        metadata_json = [
            asdict(meta)
            for meta in self.metadata
        ]

        with open(
            metadata_path,
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                metadata_json,
                file,
                ensure_ascii=False,
                indent=2,
            )

    # ---------------------------------------------------------
    # Load
    # ---------------------------------------------------------

    @classmethod
    def load(
        cls,
        index_path: str,
        metadata_path: str,
    ) -> "FAISSVectorStore":
        """
        Load vector store from disk.
        """

        if not os.path.exists(
            index_path
        ):

            raise FileNotFoundError(
                f"Missing FAISS index: "
                f"{index_path}"
            )

        if not os.path.exists(
            metadata_path
        ):

            raise FileNotFoundError(
                f"Missing metadata file: "
                f"{metadata_path}"
            )

        # -----------------------------------------------------
        # Create instance
        # -----------------------------------------------------

        store = cls()

        # -----------------------------------------------------
        # Load FAISS index
        # -----------------------------------------------------

        store.index = faiss.read_index(
            index_path
        )

        # -----------------------------------------------------
        # Load metadata
        # -----------------------------------------------------

        with open(
            metadata_path,
            "r",
            encoding="utf-8",
        ) as file:

            metadata_json = json.load(
                file
            )

        store.metadata = [
            ChunkMetadata(**meta)
            for meta in metadata_json
        ]

        return store

    # ---------------------------------------------------------
    # Utilities
    # ---------------------------------------------------------

    @property
    def size(self) -> int:
        """
        Number of indexed chunks.
        """

        return self.index.ntotal

    def clear(self) -> None:
        """
        Reset vector store.
        """

        self.index.reset()

        self.metadata.clear()