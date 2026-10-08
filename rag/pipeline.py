from __future__ import annotations
import hashlib

from typing import Dict, List, Optional
from preprocessing.chunking import chunk_text
from preprocessing.language_detection import detect_language
from rag.cache import FileCache, build_cache_key
from rag.generator import RAGGenerator
from rag.reranker import CrossEncoderReranker
from rag.retriever import SemanticRetriever
from rag.vector_store import FAISSVectorStore


class RAGPipeline:
    """
    End-to-end multilingual RAG pipeline.

    Pipeline
    -----------------------------------------
    document
        ↓
    semantic chunking
        ↓
    embeddings
        ↓
    FAISS indexing
        ↓
    semantic retrieval
        ↓
    reranking
        ↓
    grounded generation
    -----------------------------------------
    """

    def __init__(
        self,
        tokenizer,
        use_reranker: bool = True,
        use_cache: bool = True,
        cache_dir: str = "cache/rag",
    ):

        self.vector_store = FAISSVectorStore()

        self.retriever = SemanticRetriever(
            self.vector_store
        )

        self.use_reranker = use_reranker

        self.reranker = (
            CrossEncoderReranker()
            if use_reranker
            else None
        )

        self.generator = (
            RAGGenerator(
                multilingual=True
            )
        )

        self.use_cache = use_cache

        self.cache = (
            FileCache(cache_dir=cache_dir)
            if use_cache
            else None
        )

        self._indexed_chunks_key = None

    # ---------------------------------------------------------
    # Index Document
    # ---------------------------------------------------------

    def index_document(
        self,
        text: str,
        source: Optional[str] = None,
        language: Optional[str] = None,
    ) -> int:
        text = (text or "").strip()
        if not text:
            return 0

        if language is None:
            language = detect_language(text)

        chunks = chunk_text(
            text=text,
            language=language,
            chunk_size=350,
            overlap=60,
        )
        
        if not chunks:
            return 0

        self.vector_store.add_documents(
            chunks=chunks,
            source=source,
            language=language,
        )
        return len(chunks)

    # ---------------------------------------------------------
    # Retrieve Context
    # ---------------------------------------------------------

    def retrieve_context(
        self,
        question: str,
        top_k: int = 5,
        language: Optional[str] = None,
    ) -> Dict:
        retrieved = self.retriever.retrieve(
            query=question,
            top_k=top_k,
            language=language,
        )

        # Retrieval fallback: never return empty context when index has chunks.
        if not retrieved and self.vector_store.metadata:
            fallback_items = self.vector_store.metadata[:top_k]
            retrieved = [
                {
                    "score": 0.0,
                    "text": item.text,
                    "chunk_id": item.chunk_id,
                    "source": item.source,
                    "section": item.section,
                    "page": item.page,
                    "language": item.language,
                }
                for item in fallback_items
                if item.text and item.text.strip()
            ]

        if len(retrieved) <= 2:
            use_reranker = False
        else:
            use_reranker = self.use_reranker
            
        if use_reranker and self.reranker and retrieved:
            retrieved = self.reranker.rerank(
                query=question,
                retrieved_chunks=retrieved,
            )

        context = "\n\n".join(
            f"[Chunk {i}]\n{r['text']}"
            for i, r in enumerate(retrieved, start=1)
        )

        return {"context": context, "retrieved_chunks": retrieved}

    # ---------------------------------------------------------
    # Ask Question
    # ---------------------------------------------------------

    def ask(
        self,
        question: str,
        top_k: int = 5,
        language: Optional[str] = None,
        use_cache: Optional[bool] = None,
    ) -> Dict:
        question = (question or "").strip()
        if not question:
            return {"answer": "Please provide a question.", "context": "", "retrieved_chunks": []}

        if language is None:
            language = detect_language(question)

        cache_enabled = self.use_cache if use_cache is None else use_cache
        cache_key     = build_cache_key(question, top_k, language)

        if cache_enabled and self.cache:
            cached = self.cache.load(cache_key)
            if cached is not None:
                cached["cached"] = True
                return cached

        retrieval        = self.retrieve_context(question=question, top_k=top_k, language=language)
        context          = retrieval["context"]
        retrieved_chunks = retrieval["retrieved_chunks"]

        answer = self.generator.generate_answer(
            question=question,
            context=context,
            language=language,
        )

        result = {
            "answer":           answer,
            "context":          context,
            "retrieved_chunks": retrieved_chunks,
            "cached":           False,
        }

        if cache_enabled and self.cache:
            self.cache.save(cache_key, result)

        return result

    # ---------------------------------------------------------
    # Persistence
    # ---------------------------------------------------------

    def save_index(self, index_path: str, metadata_path: str) -> None:
        self.vector_store.save(index_path=index_path, metadata_path=metadata_path)

    def load_index(self, index_path: str, metadata_path: str) -> None:
        self.vector_store = FAISSVectorStore.load(
            index_path=index_path,
            metadata_path=metadata_path,
        )
        self.retriever = SemanticRetriever(self.vector_store)

    @property
    def indexed_chunks(self) -> int:
        return self.vector_store.size

    def clear_index(self) -> None:
        self.vector_store.clear()
        self._indexed_chunks_key = None


# ---------------------------------------------------------
# Backward Compatibility Wrapper
# ---------------------------------------------------------

_default_pipeline: Optional[RAGPipeline] = None

# FIX: track which chunks are currently indexed in the global
# pipeline so we only re-index when the document actually changes.
_indexed_chunks_hash: Optional[int] = None


def answer_question_with_rag(
    question: str,
    chunks: List[str],
    top_k: int = 5,
) -> Dict:
    """
    Compatibility wrapper for qa_pipeline.py.

    FIXES
    ---------------------------------------------------------
    1. Avoids recursive re-chunking
    2. Prevents oversized FAISS embeddings
    3. Reuses existing index when chunks unchanged
    4. Dramatically improves QA retrieval quality
    ---------------------------------------------------------
    """

    global _default_pipeline, _indexed_chunks_hash

    if _default_pipeline is None:

        from transformers import (
            AutoTokenizer
        )

        tokenizer = (
            AutoTokenizer
            .from_pretrained(
                "google/flan-t5-base"
            )
        )

        _default_pipeline = (
            RAGPipeline(
                tokenizer=tokenizer
            )
        )

    pipeline = _default_pipeline

    # ---------------------------------------------------------
    # Stable chunk hash
    # ---------------------------------------------------------

    chunks_hash = hashlib.md5(
        "||".join(chunks)
        .encode("utf-8")

    ).hexdigest()

    # ---------------------------------------------------------
    # Only rebuild FAISS if chunks changed
    # ---------------------------------------------------------

    if chunks_hash != _indexed_chunks_hash:

        pipeline.clear_index()

        # -----------------------------------------------------
        # IMPORTANT FIX:
        # chunks are ALREADY chunked upstream
        # DO NOT re-run SemanticChunker
        # -----------------------------------------------------

        pipeline.vector_store.add_documents(

            chunks=chunks,

            source=None,

            language=detect_language(
                " ".join(chunks[:3])
            )
        )

        _indexed_chunks_hash = (
            chunks_hash
        )

    return pipeline.ask(
        question=question,
        top_k=top_k,
        use_cache=False,
    )
