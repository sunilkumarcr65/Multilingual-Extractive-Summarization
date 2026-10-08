from __future__ import annotations

import time
from typing import Any, Dict

from document_processing.loaders import load_document_text
from formatting.response_formatter import build_error_response, build_qa_response
from preprocessing.chunking import chunk_text
from preprocessing.language_detection import detect_language
from rag.pipeline import answer_question_with_rag
from utils.text.pipeline import preprocess_text


# ---------------------------------------------------------
# Document chunk cache
# Keyed by file_path so switching documents invalidates it.
# Avoids re-loading + re-chunking on every question.
# ---------------------------------------------------------

_chunk_cache: Dict[str, list] = {}


# ---------------------------------------------------------
# Validation
# ---------------------------------------------------------

def validate_question(question: str) -> str:
    question = (question or "").strip()
    if not question:
        raise ValueError("Question cannot be empty.")
    return question


# ---------------------------------------------------------
# Context Preparation  (with caching)
# ---------------------------------------------------------

def prepare_document_chunks(
    file_path: str,
    ocr_language: str = "eng",
) -> list:
    """
    Load, preprocess, and chunk document.

    FIX: cached by file_path — the document is only loaded and
    chunked once per file.  Every subsequent question on the same
    file returns the cached chunks instantly, removing the biggest
    latency source from the QA loop.
    """

    if file_path in _chunk_cache:
        return _chunk_cache[file_path]

    document_text = load_document_text(
        file_path=file_path,
        ocr_language=ocr_language,
    )

    if not (document_text and document_text.strip()):
        return []

    processed_sentences = preprocess_text(document_text)
    processed_text = " ".join(processed_sentences)

    # FIX: pass language explicitly — chunk_text signature requires it
    # and omitting it caused a silent crash in some chunker backends.
    detected_language = detect_language(processed_text)
    if detected_language == "unknown":
        detected_language = "en"

    chunks = chunk_text(
        text=processed_text,
        language=detected_language,
        chunk_size=180,
        overlap=30,
    )

    _chunk_cache[file_path] = chunks
    return chunks


# ---------------------------------------------------------
# Cache invalidation helper
# Called by summarizer_tab when a new file is uploaded.
# ---------------------------------------------------------

def clear_chunk_cache(file_path: str | None = None) -> None:
    """
    Clear cached chunks.
    Pass file_path to evict one entry, or None to evict all.
    """
    if file_path is None:
        _chunk_cache.clear()
    else:
        _chunk_cache.pop(file_path, None)


# ---------------------------------------------------------
# Main QA Pipeline
# ---------------------------------------------------------

def answer_question(
    file_path: str,
    question: str,
    ocr_language: str = "eng",
    top_k: int = 5,
) -> Dict[str, Any]:
    """
    RAG-based QA pipeline.

    Pipeline
    -----------------------------------------
    document loading  (cached after first call)
        ↓
    preprocessing     (cached after first call)
        ↓
    chunking          (cached after first call)
        ↓
    retrieval
        ↓
    reranking
        ↓
    answer generation
    -----------------------------------------
    """

    start_time = time.time()

    try:
        question = validate_question(question)

        # FIX: chunks are now fetched from cache if available —
        # no re-loading or re-chunking on repeated questions.
        chunks = prepare_document_chunks(
            file_path=file_path,
            ocr_language=ocr_language,
        )

        if not chunks:
            return build_error_response(
                message="No readable text found in document.",
                error_type="empty_document",
            )
        
        print("\n=== QA CHUNKS ===")
        print(len(chunks))
        print(chunks[:2])
        print("=================\n")

        rag_result = answer_question_with_rag(
            question=question,
            chunks=chunks,
            top_k=top_k,
        )

        processing_time = time.time() - start_time

        return build_qa_response(
            question=question,
            answer=rag_result.get("answer", ""),
            context=rag_result.get("context", ""),
            retrieved_chunks=rag_result.get("retrieved_chunks", []),
        ) | {"processing_time_seconds": round(processing_time, 3)}

    except Exception as error:
        return build_error_response(
            message=str(error),
            error_type="qa_pipeline_error",
        )
