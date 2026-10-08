from __future__ import annotations

import re
import time
from typing import Any, Dict, List, Optional

from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer

from abstractive.service import (
    generate_abstractive_summary,
)

from document_processing.document_classifier import (
    classify_document,
)

from document_processing.loaders import (
    load_document_text,
)

from formatting.markdown_formatter import (
    format_summary_markdown,
)

from formatting.response_formatter import (
    build_error_response,
    build_summary_response,
)

from orchestration.model_router import (
    get_compression_ratio,
    select_document_model,
    select_summarization_mode,
)

from orchestration.sentiment_manager import (
    analyze_document_sentiment,
)
from orchestration.contract_insights import (
    build_contract_insights,
)

from pipelines.translation_pipeline import (
    translate_document_text,
)

from preprocessing.chunking import (
    chunk_text,
)
from preprocessing.language_detection import (
    detect_language,
)

from extractive.extractor import (
    generate_extractive_summary,
)

from utils.text.pipeline import (
    preprocess_text,
)

from utils.text.sentence_splitter import (
    split_sentences,
)

from utils.text.hallucination_filter import (
    remove_hallucinated_sentences,
)
from metrics.evaluation_metrics import (
    compute_metrics,
)

from preprocessing.section_detection import (
    detect_sections,
)

from preprocessing.section_priority import (
    rank_sections,
)

import gc
import torch

# ---------------------------------------------------------
# Constants
# ---------------------------------------------------------

MAX_SAFE_INPUT_TOKENS = 420

RESUME_CHUNK_SIZE = 500

LEGAL_HINTS = {
    "agreement",
    "statement of work",
    "payments",
    "independent contractor",
    "ownership",
    "confidential",
    "data protection",
    "warranties",
    "termination",
    "limitation of liability",
    "governing law",
    "arbitration",
}


# ---------------------------------------------------------
# Tokenizer Cache
# ---------------------------------------------------------

_tokenizer_cache: Dict[str, Any] = {}
_embedding_model = None

def _get_tokenizer(
    model_name: str,
):

    if model_name not in _tokenizer_cache:

        _tokenizer_cache[
            model_name
        ] = AutoTokenizer.from_pretrained(
            model_name
        )

    return _tokenizer_cache[
        model_name
    ]

def get_embedding_model():

    global _embedding_model

    if _embedding_model is None:

        print(
            "[PIPELINE] Loading semantic dedup model..."
        )

        _embedding_model = SentenceTransformer(
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )

    return _embedding_model

# ---------------------------------------------------------
# Token Count
# ---------------------------------------------------------

def count_tokens(
    text: str,
    model_name: str,
) -> int:

    try:

        tokenizer = _get_tokenizer(
            model_name
        )

        tokens = tokenizer.encode(
            text,
            add_special_tokens=False,
        )

        return len(tokens)

    except Exception:

        return int(
            len(text.split()) * 1.3
        )

# ---------------------------------------------------------
# Safe Chunking
# ---------------------------------------------------------

def create_safe_chunks(
    text: str,
    language: str,
    model_name: str,
    document_type: str,
) -> List[str]:

    token_count = count_tokens(
        text,
        model_name,
    )

    # -------------------------------------------------
    # Small enough
    # -------------------------------------------------

    if token_count <= MAX_SAFE_INPUT_TOKENS:

        return [text]

    # -------------------------------------------------
    # Resume handling
    # -------------------------------------------------

    if document_type == "resume":

        return chunk_text(
            text=text,
            language=language,
            chunk_size=450,
            overlap=40,
        )

    # -------------------------------------------------
    # General documents
    # MUCH LOWER OVERLAP
    # -------------------------------------------------

    return chunk_text(
        text=text,
        language=language,
        chunk_size=520,
        overlap=40,
    )


# ---------------------------------------------------------
# Chunk Summarization
# ---------------------------------------------------------

def summarize_chunks(
    chunks: List[str],
    summarization_mode: str,
    source_language: str,
    compression_ratio: float,
    model_name: str,
    document_type: str,
) -> List[str]:

    summaries: List[str] = []

    for chunk in chunks:

        chunk = (
            chunk or ""
        ).strip()

        if not chunk:
            continue

        # -------------------------------------------------
        # Resume-specific hybrid summarization
        # -------------------------------------------------

        if document_type == "resume":

            extractive = generate_extractive_summary(

                text=chunk,

                compression_ratio=0.6,
            )

            summary = generate_abstractive_summary(

                text=extractive,

                source_language=source_language,

                model_name=model_name,

                document_type=document_type,

                synthesis_mode=True,

                max_new_tokens=120,

                min_new_tokens=40,
            )

            summary = (summary or "").strip()

            if summary:

                summaries.append(summary)

            continue
        # -------------------------------------------------
        # Extractive
        # -------------------------------------------------

        if summarization_mode == "extractive":

            summary = (
                generate_extractive_summary(
                    text=chunk,
                    compression_ratio=compression_ratio,
                )
            )
        
            print("=" * 80)
            print("[EXTRACTIVE SUMMARY]")
            print("=" * 80)

            print(
                f"[SUMMARY GENERATED] "
                f"{len(summary.split())} words"
            )

        # -------------------------------------------------
        # Hybrid
        # -------------------------------------------------

        elif summarization_mode == "hybrid":

            extractive = (
                generate_extractive_summary(

                    text=chunk,

                    compression_ratio=(

                        0.65

                        if document_type == "resume"

                        else min(
                            compression_ratio * 1.6,
                            0.8,
                        )
                    ),
                )
            )

            summary = (
                generate_abstractive_summary(
                    text=extractive,
                    source_language=source_language,
                    model_name=model_name,
                    document_type=document_type,
                    synthesis_mode=True,
                    max_new_tokens=(
                        140
                        if document_type == "resume"
                        else 220
                    ),
                    min_new_tokens=(
                        50
                        if document_type == "resume"
                        else 80
                    ),
                )
            )

        # -------------------------------------------------
        # Pure abstractive
        # -------------------------------------------------

        else:

            summary = (
                generate_abstractive_summary(
                    text=chunk,
                    source_language=source_language,
                    model_name=model_name,
                    document_type=document_type,
                    synthesis_mode=False,
                    max_new_tokens=180,
                    min_new_tokens=70,
                )
            )

            print("=" * 80)
            print("[ABSTRACTIVE REWRITE]")
            print("=" * 80)

            print(
                f"[SUMMARY GENERATED] "
                f"{len(summary.split())} words"
            )

        summary = (
            summary or ""
        ).strip()

        if summary:

            summaries.append(
                summary
            )

    return summaries


# ---------------------------------------------------------
# Recursive Hierarchical Synthesis
# ---------------------------------------------------------

def synthesize_final_summary(
    chunk_summaries: List[str],
    source_language: str,
    model_name: str,
    document_type: str,
    recursion_group_size: int = 6,
) -> str:

    if not chunk_summaries:
        return ""

    # -----------------------------------------------------
    # Resume handling
    # -----------------------------------------------------

    if document_type == "resume":

        seen = set()

        merged = []

        for item in chunk_summaries:

            normalized = (
                " ".join(item.split())
                .strip()
                .lower()
            )

            if (
                not normalized
                or normalized in seen
            ):
                continue

            seen.add(normalized)

            merged.append(
                " ".join(item.split()).strip()
            )

        return "\n".join(merged).strip()

    # -----------------------------------------------------
    # Recursive summarization
    # -----------------------------------------------------

    current_level = chunk_summaries

    recursion_depth = 1

    while len(current_level) > 1:

        print(
            f"[RECURSIVE SYNTHESIS] "
            f"Depth={recursion_depth} | "
            f"Items={len(current_level)}"
        )
        
        # -------------------------------------------------
        # Stop early if summaries are already condensed
        # -------------------------------------------------

        avg_words = sum(
            len(x.split())
            for x in current_level
        ) / max(len(current_level), 1)

        if (
            avg_words < 160
            and len(current_level) <= 4
        ):

            print(
                "[RECURSIVE SYNTHESIS] "
                "Stopping early."
            )

            break
        next_level = []

        # -------------------------------------------------
        # Group summaries
        # -------------------------------------------------

        for i in range(
            0,
            len(current_level),
            recursion_group_size,
        ):

            group = current_level[
                i : i + recursion_group_size
            ]

            combined = "\n\n".join(group)

            # -------------------------------------------------
            # Prevent tokenizer overflow
            # -------------------------------------------------

            max_recursive_tokens = 420

            while (
                count_tokens(
                    combined,
                    model_name,
                ) > max_recursive_tokens
                and len(group) > 1
            ):

                group = group[:-1]

                combined = "\n\n".join(group)

            combined = combined.strip()

            if not combined:
                continue

            token_count = count_tokens(
                combined,
                model_name,
            )

            print(
                f"[GROUP TOKENS] "
                f"{token_count}"
            )

            # ---------------------------------------------
            # Dynamic generation sizing
            # ---------------------------------------------

            dynamic_max = max(
                220,
                min(
                    420,
                    int(token_count * 0.75),
                ),
            )

            dynamic_min = max(
                120,
                min(
                    220,
                    int(dynamic_max * 0.55),
                ),
            )

            # ---------------------------------------------
            # Recursive synthesis
            # ---------------------------------------------

            synthesized = (
                generate_abstractive_summary(

                    text=combined,

                    source_language=source_language,

                    model_name=model_name,

                    document_type=document_type,

                    synthesis_mode=True,

                    max_new_tokens=dynamic_max,

                    min_new_tokens=dynamic_min,
                )
            )

            synthesized = (
                synthesized or ""
            ).strip()

            if synthesized:

                next_level.append(
                    synthesized
                )

        # -------------------------------------------------
        # Safety fallback
        # -------------------------------------------------

        if not next_level:

            break

        # -------------------------------------------------
        # Prevent infinite recursion
        # -------------------------------------------------

        if len(next_level) == len(current_level):

            break

        current_level = next_level

        recursion_depth += 1

    # -----------------------------------------------------
    # Final output
    # -----------------------------------------------------

    # -----------------------------------------------------
    # Final merge
    # -----------------------------------------------------

    final_summary = " ".join(
        current_level
    ).strip()

    # Remove repeated whitespace
    final_summary = re.sub(
        r"\s+",
        " ",
        final_summary,
    )

    # Remove repeated sentences
    seen = set()

    deduped = []

    for sentence in split_sentences(final_summary):

        normalized = (
            sentence.lower().strip()
        )

        if normalized in seen:
            continue

        seen.add(normalized)

        deduped.append(sentence)

    final_summary = " ".join(deduped)

    # -----------------------------------------------------
    # Semantic redundancy cleanup
    # -----------------------------------------------------

    semantic_sentences = split_sentences(
        final_summary
    )

    semantic_sentences = [

        s.strip()

        for s in semantic_sentences

        if len(s.split()) >= 6
    ]

    if semantic_sentences:

        embedding_model = get_embedding_model()

        filtered_sentences = []

        stored_embeddings = []

        from sklearn.metrics.pairwise import cosine_similarity

        batch_embeddings = embedding_model.encode(
            semantic_sentences,
            normalize_embeddings=True,
            batch_size=32,
            show_progress_bar=False,
        )

        for idx, sentence in enumerate(
            semantic_sentences
        ):

            current_embedding = (
                batch_embeddings[idx]
            )

            is_duplicate = False

            for previous_embedding in stored_embeddings:

                similarity = cosine_similarity(
                    [current_embedding],
                    [previous_embedding],
                )[0][0]

                if similarity > 0.90:

                    is_duplicate = True
                    break

            if not is_duplicate:

                filtered_sentences.append(
                    sentence
                )

                stored_embeddings.append(
                    current_embedding
                )

        final_summary = " ".join(
            filtered_sentences
        )

        # -----------------------------------------------------
        # Academic cleanup
        # -----------------------------------------------------

        cleanup_patterns = [

            r"Figure\s+\d+[\.\d]*",

            r"Table\s+\d+[\.\d]*",

            r"Equation\s+\d+[\.\d]*",

            r"\[\d+\]",

            r"\bGeneral\b",

            r"CHAPTER\s+\d+",

            r"Page\s+\d+",

        ]

        for pattern in cleanup_patterns:

            final_summary = re.sub(
                pattern,
                "",
                final_summary,
                flags=re.IGNORECASE,
            )

        final_summary = re.sub(
            r"\s+",
            " ",
            final_summary,
        ).strip()

        print(
            f"[FINAL SUMMARY LENGTH] "
            f"{len(final_summary.split())} words"
        )

        return final_summary

# ---------------------------------------------------------
# Metrics
# ---------------------------------------------------------

def compute_basic_metrics(
    original_text: str,
    summary_text: str,
) -> Dict[str, Any]:

    original_words = len(
        original_text.split()
    )

    summary_words = len(
        summary_text.split()
    )

    compression = round(
        1 - (
            summary_words
            / max(original_words, 1)
        ),
        3,
    )

    return {

        "original_word_count":
            original_words,

        "summary_word_count":
            summary_words,

        "compression_ratio":
            compression,
    }


def enforce_target_compression(
    summary_text: str,
    original_text: str,
    target_ratio: float,
    min_sentences: int = 3,
) -> str:
    """
    Keep final summary close to requested retention ratio.
    target_ratio = fraction of original words to keep.
    """

    summary_text = (summary_text or "").strip()
    original_text = (original_text or "").strip()

    if not summary_text or not original_text:
        return summary_text

    target_ratio = max(0.05, min(0.9, float(target_ratio)))
    original_words = original_text.split()
    target_words = max(40, int(len(original_words) * target_ratio))

    current_words = summary_text.split()
    if len(current_words) <= target_words:
        return summary_text

    sentences = split_sentences(summary_text)
    if not sentences:
        return " ".join(current_words[:target_words]).strip()

    kept: List[str] = []
    running_words = 0
    for sentence in sentences:
        sent_words = len(sentence.split())
        if kept and running_words + sent_words > target_words:
            break
        kept.append(sentence)
        running_words += sent_words

    if len(kept) < min_sentences:
        kept = sentences[:min(min_sentences, len(sentences))]

    return " ".join(kept).strip()


def is_legal_like_document(text: str) -> bool:
    text_lower = (text or "").lower()
    hits = sum(hint in text_lower for hint in LEGAL_HINTS)
    return hits >= 3

# ---------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------

def summarize_document(
    file_path: str,
    summarization_mode: str = "hybrid",
    target_language: str = "en",
    compression_ratio: float = 0.40,
    max_sentences: int = 12,
    abstractive_model: Optional[str] = None,
    ocr_language: str = "eng",
    debug: bool = False,
    **kwargs,
) -> Dict[str, Any]:

    start_time = time.time()

    try:

        # -----------------------------------------------------
        # Load document
        # -----------------------------------------------------

        document_text = (
            load_document_text(
                file_path=file_path,
                ocr_language=ocr_language,
            )
        )

        if not document_text.strip():

            return build_error_response(
                "No readable text found.",
                "empty_document",
            )

        # -----------------------------------------------------
        # Early document classification
        # -----------------------------------------------------

        document_type_preview = classify_document(
            document_text
        )

        # -----------------------------------------------------
        # Resume-safe preprocessing
        # -----------------------------------------------------

        if document_type_preview == "resume":

            processed_text = (
                document_text
                .replace("\x00", " ")
                .strip()
            )

        else:

            processed_sentences = preprocess_text(
                document_text,
                preserve_paragraphs=True,
                deduplicate=False,
            )

            processed_text = "\n".join(
                processed_sentences
            ).strip()

        if not processed_text:

            return build_error_response(
                "Document empty after preprocessing.",
                "preprocessing_error",
            )

        # -----------------------------------------------------
        # Classification
        # -----------------------------------------------------

        document_type = document_type_preview

        source_language = detect_language(
            processed_text
        )
        if not source_language:
            source_language = "unknown"
        source_language = source_language.lower()

        original_source_language = source_language
        pivoted_to_english = False

        # -----------------------------------------------------
        # Quality-first language pivot:
        # non-English docs -> English -> summarize -> translate out
        # -----------------------------------------------------

        if source_language not in {"en", "english", "unknown"}:
            pivot_result = translate_document_text(
                text=processed_text,
                source_language=source_language,
                target_language="en",
            )
            pivot_text = (
                pivot_result.get("translated_text", "")
                if isinstance(pivot_result, dict)
                else ""
            )
            if pivot_text and pivot_text.strip():
                processed_text = pivot_text.strip()
                source_language = "en"
                pivoted_to_english = True

        legal_like = (
            document_type == "general"
            and is_legal_like_document(processed_text)
        )

        # -----------------------------------------------------
        # Resume optimization
        # -----------------------------------------------------

        # Respect user-provided max_sentences for all document types.

        # -----------------------------------------------------
        # Smart routing
        # -----------------------------------------------------

        summarization_mode = (
            select_summarization_mode(
                document_type=document_type,
                user_mode=summarization_mode,
            )
        )

        # Legal contracts are better served by factual extractive summaries.
        if legal_like and summarization_mode != "extractive":
            summarization_mode = "extractive"

        compression_ratio = (
            get_compression_ratio(
                document_type=document_type,
                user_ratio=compression_ratio,
            )
        )

        model_name = (
            select_document_model(
                document_type=document_type,
                language=source_language,
                override_model=abstractive_model,
            )
        )

        print("\n" + "=" * 80)
        print("[RAW DOCUMENT TEXT]")
        print("=" * 80)

        print(document_text[:5000])

        print("\n")

        # -----------------------------------------------------
        # Detect + rank sections
        # -----------------------------------------------------

        sections = detect_sections(
            processed_text
        )

        ranked_sections = rank_sections(
            sections
        )

        # -----------------------------------------------------
        # Keep ALL meaningful sections
        # -----------------------------------------------------

        filtered_sections = []

        for heading, content, priority in ranked_sections:

            content = (content or "").strip()

            # Skip tiny/noisy sections
            if len(content.split()) < 80:
                continue

            # Skip references
            heading_lower = heading.lower()

            if any(
                bad in heading_lower
                for bad in [
                    "reference",
                    "bibliography",
                    "citation",
                ]
            ):
                continue

            filtered_sections.append(
                (heading, content, priority)
            )

        chunks = []

        for heading, content, priority in filtered_sections:

            section_chunks = create_safe_chunks(

                text=content,

                language=source_language,

                model_name=model_name,

                document_type=document_type,
            )

            # -------------------------------------------------
            # Preserve heading context
            # -------------------------------------------------

            section_chunks = [

                f"{heading}\n\n{chunk}"

                for chunk in section_chunks
            ]

            chunks.extend(section_chunks)

        if not chunks:

            return build_error_response(
                "Chunking failed.",
                "chunking_error",
            )

        print("=" * 80)
        print("[CHUNKS]")
        print("=" * 80)

        print(f"\n[INFO] Total chunks: {len(chunks)}")

        for idx, chunk in enumerate(chunks, start=1):

            print(f"\n--- CHUNK {idx} ---\n")

            print(chunk[:2000])

        # -----------------------------------------------------
        # Chunk summaries
        # -----------------------------------------------------

        chunk_summaries = summarize_chunks(
            chunks=chunks,
            summarization_mode=summarization_mode,
            source_language=source_language,
            compression_ratio=compression_ratio,
            model_name=model_name,
            document_type=document_type,
        )

        if not chunk_summaries:

            return build_error_response(
                "Summarization failed.",
                "summarization_error",
            )

        # -----------------------------------------------------
        # Cleanup
        # -----------------------------------------------------

        cleaned_chunks = []

        for chunk_summary in chunk_summaries:

            chunk_summary = (
                chunk_summary or ""
            ).strip()

            if not chunk_summary:
                continue

            chunk_summary = " ".join(
                chunk_summary.split()
            )

            cleaned_chunks.append(
                chunk_summary
            )

        chunk_summaries = cleaned_chunks

        # -----------------------------------------------------
        # Final synthesis
        # -----------------------------------------------------

        if summarization_mode == "extractive":

            summary_en = "\n\n".join(
                chunk_summaries
            )

        else:

            summary_en = (
                synthesize_final_summary(
                    chunk_summaries=chunk_summaries,
                    source_language=source_language,
                    model_name=model_name,
                    document_type=document_type,
                )
            )

        summary_en = (
            summary_en or ""
        ).strip()

        # -----------------------------------------------------
        # Remove duplicate sentences
        # -----------------------------------------------------

        deduped_sentences = []

        seen = set()

        for sentence in split_sentences(summary_en):

            normalized = (
                sentence.lower()
                .strip()
            )

            if normalized in seen:
                continue

            seen.add(normalized)

            deduped_sentences.append(
                sentence
            )

        summary_en = " ".join(
            deduped_sentences
        )
        
        summary_en = enforce_target_compression(
            summary_text=summary_en,
            original_text=processed_text,
            target_ratio=compression_ratio,
            min_sentences=3,
        )

        # -----------------------------------------------------
        # Hallucination filtering
        # -----------------------------------------------------

        if (
            document_type != "resume"
            and summarization_mode != "extractive"
        ):

            filtered_summary = (
                remove_hallucinated_sentences(
                    generated_text=summary_en,
                    source_text=processed_text,
                )
            )

            if filtered_summary.strip():

                summary_en = filtered_summary

        # -----------------------------------------------------
        # Extractive truncation
        # -----------------------------------------------------

        if (
            summarization_mode
            == "extractive"
        ):

            sentences = split_sentences(
                summary_en
            )

            if len(sentences) > max_sentences:

                summary_en = "\n".join(
                    sentences[:max_sentences]
                )

        # -----------------------------------------------------
        # Sentiment
        # -----------------------------------------------------

        if document_type == "resume":

            sentiment = {
                "positive": 0.05,
                "neutral": 0.90,
                "negative": 0.05,
            }

        else:

            sentiment = (
                analyze_document_sentiment(
                    summary_en
                )
            )

        # -----------------------------------------------------
        # Metrics
        # -----------------------------------------------------

        metrics = compute_metrics(
            original_text=processed_text,
            summary_text=summary_en,
            reference_summary=None,
            lang=source_language,
        )

        original_words = max(1, len(processed_text.split()))
        summary_words = len(summary_en.split())
        retention_ratio = round(summary_words / original_words, 4)
        reduction_ratio = round(1 - retention_ratio, 4)
        metrics["Retention Ratio"] = retention_ratio
        metrics["Compression Reduction"] = reduction_ratio

        # -----------------------------------------------------
        # Markdown formatting
        # -----------------------------------------------------

        formatted_summary = (
            format_summary_markdown(
                summary_en,
                document_type=document_type,
            )
        )

        contract_insights = {}
        if legal_like:
            contract_insights = build_contract_insights(
                processed_text
            )

        # -----------------------------------------------------
        # Translation
        # -----------------------------------------------------

        summary_translated = None

        if target_language:
            target_language = target_language.lower()

        # If we pivoted to English, translate summary to requested output
        # language (including back to original source when needed).
        if (
            target_language
            and target_language != "en"
        ):

            translation_result = (
                translate_document_text(
                    text=summary_en,
                    source_language="en",
                    target_language=target_language,
                )
            )

            summary_translated = (
                translation_result.get(
                    "translated_text"
                )
            )

        # -----------------------------------------------------
        # Final response
        # -----------------------------------------------------

        processing_time = round(
            time.time() - start_time,
            3,
        )

        response = build_summary_response(

            summary_markdown=formatted_summary,

            sections=[],

            overall_sentiment=sentiment,

            metadata={

                "document_type":
                    document_type,

                "model_used":
                    model_name,

                "chunk_count":
                    len(chunks),

                "processing_time_seconds":
                    processing_time,

                "summarization_mode":
                    summarization_mode,

                "target_compression_ratio":
                    round(float(compression_ratio), 4),

                "actual_retention_ratio":
                    retention_ratio,

                "actual_reduction_ratio":
                    reduction_ratio,

                "source_language_detected":
                    original_source_language,

                "pivoted_to_english":
                    pivoted_to_english,
            },
        )

        response["summary_en"] = (
            summary_en
        )

        response[
            "summary_translated"
        ] = summary_translated

        response["metrics"] = metrics

        response["document_text"] = (
            processed_text
        )

        response["document_type"] = (
            document_type
        )

        response["contract_insights"] = (
            contract_insights
        )

        return response

    except Exception as error:

        return build_error_response(
            message=str(error),
            error_type=(
                "summarization_pipeline_error"
            ),
        )

gc.collect()

if torch.cuda.is_available():
    torch.cuda.empty_cache()
