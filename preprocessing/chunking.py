from __future__ import annotations

import re
from typing import List

from transformers import AutoTokenizer


# ---------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------

TOKENIZER = AutoTokenizer.from_pretrained(
    "google/flan-t5-base"
)


# ---------------------------------------------------------
# Config
# ---------------------------------------------------------

DEFAULT_CHUNK_SIZE = 320

DEFAULT_OVERLAP = 72

MAX_EMBEDDING_TOKENS = 380


# ---------------------------------------------------------
# Generic Section Headers
# ---------------------------------------------------------

SECTION_HEADERS = [

    "ABSTRACT",
    "INTRODUCTION",
    "CONCLUSION",
    "REFERENCES",

    "EDUCATION",
    "EXPERIENCE",
    "WORK EXPERIENCE",
    "PROJECTS",
    "SKILLS",
    "ACHIEVEMENTS",
    "CERTIFICATIONS",

    "CHAPTER",
    "OVERVIEW",
    "SUMMARY",
    "RESULTS",
]

BLOCKED_SECTIONS = [
    "references",
    "bibliography",
    "citation",
    "certificate",
    "declaration",
    "acknowledgement",
    "acknowledgments",
    "table of contents",
]


# ---------------------------------------------------------
# Real Token Counter
# ---------------------------------------------------------

def token_count(
    text: str,
) -> int:

    return len(
        TOKENIZER.encode(
            text,
            add_special_tokens=False,
        )
    )


# ---------------------------------------------------------
# Sentence Splitter
# ---------------------------------------------------------

def split_sentences(
    text: str,
) -> List[str]:

    text = re.sub(
        r"\s+",
        " ",
        text,
    ).strip()

    if not text:
        return []

    sentences = re.split(
        r"(?<=[.!?])\s+(?=[A-Z0-9])",
        text,
    )

    cleaned = []

    for sentence in sentences:

        sentence = sentence.strip()

        if len(sentence.split()) < 4:
            continue

        cleaned.append(sentence)

    return cleaned

# ---------------------------------------------------------
# Section Split
# ---------------------------------------------------------

def split_by_sections(
    text: str,
) -> List[str]:

    header_pattern = (
        r"^\s*(?:"
        + "|".join(
            re.escape(h)
            for h in SECTION_HEADERS
        )
        + r")"
    )

    lines = text.splitlines()

    sections = []

    current = []

    for raw_line in lines:

        line = raw_line.strip()
        
        normalized = line.lower()

        if any(
            blocked in normalized
            for blocked in BLOCKED_SECTIONS
        ):
            continue
        
        if not line:
            continue

        if re.match(
            header_pattern,
            line,
            flags=re.IGNORECASE,
        ):

            if current:

                sections.append(
                    "\n".join(current)
                )

            current = [line]

        else:

            current.append(line)

    if current:

        sections.append(
            "\n".join(current)
        )

    return sections


# ---------------------------------------------------------
# Token-aware Chunk Builder
# ---------------------------------------------------------

def build_chunks_from_sentences(
    sentences: List[str],
    chunk_size: int,
    overlap: int,
) -> List[str]:

    chunks = []

    current_chunk = []

    current_tokens = 0

    for sentence in sentences:

        sentence_tokens = token_count(
            sentence
        )

        # Skip pathological sentence
        if sentence_tokens > chunk_size:

            continue

        if (
            current_tokens
            + sentence_tokens
            > chunk_size
        ):

            chunk_text = " ".join(
                current_chunk
            )

            chunks.append(chunk_text)

            # overlap logic
            overlap_sentences = []

            overlap_tokens = 0

            for prev_sentence in reversed(
                current_chunk
            ):

                prev_tokens = token_count(
                    prev_sentence
                )

                if (
                    overlap_tokens
                    + prev_tokens
                    > overlap
                ):
                    break

                overlap_sentences.insert(
                    0,
                    prev_sentence,
                )

                overlap_tokens += prev_tokens

            current_chunk = overlap_sentences

            current_tokens = overlap_tokens

        current_chunk.append(sentence)

        current_tokens += sentence_tokens

    if current_chunk:

        chunks.append(
            " ".join(current_chunk)
        )

    return chunks


# ---------------------------------------------------------
# Main Chunker
# ---------------------------------------------------------

def chunk_text(
    text: str,
    language: str = "en",
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
) -> List[str]:

    text = (text or "").strip()

    if not text:
        return []
    
    document_tokens = token_count(
        text
    )

    # -----------------------------------------------------
    # Adaptive chunk sizing
    # -----------------------------------------------------

    if document_tokens > 12000:

        chunk_size = 280

    elif document_tokens > 6000:

        chunk_size = 320

    else:

        chunk_size = DEFAULT_CHUNK_SIZE

    sections = split_by_sections(
        text
    )

    final_chunks = []

    for section in sections:
        
        if token_count(section) < 40:
            continue
        
        sentences = split_sentences(
            section
        )

        section_chunks = (
            build_chunks_from_sentences(
                sentences=sentences,
                chunk_size=chunk_size,
                overlap=overlap,
            )
        )

        final_chunks.extend(
            section_chunks
        )

    # cleanup
    cleaned = []

    for chunk in final_chunks:

        chunk = chunk.strip()

        if (
            chunk
            and chunk.lower() not in {
                c.lower() for c in cleaned
            }
            and token_count(chunk)
            <= MAX_EMBEDDING_TOKENS
        ):

            cleaned.append(chunk)

    return cleaned