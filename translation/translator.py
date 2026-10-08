from __future__ import annotations

from functools import lru_cache
import platform
from typing import Optional

import torch
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
)

from translation.batching import (
    MAX_MODEL_TOKENS,
    build_translation_batches,
)

from preprocessing.language_detection import (
    detect_language,
)

from translation.language_map import (
    get_language_code,
    is_supported_language,
)

from translation.model import (
    DEVICE,
)

from translation.utils import (
    flatten_paragraphs,
    preserve_paragraph_structure,
    split_paragraphs,
)


# ---------------------------------------------------------
# Translation
# ---------------------------------------------------------

PAIR_MODEL_MAP = {
    ("en", "hi"): "Helsinki-NLP/opus-mt-en-hi",
    ("hi", "en"): "Helsinki-NLP/opus-mt-hi-en",
    ("en", "fr"): "Helsinki-NLP/opus-mt-en-fr",
    ("fr", "en"): "Helsinki-NLP/opus-mt-fr-en",
    ("en", "de"): "Helsinki-NLP/opus-mt-en-de",
    ("de", "en"): "Helsinki-NLP/opus-mt-de-en",
    ("en", "es"): "Helsinki-NLP/opus-mt-en-es",
    ("es", "en"): "Helsinki-NLP/opus-mt-es-en",
    ("en", "pt"): "Helsinki-NLP/opus-mt-en-pt",
    ("pt", "en"): "Helsinki-NLP/opus-mt-pt-en",
    ("en", "ru"): "Helsinki-NLP/opus-mt-en-ru",
    ("ru", "en"): "Helsinki-NLP/opus-mt-ru-en",
    ("en", "zh"): "Helsinki-NLP/opus-mt-en-zh",
    ("zh", "en"): "Helsinki-NLP/opus-mt-zh-en",
    ("en", "ar"): "Helsinki-NLP/opus-mt-en-ar",
    ("ar", "en"): "Helsinki-NLP/opus-mt-ar-en",
    ("en", "bn"): "Helsinki-NLP/opus-mt-en-bn",
    ("bn", "en"): "Helsinki-NLP/opus-mt-bn-en",
}

NLLB_MODEL = "facebook/nllb-200-distilled-600M"
ENABLE_NLLB_FALLBACK = False

NLLB_LANGUAGE_MAP = {
    "en": "eng_Latn",
    "hi": "hin_Deva",
    "es": "spa_Latn",
    "fr": "fra_Latn",
    "de": "deu_Latn",
    "pt": "por_Latn",
    "ar": "arb_Arab",
    "zh": "zho_Hans",
    "ru": "rus_Cyrl",
    "bn": "ben_Beng",
    "mr": "mar_Deva",
    "ta": "tam_Taml",
    "te": "tel_Telu",
}


@lru_cache(maxsize=16)
def load_pair_translation_model(model_name: str):
    print(f"[TRANSLATION] Loading pair model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(DEVICE)
    model.eval()
    return tokenizer, model


@lru_cache(maxsize=1)
def load_nllb_model():
    print(f"[TRANSLATION] Loading multilingual fallback: {NLLB_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(NLLB_MODEL, use_fast=False)
    model = AutoModelForSeq2SeqLM.from_pretrained(NLLB_MODEL).to(DEVICE)
    model.eval()
    return tokenizer, model


def translate_with_pair_model(
    text: str,
    source_language: str,
    target_language: str,
    batch_size: int = 8,
) -> str:
    model_name = PAIR_MODEL_MAP.get((source_language, target_language))
    if not model_name:
        return ""

    tokenizer, model = load_pair_translation_model(model_name)

    paragraphs = split_paragraphs(text)
    if not paragraphs:
        return ""

    sentences, sentences_per_paragraph = flatten_paragraphs(paragraphs)
    if not sentences:
        return ""

    translated_sentences = []
    for i in range(0, len(sentences), batch_size):
        batch = sentences[i:i + batch_size]
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_MODEL_TOKENS,
        ).to(DEVICE)
        generated = model.generate(
            **inputs,
            num_beams=4,
            early_stopping=True,
            no_repeat_ngram_size=3,
            length_penalty=1.0,
        )
        decoded = [
            tokenizer.decode(ids, skip_special_tokens=True).strip()
            for ids in generated
        ]
        translated_sentences.extend(decoded)

    return preserve_paragraph_structure(
        translated_sentences=translated_sentences,
        sentences_per_paragraph=sentences_per_paragraph,
    ).strip()


def translate_with_pivot_english(
    text: str,
    source_language: str,
    target_language: str,
    batch_size: int = 8,
) -> str:
    if source_language == "en" or target_language == "en":
        return ""

    step1 = translate_with_pair_model(
        text=text,
        source_language=source_language,
        target_language="en",
        batch_size=batch_size,
    )
    if not step1:
        return ""

    step2 = translate_with_pair_model(
        text=step1,
        source_language="en",
        target_language=target_language,
        batch_size=batch_size,
    )
    return step2 or ""


def translate_with_nllb(
    text: str,
    source_language: str,
    target_language: str,
    batch_size: int = 8,
) -> str:
    src_code = NLLB_LANGUAGE_MAP.get(source_language)
    tgt_code = NLLB_LANGUAGE_MAP.get(target_language)
    if not src_code or not tgt_code:
        return ""

    # NLLB can be unstable on some CPU/Windows environments.
    # Keep this path opt-in unless explicitly enabled.
    if not ENABLE_NLLB_FALLBACK:
        return ""

    if DEVICE == "cpu" and platform.system().lower() == "windows":
        return ""

    tokenizer, model = load_nllb_model()

    paragraphs = split_paragraphs(text)
    if not paragraphs:
        return ""

    sentences, sentences_per_paragraph = flatten_paragraphs(paragraphs)
    if not sentences:
        return ""

    translated_sentences = []
    for i in range(0, len(sentences), batch_size):
        batch = sentences[i:i + batch_size]
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_MODEL_TOKENS,
        ).to(DEVICE)
        forced_bos = tokenizer.convert_tokens_to_ids(tgt_code)
        generated = model.generate(
            **inputs,
            forced_bos_token_id=forced_bos,
            num_beams=4,
            early_stopping=True,
            no_repeat_ngram_size=3,
            length_penalty=1.0,
        )
        decoded = [
            tokenizer.decode(ids, skip_special_tokens=True).strip()
            for ids in generated
        ]
        translated_sentences.extend(decoded)

    return preserve_paragraph_structure(
        translated_sentences=translated_sentences,
        sentences_per_paragraph=sentences_per_paragraph,
    ).strip()


@torch.inference_mode()
def translate_text(
    text: str,
    source_language: Optional[str] = None,
    target_language: str = "en",
    model_name: Optional[str] = None,
    batch_size: int = 8,
) -> str:
    """
    Multilingual translation.

    Features
    -----------------------------------------
    - automatic language detection
    - token-aware batching
    - multilingual translation
    - paragraph preservation
    - GPU acceleration
    -----------------------------------------
    """

    text = (
        text or ""
    ).strip()

    if not text:

        return f"[Translation Failed]\n\n{text}"

    # ---------------------------------------------------------
    # Detect source language
    # ---------------------------------------------------------

    if source_language is None:

        source_language = (
            detect_language(text)
        )

    source_language = (
        source_language
        .strip()
        .lower()
    )

    target_language = (
        target_language
        .strip()
        .lower()
    )

    # ---------------------------------------------------------
    # No-op
    # ---------------------------------------------------------

    if (
        source_language
        == target_language
    ):
        return text

    # ---------------------------------------------------------
    # Support validation
    # ---------------------------------------------------------

    if not is_supported_language(
        source_language
    ):

        raise ValueError(
            f"Unsupported source language: "
            f"{source_language}"
        )

    if not is_supported_language(
        target_language
    ):

        raise ValueError(
            f"Unsupported target language: "
            f"{target_language}"
        )


    source_code = (
        get_language_code(
            source_language
        )
    )

    target_code = (
        get_language_code(
            target_language
        )
    )

    # ---------------------------------------------------------
    # Preferred lightweight pair models (more stable on CPU)
    # ---------------------------------------------------------

    try:
        translated = translate_with_pair_model(
            text=text,
            source_language=source_language,
            target_language=target_language,
            batch_size=batch_size,
        )
        if translated:
            return translated
    except Exception as error:
        print(
            "[TRANSLATION] "
            f"Pair model failed: {error}"
        )

    # ---------------------------------------------------------
    # Second choice: pivot translation through English
    # ---------------------------------------------------------

    try:
        translated = translate_with_pivot_english(
            text=text,
            source_language=source_language,
            target_language=target_language,
            batch_size=batch_size,
        )
        if translated:
            return translated
    except Exception as error:
        print(
            "[TRANSLATION] "
            f"Pivot translation failed: {error}"
        )

    # ---------------------------------------------------------
    # Third choice: multilingual NLLB fallback (opt-in)
    # ---------------------------------------------------------

    try:
        translated = translate_with_nllb(
            text=text,
            source_language=source_language,
            target_language=target_language,
            batch_size=batch_size,
        )
        if translated:
            return translated
    except Exception as error:
        print(
            "[TRANSLATION] "
            f"NLLB fallback failed: {error}"
        )

    # ---------------------------------------------------------
    # Preserve paragraphs
    # ---------------------------------------------------------

    paragraphs = split_paragraphs(
        text
    )

    if not paragraphs:

        return f"[Translation Failed]\n\n{text}"

    (
        sentences,
        sentences_per_paragraph,
    ) = flatten_paragraphs(
        paragraphs
    )

    if not sentences:

        return f"[Translation Failed]\n\n{text}"

    # ---------------------------------------------------------
    # Build batches
    # ---------------------------------------------------------

    # ---------------------------------------------------------
    # Last-resort fallback to no-op translation to avoid crash.
    # ---------------------------------------------------------

    print(
        "[TRANSLATION] No stable model path for "
        f"{source_language}->{target_language}. Returning original text."
    )
    return text


# ---------------------------------------------------------
# Convenience Wrapper
# ---------------------------------------------------------

def translate_to_english(
    text: str,
    source_language: Optional[str] = None,
) -> str:
    """
    Translate any language to English.
    """

    return translate_text(
        text=text,
        source_language=(
            source_language
        ),
        target_language="en",
    )


# ---------------------------------------------------------
# Convenience Wrapper
# ---------------------------------------------------------

def translate_from_english(
    text: str,
    target_language: str,
) -> str:
    """
    Translate English to target language.
    """

    return translate_text(
        text=text,
        source_language="en",
        target_language=(
            target_language
        ),
    )
