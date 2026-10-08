from __future__ import annotations

from functools import lru_cache
import re
from typing import Optional

import torch
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
)

from preprocessing.language_detection import (
    detect_language,
)


# ---------------------------------------------------------
# Device
# ---------------------------------------------------------

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ---------------------------------------------------------
# Model Config
# ---------------------------------------------------------

DEFAULT_GENERATION_MODEL = (
    "google/flan-t5-large"
)

MULTILINGUAL_GENERATION_MODEL = (
    "google/flan-t5-base"
)


# ---------------------------------------------------------
# Prompt Templates
# ---------------------------------------------------------

QA_PROMPT_TEMPLATE = """
You are a document question-answering assistant.

Use ONLY the provided context.

Rules:
- Do not invent information
- If information is missing, say so
- Quote exact facts when possible
- Prefer concise factual answers
- Answer in plain text only
- Do not output placeholders like <extra_id_0>

Context:
{context}

Question:
{question}

Answer:
"""


# ---------------------------------------------------------
# Model Loader
# ---------------------------------------------------------

@lru_cache(maxsize=2)
def load_generation_model(
    multilingual: bool = True,
):
    """
    Load QA generation model.
    """

    model_name = (
        MULTILINGUAL_GENERATION_MODEL
        if multilingual
        else DEFAULT_GENERATION_MODEL
    )

    print(
        f"[GENERATOR] "
        f"Loading model: {model_name}"
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_name
    )

    model = (
        AutoModelForSeq2SeqLM
        .from_pretrained(model_name)
        .to(DEVICE)
    )

    model.eval()

    return tokenizer, model


# ---------------------------------------------------------
# Generator
# ---------------------------------------------------------

class RAGGenerator:
    """
    Grounded answer generator.

    Uses:
    -----------------------------------------
    retrieval context
    +
    instruction-tuned generation
    -----------------------------------------

    Supports:
    - multilingual QA
    - grounded generation
    - hallucination reduction
    """

    def __init__(
        self,
        multilingual: bool = True,
        max_input_tokens: int = 1024,
        max_new_tokens: int = 256,
    ):
        self.multilingual = (
            multilingual
        )

        self.max_input_tokens = (
            max_input_tokens
        )

        self.max_new_tokens = (
            max_new_tokens
        )

        (
            self.tokenizer,
            self.model,
        ) = load_generation_model(
            multilingual=multilingual
        )

    # ---------------------------------------------------------
    # Prompt Builder
    # ---------------------------------------------------------

    def build_prompt(
        self,
        question: str,
        context: str,
    ) -> str:
        """
        Construct grounded QA prompt.
        """

        return (
            QA_PROMPT_TEMPLATE
            .strip()
            .format(
                context=context.strip(),
                question=question.strip(),
            )
        )

    # ---------------------------------------------------------
    # Context Trimming
    # ---------------------------------------------------------

    def trim_context(
        self,
        prompt: str,
    ) -> str:
        """
        Trim prompt to model-safe length.
        """

        tokenized = (
            self.tokenizer.encode(
                prompt,
                truncation=True,
                max_length=(
                    self.max_input_tokens
                ),
                return_tensors="pt",
            )
        )

        trimmed = (
            self.tokenizer.decode(
                tokenized[0],
                skip_special_tokens=True,
            )
        )

        return trimmed

    # ---------------------------------------------------------
    # Main Generation
    # ---------------------------------------------------------

    def generate_answer(
        self,
        question: str,
        context: str,
        language: Optional[str] = None,
    ) -> str:
        """
        Generate grounded answer.
        """

        question = (
            question or ""
        ).strip()

        context = (
            context or ""
        ).strip()

        if not question:

            return (
                "Please provide a question."
            )

        if not context:

            return (
                "I could not find "
                "relevant information "
                "in the document."
            )

        # -----------------------------------------------------
        # Auto language detection
        # -----------------------------------------------------

        if language is None:

            language = detect_language(
                question
            )

        # -----------------------------------------------------
        # Prompt
        # -----------------------------------------------------

        prompt = self.build_prompt(
            question=question,
            context=context,
        )

        prompt = self.trim_context(
            prompt
        )

        # -----------------------------------------------------
        # Tokenization
        # -----------------------------------------------------

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=(
                self.max_input_tokens
            ),
        ).to(DEVICE)

        # -----------------------------------------------------
        # Generation
        # -----------------------------------------------------

        with torch.no_grad():

            outputs = self.model.generate(
                **inputs,

                max_new_tokens=(
                    self.max_new_tokens
                ),

                num_beams=4,

                early_stopping=True,

                no_repeat_ngram_size=3,

                repetition_penalty=1.2,

                length_penalty=1.0,
            )

        answer = (
            self.tokenizer.decode(
                outputs[0],
                skip_special_tokens=True,
            )
            .strip()
        )

        if self._is_unusable_answer(answer):
            fallback = self._fallback_from_context(
                question=question,
                context=context,
            )
            if fallback:
                return fallback

        # -----------------------------------------------------
        # Empty fallback
        # -----------------------------------------------------

        if not answer:

            return (
                "I could not generate "
                "an answer from the "
                "document."
            )

        return answer

    @staticmethod
    def _is_unusable_answer(answer: str) -> bool:
        text = (answer or "").strip().lower()
        if not text:
            return True
        if "<extra_id_" in text:
            return True
        if "read more" in text:
            return True
        if len(text.split()) < 2:
            return True
        return False

    @staticmethod
    def _fallback_from_context(
        question: str,
        context: str,
    ) -> str:
        q = (question or "").lower()
        ctx = (context or "").strip()
        if not ctx:
            return ""

        lines = [
            line.strip()
            for line in ctx.splitlines()
            if line.strip()
            and not line.strip().startswith("[Chunk")
        ]

        if not lines:
            return ""

        context_text = " ".join(lines)
        sentences = [
            s.strip()
            for s in re.split(r"(?<=[.!?])\s+", context_text)
            if s.strip()
        ]
        if not sentences:
            return lines[0][:220].strip()

        query_tokens = {
            t for t in re.findall(r"[a-z0-9]+", q)
            if len(t) > 2
            and t
            not in {
                "what", "which", "where", "when", "who",
                "whom", "whose", "name", "about", "from",
                "this", "that", "does", "have", "with",
                "candidate", "resume", "document",
            }
        }

        best_sentence = sentences[0]
        best_score = -1.0
        for sentence in sentences:
            s_tokens = set(re.findall(r"[a-z0-9]+", sentence.lower()))
            if not s_tokens:
                continue
            overlap = len(query_tokens & s_tokens)
            density = overlap / max(len(s_tokens), 1)
            score = overlap + density
            if score > best_score:
                best_score = score
                best_sentence = sentence

        # If query tokens are very sparse (e.g. "name of candidate"),
        # use the first context line as a factual fallback.
        if best_score <= 0.0:
            head = lines[0]
            head = re.split(r"\||\d{7,}|@", head)[0].strip()
            match = re.search(r"[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+", head)
            if match:
                return match.group(0)

        return best_sentence[:260].strip()
