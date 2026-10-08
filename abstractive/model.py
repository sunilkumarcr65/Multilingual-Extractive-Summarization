from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
)


@dataclass
class GenerationConfig:
    max_new_tokens: int = 180
    min_length: int = 50

    num_beams: int = 6
    length_penalty: float = 1.5
    repetition_penalty: float = 1.5
    no_repeat_ngram_size: int = 3

    early_stopping: bool = True

    do_sample: bool = True
    temperature: float = 0.7
    top_k: int = 50
    top_p: float = 0.95


class SummarizationModel:
    """
    Handles:
    - tokenizer loading
    - model loading
    - text encoding
    - summary generation

    Recommended models:
    - google/flan-t5-small
    - csebuetnlp/mT5_multilingual_XLSum
    - google/mt5-base
    - facebook/mbart-large-50-many-to-many-mmt
    """

    MODEL_MAX_TOKENS = {
        "longt5": 3500,
        "bigbird": 3500,
        "pegasus": 1024,
        "t5": 1024,
        "flan": 1024,
        "mt5": 1024,
        "bart": 1024,
        "mbart": 1024,
    }

    DEFAULT_MAX_TOKENS = 1024

    def __init__(
        self,
        model_name: str = "google/flan-t5-small",
        max_input_tokens: Optional[int] = None,
    ):
        self.model_name = model_name

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        print(f"[MODEL] Using device: {self.device}")
        print(f"[MODEL] Loading: {model_name}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name
        ).to(self.device)

        self.model.eval()

        lower_name = model_name.lower()

        self.is_t5_family = any(
            x in lower_name
            for x in ("t5", "flan", "mt5")
        )

        self.max_input_tokens = (
            max_input_tokens
            or next(
                (
                    v
                    for k, v in self.MODEL_MAX_TOKENS.items()
                    if k in lower_name
                ),
                self.DEFAULT_MAX_TOKENS,
            )
        )

    # ---------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------

    def _prefix(self, text: str) -> str:
        """
        T5-family models require task prefixes.
        """
        if self.is_t5_family:
            return f"summarize: {text}"

        return text

    # ---------------------------------------------------------
    # Public API
    # ---------------------------------------------------------

    def token_length(self, text: str) -> int:
        return len(
            self.tokenizer.encode(
                text,
                add_special_tokens=False
            )
        )

    def encode(self, text: str) -> dict:
        """
        Tokenize and prepare tensors.
        """

        encoded = self.tokenizer(
            self._prefix(text),
            return_tensors="pt",
            truncation=True,
            max_length=self.max_input_tokens,
        )

        return encoded.to(self.device)

    @torch.inference_mode()
    def generate_summary(
        self,
        text: str,
        config: Optional[GenerationConfig] = None,
    ) -> str:
        """
        Generate summary from input text.
        """

        if not text.strip():
            return ""

        config = config or GenerationConfig()

        inputs = self.encode(text)

        gen_kwargs = {
            "max_new_tokens": config.max_new_tokens,
            "min_length": config.min_length,
            "num_beams": config.num_beams,
            "length_penalty": config.length_penalty,
            "repetition_penalty": config.repetition_penalty,
            "no_repeat_ngram_size": config.no_repeat_ngram_size,
            "early_stopping": config.early_stopping,
            "do_sample": config.do_sample,
        }

        if config.do_sample:
            gen_kwargs.update(
                {
                    "temperature": config.temperature,
                    "top_k": config.top_k,
                    "top_p": config.top_p,
                }
            )

        generated_ids = self.model.generate(
            **inputs,
            **gen_kwargs,
        )

        summary = self.tokenizer.decode(
            generated_ids[0],
            skip_special_tokens=True,
        )

        return summary.strip()
