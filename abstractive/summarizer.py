from __future__ import annotations

import re
from typing import List, Optional

from .model import (
    GenerationConfig
)

from .postprocessing import (
    SummaryPostProcessor,
)

from utils.text.cleaning import (
    clean_text,
    remove_ocr_artifacts,
    repair_resume_spacing,
)

from peft import PeftModel

from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
)

import torch

# ---------------------------------------------------------
# Resume Hallucination Guard
# ---------------------------------------------------------

RESUME_SYSTEM_PROMPT = """
Summarize this resume into a concise professional candidate profile.

Focus on:
- technical expertise
- engineering impact
- software development strengths
- measurable achievements
- projects
- work experience

Write naturally and professionally.
"""
PROMPT_LEAK_PATTERNS = [
    r"(?i)\bwrite a resume summary\b.*?(?=[\.\n]|$)",
    r"(?i)\bresume content\s*:?",
    r"(?i)\bprofessional resume summary\s*:?",
    r"(?i)\bfocus on\s*:?\s*education.*?achievements\.?",
]


# ---------------------------------------------------------
# Summarizer
# ---------------------------------------------------------

class AbstractiveSummarizer:
    """
    Safe abstractive summarizer.
    """

    def __init__(
        self,
        model_name: str = (
            "google/flan-t5-base"
        ),
        max_input_tokens: int = 480,
    ):

        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        self.max_input_tokens = (
            max_input_tokens
        )

        print(
            f"[ABSTRACTIVE] "
            f"Loading summarizer: "
            f"{model_name}"
        )

        # -----------------------------------------------------
        # Tokenizer
        # -----------------------------------------------------

        print(
            f"[ABSTRACTIVE] "
            f"Loading tokenizer..."
        )

        self.tokenizer = (
            AutoTokenizer.from_pretrained(
                model_name
            )
        )

        # -----------------------------------------------------
        # Base Model
        # -----------------------------------------------------

        print(
            f"[ABSTRACTIVE] "
            f"Loading base model..."
        )

        base_model = (
            AutoModelForSeq2SeqLM.from_pretrained(
                model_name,

                torch_dtype=(
                    torch.float16
                    if torch.cuda.is_available()
                    else torch.float32
                ),
            )
        )

        # -----------------------------------------------------
        # Load NEW multidomain LoRA
        # -----------------------------------------------------

        print(
            f"[ABSTRACTIVE] "
            f"Loading LoRA adapters..."
        )

        self.model = (
            PeftModel.from_pretrained(
                base_model,
                "./models/flan_t5_base_multidomain_lora",
            )
        )

        # -----------------------------------------------------
        # Device
        # -----------------------------------------------------

        self.model.to(
            self.device
        )

        self.model.eval()

        # -----------------------------------------------------
        # Postprocessor
        # -----------------------------------------------------

        self.postprocessor = (
            SummaryPostProcessor()
        )

    # ---------------------------------------------------------
    # Generation Config
    # ---------------------------------------------------------

    def dynamic_generation_config(
        self,
        text: str,
        max_new_tokens: Optional[int] = None,
        min_new_tokens: Optional[int] = None,
        synthesis_mode: bool = False,
    ) -> GenerationConfig:

        token_count = len(
            self.tokenizer.encode(text)
        )

        # -----------------------------------------------------
        # Synthesis Mode
        # -----------------------------------------------------

        if synthesis_mode:

            dynamic_max = max(
                180,
                min(
                    320,
                    int(token_count * 0.40),
                ),
            )

            dynamic_min = max(
                80,
                min(
                    140,
                    dynamic_max // 3,
                ),
            )

            return GenerationConfig(

                max_new_tokens=(
                    max_new_tokens
                    or dynamic_max
                ),

                min_length=(
                    min_new_tokens
                    or dynamic_min
                ),

                num_beams=6,

                do_sample=False,

                early_stopping=True,

                repetition_penalty=1.2,

                no_repeat_ngram_size=4,

                length_penalty=1.15,
            )

        # -----------------------------------------------------
        # Standard
        # -----------------------------------------------------

        dynamic_max = max(
            80,
            min(
                220,
                int(token_count * 0.28),
            ),
        )

        dynamic_min = max(
            30,
            min(
                80,
                dynamic_max // 3,
            ),
        )

        return GenerationConfig(

            max_new_tokens=(
                max_new_tokens
                or dynamic_max
            ),

            min_length=(
                min_new_tokens
                or dynamic_min
            ),

            num_beams=4,

            do_sample=False,

            early_stopping=True,

            repetition_penalty=1.35,

            no_repeat_ngram_size=3,

            length_penalty=1.15,
        )

    # ---------------------------------------------------------
    # Resume Cleaning
    # ---------------------------------------------------------

    def clean_resume_text(
        self,
        text: str,
    ) -> str:
        """
        Aggressive resume cleanup
        before abstractive rewriting.
        """

        text = re.sub(
            r"https?://\S+",
            " ",
            text,
        )

        text = re.sub(
            r"\S+@\S+",
            " ",
            text,
        )

        text = re.sub(
            r"\+?\d[\d\s\-]{7,}",
            " ",
            text,
        )

        text = re.sub(
            r"\|\s*",
            ", ",
            text,
        )

        text = re.sub(
            r"\s+",
            " ",
            text,
        )

        return text.strip()

    # ---------------------------------------------------------
    # Build Prompt
    # ---------------------------------------------------------

    def build_prompt(
        self,
        text: str,
        document_type: Optional[str] = None,
        synthesis_mode: bool = False,
    ) -> str:

        # -----------------------------------------------------
        # Resume
        # -----------------------------------------------------

        if document_type == "resume":

            return f"""
    Summarize the following resume into a concise professional candidate profile.

    Focus on:
    - technical skills
    - engineering contributions
    - impactful projects
    - measurable achievements
    - software development experience

    Write naturally and professionally.

    Resume:
    {text}

    Summary:
    """

        # -----------------------------------------------------
        # Research Paper
        # -----------------------------------------------------

        if document_type == "research_paper":

            if synthesis_mode:

                return f"""
    Generate a coherent research paper summary.

    Focus on:
    - system architecture
    - methodology
    - algorithms
    - embeddings
    - multilingual processing
    - evaluation metrics
    - RAG pipeline
    - experimental findings

    Avoid references, certificates, acknowledgements, and repetitive text.

    Research Content:
    {text}

    Research Summary:
    """

            return f"""
    Summarize the following research document.

    Focus on:
    - core methodology
    - technical contributions
    - architecture
    - algorithms
    - evaluation
    - key findings

    Document:
    {text}

    Summary:
    """

        # -----------------------------------------------------
        # General Documents
        # -----------------------------------------------------

        return f"""
    Generate a concise factual summary of the following document.

    Focus on the most important information.

    Document:
    {text}

    Summary:
    """

    # ---------------------------------------------------------
    # Main Summarization
    # ---------------------------------------------------------

    def summarize(
        self,
        text: str,
        max_new_tokens: int = 180,
        min_new_tokens: int = 60,
        synthesis_mode: bool = False,
        document_type: Optional[str] = None,
    ) -> str:

        text = (
            text or ""
        ).strip()

        if not text:
            return ""

        # -----------------------------------------------------
        # Cleaning
        # -----------------------------------------------------

        text = clean_text(
            text,
            preserve_paragraphs=True,
        )

        text = remove_ocr_artifacts(
            text
        )

        text = repair_resume_spacing(
            text
        )

        # -----------------------------------------------------
        # Resume cleanup
        # -----------------------------------------------------

        if document_type == "resume":

            text = self.clean_resume_text(
                text
            )

        # -----------------------------------------------------
        # Prompt
        # -----------------------------------------------------

        prompt = self.build_prompt(
            text=text,
            document_type=document_type,
            synthesis_mode=synthesis_mode,
        )

        # -----------------------------------------------------
        # Config
        # -----------------------------------------------------

        config = (
            self.dynamic_generation_config(

                text=text,

                max_new_tokens=(
                    max_new_tokens
                ),

                min_new_tokens=(
                    min_new_tokens
                ),

                synthesis_mode=(
                    synthesis_mode
                ),
            )
        )

        # -----------------------------------------------------
        # Generate
        # -----------------------------------------------------

        # -----------------------------------------------------
        # Safe token budgeting
        # -----------------------------------------------------

        safe_max_tokens = 480

        encoded = self.tokenizer(
            prompt,
            truncation=True,
            max_length=safe_max_tokens,
            return_tensors="pt",
        )

        inputs = {
            key: value.to(self.device)
            for key, value in encoded.items()
        }

        generated = self.model.generate(
            **inputs,

            max_new_tokens=(
                config.max_new_tokens
            ),

            min_length=(
                config.min_length
            ),

            num_beams=(
                config.num_beams
            ),

            repetition_penalty=(
                config.repetition_penalty
            ),

            no_repeat_ngram_size=(
                config.no_repeat_ngram_size
            ),

            length_penalty=(
                config.length_penalty
            ),

            early_stopping=(
                config.early_stopping
            ),
        )

        summary = self.tokenizer.decode(
            generated[0],
            skip_special_tokens=True,
        )

        if not summary.strip():
            return ""

        # -----------------------------------------------------
        # Postprocess
        # -----------------------------------------------------

        summary = (
            self.postprocessor.process(
                summary,
                text,
            )
        )

        if document_type == "resume":
            summary = self._sanitize_resume_output(
                summary
            )

            # Remove repetitive instruction leakage
            summary = re.sub(
                r"(?i)always use information explicitly present in the input\.?",
                " ",
                summary,
            )

            summary = re.sub(
                r"(?i)never invent.*?(?=[\.\n]|$)",
                " ",
                summary,
            )

            summary = re.sub(
                r"\s+",
                " ",
                summary,
            ).strip()

        return summary.strip()

    def _sanitize_resume_output(
        self,
        text: str,
    ) -> str:
        cleaned = text or ""
        for pattern in PROMPT_LEAK_PATTERNS:
            cleaned = re.sub(
                pattern,
                " ",
                cleaned,
            )
        cleaned = re.sub(
            r"\s+",
            " ",
            cleaned,
        ).strip()
        cleaned = re.sub(
            r"^[\s\.,;:\-]+",
            "",
            cleaned,
        ).strip()
        return cleaned

    # ---------------------------------------------------------
    # Batch
    # ---------------------------------------------------------

    def summarize_batch(
        self,
        texts: List[str],
    ) -> List[str]:

        return [
            self.summarize(text)
            for text in texts
        ]
