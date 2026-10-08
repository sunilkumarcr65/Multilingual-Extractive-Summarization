# training/model.py

from __future__ import annotations

from typing import Tuple

import torch

from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    PeftModel,
)

from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
)

from training.config import (
    DEVICE,
    FINAL_MODEL_DIR,
    LORA_ALPHA,
    LORA_DROPOUT,
    LORA_RANK,
    LORA_TARGET_MODULES,
    MODEL_NAME,
)


# ---------------------------------------------------------
# TOKENIZER
# ---------------------------------------------------------

def load_tokenizer(
    model_name: str = MODEL_NAME,
):

    print(f"[MODEL] Loading tokenizer: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
    )

    return tokenizer


# ---------------------------------------------------------
# BASE MODEL
# ---------------------------------------------------------

def load_base_model(
    model_name: str = MODEL_NAME,
):

    print(f"[MODEL] Loading base model: {model_name}")

    if torch.cuda.is_available():

        torch.backends.cuda.matmul.allow_tf32 = True

        torch.cuda.empty_cache()

    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        torch_dtype=(
            torch.float16
            if torch.cuda.is_available()
            else torch.float32
        ),
    )

    model = model.to(DEVICE)

    # -----------------------------------------------------
    # Memory optimization
    # -----------------------------------------------------

    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.config.use_cache = False

    return model


# ---------------------------------------------------------
# LoRA CONFIG
# ---------------------------------------------------------

def build_lora_config():

    print("[MODEL] Building LoRA config")

    lora_config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        target_modules=LORA_TARGET_MODULES,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type=TaskType.SEQ_2_SEQ_LM,
        inference_mode=False,
    )

    return lora_config


# ---------------------------------------------------------
# APPLY LoRA
# ---------------------------------------------------------

def apply_lora(
    model,
):

    lora_config = build_lora_config()

    model = get_peft_model(
        model,
        lora_config,
    )

    model.print_trainable_parameters()

    return model


# ---------------------------------------------------------
# FULL TRAINING STACK
# ---------------------------------------------------------

def load_training_stack(
    model_name: str = MODEL_NAME,
) -> Tuple:

    tokenizer = load_tokenizer(
        model_name,
    )

    model = load_base_model(
        model_name,
    )

    model.enable_input_require_grads()

    model = apply_lora(
        model,
    )

    return tokenizer, model


# ---------------------------------------------------------
# LOAD FINETUNED MODEL
# ---------------------------------------------------------

def load_finetuned_model(
    base_model_name: str = MODEL_NAME,
    lora_model_dir: str = FINAL_MODEL_DIR,
):

    print(
        f"[MODEL] Loading fine-tuned model from: {lora_model_dir}"
    )

    tokenizer = load_tokenizer(
        base_model_name,
    )

    base_model = AutoModelForSeq2SeqLM.from_pretrained(
        base_model_name,
        torch_dtype=(
            torch.float16
            if torch.cuda.is_available()
            else torch.float32
        ),
    )

    base_model = base_model.to(DEVICE)

    model = PeftModel.from_pretrained(
        base_model,
        lora_model_dir,
    )

    model.eval()

    return tokenizer, model


# ---------------------------------------------------------
# DEVICE INFO
# ---------------------------------------------------------

def print_device_info():

    print(f"[DEVICE] Using: {DEVICE}")

    if torch.cuda.is_available():

        gpu_name = torch.cuda.get_device_name(0)

        vram_gb = (
            torch.cuda.get_device_properties(0).total_memory
            / (1024 ** 3)
        )

        print(f"[DEVICE] GPU: {gpu_name}")

        print(f"[DEVICE] VRAM: {vram_gb:.2f} GB")

    else:

        print(
            "[DEVICE] WARNING: CUDA not available."
        )