# training/train.py

from __future__ import annotations

import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import random
import sys

import numpy as np
import torch

from datasets import (
    load_from_disk,
)

from training.config import (
    FINAL_MODEL_DIR,
    SEED,
)

from training.model import (
    load_training_stack,
    print_device_info,
)

from training.preprocessing import (
    preprocess_batch,
)

from training.trainer import (
    build_trainer,
)


# =========================================================
# DATASET PATH
# =========================================================

MULTIDOMAIN_DATASET_PATH = (
    "./training/data/merged/multidomain_dataset"
)


# =========================================================
# REPRODUCIBILITY
# =========================================================

def set_seed(seed: int = SEED):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(seed)


# =========================================================
# GPU VERIFICATION
# =========================================================

def verify_gpu():

    print("\n" + "=" * 50)
    print("GPU VERIFICATION")
    print("=" * 50)

    if not torch.cuda.is_available():

        print(
            "ERROR: CUDA GPU not detected."
        )

        sys.exit(1)

    gpu_name = torch.cuda.get_device_name(0)

    vram_bytes = (
        torch.cuda.get_device_properties(0)
        .total_memory
    )

    vram_gb = (
        vram_bytes / (1024 ** 3)
    )

    print(f"GPU detected:  {gpu_name}")

    print(f"VRAM total:    {vram_gb:.2f} GB")

    print(f"CUDA version:  {torch.version.cuda}")

    print(f"PyTorch:       {torch.__version__}")

    print("=" * 50 + "\n")


# =========================================================
# LOAD MULTI-DOMAIN DATASET
# =========================================================

def load_multidomain_dataset():

    print(
        "\n[DATASET] Loading multi-domain dataset"
    )

    dataset = load_from_disk(
        MULTIDOMAIN_DATASET_PATH
    )

    dataset = dataset.shuffle(
        seed=SEED
    )

    split_dataset = (
        dataset.train_test_split(
            test_size=0.1,
            seed=SEED,
        )
    )

    train_dataset = (
        split_dataset["train"]
    )

    validation_dataset = (
        split_dataset["test"]
    )

    print(
        f"[DATASET] Train: "
        f"{len(train_dataset)}"
    )

    print(
        f"[DATASET] Validation: "
        f"{len(validation_dataset)}"
    )

    print(
        f"[DATASET] Features: "
        f"{train_dataset.column_names}"
    )

    return (
        train_dataset,
        validation_dataset,
    )


# =========================================================
# TOKENIZATION
# =========================================================

def tokenize_datasets(
    train_dataset,
    validation_dataset,
    tokenizer,
):

    print(
        "\n[TRAINING] Tokenizing datasets..."
    )

    train_tokenized = train_dataset.map(

        lambda batch: preprocess_batch(
            batch=batch,
            tokenizer=tokenizer,
        ),

        batched=True,

        remove_columns=train_dataset.column_names,
    )

    validation_tokenized = validation_dataset.map(

        lambda batch: preprocess_batch(
            batch=batch,
            tokenizer=tokenizer,
        ),

        batched=True,

        remove_columns=validation_dataset.column_names,
    )

    return (
        train_tokenized,
        validation_tokenized,
    )


# =========================================================
# SAVE MODEL
# =========================================================

def save_model(
    model,
    tokenizer,
):

    print(
        f"\n[TRAINING] Saving model to:"
        f" {FINAL_MODEL_DIR}"
    )

    model.save_pretrained(
        FINAL_MODEL_DIR
    )

    tokenizer.save_pretrained(
        FINAL_MODEL_DIR
    )

    print(
        "[TRAINING] Model saved successfully."
    )


# =========================================================
# MAIN
# =========================================================

def main():

    print("STEP 1")
    set_seed()

    print("STEP 2")
    verify_gpu()

    print("STEP 3")
    print_device_info()

    print("STEP 4")
    train_dataset, validation_dataset = (
        load_multidomain_dataset()
    )

    print("STEP 5")
    tokenizer, model = (
        load_training_stack()
    )

    print("STEP 6")
    train_tokenized, validation_tokenized = (
        tokenize_datasets(
            train_dataset=train_dataset,
            validation_dataset=validation_dataset,
            tokenizer=tokenizer,
        )
    )

    print("STEP 7")
    trainer = build_trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_tokenized,
        validation_dataset=validation_tokenized,
    )

    print("STEP 8")
    trainer.train()

    print("STEP 9")
    save_model(
        model=trainer.model,
        tokenizer=tokenizer,
    )

    print("\nDONE")


# =========================================================
# ENTRY
# =========================================================

if __name__ == "__main__":

    import multiprocessing

    multiprocessing.freeze_support()

    main()