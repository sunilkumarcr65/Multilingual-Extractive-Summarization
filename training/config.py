# training/config.py

from __future__ import annotations
import torch


# ---------------------------------------------------------
# MODEL
# ---------------------------------------------------------

MODEL_NAME = "google/flan-t5-base"

MODEL_TASK_PREFIX = "summarize: "


# ---------------------------------------------------------
# DATASET
# ---------------------------------------------------------

DATASET_NAME = "csebuetnlp/xlsum"

DATASET_LANGUAGE = "english"

TRAIN_SAMPLES = 5000

VALIDATION_SAMPLES = 200

MULTILINGUAL_LANGUAGES = [
    "english",
    "hindi",
    "gujarati",
    "bengali",
]

MULTILINGUAL_SAMPLES_PER_LANG = 500


# ---------------------------------------------------------
# TOKENIZATION
# ---------------------------------------------------------

# IMPORTANT:
# flan-t5-base on 4GB VRAM
# cannot handle massive sequence lengths
# during training safely

MAX_SOURCE_LENGTH = 384

MAX_TARGET_LENGTH = 96

PADDING = "max_length"

TRUNCATION = True


# ---------------------------------------------------------
# LoRA
# ---------------------------------------------------------

LORA_RANK = 8

LORA_ALPHA = 16

LORA_DROPOUT = 0.1

LORA_TARGET_MODULES = [
    "q",
    "v",
]


# ---------------------------------------------------------
# TRAINING
# ---------------------------------------------------------

OUTPUT_DIR = "./training/checkpoints"

FINAL_MODEL_DIR = (
    "./models/flan_t5_base_multidomain_lora"
)

LEARNING_RATE = 5e-5

NUM_EPOCHS = 3

# MUST stay 1 for 4GB VRAM
PER_DEVICE_BATCH_SIZE = 1

# Simulates larger batch
GRADIENT_ACCUMULATION_STEPS = 32

MAX_GRAD_NORM = 0.5

WEIGHT_DECAY = 0.01

WARMUP_RATIO = 0.05


# ---------------------------------------------------------
# LOGGING
# ---------------------------------------------------------

LOGGING_STEPS = 25

EVAL_STEPS = 250

SAVE_STEPS = 250

SAVE_TOTAL_LIMIT = 2


# ---------------------------------------------------------
# MIXED PRECISION
# ---------------------------------------------------------

# RTX 3050 supports fp16
USE_FP16 = False

USE_BF16 = False


# ---------------------------------------------------------
# DEVICE
# ---------------------------------------------------------

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ---------------------------------------------------------
# GENERATION
# ---------------------------------------------------------

NUM_BEAMS = 4

LENGTH_PENALTY = 1.2

NO_REPEAT_NGRAM_SIZE = 3


# ---------------------------------------------------------
# EARLY STOPPING
# ---------------------------------------------------------

EARLY_STOPPING_PATIENCE = 3


# ---------------------------------------------------------
# TRACKING
# ---------------------------------------------------------

REPORT_TO = "none"


# ---------------------------------------------------------
# REPRODUCIBILITY
# ---------------------------------------------------------

SEED = 42