# training/callbacks.py

from __future__ import annotations

import math
import time
from typing import Dict

from transformers import (
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)


# ---------------------------------------------------------
# Training Timer
# ---------------------------------------------------------

class TrainingTimerCallback(
    TrainerCallback,
):
    """
    Measure training duration.
    """

    def __init__(self):

        self.start_time = None

    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):

        self.start_time = time.time()

        print(
            "[CALLBACK] "
            "Training started."
        )

    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):

        if self.start_time is None:
            return

        elapsed = (
            time.time()
            - self.start_time
        )

        hours = elapsed / 3600

        print(
            f"[CALLBACK] "
            f"Training completed "
            f"in {hours:.2f}h"
        )


# ---------------------------------------------------------
# Loss Logger
# ---------------------------------------------------------

class LossLoggerCallback(
    TrainerCallback,
):
    """
    Log training/eval loss.
    """

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: Dict = None,
        **kwargs,
    ):

        if not logs:
            return

        step = state.global_step

        train_loss = logs.get(
            "loss"
        )

        eval_loss = logs.get(
            "eval_loss"
        )

        learning_rate = logs.get(
            "learning_rate"
        )

        if train_loss is not None:

            print(
                f"[TRAIN] "
                f"Step={step} | "
                f"Loss={train_loss:.4f}"
            )

        if eval_loss is not None:

            print(
                f"[EVAL] "
                f"Step={step} | "
                f"EvalLoss={eval_loss:.4f}"
            )

        if learning_rate is not None:

            print(
                f"[LR] "
                f"{learning_rate:.8f}"
            )


# ---------------------------------------------------------
# NaN Guard
# ---------------------------------------------------------

class NaNDetectorCallback(
    TrainerCallback,
):
    """
    Stop training if NaN loss appears.
    """

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: Dict = None,
        **kwargs,
    ):

        if not logs:
            return

        loss = logs.get("loss")

        if (
            loss is not None
            and math.isnan(loss)
        ):

            print(
                "[CALLBACK] "
                "NaN loss detected. "
                "Stopping training."
            )

            control.should_training_stop = (
                True
            )


# ---------------------------------------------------------
# GPU Memory Monitor
# ---------------------------------------------------------

class GPUMemoryCallback(
    TrainerCallback,
):
    """
    Monitor GPU memory usage.
    """

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: Dict = None,
        **kwargs,
    ):

        try:
            import torch

            if not (
                torch.cuda.is_available()
            ):
                return

            allocated = (
                torch.cuda.memory_allocated()
                / (1024 ** 3)
            )

            reserved = (
                torch.cuda.memory_reserved()
                / (1024 ** 3)
            )

            print(
                f"[GPU] "
                f"Allocated="
                f"{allocated:.2f}GB | "
                f"Reserved="
                f"{reserved:.2f}GB"
            )

        except Exception:
            pass