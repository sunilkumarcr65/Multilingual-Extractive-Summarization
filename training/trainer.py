# training/trainer.py

from __future__ import annotations

from transformers import (
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

from training.callbacks import (
    LossLoggerCallback,
    TrainingTimerCallback,
)

from training.config import (
    EVAL_STEPS,
    GRADIENT_ACCUMULATION_STEPS,
    OUTPUT_DIR,
    PER_DEVICE_BATCH_SIZE,
    REPORT_TO,
    SAVE_STEPS,
    SAVE_TOTAL_LIMIT,
    USE_BF16,
    USE_FP16,
)

from training.evaluation import (
    compute_metrics_builder,
)


# =========================================================
# TRAINING ARGUMENTS
# =========================================================

def build_training_arguments():

    return Seq2SeqTrainingArguments(

        output_dir=OUTPUT_DIR,

        # -------------------------------------------------
        # Training
        # -------------------------------------------------

        per_device_train_batch_size=(
            PER_DEVICE_BATCH_SIZE
        ),

        per_device_eval_batch_size=1,

        gradient_accumulation_steps=(
            GRADIENT_ACCUMULATION_STEPS
        ),

        num_train_epochs=3,

        # -------------------------------------------------
        # Evaluation
        # -------------------------------------------------

        evaluation_strategy="steps",

        eval_steps=EVAL_STEPS,

        save_strategy="steps",

        save_steps=SAVE_STEPS,

        save_total_limit=SAVE_TOTAL_LIMIT,

        load_best_model_at_end=True,

        metric_for_best_model="eval_loss",

        greater_is_better=False,

        predict_with_generate=True,

        # -------------------------------------------------
        # Logging
        # -------------------------------------------------

        logging_steps=50,

        report_to=REPORT_TO,

        # -------------------------------------------------
        # Precision
        # -------------------------------------------------

        fp16=USE_FP16,

        bf16=USE_BF16,

        # -------------------------------------------------
        # Stability
        # -------------------------------------------------

        dataloader_pin_memory=False,
    )


# =========================================================
# DATA COLLATOR
# =========================================================

def build_data_collator(
    tokenizer,
    model,
):

    return DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding="longest",
    )


# =========================================================
# CALLBACKS
# =========================================================

def build_callbacks():

    return [
        TrainingTimerCallback(),
        LossLoggerCallback(),
    ]


# =========================================================
# TRAINER
# =========================================================

def build_trainer(
    model,
    tokenizer,
    train_dataset,
    validation_dataset,
):

    training_args = (
        build_training_arguments()
    )

    collator = build_data_collator(
        tokenizer=tokenizer,
        model=model,
    )

    callbacks = build_callbacks()

    trainer = Seq2SeqTrainer(

        model=model,

        args=training_args,

        train_dataset=train_dataset,

        eval_dataset=validation_dataset,

        tokenizer=tokenizer,

        data_collator=collator,

        callbacks=callbacks,

        compute_metrics=(
            compute_metrics_builder(
                tokenizer
            )
        ),
    )

    return trainer