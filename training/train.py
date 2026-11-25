"""
train.py — QLoRA fine-tuning of Qwen2.5-7B-Instruct on Arabic diacritization.

Hardware target: RTX 3060 12 GB VRAM
Estimated time: ~8-14 hours for 3 epochs on Tashkeela

Usage:
    source /mnt/e/harakaat-lora/venv/bin/activate
    cd /mnt/e/harakaat-lora
    python training/train.py

OOM fallback: edit CONFIG section — reduce max_seq_length or switch to 3B model.
"""
import os
import sys
import json
import time
import logging
from pathlib import Path
from dataclasses import dataclass

# ── Environment ──────────────────────────────────────────────────────────────
PROJ_DIR = Path("/mnt/e/harakaat-lora")
os.environ["HF_HOME"] = str(PROJ_DIR / ".cache" / "huggingface")
os.environ["HF_DATASETS_CACHE"] = str(PROJ_DIR / ".cache" / "huggingface" / "datasets")
# CUDA for WSL2
os.environ.setdefault("LD_LIBRARY_PATH", "/usr/lib/wsl/lib")

import torch
import transformers
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from datasets import Dataset
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)


# ── CONFIG — edit here if OOM ─────────────────────────────────────────────────
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
# OOM fallback: "Qwen/Qwen2.5-3B-Instruct"

MAX_SEQ_LENGTH = 512          # Reduce to 384 or 256 if OOM
BATCH_SIZE = 1                # Keep at 1 for 12 GB VRAM
GRAD_ACCUM = 16               # Effective batch = BATCH_SIZE * GRAD_ACCUM = 16
NUM_EPOCHS = 3
LEARNING_RATE = 2e-4
LORA_R = 16                   # Reduce to 8 if OOM
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
WARMUP_RATIO = 0.03
USE_BF16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()

OUTPUT_DIR = PROJ_DIR / "outputs" / "harakaat-adapter"
LOG_DIR = PROJ_DIR / "results"

TRAIN_FILE = PROJ_DIR / "data" / "train.jsonl"
VAL_FILE = PROJ_DIR / "data" / "val.jsonl"
# ─────────────────────────────────────────────────────────────────────────────


def check_gpu():
    if not torch.cuda.is_available():
        logger.error("CUDA not available. Check LD_LIBRARY_PATH=/usr/lib/wsl/lib")
        logger.error("Run: export LD_LIBRARY_PATH=/usr/lib/wsl/lib && python training/train.py")
        sys.exit(1)
    gpu = torch.cuda.get_device_name(0)
    vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    logger.info(f"GPU: {gpu}  VRAM: {vram:.1f} GB  BF16: {USE_BF16}")
    return vram


def load_jsonl(path: Path) -> list[dict]:
    with open(path, encoding='utf-8') as f:
        return [json.loads(l) for l in f if l.strip()]


def format_prompt(example: dict) -> str:
    """Qwen2.5 ChatML format."""
    return (
        "<|im_start|>system\n"
        "أنت نظام متخصص في إضافة التشكيل الكامل للنصوص العربية الفصحى.\n"
        "<|im_end|>\n"
        "<|im_start|>user\n"
        f"{example['instruction']}\n{example['input']}\n"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
        f"{example['output']}\n"
        "<|im_end|>"
    )


def build_dataset(records: list[dict]) -> Dataset:
    texts = [format_prompt(r) for r in records]
    return Dataset.from_dict({"text": texts})


def save_loss_plot(log_history: list[dict]):
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    train_steps, train_losses = [], []
    val_steps, val_losses = [], []

    for entry in log_history:
        if "loss" in entry and "step" in entry:
            train_steps.append(entry["step"])
            train_losses.append(entry["loss"])
        if "eval_loss" in entry and "step" in entry:
            val_steps.append(entry["step"])
            val_losses.append(entry["eval_loss"])

    fig, ax = plt.subplots(figsize=(10, 5))
    if train_losses:
        ax.plot(train_steps, train_losses, label="Train loss", color="steelblue")
    if val_losses:
        ax.plot(val_steps, val_losses, label="Val loss", color="tomato", linestyle='--')
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title("QLoRA Training: Qwen2.5-7B → harakaat")
    ax.legend()
    ax.grid(alpha=0.3)
    out_path = LOG_DIR / "loss_curve.png"
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Loss curve saved to {out_path}")


def main():
    start_time = time.time()
    vram = check_gpu()

    # Warn if VRAM is marginal for 7B
    if vram < 11.5:
        logger.warning(f"Only {vram:.1f} GB VRAM. Consider switching to 3B model.")

    logger.info(f"Loading model: {MODEL_ID}")
    logger.info(f"Config: max_seq={MAX_SEQ_LENGTH}, lora_r={LORA_R}, bf16={USE_BF16}")

    # ── Quantization config (4-bit NF4) ──────────────────────────────────────
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if USE_BF16 else torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    # ── Load tokenizer ────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # ── Load model ────────────────────────────────────────────────────────────
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if USE_BF16 else torch.float16,
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1

    # Log memory after model load
    alloc = torch.cuda.memory_allocated(0) / 1e9
    reserved = torch.cuda.memory_reserved(0) / 1e9
    logger.info(f"After model load — allocated: {alloc:.2f} GB, reserved: {reserved:.2f} GB")

    # ── LoRA config ───────────────────────────────────────────────────────────
    # Target the attention projection layers
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ── Data ──────────────────────────────────────────────────────────────────
    if not TRAIN_FILE.exists():
        logger.error(f"Training data not found: {TRAIN_FILE}")
        logger.error("Run: python data/prepare_dataset.py")
        sys.exit(1)

    train_records = load_jsonl(TRAIN_FILE)
    val_records = load_jsonl(VAL_FILE)
    logger.info(f"Train: {len(train_records):,}  Val: {len(val_records):,}")

    train_ds = build_dataset(train_records)
    val_ds = build_dataset(val_records)

    # ── Training arguments ────────────────────────────────────────────────────
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LEARNING_RATE,
        bf16=USE_BF16,
        fp16=not USE_BF16,
        logging_steps=50,
        evaluation_strategy="steps",
        eval_steps=200,
        save_steps=200,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="paged_adamw_32bit",
        lr_scheduler_type="cosine",
        warmup_ratio=WARMUP_RATIO,
        max_grad_norm=0.3,
        report_to="none",
        logging_dir=str(LOG_DIR / "tensorboard"),
        dataloader_num_workers=0,
    )

    # ── Trainer ───────────────────────────────────────────────────────────────
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LENGTH,
        packing=False,
    )

    # ── Train ─────────────────────────────────────────────────────────────────
    logger.info("Starting training...")
    logger.info(f"Effective batch size: {BATCH_SIZE * GRAD_ACCUM}")
    logger.info(f"Steps per epoch: ~{len(train_records) // (BATCH_SIZE * GRAD_ACCUM)}")

    result = trainer.train()
    elapsed = time.time() - start_time

    logger.info(f"Training complete!")
    logger.info(f"  Total time: {elapsed/3600:.2f} hours")
    logger.info(f"  Final train loss: {result.training_loss:.4f}")
    logger.info(f"  Steps: {result.global_step}")

    # ── Save adapter ──────────────────────────────────────────────────────────
    trainer.save_model(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))
    logger.info(f"Adapter saved to {OUTPUT_DIR}")

    # ── Loss curve ────────────────────────────────────────────────────────────
    save_loss_plot(trainer.state.log_history)

    # ── Save training summary ──────────────────────────────────────────────────
    summary = {
        "model": MODEL_ID,
        "lora_r": LORA_R,
        "lora_alpha": LORA_ALPHA,
        "max_seq_length": MAX_SEQ_LENGTH,
        "num_epochs": NUM_EPOCHS,
        "train_examples": len(train_records),
        "val_examples": len(val_records),
        "final_train_loss": result.training_loss,
        "total_steps": result.global_step,
        "training_hours": round(elapsed / 3600, 2),
        "gpu": torch.cuda.get_device_name(0),
        "vram_gb": vram,
        "bf16": USE_BF16,
    }
    summary_path = LOG_DIR / "training_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Summary saved to {summary_path}")

    print("\n=== Training complete ===")
    print(f"Adapter: {OUTPUT_DIR}")
    print(f"Time:    {elapsed/3600:.2f} hours")
    print(f"Loss:    {result.training_loss:.4f}")
    print("\nNext: python evaluation/evaluate.py")


if __name__ == "__main__":
    main()
