"""
SFT Training Script for Cashflow Multi-Agent Sub-Agents.

Designed to run on Kaggle/Colab with GPU (T4/P100).
Uses Unsloth for 4-bit LoRA fine-tuning.

WORKFLOW:
  1. Upload your JSONL data files to Kaggle (from generate_sft_data.py)
  2. Run this notebook/script
  3. Download the LoRA adapter weights
  4. Load locally for inference

Usage in Kaggle notebook:
  !pip install unsloth datasets trl
  # Then run this script or paste cells
"""

import os
import json
import inspect
import torch
from datasets import Dataset

# ═══════════════════════════════════════════════════════
# CONFIG — Change these for your setup
# ═══════════════════════════════════════════════════════
MODEL_NAME = "unsloth/gemma-2-2b-it-bnb-4bit"  # Open access, fits on T4
MAX_SEQ_LENGTH = 2048
LORA_R = 16
LORA_ALPHA = 16
BATCH_SIZE = 2
GRAD_ACCUM = 4
MAX_STEPS = 100       # Increase for better quality (500-1000 recommended)
LEARNING_RATE = 2e-4
OUTPUT_DIR = "cashflow_agent_lora"

# Which agent to train: "expenditure", "revenue", "risk", or "cfo"
AGENT_TO_TRAIN = os.environ.get("AGENT", "cfo")
DATA_PATH = os.environ.get("DATA_PATH", f"data/{AGENT_TO_TRAIN}_sft.jsonl")


def load_data(path):
    """Load JSONL chat data."""
    samples = []
    with open(path, "r") as f:
        for line in f:
            samples.append(json.loads(line.strip()))
    return samples


def main():
    print(f"Training agent: {AGENT_TO_TRAIN}")
    print(f"Data path: {DATA_PATH}")
    print(f"Model: {MODEL_NAME}")
    print(f"Output: {OUTPUT_DIR}")

    # ─── Check for GPU ───
    if not torch.cuda.is_available():
        print("WARNING: No GPU detected. Training will be very slow.")
        print("This script is designed for Kaggle/Colab with GPU runtime.")

    # ─── Load Unsloth ───
    from unsloth import FastLanguageModel
    from unsloth.chat_templates import get_chat_template

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=True,
    )

    # ─── Setup Gemma-2 Chat Template ───
    tokenizer = get_chat_template(
        tokenizer,
        chat_template="gemma",
    )

    # ─── Add LoRA Adapters ───
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_R,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=LORA_ALPHA,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    # ─── Load & Format Data ───
    raw_data = load_data(DATA_PATH)
    print(f"Loaded {len(raw_data)} training samples")

    dataset = Dataset.from_list(raw_data)

    def apply_template(examples):
        texts = []
        for messages in examples["messages"]:
            texts.append(tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            ))
        return {"text": texts}

    dataset = dataset.map(apply_template, batched=True, remove_columns=dataset.column_names)

    # ─── Train ───
    from trl import SFTTrainer, SFTConfig

    # FIX: build config dict and strip any keys SFTConfig doesn't accept
    # so version mismatches with push_to_hub_token / hub_token never crash.
    _sft_fields = set(inspect.signature(SFTConfig.__init__).parameters.keys())
    _sft_kwargs = {k: v for k, v in {
        "output_dir":                    f"outputs/{AGENT_TO_TRAIN}",
        "per_device_train_batch_size":   BATCH_SIZE,
        "gradient_accumulation_steps":   GRAD_ACCUM,
        "warmup_steps":                  5,
        "max_steps":                     MAX_STEPS,
        "learning_rate":                 LEARNING_RATE,
        "fp16":                          not torch.cuda.is_bf16_supported(),
        "bf16":                          torch.cuda.is_bf16_supported(),
        "logging_steps":                 5,
        "optim":                         "adamw_8bit",
        "weight_decay":                  0.01,
        "lr_scheduler_type":             "linear",
        "seed":                          42,
        "report_to":                     "none",
        "push_to_hub":                   False,
        "dataset_text_field":            "text",
        "max_seq_length":                MAX_SEQ_LENGTH,
        "dataset_num_proc":              2,
    }.items() if k in _sft_fields}

    sft_config = SFTConfig(**_sft_kwargs)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=sft_config,
    )

    print("Starting training...")
    stats = trainer.train()
    print(f"Training complete! Loss: {stats.training_loss:.4f}")

    # ─── Save ───
    save_path = f"{OUTPUT_DIR}/{AGENT_TO_TRAIN}"
    model.save_pretrained_merged(save_path, tokenizer, save_method="lora")
    print(f"Saved LoRA adapter to: {save_path}")

    # Also save as merged 16-bit for easy inference
    merged_path = f"{OUTPUT_DIR}/{AGENT_TO_TRAIN}_merged"
    model.save_pretrained_merged(merged_path, tokenizer, save_method="merged_16bit")
    print(f"Saved merged model to: {merged_path}")

    print("\n" + "="*50)
    print(f"DONE! Download '{save_path}' for local inference.")
    print("="*50)


if __name__ == "__main__":
    main()