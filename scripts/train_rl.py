"""
RL Training Script for the CFO Agent.

Uses GRPO (Group Relative Policy Optimization) from HuggingFace TRL.
GRPO is ideal for RL from environment rewards — no separate reward model needed.

WORKFLOW:
  1. Generate episodes using the environment (collect prompts + rewards)
  2. Train the CFO model to maximize cumulative reward
  3. Download weights for local inference

Designed for Kaggle/Colab with GPU.

Usage:
  !pip install unsloth trl datasets
  # Set AGENT=cfo, DATA_PATH=data/cfo_sft.jsonl
  # Then run this script
"""

import os
import sys
import json
import inspect
import random
import torch
from datasets import Dataset

# ═══════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════
MODEL_NAME = "unsloth/gemma-2-2b-it-bnb-4bit"
MAX_SEQ_LENGTH = 1024
LORA_R = 16
BATCH_SIZE = 4
NUM_GENERATIONS = 4    # GRPO generates multiple completions per prompt
MAX_STEPS = 60
LEARNING_RATE = 5e-6
OUTPUT_DIR = "cashflow_cfo_rl"

# Path to transitions from inference.py
TRANSITIONS_PATH = os.environ.get("TRANSITIONS_PATH", "transitions.jsonl")


def load_transitions(path):
    """Load environment transitions as RL training prompts."""
    prompts = []
    with open(path, "r") as f:
        for line in f:
            t = json.loads(line.strip())
            advisor_str = "\n".join([
                f"[{k}]: {json.dumps(v) if isinstance(v, dict) else v}"
                for k, v in t.get("advisor_memos", {}).items()
            ])
            prompt_text = (
                f"State: {t['state_summary']}\n"
                f"Advisors:\n{advisor_str}\n"
                f"Decide the best action (pay/defer/partial/negotiate/credit):"
            )
            prompts.append({
                "prompt": [{"role": "user", "content": prompt_text}],
                "reward": t["reward"],
                "action": t["action"],
            })
    return prompts


def reward_function(completions, prompts_data):
    """
    Custom reward function for GRPO.
    Scores each LLM completion based on:
      1. Valid JSON output (+1)
      2. Correct action type (+2)
      3. Reasonable reasoning (+1)
      4. Matches expert action from transitions (+3)
    """
    rewards = []
    for i, completion in enumerate(completions):
        score = 0.0
        text = completion if isinstance(completion, str) else completion[0]

        try:
            parsed = json.loads(text)
            score += 1.0

            if parsed.get("type") in ["pay", "defer", "partial", "negotiate", "credit"]:
                score += 2.0

            if parsed.get("reasoning") and len(parsed["reasoning"]) > 10:
                score += 1.0

            if i < len(prompts_data):
                expert = prompts_data[i]["action"]
                if parsed.get("type") == expert.get("type"):
                    score += 3.0

        except (json.JSONDecodeError, TypeError):
            score -= 1.0

        if i < len(prompts_data):
            env_reward = prompts_data[i]["reward"]
            score += env_reward * 0.01

        rewards.append(score)

    return rewards


def main():
    print(f"RL Training for CFO Agent")
    print(f"Model: {MODEL_NAME}")
    print(f"Transitions: {TRANSITIONS_PATH}")

    if not torch.cuda.is_available():
        print("WARNING: No GPU. This script needs Kaggle/Colab GPU runtime.")

    # ─── Load Model ───
    from unsloth import FastLanguageModel
    from unsloth.chat_templates import get_chat_template

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=True,
    )

    tokenizer = get_chat_template(
        tokenizer,
        chat_template="gemma",
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_R,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    # ─── Load Transition Data ───
    if not os.path.exists(TRANSITIONS_PATH):
        print(f"ERROR: {TRANSITIONS_PATH} not found.")
        print("Run inference.py first to generate transitions.")
        print("  python inference.py")
        sys.exit(1)

    raw_data = load_transitions(TRANSITIONS_PATH)
    print(f"Loaded {len(raw_data)} transitions")

    dataset = Dataset.from_list([{"prompt": d["prompt"]} for d in raw_data])

    # ─── GRPO Training ───
    from trl import GRPOConfig, GRPOTrainer

    # FIX: removed broken monkeypatch (setting class attributes on GRPOConfig
    # doesn't prevent them being rejected as kwargs by __init__).
    # Instead, build the config dict and strip any keys the installed
    # version of GRPOConfig doesn't accept — version-safe across all TRL releases.
    _grpo_fields = set(inspect.signature(GRPOConfig.__init__).parameters.keys())
    _grpo_kwargs = {k: v for k, v in {
        "output_dir":                    "outputs/cfo_rl",
        "num_generations":               NUM_GENERATIONS,
        "max_completion_length":         256,
        "max_prompt_length":             MAX_SEQ_LENGTH - 256,
        "per_device_train_batch_size":   BATCH_SIZE,
        "gradient_accumulation_steps":   2,
        "max_steps":                     MAX_STEPS,
        "learning_rate":                 LEARNING_RATE,
        "logging_steps":                 5,
        "report_to":                     "none",
        "seed":                          42,
        "push_to_hub":                   False,
    }.items() if k in _grpo_fields}

    grpo_config = GRPOConfig(**_grpo_kwargs)

    def compute_rewards(completions, **kwargs):
        """GRPO reward function — scores each completion."""
        texts = []
        for c in completions:
            if hasattr(c, "text"):
                texts.append(c.text)
            elif isinstance(c, list):
                texts.append(tokenizer.decode(c, skip_special_tokens=True))
            else:
                texts.append(str(c))
        scores = reward_function(texts, raw_data)
        return [torch.tensor(s) for s in scores]

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        config=grpo_config,
        train_dataset=dataset,
        reward_funcs=compute_rewards,
    )

    print("Starting GRPO training...")
    trainer.train()
    print("Training complete!")

    # ─── Save ───
    save_path = f"{OUTPUT_DIR}/cfo_rl_lora"
    model.save_pretrained_merged(save_path, tokenizer, save_method="lora")
    print(f"Saved RL-trained CFO LoRA to: {save_path}")

    print("\n" + "="*50)
    print(f"DONE! Download '{save_path}' for local inference.")
    print("="*50)


if __name__ == "__main__":
    main()