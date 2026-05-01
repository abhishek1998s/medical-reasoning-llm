"""
train_sft.py — Medical Reasoning Fine-Tuning Pipeline
=====================================================

Fine-tunes a small instruct model (Qwen2.5-1.5B-Instruct by default) on the
OpenMed Medical-Reasoning-SFT dataset using Unsloth + QLoRA.

Supports three tracks via --track:
    A_full   : Full chain-of-thought reasoning + answer
    A_short  : Truncated CoT (first ~150 tokens of reasoning) + answer
    B        : Answer only (no reasoning)

Day 2 — Track B baseline on Kaggle P100:
    python train_sft.py --track B --num_samples 5000

Day 3 — Track A full + short on Kaggle P100:
    python train_sft.py --track A_full --num_samples 5000 --max_seq_length 4096 \
        --batch_size 1 --grad_accum 16
    python train_sft.py --track A_short --num_samples 5000

Day 4 — Final scaled run on C-DAC A30 / A100:
    python train_sft.py \
        --model unsloth/Llama-3.2-3B-Instruct-bnb-4bit \
        --track A_full \
        --num_samples 30000 \
        --max_seq_length 4096 \
        --batch_size 8 --grad_accum 4 \
        --push_to_hub --hub_repo your-username/llama32-3b-medreason-trackA-final

Tested with: unsloth >= 2024.10, trl >= 0.12, transformers >= 4.46
"""

import argparse
import os
import json
from pathlib import Path

# Unsloth must be imported BEFORE transformers/trl (it patches them).
from unsloth import FastLanguageModel, is_bfloat16_supported
from unsloth.chat_templates import get_chat_template

import torch
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig


# ============================================================
# 1. Dataset formatting — three tracks
# ============================================================

def _truncate_to_n_tokens(text: str, tokenizer, max_tokens: int) -> str:
    """Truncate text to roughly max_tokens tokens (best-effort, keeps prefix)."""
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) <= max_tokens:
        return text
    return tokenizer.decode(ids[:max_tokens], skip_special_tokens=True) + " ..."


def make_formatter(track: str, tokenizer, short_cot_tokens: int = 150):
    """
    Return a function row -> {"messages": [...]}  re-shaped per track.

    OpenMed schema: each row has a `messages` list. Assistant turns carry both
        - `content`           : the final answer (clean string)
        - `reasoning_content` : the GPT-OSS-120B chain-of-thought
    """
    def fmt(example):
        out = []
        for m in example["messages"]:
            role = m["role"]
            if role == "user":
                out.append({"role": "user", "content": m["content"]})
                continue

            if role == "assistant":
                content   = (m.get("content") or "").strip()
                reasoning = (m.get("reasoning_content") or "").strip()

                if track == "B":
                    final = content                                                  # answer only
                elif track == "A_full":
                    final = f"<think>\n{reasoning}\n</think>\n\n{content}"          # full CoT
                elif track == "A_short":
                    short = _truncate_to_n_tokens(reasoning, tokenizer, short_cot_tokens)
                    final = f"<think>\n{short}\n</think>\n\n{content}"             # short CoT
                else:
                    raise ValueError(f"Unknown track: {track}")

                out.append({"role": "assistant", "content": final})
                continue

            # System/tool messages pass through.
            out.append(m)

        return {"messages": out}

    return fmt


def render_chat_template(examples, tokenizer):
    """Map function: render the messages list into a single 'text' field."""
    texts = [
        tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
        for msgs in examples["messages"]
    ]
    return {"text": texts}


# ============================================================
# 2. Main
# ============================================================

def main():
    ap = argparse.ArgumentParser()
    # ---- Model & data ----
    ap.add_argument("--model", default="unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit",
                    help="Use the unsloth/*-bnb-4bit variants for QLoRA.")
    ap.add_argument("--track", required=True, choices=["A_full", "A_short", "B"])
    ap.add_argument("--dataset", default="OpenMed/Medical-Reasoning-SFT-GPT-OSS-120B-V2")
    ap.add_argument("--num_samples", type=int, default=5000)
    ap.add_argument("--short_cot_tokens", type=int, default=150)
    # ---- Training hyperparams ----
    ap.add_argument("--max_seq_length", type=int, default=2048)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=16)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    # ---- Outputs ----
    ap.add_argument("--output_dir", default="./output")
    ap.add_argument("--run_name", default=None)
    ap.add_argument("--push_to_hub", action="store_true")
    ap.add_argument("--hub_repo", default=None,
                    help="e.g. abhi/qwen25-1.5b-medreason-trackB-v0")
    ap.add_argument("--wandb_project", default="medical-reasoning-sft")
    args = ap.parse_args()

    # Auto run-name
    if args.run_name is None:
        model_short = args.model.split("/")[-1].split("-bnb")[0]
        args.run_name = f"{model_short}-{args.track}-{args.num_samples}"

    os.environ["WANDB_PROJECT"] = args.wandb_project
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"Run:    {args.run_name}")
    print(f"Model:  {args.model}")
    print(f"Track:  {args.track}")
    print(f"N:      {args.num_samples}")
    print(f"Seq:    {args.max_seq_length}")
    print(f"Batch:  {args.batch_size} x grad_accum {args.grad_accum}"
          f" = effective {args.batch_size * args.grad_accum}")
    print("=" * 60)

    # ---- Load model + tokenizer (4-bit QLoRA) ----
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=args.max_seq_length,
        dtype=None,                  # auto-detect bf16/fp16
        load_in_4bit=True,
    )

    # Pick the right chat template
    name = args.model.lower()
    if "qwen" in name:
        chat_template_name = "qwen-2.5"
    elif "llama-3" in name or "llama3" in name:
        chat_template_name = "llama-3.1"   # works for 3.0/3.1/3.2
    elif "phi-3" in name:
        chat_template_name = "phi-3"
    else:
        chat_template_name = "chatml"      # safe fallback

    tokenizer = get_chat_template(tokenizer, chat_template=chat_template_name)
    tokenizer.padding_side = "right"       # right-pad for training

    # ---- Attach LoRA ----
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
        use_rslora=False,
    )

    # ---- Dataset ----
    print("\n[data] loading…")
    ds = load_dataset(args.dataset, split="train")
    print(f"[data] full size: {len(ds)}")

    ds = ds.shuffle(seed=args.seed).select(range(min(args.num_samples, len(ds))))
    print(f"[data] using:    {len(ds)}")

    # Reformat per track
    formatter = make_formatter(args.track, tokenizer, args.short_cot_tokens)
    ds = ds.map(formatter)

    # Render chat template -> 'text'
    ds = ds.map(
        lambda ex: render_chat_template(ex, tokenizer),
        batched=True,
        remove_columns=ds.column_names,
    )

    print("\n[data] sample after formatting:")
    print("-" * 60)
    print(ds[0]["text"][:1200])
    print("-" * 60)

    # 95/5 split
    split = ds.train_test_split(test_size=0.05, seed=args.seed)
    train_ds, eval_ds = split["train"], split["test"]
    print(f"[data] train: {len(train_ds)}  eval: {len(eval_ds)}")

    # ---- SFT config ----
    sft_config = SFTConfig(
        output_dir=args.output_dir,
        run_name=args.run_name,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        warmup_ratio=0.03,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=2,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        seed=args.seed,
        max_seq_length=args.max_seq_length,
        dataset_text_field="text",
        packing=False,                    # safer than packing for chat data
        report_to="wandb",
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        args=sft_config,
    )

    # GPU memory before
    if torch.cuda.is_available():
        print(f"\n[gpu] {torch.cuda.get_device_name(0)} | "
              f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # ---- Train ----
    print("\n[train] starting…")
    train_stats = trainer.train()
    print("[train] done.")
    print(train_stats.metrics)

    # ---- Save adapter + metadata ----
    save_path = Path(args.output_dir) / "final_adapter"
    model.save_pretrained(str(save_path))
    tokenizer.save_pretrained(str(save_path))

    meta = {
        "model": args.model,
        "track": args.track,
        "num_samples": args.num_samples,
        "max_seq_length": args.max_seq_length,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "lr": args.lr,
        "epochs": args.epochs,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "chat_template": chat_template_name,
        "train_runtime_sec": train_stats.metrics.get("train_runtime"),
        "train_loss": train_stats.metrics.get("train_loss"),
        "eval_loss": train_stats.metrics.get("eval_loss"),
    }
    with open(save_path / "training_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    # ---- Optional: push to HF Hub ----
    if args.push_to_hub and args.hub_repo:
        print(f"\n[hub] pushing to {args.hub_repo}")
        model.push_to_hub(args.hub_repo, token=os.environ.get("HF_TOKEN"))
        tokenizer.push_to_hub(args.hub_repo, token=os.environ.get("HF_TOKEN"))

    print(f"\n✅ adapter saved → {save_path}")
    print("   next step: run inference and produce a predictions CSV "
          "with columns [question, reference, prediction], then feed it to "
          "llm_judge.py")


if __name__ == "__main__":
    main()
