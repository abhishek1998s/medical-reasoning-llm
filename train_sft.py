"""
train_sft.py — Medical Reasoning Fine-Tuning Pipeline
=====================================================

Fine-tunes a small instruct model (Qwen2.5-1.5B-Instruct by default) on the
OpenMed Medical-Reasoning-SFT dataset using Unsloth + QLoRA.

Supports three tracks via --track:
    A_full   : Full chain-of-thought reasoning + final answer
    A_short  : Truncated CoT (first ~150 tokens, sentence-boundary) + final answer
    B        : Answer only (no reasoning)

Output format for Track A (both variants):
    Clinical rationale:
    <reasoning>

    Final answer:
    <answer>

Day 2 — Track B baseline on Kaggle T4:
    python train_sft.py --track B --num_samples 3000

Day 3 — Track A short-CoT on Kaggle T4:
    python train_sft.py --track A_short --num_samples 3000

Day 4 — Final scaled run on A30 / A100:
    python train_sft.py \\
        --model unsloth/Llama-3.2-3B-Instruct-bnb-4bit \\
        --track A_short \\
        --num_samples 30000 \\
        --max_seq_length 4096 \\
        --batch_size 8 --grad_accum 4 \\
        --push_to_hub --hub_repo your-username/llama32-3b-medreason-trackA-final

Tested with: unsloth==2026.4.8, trl==0.24.0, transformers==5.5.0, peft==0.19.1
"""

import argparse
import functools
import os
import json
import logging
import warnings
from pathlib import Path

# Suppress noisy but harmless warnings from the deep learning stack.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
warnings.filterwarnings("ignore", message=".*max_new_tokens.*")
warnings.filterwarnings("ignore", message=".*AttentionMaskConverter.*")
warnings.filterwarnings("ignore", message=".*has new PAD/BOS/EOS tokens.*")
warnings.filterwarnings("ignore", message=".*Will smartly offload gradients.*")
warnings.filterwarnings("ignore", message=".*use_cache.*gradient.*checkpointing.*")
warnings.filterwarnings("ignore", message=".*TypedStorage is deprecated.*")
warnings.filterwarnings("ignore", message=".*processing_class.*")
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")
warnings.filterwarnings("ignore", category=FutureWarning, module="peft")
warnings.filterwarnings("ignore", category=FutureWarning, module="trl")
warnings.filterwarnings("ignore", category=UserWarning,   module="torch")
for _lg in ("transformers", "datasets", "peft", "accelerate"):
    logging.getLogger(_lg).setLevel(logging.ERROR)

# Unsloth must be imported BEFORE transformers/trl (it patches them).
from unsloth import FastLanguageModel, is_bfloat16_supported
from unsloth.chat_templates import get_chat_template, train_on_responses_only

import torch
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig

# transformers v5 fires `use_return_dict` etc. via its own warning_once() —
# which uses internal logging, not Python's warnings module. Both pipes must
# be silenced; the warnings.filterwarnings calls above catch the former,
# this catches the latter.
import transformers
transformers.logging.set_verbosity_error()

from src.data_formatting import format_for_track_a, format_for_track_b
from src.splits import shuffle_filter_split


# ============================================================
# 1. Chat-template response markers (model-specific)
# ============================================================

# train_on_responses_only needs these to identify where the assistant turn
# starts so it can mask user/system tokens with label=-100.
_RESPONSE_PARTS: dict[str, tuple[str, str]] = {
    "qwen-2.5":  ("<|im_start|>user\n",                          "<|im_start|>assistant\n"),
    "llama-3.1": ("<|start_header_id|>user<|end_header_id|>\n\n",
                  "<|start_header_id|>assistant<|end_header_id|>\n\n"),
    "phi-3":     ("<|user|>\n",                                   "<|assistant|>\n"),
    "chatml":    ("<|im_start|>user\n",                          "<|im_start|>assistant\n"),
}


def render_chat_template(examples, tokenizer):
    """Map function: render the messages list into a single 'text' field."""
    texts = [
        tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
        for msgs in examples["messages"]
    ]
    return {"text": texts}


# ============================================================
# 2. Masking preflight check
# ============================================================

def _verify_masking(trainer) -> None:
    sample = trainer.train_dataset[0]
    labels = list(sample.get("labels", sample.get("label", [])))
    if not labels:
        print("[mask-check] WARNING: no labels found, cannot verify masking")
        return
    n_masked = sum(1 for l in labels if l == -100)
    n_trainable = sum(1 for l in labels if l != -100)
    if n_trainable == 0:
        raise AssertionError(
            f"MASKING ERROR: all {len(labels)} labels are -100. "
            "train_on_responses_only masked everything — check response_part marker."
        )
    if n_masked == 0:
        raise AssertionError(
            f"MASKING ERROR: no labels are -100. "
            "User/system tokens are not being masked — check instruction_part marker."
        )
    pct = n_trainable / len(labels) * 100
    print(f"[mask-check] OK: {n_trainable}/{len(labels)} tokens trainable ({pct:.1f}% assistant)")


# ============================================================
# 3. Main
# ============================================================

def main():
    ap = argparse.ArgumentParser()
    # ---- Model & data ----
    ap.add_argument("--model", default="unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit",
                    help="Use the unsloth/*-bnb-4bit variants for QLoRA.")
    ap.add_argument("--track", required=True, choices=["A_full", "A_short", "B"])
    ap.add_argument("--dataset", default="OpenMed/Medical-Reasoning-SFT-GPT-OSS-120B-V2")
    ap.add_argument("--num_samples", type=int, default=3000)
    ap.add_argument("--short_cot_tokens", type=int, default=150)
    # ---- Training hyperparams ----
    ap.add_argument("--max_seq_length", type=int, default=2048)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=16)
    ap.add_argument("--lora_dropout", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_rows", type=int, default=None)
    ap.add_argument("--eval_steps", type=int, default=100)
    ap.add_argument("--save_steps", type=int, default=200)
    ap.add_argument("--logging_steps", type=int, default=10)
    ap.add_argument("--warmup_ratio", type=float, default=0.03)
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

    num_val = max(1, int(args.num_samples * 0.05))
    cot_budget = args.short_cot_tokens if args.track == "A_short" else None

    # Reformat per track using the tested formatters from src/data_formatting.py.
    # A_full passes a huge token limit so truncate_to_n_tokens is effectively a no-op.
    if args.track == "B":
        formatter = format_for_track_b
    elif args.track == "A_short":
        formatter = functools.partial(
            format_for_track_a,
            tokenizer=tokenizer,
            short_cot_max_tokens=args.short_cot_tokens,
        )
    else:  # A_full — keep the entire chain-of-thought
        formatter = functools.partial(
            format_for_track_a,
            tokenizer=tokenizer,
            short_cot_max_tokens=99_999,
        )

    train_ds, eval_ds, _ = shuffle_filter_split(
        ds,
        shuffle_seed=args.seed,
        num_train=args.num_samples,
        num_val=num_val,
        num_test=0,
        max_rows=args.max_rows,
        cot_budget=cot_budget,
        save_indices_path=str(Path(args.output_dir) / "split_indices.json"),
    )
    print(f"[data] train: {len(train_ds)}  eval: {len(eval_ds)}")

    train_ds = train_ds.map(formatter, load_from_cache_file=False)
    eval_ds  = eval_ds.map(formatter, load_from_cache_file=False)

    # Render chat template -> 'text'
    train_ds = train_ds.map(
        lambda ex: render_chat_template(ex, tokenizer),
        batched=True,
        remove_columns=train_ds.column_names,
        load_from_cache_file=False,
    )
    eval_ds = eval_ds.map(
        lambda ex: render_chat_template(ex, tokenizer),
        batched=True,
        remove_columns=eval_ds.column_names,
        load_from_cache_file=False,
    )

    print("\n[data] sample after formatting:")
    print("-" * 60)
    print(train_ds[0]["text"][:1200])
    print("-" * 60)

    # ---- SFT config ----
    # assistant_only_loss=False because we apply Unsloth's train_on_responses_only
    # after building the trainer — the Unsloth compiled wrapper conflicts with
    # TRL's native assistant_only_loss path (same pattern as notebook 02).
    # warmup_ratio is deprecated in transformers v5 / TRL — convert it to an
    # absolute warmup_steps using math.ceil (transformers' own convention).
    import math
    total_optim_steps = max(1, int(len(train_ds) * args.epochs
                                    / (args.batch_size * args.grad_accum)))
    warmup_steps = max(1, math.ceil(args.warmup_ratio * total_optim_steps))
    print(f"[train] total_optim_steps={total_optim_steps}  warmup_steps={warmup_steps}")
    # Eval runs without gradient checkpointing, so at batch_size=2 with
    # max_seq_length=4096 it OOMs on a T4 (16 GB). Force eval batch to 1.
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    sft_config = SFTConfig(
        output_dir=args.output_dir,
        run_name=args.run_name,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.grad_accum,
        warmup_steps=warmup_steps,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=args.logging_steps,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=2,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        seed=args.seed,
        max_length=args.max_seq_length,
        dataset_text_field="text",
        packing=False,                    # safer than packing for chat data
        assistant_only_loss=False,
        report_to="wandb",
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        args=sft_config,
    )

    # Mask user/system tokens — only assistant tokens contribute to the loss.
    instr_part, resp_part = _RESPONSE_PARTS.get(chat_template_name,
                                                 _RESPONSE_PARTS["chatml"])
    trainer = train_on_responses_only(
        trainer,
        instruction_part=instr_part,
        response_part=resp_part,
    )
    _verify_masking(trainer)

    # GPU memory before
    if torch.cuda.is_available():
        print(f"\n[gpu] {torch.cuda.get_device_name(0)} | "
              f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # ---- Train ----
    print("\n[train] starting…")
    train_stats = trainer.train()
    print("[train] done.")
    print(train_stats.metrics)

    # Read eval_loss from log_history rather than train_stats.metrics — in
    # transformers v5 the metrics dict doesn't include eval_loss after train().
    eval_entries = [e for e in trainer.state.log_history if "eval_loss" in e]
    final_eval_loss = eval_entries[-1]["eval_loss"] if eval_entries else None

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
        "eval_loss": final_eval_loss,
        "dataset_stats": {
            "n_train": len(train_ds),
            "n_eval": len(eval_ds),
            "max_rows_cap": args.max_rows,
            "cot_budget_tokens": cot_budget,
        },
        "packages": {
            "unsloth": getattr(__import__("unsloth"), "__version__", "?"),
            "transformers": getattr(__import__("transformers"), "__version__", "?"),
            "trl": getattr(__import__("trl"), "__version__", "?"),
            "peft": getattr(__import__("peft"), "__version__", "?"),
        },
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
