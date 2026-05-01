"""Build notebooks/01_setup_and_data_exploration.ipynb.

Run with:
    python scripts/build_notebook_01.py

Produces a deterministic, unexecuted notebook with markdown + code cells
ready to be run on Kaggle (T4 GPU, internet on).
"""
from __future__ import annotations
import json
from pathlib import Path

NB_PATH = Path("notebooks/01_setup_and_data_exploration.ipynb")
NB_PATH.parent.mkdir(parents=True, exist_ok=True)


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "metadata": {},
        "source": text,
        "outputs": [],
        "execution_count": None,
    }


cells: list[dict] = []

# ─────────────────────────────────────────────────────────────────────────────
# Section 0 — header
# ─────────────────────────────────────────────────────────────────────────────
cells.append(md("""# Notebook 01 — Setup and Data Exploration

**Day 1 of 7 (2026-05-02).** Goal: a working Kaggle environment, a first look at the OpenMed medical-reasoning dataset, and a sanity-check that the Track A and Track B formatters produce sensible outputs on real rows.

This notebook does **not** train anything. Training starts in Notebook 02.

## What you'll learn here
1. How tokenizers turn text into IDs and back, and what a chat template does.
2. The structure of the OpenMed dataset (`messages`, `content`, `reasoning_content`).
3. What 4-bit quantization does to model memory, and why it matters on a 16 GB T4.
4. How Track A's "Clinical rationale + Final answer" output looks vs Track B's answer-only output.
5. A single end-to-end inference smoke test (base model, no fine-tuning yet).

## Required Kaggle environment
- Accelerator: **GPU T4 x1**
- Internet: **On**
- Kaggle Secrets set: `HF_TOKEN`, `WANDB_API_KEY`, `GROQ_API_KEY`

## Plan reference
[`docs/superpowers/plans/2026-05-02-plan1-bootstrap-and-notebook01.md`](../docs/superpowers/plans/2026-05-02-plan1-bootstrap-and-notebook01.md) — Phase B."""))

# ─────────────────────────────────────────────────────────────────────────────
# Section 1 — bootstrap
# ─────────────────────────────────────────────────────────────────────────────
cells.append(md("""## 1. Bootstrap: clone repo, install deps

Kaggle's preinstalled environment has *most* of what we need but not all. We pin our exact versions from `requirements.txt`. Total install ≈ 2–4 min.

> **Note:** the project repo is public, so no GitHub auth is needed for `git clone`. Model adapters stay private on Hugging Face Hub."""))

cells.append(code("""# Clone the project so we can `import src.data_formatting` etc.
import subprocess, sys, os

REPO_URL = "https://github.com/abhishek1998s/medical-reasoning-llm.git"
REPO_DIR = "/kaggle/working/medical-reasoning-llm"

if not os.path.exists(REPO_DIR):
    subprocess.run(["git", "clone", REPO_URL, REPO_DIR], check=True)

# Make src/ importable from any cell below
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

print("Repo cloned to:", REPO_DIR)
print("Contents:", os.listdir(REPO_DIR))"""))

cells.append(code("""# Install the pinned project requirements.
# The {REPO_DIR} below is substituted by Jupyter's `!` shell magic.
!pip install -q -r {REPO_DIR}/requirements.txt"""))

cells.append(code("""# Verify imports succeed and record installed versions.
# These versions go into outputs/<track>/training_meta.json on training day
# so the experiment is reproducible.
import importlib

versions = {}
for pkg in ["unsloth", "transformers", "trl", "peft", "datasets",
            "bitsandbytes", "accelerate", "wandb"]:
    try:
        m = importlib.import_module(pkg)
        versions[pkg] = getattr(m, "__version__", "?")
    except Exception as e:
        versions[pkg] = f"FAILED: {e.__class__.__name__}"

for k, v in versions.items():
    print(f"  {k:20s} {v}")"""))

# ─────────────────────────────────────────────────────────────────────────────
# Section 2 — secrets
# ─────────────────────────────────────────────────────────────────────────────
cells.append(md("""## 2. Secrets

We use **Kaggle Secrets** (lock icon in the right sidebar of Kaggle's notebook editor). Three keys:

| Secret name | Purpose | When used |
|---|---|---|
| `HF_TOKEN` | Hugging Face token with **write** access | Day 2 — push private LoRA adapters |
| `WANDB_API_KEY` | Weights & Biases | Day 2/3 — track training loss curves |
| `GROQ_API_KEY` | Groq API | Day 6 — LLM-as-judge on predictions |

If a secret is missing, the corresponding step is skipped — we don't crash the whole notebook."""))

cells.append(code("""# Load all three Kaggle Secrets, tolerating missing ones.
from kaggle_secrets import UserSecretsClient

secrets = UserSecretsClient()

def _try_get(name):
    try:
        return secrets.get_secret(name)
    except Exception as e:
        print(f"  [skip] {name}: {e.__class__.__name__}")
        return None

os.environ["HF_TOKEN"]      = _try_get("HF_TOKEN")      or ""
os.environ["WANDB_API_KEY"] = _try_get("WANDB_API_KEY") or ""
os.environ["GROQ_API_KEY"]  = _try_get("GROQ_API_KEY")  or ""

print()
print("HF_TOKEN set:      ", bool(os.environ["HF_TOKEN"]))
print("WANDB_API_KEY set: ", bool(os.environ["WANDB_API_KEY"]))
print("GROQ_API_KEY set:  ", bool(os.environ["GROQ_API_KEY"]))"""))

cells.append(code("""# Hugging Face login — needed even for reading the OpenMed dataset
# (some HF mirrors return 401 without a token).
from huggingface_hub import login as hf_login

if os.environ["HF_TOKEN"]:
    hf_login(token=os.environ["HF_TOKEN"], add_to_git_credential=False)
    print("HF login OK")
else:
    print("[warning] HF_TOKEN not set — dataset download may fail or rate-limit")"""))

# ─────────────────────────────────────────────────────────────────────────────
# Section 3 — dataset exploration
# ─────────────────────────────────────────────────────────────────────────────
cells.append(md("""## 3. Dataset — `OpenMed/Medical-Reasoning-SFT-GPT-OSS-120B-V2`

Synthetic medical QA distilled from GPT-OSS-120B. Each row is a single-turn conversation:

```python
{
    "messages": [
        {"role": "user", "content": "<medical question>"},
        {"role": "assistant",
         "content":           "<final answer>",
         "reasoning_content": "<chain-of-thought>"}
    ]
}
```

**Track A** uses both `content` and a truncated `reasoning_content`.
**Track B** uses only `content`.
Same questions; different output formats."""))

cells.append(code("""# Load the dataset.
from datasets import load_dataset

ds = load_dataset(
    "OpenMed/Medical-Reasoning-SFT-GPT-OSS-120B-V2",
    split="train",
    token=os.environ["HF_TOKEN"] or None,
)
print(f"Total rows: {len(ds):,}")
print(f"Columns: {ds.column_names}")"""))

cells.append(code("""# Peek at the first row's structure.
sample = ds[0]
print("Number of messages:", len(sample["messages"]))
for i, m in enumerate(sample["messages"]):
    print(f"\\n--- message[{i}]  role={m['role']} ---")
    content = (m.get("content") or "")[:300]
    print("content       :", content, "..." if content else "")
    if m["role"] == "assistant":
        reasoning = (m.get("reasoning_content") or "")[:300]
        print("reasoning_content:", reasoning, "..." if reasoning else "")"""))

cells.append(code("""# Length statistics — sanity-check that our 150-token short-CoT cap
# actually truncates anything (i.e. that most reasoning blocks are >150 tok).
import numpy as np
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")

# Sample 200 rows for speed (full corpus length stats would take a few minutes).
sample_indices = np.random.RandomState(42).choice(len(ds), size=200, replace=False)

reasoning_lens, answer_lens = [], []
for i in sample_indices:
    row = ds[int(i)]
    asst = next(m for m in row["messages"] if m["role"] == "assistant")
    reasoning_lens.append(len(tok.encode(asst.get("reasoning_content") or "",
                                          add_special_tokens=False)))
    answer_lens.append(len(tok.encode(asst.get("content") or "",
                                       add_special_tokens=False)))

print("reasoning_content tokens:")
print(f"   mean   {np.mean(reasoning_lens):.0f}")
print(f"   median {np.median(reasoning_lens):.0f}")
print(f"   p95    {np.percentile(reasoning_lens, 95):.0f}")
print(f"   max    {np.max(reasoning_lens):.0f}")

print("\\ncontent (final answer) tokens:")
print(f"   mean   {np.mean(answer_lens):.0f}")
print(f"   median {np.median(answer_lens):.0f}")
print(f"   p95    {np.percentile(answer_lens, 95):.0f}")
print(f"   max    {np.max(answer_lens):.0f}")

frac_over = np.mean(np.array(reasoning_lens) > 150)
print(f"\\nFraction of rows with reasoning > 150 tokens: {frac_over:.1%}")"""))

# ─────────────────────────────────────────────────────────────────────────────
# Section 4 — formatter sanity check
# ─────────────────────────────────────────────────────────────────────────────
cells.append(md("""## 4. Formatters in action

Before training, we visually verify both formatters produce sensible output on a real row. We pick one with a long reasoning so we can see truncation actually trigger."""))

cells.append(code("""# Pick a row whose reasoning is > 200 tokens, so 150-token truncation matters.
long_idx = None
rl = None
for idx in sample_indices:
    row = ds[int(idx)]
    asst = next(m for m in row["messages"] if m["role"] == "assistant")
    rl = len(tok.encode(asst.get("reasoning_content") or "",
                         add_special_tokens=False))
    if rl > 200:
        long_idx = int(idx)
        break

assert long_idx is not None, "No long-reasoning row found in sample — increase sample size"
print(f"Using row {long_idx} (reasoning = {rl} tokens)")"""))

cells.append(code("""# Apply the Track A formatter.
from src.data_formatting import format_for_track_a, format_for_track_b

row = ds[long_idx]

out_a = format_for_track_a(row, tok, short_cot_max_tokens=150)
asst_a = out_a["messages"][-1]["content"]

print("=" * 80)
print("TRACK A (Clinical rationale + Final answer, short CoT capped at 150 tokens)")
print("=" * 80)
print(asst_a[:1500])
print("...")
print(f"\\n  -> assistant length (Track A): "
      f"{len(tok.encode(asst_a, add_special_tokens=False))} tokens")"""))

cells.append(code("""# Apply the Track B formatter.
out_b = format_for_track_b(row)
asst_b = out_b["messages"][-1]["content"]

print("=" * 80)
print("TRACK B (final answer only, no reasoning)")
print("=" * 80)
print(asst_b[:1500])
print(f"\\n  -> assistant length (Track B): "
      f"{len(tok.encode(asst_b, add_special_tokens=False))} tokens")"""))

cells.append(code("""# Sanity assertions — these MUST pass before we trust the formatters in training.
assert "Clinical rationale:" in asst_a
assert "Final answer:" in asst_a
assert asst_a.rstrip().endswith(row["messages"][-1]["content"].strip())

assert "Clinical rationale:" not in asst_b
assert "Final answer:" not in asst_b
assert asst_b == row["messages"][-1]["content"].strip()

print("formatter sanity checks passed")"""))

# ─────────────────────────────────────────────────────────────────────────────
# Section 5 — chat template
# ─────────────────────────────────────────────────────────────────────────────
cells.append(md("""## 5. Chat templates

`tokenizer.apply_chat_template(messages, tokenize=False)` turns
`[{"role":"user","content":...}, {"role":"assistant","content":...}]` into a single string the model knows how to parse.
Qwen2.5 uses an `<|im_start|>role\\n…<|im_end|>` wrapping.

> **Mismatch between the chat template at training and inference is the #1 silent fine-tuning bug.** That's why we use the same Unsloth `get_chat_template(tokenizer, "qwen-2.5")` everywhere."""))

cells.append(code("""# Apply the Unsloth-augmented Qwen-2.5 chat template and render both tracks.
from unsloth.chat_templates import get_chat_template

tok_chat = get_chat_template(tok, chat_template="qwen-2.5")

text_a = tok_chat.apply_chat_template(out_a["messages"], tokenize=False,
                                      add_generation_prompt=False)
text_b = tok_chat.apply_chat_template(out_b["messages"], tokenize=False,
                                      add_generation_prompt=False)

print("=" * 80)
print("CHAT-TEMPLATED — TRACK A (first 1200 chars)")
print("=" * 80)
print(text_a[:1200])
print()
print("=" * 80)
print("CHAT-TEMPLATED — TRACK B (first 1200 chars)")
print("=" * 80)
print(text_b[:1200])"""))

# ─────────────────────────────────────────────────────────────────────────────
# Section 6 — memory math (markdown only)
# ─────────────────────────────────────────────────────────────────────────────
cells.append(md("""## 6. Memory math — why 4-bit?

Qwen2.5-1.5B has ≈1.54 B parameters. Memory cost depends on the dtype:

| Dtype | Bytes/param | Total weights memory |
|---|---|---|
| fp32 | 4 | ~6.2 GB |
| fp16 / bf16 | 2 | ~3.1 GB |
| int8 | 1 | ~1.5 GB |
| int4 (NF4) | 0.5 | ~0.78 GB |

But that's just **weights**. Training also needs activations, gradients, and optimizer state. Adam's state is roughly 2× weights in fp32 → another ~12 GB on top of weights for full fine-tuning. On a 16 GB T4 → fp16 fine-tuning is infeasible.

**QLoRA's trick**: keep base weights frozen at 4-bit (~0.78 GB), train only ~18 M LoRA parameters on top (~70 MB in fp16). Adam state is on the LoRA weights, not the base. Total VRAM expected: ~5–8 GB during training. Plenty of room on a T4.

This is why the assignment uses QLoRA on a small GPU — it makes a fine-tune *possible* in our environment, not just *easier*."""))

# ─────────────────────────────────────────────────────────────────────────────
# Section 7 — load model
# ─────────────────────────────────────────────────────────────────────────────
cells.append(md("""## 7. Load `unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit`

We load the model in 4-bit and check VRAM. Expected: ~1–1.5 GB allocated for the 4-bit base on a fresh T4."""))

cells.append(code("""import torch
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name      = "unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit",
    max_seq_length  = 2048,
    dtype           = None,        # auto-detect fp16/bf16
    load_in_4bit    = True,
)
tokenizer = get_chat_template(tokenizer, chat_template="qwen-2.5")
print("Model loaded.")"""))

cells.append(code("""def _vram_used_gb():
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.memory_allocated() / 1e9

print(f"Allocated VRAM after model load: {_vram_used_gb():.2f} GB")
print(f"Total GPU VRAM:                  "
      f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
print(f"Device:                          {torch.cuda.get_device_name(0)}")"""))

# ─────────────────────────────────────────────────────────────────────────────
# Section 8 — inference smoke test
# ─────────────────────────────────────────────────────────────────────────────
cells.append(md("""## 8. Smoke test: ask the *base* model a question

We're testing only that generation works end-to-end **before** any fine-tuning. The base model has not seen our medical-reasoning training data yet, so the answer may be generic. Inference should take 3–10 seconds on a T4.

> If this fails with an OOM, restart the kernel (Run -> Restart) and re-run from Section 1. Usually it's leftover state from a previous Kaggle session."""))

cells.append(code("""import time

FastLanguageModel.for_inference(model)   # ~2x faster generation

prompt_messages = [
    {"role": "user",
     "content": "What are the first-line treatments for hypertension in a 55-year-old non-diabetic patient?"}
]
prompt_text = tokenizer.apply_chat_template(
    prompt_messages, tokenize=False, add_generation_prompt=True,
)
inputs = tokenizer(prompt_text, return_tensors="pt").to("cuda")

start = time.time()
with torch.no_grad():
    out_ids = model.generate(
        **inputs,
        max_new_tokens=400,
        do_sample=False,
        temperature=0.0,
    )
elapsed = time.time() - start

prompt_len = inputs["input_ids"].shape[1]
new_ids    = out_ids[0, prompt_len:]
response   = tokenizer.decode(new_ids, skip_special_tokens=True)

print(f"Generated {len(new_ids)} new tokens in {elapsed:.1f}s "
      f"({len(new_ids)/elapsed:.1f} tok/s)")
print("=" * 80)
print(response)"""))

# ─────────────────────────────────────────────────────────────────────────────
# Section 9 — Day-1 takeaways
# ─────────────────────────────────────────────────────────────────────────────
cells.append(md("""## 9. Day-1 takeaways (fill in your own numbers)

- Dataset has **<NN>** rows. Median reasoning is **<NN>** tokens; **<NN>%** exceed our 150-token cap.
- Model loads in **<X.X> GB** VRAM in 4-bit (out of 16 GB on T4).
- Base-model inference works end-to-end; baseline tok/s ≈ **<NN>**.
- Track A formatter produces `Clinical rationale: … Final answer: …` outputs as expected.
- Track B formatter produces answer-only outputs as expected.

**Tomorrow (Day 2):** train Track B (the simpler ablation) end-to-end.

Replace the `<NN>` placeholders with your actual numbers from the cells above before downloading + committing the notebook."""))

# ─────────────────────────────────────────────────────────────────────────────
# Build notebook
# ─────────────────────────────────────────────────────────────────────────────
nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.11",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

NB_PATH.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"Wrote {NB_PATH}  ({len(cells)} cells)")
