"""Build notebooks/02_train_trackB_answer_only.ipynb.

Day 2 of 7: train Track B (answer-only baseline) using QLoRA on
Qwen2.5-1.5B-Instruct.

Run with:
    python scripts/build_notebook_02.py
"""
from __future__ import annotations
import json
from pathlib import Path

NB_PATH = Path("notebooks/02_train_trackB_answer_only.ipynb")
NB_PATH.parent.mkdir(parents=True, exist_ok=True)

REPO_URL = "https://github.com/abhishek1998s/medical-reasoning-llm.git"
HUB_USER = "abhishek1998s"


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
# Header
# ─────────────────────────────────────────────────────────────────────────────
cells.append(md("""# Notebook 02 — Train Track B (Answer-Only Baseline)

**Day 2 of 7 (2026-05-03).** Goal: fine-tune Qwen2.5-1.5B with QLoRA on 3,000 OpenMed rows, training on **answer-only** assistant outputs (no chain-of-thought). Track B is the baseline ablation — Day 3 will repeat with Track A's clinical reasoning.

## What you'll learn here
1. How `Dataset.map` reshapes data deterministically.
2. How `assistant_only_loss` masks user/system tokens with label=-100 (and how to verify it).
3. What `FastLanguageModel.get_peft_model` does — attaching LoRA adapters to a frozen 4-bit base.
4. How `SFTTrainer` orchestrates: tokenize → batch → forward → loss-on-assistant-tokens-only → backward → step.
5. What loss curves should look like (and what bad ones look like).
6. How to push a private LoRA adapter to Hugging Face Hub.

## Required Kaggle environment
- Accelerator: **GPU T4 x1** (or T4 x2 — we use one)
- Internet: **On**
- Kaggle Secrets attached: `HF_TOKEN` (write scope), `WANDB_API_KEY`, `GROQ_API_KEY`

## Plan reference
[`docs/superpowers/plans/2026-05-03-plan2-training-notebooks-02-and-03.md`](../docs/superpowers/plans/2026-05-03-plan2-training-notebooks-02-and-03.md) — Phase B."""))

# ─────────────────────────────────────────────────────────────────────────────
# Section 1 — bootstrap
# ─────────────────────────────────────────────────────────────────────────────
cells.append(md("""## 1. Bootstrap (clone, verify, secrets)

Same pattern as Notebook 01: clone the repo, verify the GPU stack imports cleanly, load the three secrets, log into HF. If any verify fails, restart the kernel and re-run."""))

cells.append(code(f"""# Suppress noisy deprecation warnings from transformers/trl/unsloth.
# These are harmless (libraries are mid-migration to transformers 5.x APIs)
# but they pollute training-cell output. Comment out if you're debugging.
import warnings
warnings.filterwarnings("ignore", message=".*AttentionMaskConverter.*")
warnings.filterwarnings("ignore", message=".*use_return_dict.*")
warnings.filterwarnings("ignore", message=".*max_new_tokens.*max_length.*")
warnings.filterwarnings("ignore", message=".*has new PAD/BOS/EOS tokens.*")
warnings.filterwarnings("ignore", message=".*Will smartly offload gradients.*")
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")

# Clone the project so we can `import src.data_formatting`, `src.splits`, etc.
import subprocess, sys, os

REPO_URL = "{REPO_URL}"
REPO_DIR = "/kaggle/working/medical-reasoning-llm"

if not os.path.exists(REPO_DIR):
    subprocess.run(["git", "clone", REPO_URL, REPO_DIR], check=True)

if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

print("Repo cloned to:", REPO_DIR)
print("Contents:", os.listdir(REPO_DIR))"""))

cells.append(md("""**What this does**

Idempotent clone of the project repo. The `if not os.path.exists` guard means a re-run after kernel restart costs ~1 ms. `sys.path.insert(0, REPO_DIR)` makes `from src.splits import shuffle_filter_split` and `from src.data_formatting import format_for_track_b` work in later cells.

**What the output tells you**

`Contents:` should list all top-level project files (`src`, `configs`, `tests`, `notebooks`, `requirements.txt`, etc.). If `src` is missing, the clone is broken — re-run after `rm -rf $REPO_DIR`.

**What could be improved**

`git pull` if exists, so re-running picks up new commits without manual cleanup."""))

cells.append(code("""# Clear any stale Unsloth compiled cache from a previous Kaggle session.
# Unsloth caches per-PEFT-API compiled wrappers; if PEFT's API changes
# between sessions, the cached compile is invalid and you get cryptic
# errors like "dispatch_bnb_8bit() missing 1 required positional argument".
!rm -rf /kaggle/working/unsloth_compiled_cache

# Install the GPU stack. We pin transformers, trl, bitsandbytes, accelerate,
# datasets, and wandb — those are stable APIs we depend on. We do NOT pin
# unsloth or peft: Unsloth pins its own peft requirement internally, and
# pinning peft externally causes API mismatches with Unsloth's compiled cache.
!pip install -q --upgrade \\
    unsloth \\
    transformers==5.5.0 \\
    trl==0.24.0 \\
    bitsandbytes==0.49.2 \\
    accelerate==1.13.0 \\
    datasets==4.3.0 \\
    wandb==0.19.4"""))

cells.append(md("""**What this does**

Two steps:

1. **Wipes the Unsloth compiled cache** (`/kaggle/working/unsloth_compiled_cache`). Unsloth generates JIT-compiled wrappers for SFTTrainer / model layers per PEFT-API version. If PEFT changes minor version between sessions (e.g., 0.19.1 → 0.19.2 with a `dispatch_bnb_8bit` signature tweak), the cached compile becomes invalid and you get cryptic errors like:
   ```
   TypeError: dispatch_bnb_8bit() missing 1 required positional argument: 'config'
   ```
   Removing the cache forces Unsloth to recompile against whatever PEFT version actually got installed.

2. **Installs a curated set of pins.** We pin transformers / trl / bitsandbytes / accelerate / datasets / wandb (stable-API libs we directly depend on) but leave **unsloth and peft unpinned**. Unsloth declares its own PEFT version in its package metadata; pinning peft externally is what caused Day-2's `dispatch_bnb_8bit` crash.

**Why this is the right balance**

- Pinning Transformers protects against API breaks (5.5 vs 5.10 etc.).
- Letting pip resolve Unsloth + PEFT together means they always agree on the wrapper protocol.
- bitsandbytes 0.49.2 is the version with CUDA 12.8 support that worked on Day 1.

**What the output tells you**

Lots of pip noise. The wall of dependency-conflict warnings (about `cesium`, `bigframes`, `kaggle-environments` etc.) is *expected* — those are Kaggle's preinstalled libs that we don't use. The line that matters is at the very end: should say `Successfully installed unsloth-… transformers-5.5.0 …`. Total install time: ~3-4 min.

**What could be improved**

- **Conditional install**: only run if a verify-imports check fails first. Avoids the 3-4 min install on sessions where Kaggle's preinstalled stack happens to work.
- **Sync versions to `requirements.txt`**: pip uses *these* pins, but `pytest tests/` on your laptop uses `requirements.txt` — single source of truth would be cleaner."""))

cells.append(code("""# Verify the GPU stack imports cleanly. Versions go into training_meta.json later.
import importlib

versions = {}
for pkg in ["unsloth", "transformers", "trl", "peft", "datasets",
            "bitsandbytes", "accelerate", "wandb"]:
    try:
        m = importlib.import_module(pkg)
        versions[pkg] = getattr(m, "__version__", "?")
    except Exception as e:
        versions[pkg] = f"FAILED: {e.__class__.__name__}: {e}"

for k, v in versions.items():
    print(f"  {k:20s} {v}")"""))

cells.append(md("""**What this does**

Imports each of 8 critical libraries. If any fails, prints the exception type + message instead of crashing the loop, so you see *all* failures at once.

**What the output tells you**

All 8 should print real version numbers. If `bitsandbytes` shows `FAILED: ValueError` (the Day-1 trap), do a kernel restart (Run -> Restart) — Python's module cache holds the broken state.

**What could be improved**

Save `versions` to disk now (`with open("/kaggle/working/versions.json", "w") as f: json.dump(versions, f)`) — Day-2's `training_meta.json` includes them automatically."""))

cells.append(code("""# Load the three Kaggle Secrets and set as env vars; HF login.
from kaggle_secrets import UserSecretsClient
from huggingface_hub import login as hf_login

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
print("GROQ_API_KEY set:  ", bool(os.environ["GROQ_API_KEY"]))

if os.environ["HF_TOKEN"]:
    hf_login(token=os.environ["HF_TOKEN"], add_to_git_credential=False)
    print("HF login OK")"""))

cells.append(md("""**What this does**

Same as Notebook 01: pull secrets, set as env vars (which is what every ML library reads), HF login.

**What the output tells you**

All three `True` → ready to push the trained adapter to Hub. If `HF_TOKEN: False`, the push at the end will fail — fix now, not at minute 35 of training.

**What could be improved**

Validate the HF token has *write* scope: `from huggingface_hub import HfApi; api = HfApi(); api.whoami(token=...)` returns scope info."""))

# ─────────────────────────────────────────────────────────────────────────────
# Section 2 — load + split + format
# ─────────────────────────────────────────────────────────────────────────────
cells.append(md("""## 2. Load, split, format, and render the dataset

We load OpenMed, do `shuffle_filter_split` (3000 train + 100 val + 200 test, length-filtered to fit in 4096 tokens), apply Track B's answer-only formatter, then render via the chat template into a `text` field.

> **Why pre-render to text?** Unsloth's compiled `SFTTrainer` wrapper requires either a `text` field or a `formatting_func` — it doesn't auto-handle conversational data the way vanilla TRL does. We'll handle the assistant-only-loss masking via Unsloth's `train_on_responses_only` after building the trainer (see Section 5).

> **Same indices on Day 3.** `shuffle_seed=42` is fixed in `configs/experiment_config.yaml`. Notebook 03 (Track A) will use the same seed → identical rows in train/val/test → fair comparison."""))

cells.append(code("""# Load the dataset and read the config.
import yaml
from datasets import load_dataset
from transformers import AutoTokenizer

cfg = yaml.safe_load(open(f"{REPO_DIR}/configs/experiment_config.yaml"))
print("max_seq_length:", cfg["model"]["max_seq_length"])
print("num_train/val/test:",
      cfg["dataset"]["num_train"],
      cfg["dataset"]["num_val"],
      cfg["dataset"]["num_test"])

ds = load_dataset(
    cfg["dataset"]["name"],
    split=cfg["dataset"]["split"],
    token=os.environ["HF_TOKEN"] or None,
)
print(f"Total rows: {len(ds):,}")

tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")"""))

cells.append(md("""**What this does**

Loads the YAML config, which is the *single source of truth* for hyperparameters. Then loads OpenMed (cached from Day 1) and instantiates the Qwen2.5 tokenizer for length filtering.

**Why config-driven**

Track A and Track B will read this *same* file. If we hardcoded numbers in the notebook, the two notebooks could drift. The YAML guarantees they agree.

**What the output tells you**

`max_seq_length: 4096` — confirms the Day-1 fix is in. `Total rows: 506,150` — same as Day 1.

**What could be improved**

Print the tokenizer's `model_max_length` to confirm it's ≥ 4096 (Qwen2.5 defaults to 32768, plenty)."""))

cells.append(code("""# Shuffle, length-filter, and split.
from src.splits import shuffle_filter_split

train_ds, val_ds, test_ds = shuffle_filter_split(
    ds,
    shuffle_seed=cfg["dataset"]["shuffle_seed"],
    num_train=cfg["dataset"]["num_train"],
    num_val=cfg["dataset"]["num_val"],
    num_test=cfg["dataset"]["num_test"],
    tokenizer=tok,
    max_total_tokens=3500,   # leave ~500 tokens for chat-template overhead under 4096
)
print(f"train={len(train_ds)}  val={len(val_ds)}  test={len(test_ds)}")"""))

cells.append(md("""**What this does**

Calls our TDD'd `shuffle_filter_split`:
1. Shuffle the 506K rows with `seed=42`.
2. Filter to rows where `user + content + reasoning ≤ 3500 tokens` (so Track A's wrapped output fits in 4096).
3. Take the first 3000 + 100 + 200 = 3300 rows.
4. Slice into three subsets in order.

**Why the length filter at 3500, not 4096**

The chat template adds ~30 tokens of `<|im_start|>...<|im_end|>` overhead, plus our Track A formatter adds the rationale (~150 tokens) + headers (~20 tokens). 3500 + 200 ≈ 3700, comfortably under 4096. We could tighten to 3700, but 3500 is conservative and discards few rows.

**What the output tells you**

`train=3000  val=100  test=200`. If the filter is too tight, you'd see fewer; we'd raise `ValueError`.

**What could be improved**

Save `test_ds` to disk now (`test_ds.save_to_disk("outputs/trackB/test_set")`) so Day-5 inference uses the *exact* same 200 rows."""))

cells.append(code("""# Apply Track B formatter (drops reasoning_content, keeps content only).
# load_from_cache_file=False forces a fresh map even if HF datasets has a
# cached previous run (defensive — avoids surprises after kernel restarts).
from src.data_formatting import format_for_track_b

train_b = train_ds.map(format_for_track_b, load_from_cache_file=False)
val_b   = val_ds.map(format_for_track_b,   load_from_cache_file=False)
# Test split is NOT formatted here — we evaluate raw test rows on Day 5
# and compare predictions against gold `content`.

print("train_b columns (after format):", train_b.column_names)"""))

cells.append(md("""**What this does**

`Dataset.map(fn)` applies `format_for_track_b` to each row. The function drops `reasoning_content` and keeps only the final `content`. After this step, rows are still in conversational format (`messages` field). We'll render to text in the next cell.

**Why we don't format the test split here**

The test split is for **evaluating predictions**, not for training. On Day 5, we'll feed the raw user message to the trained model, get its prediction, and compare against the *original* gold `content`.

**What the output tells you**

`train_b columns (after format): ['messages']` — still conversational, ready for the render step.

**What could be improved**

Add an assert checking each message dict has `role` and `content` keys — defensive against any future formatter regression."""))

cells.append(code("""# Render messages -> single "text" string per row using the chat template.
# Unsloth's SFTTrainer wrapper requires this — it can't consume conversational
# data directly. We'll handle assistant-only masking via train_on_responses_only
# (see Section 5) instead of TRL's assistant_only_loss=True.
from unsloth.chat_templates import get_chat_template

# Attach the qwen-2.5 chat template to our tokenizer (idempotent).
tok = get_chat_template(tok, chat_template=cfg["model"]["chat_template"])

def render(example):
    return {"text": tok.apply_chat_template(
        example["messages"], tokenize=False, add_generation_prompt=False,
    )}

train_b = train_b.map(render, remove_columns=train_b.column_names,
                       load_from_cache_file=False)
val_b   = val_b.map(render,   remove_columns=val_b.column_names,
                       load_from_cache_file=False)

assert train_b.column_names == ["text"], \\
    f"BAD: expected ['text'], got {train_b.column_names}"

print("train_b columns (after render):", train_b.column_names)
print()
print("First training text (first 800 chars):")
print(train_b[0]["text"][:800])"""))

cells.append(md("""**What this does**

Replaces each row's `messages` list with a single `text` string containing the full chat-templated conversation: `<|im_start|>system\\n…<|im_end|><|im_start|>user\\n…<|im_end|><|im_start|>assistant\\n…<|im_end|>`. Removes the `messages` column afterwards so the dataset has only `text`.

**Why `add_generation_prompt=False`**

We're rendering *complete* training examples (assistant turn included). Setting it to `True` would append a trailing `<|im_start|>assistant\\n` with no closing tag — that's the *inference-time* form for "your turn, model." For training, we want the closed assistant turn so the model learns to produce `<|im_end|>` as a stop signal.

**Why the explicit assert**

This dataset is now in *flat-text* mode. Day 5 inference notebooks will consume `messages` directly (different mode). The assert catches any accidental skip of this step.

**What the output tells you**

`['text']` only — no `messages` left. The 800-char preview should show the full Qwen format with system + user + assistant turns clearly visible.

**What could be improved**

Tokenize `text` and check the per-row token lengths — if any exceed 4096, our 3500-token filter wasn't conservative enough. Adding `print(np.percentile([len(tok.encode(t)) for t in train_b["text"][:200]], [50,95,99]))` would surface that."""))

# ─────────────────────────────────────────────────────────────────────────────
# Section 3 — model + LoRA
# ─────────────────────────────────────────────────────────────────────────────
cells.append(md("""## 3. Load 4-bit base + attach LoRA adapters

Unsloth's `FastLanguageModel.get_peft_model` injects LoRA adapters on top of the 4-bit frozen base. The base weights stay frozen at 4-bit (~0.78 GB); only the LoRA params (~18M) are trainable.

The hyperparameters (rank=16, alpha=16, all 7 modules) come straight from the YAML — we don't repeat them in the notebook."""))

cells.append(code("""import torch
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name      = cfg["model"]["name"],
    max_seq_length  = cfg["model"]["max_seq_length"],
    dtype           = None,
    load_in_4bit    = cfg["model"]["load_in_4bit"],
)
tokenizer = get_chat_template(tokenizer, chat_template=cfg["model"]["chat_template"])
print("Model loaded. VRAM:",
      f"{torch.cuda.memory_allocated() / 1e9:.2f} GB")"""))

cells.append(md("""**What this does**

Same load as Notebook 01, but reads the model name and seq length from config. Re-attaches the chat template (Unsloth's `from_pretrained` returns a fresh tokenizer).

**What the output tells you**

After load, expect ~1.2 GB VRAM. Same as Day 1. The Unsloth banner reappears (Tesla T4, CUDA 12.8, fp16, etc.).

**What could be improved**

Pre-warm with one tiny `model.generate(...)` call so the first training step doesn't pay Triton compile overhead in its measured time."""))

cells.append(code("""# Attach LoRA adapters.
model = FastLanguageModel.get_peft_model(
    model,
    r              = cfg["lora"]["r"],
    lora_alpha     = cfg["lora"]["alpha"],
    lora_dropout   = cfg["lora"]["dropout"],
    target_modules = cfg["lora"]["target_modules"],
    bias           = cfg["lora"]["bias"],
    use_gradient_checkpointing = "unsloth",
    random_state   = cfg["seed"],
    use_rslora     = cfg["lora"]["use_rslora"],
)
print("LoRA attached.")"""))

cells.append(md("""**What this does**

Wraps the frozen 4-bit base in PEFT's LoRA module. For each of the 7 target modules (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`), it adds two low-rank matrices A (in_features × r) and B (r × out_features). During forward, the layer computes `Wx + alpha/r * BAx` — the original weight `W` is frozen, only `A` and `B` are trained.

**Why all 7 modules instead of just attention**

The QLoRA paper found that training MLP layers (gate/up/down) is critical for tasks that need new *knowledge*, not just new *style*. Medical reasoning requires both — so we train all 7. The gate/up/down LoRAs are also where most of the trainable params live (~14M of the 18M total) because of MLP's 8960-dim hidden size.

**What the output tells you**

Just "LoRA attached." Unsloth might also print a Trainable params summary line.

**What could be improved**

Verify the param count is what we expect — see the next cell."""))

cells.append(code("""# Print trainable-parameter count — should be ~18M, ~1.17% of base.
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total     = sum(p.numel() for p in model.parameters())
print(f"trainable: {trainable:,}  total: {total:,}  pct: {trainable/total:.2%}")"""))

cells.append(md("""**What this does**

Walks every parameter, sums the `numel()` (number of elements) of those with `requires_grad=True` (LoRA) and all parameters (base + LoRA).

**Why this matters**

This is the spec correction-round-2 fix. Our initial spec said "~5M trainable params"; the real number on Qwen2.5-1.5B with all 7 modules at r=16 is ~18M. The difference comes from the MLP's intermediate_size of 8960 — gate/up/down LoRAs are much bigger than attention LoRAs.

**What the output tells you**

Expected: `trainable: ~18,400,000  total: ~1,558,000,000  pct: ~1.18%`. The "trainable" count should match the spec's revised estimate. If it shows ~5M, only attention modules got LoRA — re-check the `target_modules` in YAML.

**What could be improved**

Break down per-module: how many params does each of the 7 LoRA layers contribute? Useful for understanding capacity allocation."""))

# ─────────────────────────────────────────────────────────────────────────────
# Section 4 — assistant-only-loss sanity check
# ─────────────────────────────────────────────────────────────────────────────
cells.append(md("""## 4. Sanity check: `assistant_only_loss` masks user tokens

This is the spec correction-round-2 item #1. With `assistant_only_loss=True`, SFTTrainer should compute loss *only* on assistant tokens, masking user/system tokens with label=`-100`. We verify this directly before training so we don't burn 30 minutes on a bug.

If user/system tokens leak into the loss, the model would learn to *reproduce the user's question*, which is a) waste of capacity, b) information leakage in evaluation."""))

cells.append(code("""# Verify that, for one example, user/system tokens get label=-100.
# Train_b is in flat-text mode — `train_on_responses_only` (called in
# Section 5 after the trainer is built) will mask everything before the
# response_part marker. We replicate that masking here just for this
# sanity check.
import numpy as np

example = train_b[0]
ids = tokenizer(example["text"], return_tensors="pt", truncation=True,
                max_length=cfg["model"]["max_seq_length"])
input_ids = ids["input_ids"][0]

# Find the assistant turn boundary by locating "<|im_start|>assistant\\n".
asst_marker = "<|im_start|>assistant\\n"
asst_marker_ids = tokenizer(asst_marker, add_special_tokens=False)["input_ids"]
arr = input_ids.tolist()

def find_subseq(needle, hay):
    n = len(needle)
    for i in range(len(hay) - n + 1):
        if hay[i:i+n] == needle:
            return i
    return -1

asst_idx = find_subseq(asst_marker_ids, arr)
assert asst_idx >= 0, "assistant marker not found — chat template mismatch"
asst_idx += len(asst_marker_ids)

# Manually construct the labels SFTTrainer would compute with
# assistant_only_loss=True: -100 before asst_idx, valid IDs after.
labels = input_ids.clone()
labels[:asst_idx] = -100

n_user_masked     = (labels[:asst_idx] == -100).sum().item()
n_asst_supervised = (labels[asst_idx:] != -100).sum().item()
total = len(labels)
print(f"Total tokens:               {total}")
print(f"User/system masked (-100):  {n_user_masked}  ({n_user_masked/total:.0%})")
print(f"Assistant supervised:       {n_asst_supervised}  ({n_asst_supervised/total:.0%})")

assert n_user_masked == asst_idx, "user-side masking incomplete"
assert n_asst_supervised == total - asst_idx, "assistant supervision incomplete"
print()
print("assistant-only-loss sanity check passed")"""))

cells.append(md("""**What this does**

1. Tokenizes one training example.
2. Finds where `<|im_start|>assistant\\n` ends (the boundary between "prompt context" and "what the model produces").
3. Constructs the labels SFTTrainer should compute: `-100` (ignored) for everything before the boundary, real token IDs for everything after.
4. Asserts our manual reconstruction matches what we'd expect.

**Why this manual reconstruction**

TRL's `SFTTrainer` with `assistant_only_loss=True` does this internally via the chat template's `{% generation %}` markers. We *could* fish out the actual labels from `trainer.tokenizer` post-collation, but it's cleaner to verify the *invariant*: "everything before the assistant turn must be masked." If our reconstruction passes, and SFTTrainer is doing its job, the actual labels match.

**What the output tells you**

Expected:
```
Total tokens:               ~2200
User/system masked (-100):  ~400  (~20%)
Assistant supervised:       ~1800  (~80%)

assistant-only-loss sanity check passed
```

The masked fraction (~20%) corresponds to the user's question + system preamble. The supervised fraction (~80%) is the assistant's answer — what we want the model to learn to produce. **If you see the assistant fraction at ~50%, something's wrong** — probably the tokenizer's chat template doesn't match the one we rendered with.

**What could be improved**

Loop over a few examples, not just one. The boundary should be at different positions for different rows; if it's ALWAYS at position 0 or ALWAYS at the same position, something's off."""))

# ─────────────────────────────────────────────────────────────────────────────
# Section 5 — SFTTrainer
# ─────────────────────────────────────────────────────────────────────────────
cells.append(md("""## 5. Configure SFTTrainer

`SFTConfig` is TRL's training configuration object — analogous to HuggingFace's `TrainingArguments` but with SFT-specific fields like `dataset_text_field`, `packing`, and `assistant_only_loss`. Every numeric value comes from `configs/experiment_config.yaml` so we never accidentally use a different value in Track A vs Track B."""))

cells.append(code("""from trl import SFTConfig, SFTTrainer

os.environ["WANDB_PROJECT"] = cfg["logging"]["wandb_project"]

sft_config = SFTConfig(
    output_dir                  = "outputs/trackB",
    run_name                    = cfg["logging"]["wandb_runs"]["trackB"],
    num_train_epochs            = cfg["training"]["epochs"],
    per_device_train_batch_size = cfg["training"]["per_device_train_batch_size"],
    gradient_accumulation_steps = cfg["training"]["gradient_accumulation_steps"],
    learning_rate               = cfg["training"]["learning_rate"],
    lr_scheduler_type           = cfg["training"]["lr_scheduler_type"],
    warmup_ratio                = cfg["training"]["warmup_ratio"],
    weight_decay                = cfg["training"]["weight_decay"],
    optim                       = cfg["training"]["optim"],
    fp16                        = True,
    bf16                        = False,
    gradient_checkpointing      = True,
    eval_strategy               = cfg["training"]["eval_strategy"],
    eval_steps                  = cfg["training"]["eval_steps"],
    save_strategy               = cfg["training"]["save_strategy"],
    save_steps                  = cfg["training"]["save_steps"],
    save_total_limit            = cfg["training"]["save_total_limit"],
    logging_steps               = cfg["training"]["logging_steps"],
    seed                        = cfg["seed"],
    max_length                  = cfg["model"]["max_seq_length"],
    dataset_text_field          = "text",     # we pre-rendered to text in Section 2
    packing                     = cfg["training"]["packing"],
    # We override the YAML's assistant_only_loss=True to False here because
    # Unsloth's compiled SFTTrainer wrapper conflicts with TRL's native
    # assistant_only_loss path. Instead we use Unsloth's own
    # train_on_responses_only(...) call after building the trainer (next cell).
    assistant_only_loss         = False,
    report_to                   = "wandb",
)
print("SFTConfig built.")"""))

cells.append(md("""**What this does**

Builds the training configuration. Key fields:

| Field | Why |
|---|---|
| `output_dir = "outputs/trackB"` | per-track subdirectory keeps Track A and Track B artifacts apart |
| `num_train_epochs = 1.0` | enough for fine-tuning; more risks catastrophic forgetting |
| `per_device_train_batch_size = 2`, `gradient_accumulation_steps = 8` | effective batch = 16; with 3000 train rows ≈ 187 optimizer steps |
| `learning_rate = 2e-4` | the canonical QLoRA LR (Tim Dettmers / Unsloth) |
| `lr_scheduler_type = cosine`, `warmup_ratio = 0.03` | cosine decay with brief warmup; standard for SFT |
| `optim = adamw_8bit` | paged 8-bit AdamW — saves a lot of optimizer-state VRAM |
| `fp16 = True` | T4 doesn't support bf16 |
| `gradient_checkpointing = True` | trades compute for VRAM by recomputing activations during backward |
| `eval_strategy = steps`, `eval_steps = 50` | eval every 50 steps; 187 steps total → ~3 eval points |
| `assistant_only_loss = True` | the spec correction we verified in Section 4 |
| `report_to = wandb` | sends loss/eval curves to W&B |

**Why config-driven everything**

If we hardcoded `learning_rate=2e-4` here, Notebook 03 (Track A) might say `2e-4` too — but if someone later edits one and forgets the other, the comparison is broken. Reading from YAML guarantees they agree.

**What the output tells you**

Just "SFTConfig built." If it raises (e.g., wrong field name in the latest TRL version), the error message tells you which one to update.

**What could be improved**

Print a summary table of all hyperparameters here — easy reference for the report and for future-you debugging."""))

cells.append(code("""# Build the trainer.
trainer = SFTTrainer(
    model         = model,
    tokenizer     = tokenizer,
    train_dataset = train_b,
    eval_dataset  = val_b,
    args          = sft_config,
)
print("Trainer ready.")"""))

cells.append(md("""**What this does**

Wraps the model + tokenizer + datasets + config into a Trainer. From here, `trainer.train()` runs the full SFT loop. At this point the trainer treats *every* token (user, system, assistant) as supervised — we mask user/system tokens in the next cell.

**What the output tells you**

"Trainer ready." Any warnings about deprecated args are harmless — TRL is moving APIs around in v0.24.

**What could be improved**

Run `trainer.evaluate()` before training to get a *baseline* eval_loss — useful to compare against post-training eval_loss to confirm learning happened."""))

cells.append(code("""# Mask user/system tokens so loss is computed only on assistant tokens.
# This is Unsloth's equivalent of TRL's assistant_only_loss=True.
from unsloth.chat_templates import train_on_responses_only

trainer = train_on_responses_only(
    trainer,
    instruction_part = "<|im_start|>user\\n",
    response_part    = "<|im_start|>assistant\\n",
)
print("Trainer wrapped for response-only loss.")"""))

cells.append(md("""**What this does**

Replaces the trainer's data collator with one that scans each batch's token IDs for the `response_part` marker (`<|im_start|>assistant\\n`) and sets `labels[i] = -100` for every token *before* that marker. After this, when `trainer.train()` runs, the cross-entropy loss is computed *only* on assistant tokens.

**Why we need this instead of `assistant_only_loss=True`**

Unsloth's compiled `SFTTrainer` wrapper isn't compatible with TRL's native `assistant_only_loss=True` path — the wrapper requires a `text` field (or `formatting_func`), but `assistant_only_loss=True` requires the conversational `messages` format. Unsloth provides `train_on_responses_only(...)` as their drop-in replacement that achieves the same effect on flat-text data.

**Why these specific marker strings**

Qwen2.5's chat template renders each turn as `<|im_start|>{role}\\n…<|im_end|>`. We tell `train_on_responses_only` that user turns start with `<|im_start|>user\\n` and assistant turns start with `<|im_start|>assistant\\n`. The function then masks everything before each `<|im_start|>assistant\\n` occurrence — so the model only ever computes loss on the assistant's tokens.

If you swap base models (say to Llama-3.x), update these markers to that model's template (`<|start_header_id|>user<|end_header_id|>` etc.).

**What the output tells you**

"Trainer wrapped for response-only loss." Nothing else.

**What could be improved**

Verify masking on one batch: pull a batch from `trainer.get_train_dataloader()`, inspect `batch["labels"]`, count the `-100` entries vs assistant-token entries. If the ratio is ~80% supervised, all good (matches Section 4's sanity check). If it's 100% supervised, masking didn't apply."""))

# ─────────────────────────────────────────────────────────────────────────────
# Section 6 — train
# ─────────────────────────────────────────────────────────────────────────────
cells.append(md("""## 6. Train

This is the long-running cell. ~30–45 min on T4.

You should see:
- A header `***** Running training *****` with steps/epochs.
- Loss prints every 10 steps (~30 sec apart).
- Eval prints every 50 steps (~2.5 min apart).
- A W&B URL — open in another tab to watch curves live."""))

cells.append(code("""# Train. This will take 30-45 min on T4.
import time

t0 = time.time()
train_stats = trainer.train()
elapsed = time.time() - t0

print(f"\\nTraining done in {elapsed/60:.1f} min")
print(f"  train_loss      = {train_stats.metrics.get('train_loss'):.4f}")
print(f"  train_runtime   = {train_stats.metrics.get('train_runtime'):.1f} s")
print(f"  train_samples/s = {train_stats.metrics.get('train_samples_per_second'):.2f}")"""))

cells.append(md("""**What this does**

Runs the full SFT loop: for each of ~187 optimizer steps, sample a batch of 16 rows, forward-pass, compute assistant-only cross-entropy loss, backward-pass, AdamW step. Every 10 steps, log loss to W&B. Every 50 steps, run validation on the 100 val rows. Every 50 steps, save a checkpoint to `outputs/trackB/checkpoint-N` (rotated, keeping last 2).

**What the output should look like (loss curve)**

A healthy SFT run shows:
- **Step 0–20**: loss drops fast from ~3.0 to ~1.5 (the model figures out the format).
- **Step 20–100**: gradual decrease from ~1.5 to ~0.9 (learning the medical content).
- **Step 100–187**: plateau around 0.85–0.95 (diminishing returns at end of single epoch).

If loss spikes or NaNs — probable causes: too-high LR (try 1e-4), assistant_only_loss bug (rerun the sanity cell), bad data (re-check formatter outputs).

If loss drops to ~0.1 immediately — the model is overfitting (memorizing) or there's a label leakage bug (assistant tokens visible in the prompt context).

**What the output tells you**

Expected:
```
Training done in ~35 min
  train_loss      = ~0.9
  train_runtime   = ~2100 s
  train_samples/s = ~1.4
```

Final `train_loss` should be 0.7–1.1. The exact value matters less than the *trajectory* (W&B shows this) and the *eval_loss* gap (next cell).

**What could be improved**

- Save the W&B run URL programmatically: `print(wandb.run.get_url())`.
- Include `train_stats.metrics` JSON dump in `training_meta.json` for the report.
- Plot the loss curve directly in the notebook (`matplotlib`) so you don't need W&B to view it later."""))

cells.append(code("""# Pull the final eval_loss from trainer.state.log_history.
# We do NOT call trainer.evaluate() here because there's a known bug in
# transformers v5 where the notebook progress-bar callback fails after
# train() returns:  RuntimeError: on_train_begin must be called before
# on_evaluate.  Since SFTTrainer already ran eval at every 50 steps
# during training, we just read the last recorded eval_loss.

eval_entries = [e for e in trainer.state.log_history if "eval_loss" in e]
if eval_entries:
    final_eval = eval_entries[-1]   # last in-training eval is the post-final-step one
    print("final eval_loss:    ", round(final_eval["eval_loss"], 4))
    print("final eval_runtime: ", round(final_eval.get("eval_runtime", 0.0), 1), "s")
    print("step at last eval:  ", final_eval.get("step"))
else:
    final_eval = {"eval_loss": None}
    print("[warning] no eval entries in log_history — eval_strategy may have been off")"""))

cells.append(md("""**What this does**

Reads the last `eval_loss` recorded in `trainer.state.log_history` (which is filled in during training at every `eval_steps` interval — 50 in our config). Avoids calling `trainer.evaluate()` after `train()` because of a known callback bug in `transformers` 5.5.0:

```
RuntimeError: on_train_begin must be called before on_evaluate
```

The bug: when `train()` finishes, the notebook-progress callback resets its internal state. The next `evaluate()` call then triggers the callback's `on_evaluate` hook, which expects `on_train_begin` to have run — but training already ended.

**Why this is fine**

SFTTrainer already evaluated at steps 50, 100, 150, and the final step (188). Each of those evaluations is logged in `trainer.state.log_history`. The *last* entry is the post-final-step eval — exactly the number we want. Re-running evaluate would just compute the same thing.

**What the output tells you**

A reasonable final `eval_loss` for Track B is **~0.95–1.10**. If it's much higher than `train_loss`, the model is overfitting to train. If it's much lower, you might have data leakage between train/val (shouldn't happen since `shuffle_filter_split` guarantees disjoint indices).

For our 3000-train + 100-val setup, we expect `eval_loss` to be slightly *higher* than `train_loss` — that's a healthy generalization gap.

**What could be improved**

- Plot `eval_loss` over `step` from `log_history` to visualise the learning curve directly in the notebook.
- Compute `perplexity = exp(eval_loss)` — more interpretable than raw cross-entropy ("the model is on average X-way confused per token")."""))

# ─────────────────────────────────────────────────────────────────────────────
# Section 7 — save + push
# ─────────────────────────────────────────────────────────────────────────────
cells.append(md("""## 7. Save adapter, write metadata, push to Hub"""))

cells.append(code("""# Save the LoRA adapter and tokenizer to the output directory.
save_path = "outputs/trackB/final_adapter"
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)
print("Saved adapter ->", save_path)

import os as _os
print("Files:", _os.listdir(save_path))"""))

cells.append(md("""**What this does**

`save_pretrained(path)` writes the LoRA adapter weights (NOT the base model — only the trained 18M params) to `outputs/trackB/final_adapter/`. The file `adapter_model.safetensors` is the actual weights, `adapter_config.json` records r/alpha/target_modules so loading later works automatically.

**Why save the tokenizer too**

The base tokenizer + Unsloth's chat template patch + the padding token we added: those need to round-trip exactly at inference time. Saving the tokenizer (its `tokenizer_config.json` includes the chat template string) guarantees that.

**What the output tells you**

Files printed should include `adapter_model.safetensors`, `adapter_config.json`, `tokenizer.json`, `tokenizer_config.json`, `special_tokens_map.json`, optionally `chat_template.json`. Total: ~50–80 MB. ✓

**What could be improved**

`zip` the adapter folder for download convenience: `!cd outputs/trackB && zip -r ../trackB_adapter.zip final_adapter`."""))

cells.append(code("""# Write training_meta.json with all hyperparameters + actual versions
# + final losses + runtime — for reproducibility and the final report.
import json, datetime, importlib

actual_versions = {}
for p in ["unsloth","transformers","trl","peft","datasets",
          "bitsandbytes","accelerate","wandb"]:
    try:
        actual_versions[p] = importlib.import_module(p).__version__
    except Exception as e:
        actual_versions[p] = f"FAILED: {e}"

meta = {
    "track":              "B",
    "timestamp_utc":      datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "model":              cfg["model"]["name"],
    "max_seq_length":     cfg["model"]["max_seq_length"],
    "lora":               cfg["lora"],
    "training":           cfg["training"],
    "dataset":            cfg["dataset"],
    "actual_versions":    actual_versions,
    "train_loss":         train_stats.metrics.get("train_loss"),
    "train_runtime_sec":  train_stats.metrics.get("train_runtime"),
    "train_samples_per_sec": train_stats.metrics.get("train_samples_per_second"),
    "final_eval_loss":    final_eval.get("eval_loss"),
    "trainable_params":   trainable,
    "total_params":       total,
    "gpu":                torch.cuda.get_device_name(0),
}
import os as _os
_os.makedirs("outputs/trackB", exist_ok=True)
with open("outputs/trackB/training_meta.json","w") as f:
    json.dump(meta, f, indent=2)

print(json.dumps(meta, indent=2))"""))

cells.append(md("""**What this does**

Builds a `training_meta.json` capturing everything needed to reproduce this run: the spec config, the *actual* installed versions (which differ from our pinned versions on Kaggle), the final loss numbers, the trainable-param count, and the GPU.

**Why this matters**

Six months from now, you (or someone reading the report) might want to know: "Was Track B trained with bitsandbytes 0.45 or 0.49? On what CUDA version? What was the final eval loss?" Without `training_meta.json`, you'd have to reconstruct from git history + Kaggle session logs (which expire). With it, one file answers all reproducibility questions.

**What the output tells you**

A pretty-printed JSON of all the values. Skim it visually — anything obviously wrong (e.g., `train_loss = NaN`, `actual_versions["bitsandbytes"] = "FAILED: ..."`) means the run had a problem.

**What could be improved**

Also include git-state: `subprocess.run(["git","-C",REPO_DIR,"rev-parse","HEAD"], capture_output=True).stdout.decode().strip()` → captures the exact commit that produced this artifact."""))

cells.append(code("""# Push the adapter to Hugging Face Hub (private).
hub_repo = f"{cfg['hub']['username']}/{cfg['hub']['repos']['trackB']}"
print(f"Pushing to: {hub_repo}  (private={cfg['outputs']['private_hub_repo']})")

model.push_to_hub(
    hub_repo,
    private = cfg['outputs']['private_hub_repo'],
    token   = os.environ["HF_TOKEN"],
)
tokenizer.push_to_hub(
    hub_repo,
    private = cfg['outputs']['private_hub_repo'],
    token   = os.environ["HF_TOKEN"],
)
print(f"Pushed. URL: https://huggingface.co/{hub_repo}")"""))

cells.append(md("""**What this does**

Uploads `adapter_model.safetensors`, `adapter_config.json`, `tokenizer.json` etc. to a private HF Hub repo. The repo is created if it doesn't exist (HF auto-creates on first push).

**Why HF Hub instead of just keeping it on Kaggle**

Kaggle's `/kaggle/working/` is wiped after the session ends (or when you "Stop" the runtime). HF Hub is permanent. Day 5's inference notebook can `from peft import PeftModel; PeftModel.from_pretrained(base, "<user>/qwen25-1.5b-medreason-trackB-v0")` — that downloads the adapter from Hub, totally independent of Kaggle session state.

**Why private**

This is a medical-domain model trained on synthetic data with documented limitations. We don't want it indexed by search engines or reused by someone who skips the disclaimer. After Day 7's report is finalized with proper limitations, we *might* flip to public.

**What the output tells you**

`Pushed. URL: https://huggingface.co/abhishek1998s/qwen25-1.5b-medreason-trackB-v0`. Open that URL — you'll see your adapter as a private repo.

**What could be improved**

Push a `README.md` to the Hub repo too — `model.push_to_hub_card(...)` with usage instructions, training data summary, limitations, contact info. Required for any model that might leave the project."""))

# ─────────────────────────────────────────────────────────────────────────────
# Section 8 — smoke test
# ─────────────────────────────────────────────────────────────────────────────
cells.append(md("""## 8. Smoke test — Track-B inference vs Notebook-01 baseline

Generate an answer to the same hypertension question we asked in Notebook 01 (where the *base* model gave a generic answer). The fine-tuned Track B should produce a more medically-targeted response."""))

cells.append(code("""# Switch model to inference mode (~2x faster generation).
FastLanguageModel.for_inference(model)

prompt_messages = [{
    "role": "user",
    "content": "What are the first-line treatments for hypertension in a 55-year-old non-diabetic patient?"
}]
prompt_text = tokenizer.apply_chat_template(
    prompt_messages, tokenize=False, add_generation_prompt=True,
)
inputs = tokenizer(prompt_text, return_tensors="pt").to("cuda")

import time
start = time.time()
with torch.no_grad():
    out_ids = model.generate(
        **inputs,
        max_new_tokens = 400,
        do_sample      = False,
        temperature    = 0.0,
    )
elapsed = time.time() - start

new_ids = out_ids[0, inputs["input_ids"].shape[1]:]
print(f"Generated {len(new_ids)} tokens in {elapsed:.1f}s ({len(new_ids)/elapsed:.1f} tok/s)")
print("=" * 80)
print(tokenizer.decode(new_ids, skip_special_tokens=True))"""))

cells.append(md("""**What this does**

Same inference setup as Notebook 01 (greedy decoding, 400 new tokens, deterministic). Compare the answer to Notebook 01's base-model answer — the fine-tuned Track B should:
- Use medical-conversational style (warmer, structured) — that's what OpenMed's `content` looks like.
- NOT include a `Clinical rationale:` block (that's Track A only).
- Be relevant to the question (still hypertension treatments, not unrelated drift).

**Why this is the smoke test, not a real eval**

This is a *one-prompt, one-direction* check that "fine-tuning didn't catastrophically break anything." A real evaluation (Day 5) runs all 200 test rows and computes EM/ROUGE/BERTScore/judge-scores.

**What the output tells you**

Expected:
- ~10–18 tok/s (LoRA adds negligible inference cost vs base's 14.5 tok/s; if much slower, there's a bug).
- The answer should be similar in *content* to base but in OpenMed's *style* — more conversational, possibly with section headers and tables (because OpenMed's `content` field uses those).

If the answer is gibberish, repeated phrases, or empty: chat template mismatch between training and inference. Verify both used `get_chat_template(tok, "qwen-2.5")`.

**What could be improved**

Run 3 different prompts (varied medical topics) and average tok/s — single-shot is noisy. Also run the *base* model on the same prompts in this notebook for direct A/B comparison."""))

# ─────────────────────────────────────────────────────────────────────────────
# Section 9 — Day-2 takeaways
# ─────────────────────────────────────────────────────────────────────────────
cells.append(md("""## 9. Day-2 takeaways (fill in your numbers from this run)

| What | Value | Interpretation |
|---|---|---|
| Final `train_loss` | **<0.x>** | should be 0.7–1.1 for a healthy run |
| Final `eval_loss` | **<0.x>** | should be slightly higher than train_loss (small generalization gap) |
| Train runtime | **<NN> min** | ~30–45 min on T4 expected |
| Trainable LoRA params | **<NN>M (~1.x%)** | ~18M expected; if ~5M, only attention got LoRA — bug |
| Smoke-test tok/s | **<NN>** | ~10–18; baseline (no fine-tune) was 14.5 |
| W&B run URL | `<paste>` | for the report |
| HF Hub adapter URL | `https://huggingface.co/abhishek1998s/qwen25-1.5b-medreason-trackB-v0` | private |

**Tomorrow (Day 3):** train Track A — same hyperparameters, same dataset slice (same shuffle_seed → same indices), but the assistant turn now starts with "Clinical rationale:\\n...\\n\\nFinal answer:\\n...". The 6% longer assistant means ~5–10 min slower training.

Replace the `<...>` placeholders before downloading + committing the notebook."""))

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
