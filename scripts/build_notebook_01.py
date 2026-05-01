"""Build notebooks/01_setup_and_data_exploration.ipynb.

Run with:
    python scripts/build_notebook_01.py

Produces a deterministic, unexecuted notebook with markdown + code cells
ready to be run on Kaggle (T4 GPU, internet on).

This is the *teaching* version: every code cell is followed by a "What
this does + why + output meaning + what to improve" markdown block.
Read it top to bottom and you should understand each line and decision.
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
# Header
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

## How to read this notebook
Every code cell is followed by an explainer markdown cell. The explainer covers:
- **What the code does** (line-by-line, plain English)
- **Why we wrote it that way** (the design choice)
- **What the output means** (how to interpret your numbers)
- **What to improve** (concrete next-level enhancements)

Don't skip the explainers — they're where the learning is.

## Required Kaggle environment
- Accelerator: **GPU T4 x1** (T4 x2 also fine — we'll use one)
- Internet: **On**
- Kaggle Secrets attached: `HF_TOKEN`, `WANDB_API_KEY`, `GROQ_API_KEY`

## Plan reference
[`docs/superpowers/plans/2026-05-02-plan1-bootstrap-and-notebook01.md`](../docs/superpowers/plans/2026-05-02-plan1-bootstrap-and-notebook01.md) — Phase B."""))

# ─────────────────────────────────────────────────────────────────────────────
# Section 1 — bootstrap
# ─────────────────────────────────────────────────────────────────────────────
cells.append(md("""## 1. Bootstrap: clone repo, install deps

Kaggle's preinstalled environment has *most* of what we need. We then verify imports work; if they don't, we recover.

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

cells.append(md("""**What this does (line by line)**

| Line | What |
|---|---|
| `import subprocess, sys, os` | shell-callable subprocess, path manipulation, filesystem |
| `REPO_URL`, `REPO_DIR` | constants — Kaggle's `/kaggle/working/` is the only writable, persistent location |
| `if not os.path.exists(REPO_DIR)` | idempotency guard — re-running this cell after a kernel restart won't re-clone |
| `subprocess.run([...], check=True)` | spawns `git clone`; `check=True` raises a `CalledProcessError` on failure (vs `!git clone` which would silently print to stderr) |
| `sys.path.insert(0, REPO_DIR)` | inserts at index 0 so our `src/` wins over any conflicting Kaggle preinstalled module |
| `print(...)` | confirmation lines — quick visual proof the clone worked |

**Why this approach**

- **`subprocess.run` over `!git clone` magic**: subprocess raises real Python exceptions; the `!` form just prints to stderr and continues, hiding failures.
- **Insert at front of `sys.path`**: deterministic resolution. Kaggle preinstalls a TON of packages; if any future package is named `src` we'd silently pick the wrong one.
- **Idempotency**: kernel restarts are common in notebook work. The guard means re-running this cell costs ~1ms instead of "destination path already exists" error.

**What the output tells you**

Expected: `Repo cloned to: /kaggle/working/medical-reasoning-llm` and a `Contents:` line listing all top-level files. If `Contents` is missing `src` or `configs`, the clone is incomplete — re-run after `rm -rf $REPO_DIR`.

**What could be improved**

- **Pull on update**: change to `git pull` if the dir exists, so re-running pulls new commits without manual cleanup. Useful if you push fixes mid-Day-1.
- **Pin to a commit SHA** instead of `main` for stricter reproducibility (`git checkout <sha>` after clone). For a learning project, `main` is fine.
- **Move `REPO_URL` to an env var** — easier to swap forks for experiments without editing the notebook."""))

cells.append(code("""# Install pinned project requirements.
# NOTE: on Kaggle's CUDA-12.8 environment, our pins are too old for some
# GPU libs (bitsandbytes, unsloth). Most are already preinstalled with
# CUDA-12.8-compatible versions, so we skip this on Kaggle.
# Uncomment below ONLY if `import unsloth` fails in the next cell.
#!pip install -q -r {REPO_DIR}/requirements.txt"""))

cells.append(md("""**What this does**

The line is intentionally **commented out** on Kaggle. If active, it would `pip install` every package in `requirements.txt` (unsloth, transformers, trl, peft, datasets, accelerate, bitsandbytes, wandb, evaluation libs, etc.).

**Why we skip it on Kaggle**

Kaggle preinstalls a working, CUDA-12.8-compatible ML stack. Our pinned versions in `requirements.txt` (e.g., `bitsandbytes==0.45.2`) were chosen against an earlier CUDA, and on Kaggle's current 12.8 they cause `libbitsandbytes_cuda128.so not found` and `GLIBCXX_3.4.32 not found` errors. The preinstalled versions are *newer* and *known to work here*.

The next cell verifies the imports actually work — that's the real test. If anything fails, we'll uncomment this line and retry.

**Why we keep `requirements.txt` despite skipping it**

For two reasons:
1. **Local dev** (Mac/Windows laptop) needs the deps to run `pytest tests/`.
2. **Reproducibility log** — Day 2 will dump the *actually-installed* versions to `training_meta.json`. We can compare those to the pins and document drift.

**What the output is**

Nothing — the cell only contains a comment. That's correct.

**What could be improved**

- **Auto-detect Kaggle vs Colab vs local** and skip/install accordingly:
  ```python
  if "KAGGLE_KERNEL_RUN_TYPE" not in os.environ:
      !pip install -q -r {REPO_DIR}/requirements.txt
  ```
- **Soften the pins** in `requirements.txt`: use `>=X,<Y` ranges for the GPU stack so a fresh install picks compatible versions. (We'll do this after Day 1.)
- **Split into `requirements-dev.txt` (CPU/local) and `requirements-gpu.txt` (Kaggle/Colab)** to make environment differences explicit."""))

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

cells.append(md("""**What this does**

For each of 8 critical packages: dynamically import it via `importlib.import_module(name)`, read its `__version__`, and store. Catch any exception and record `FAILED: <ExceptionClass>` instead of crashing.

**Why this approach**

- **`importlib` over plain `import x`**: doesn't pollute the namespace (no `import unsloth as foo` for each one) and supports loops over a list.
- **Catch all exceptions, not just `ImportError`**: we've seen `ValueError` (from bitsandbytes' CUDA detection) and `RuntimeError` (from peft's chained imports). Broad `except Exception` ensures the loop completes — we get to see *all* failures, not just the first.
- **Format with f-strings + alignment** (`{k:20s}`): readable at a glance.

**What the output tells you**

After the kernel restart on Kaggle, you saw something like:

```
  unsloth              2026.4.8
  transformers         5.5.0
  trl                  0.24.0
  peft                 0.19.1
  datasets             4.3.0
  bitsandbytes         0.49.2
  accelerate           1.13.0
  wandb                0.19.4
```

All 8 packages produced version strings, no `FAILED:` lines → the GPU stack is healthy and we can proceed.

**Note on version drift**: `transformers 5.5.0` is much newer than our pinned `4.49.0`. Transformers 5.x deprecates a few internals (notably `modeling_attn_mask_utils`) — you'll see harmless `FutureWarning`s later. If a deprecation becomes a hard error, the fix is `pip install transformers==4.x` to roll back.

**What could be improved**

- **Persist to disk**: `with open("/kaggle/working/versions.json","w") as f: json.dump(versions, f, indent=2)` — Day 2's `training_meta.json` then includes these for reproducibility.
- **Compare to pins**: read `requirements.txt`, parse the `==X.Y.Z` pins, compute drift, warn if any differ by ≥ 1 major version.
- **Probe CUDA**: also print `torch.version.cuda`, `torch.cuda.get_device_capability()` here so the env summary is one-stop. (We do this in the diagnostic cell separately.)"""))

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

cells.append(md("""**What this does**

`UserSecretsClient()` is Kaggle's API to read secrets you attached via the right-sidebar lock icon. The helper `_try_get(name)` calls `secrets.get_secret(name)` and catches the exception that would be thrown if you didn't attach that secret. Then we set each as an environment variable (`os.environ[...] = value`) — this is the standard interface every ML library reads from. The `or ""` makes a missing secret an empty string rather than `None`, so downstream code can do `if os.environ["X"]:` cleanly.

**Why env vars instead of just keeping the variables in Python**

- Hugging Face's `from_pretrained` reads `HF_TOKEN` from `os.environ`.
- W&B's `wandb.login()` reads `WANDB_API_KEY` from env.
- The `openai` library used for Groq reads keys from env by default.

By setting them once here, every subsequent library call "just works" without us threading tokens through every call.

**Why try/except per-secret**

If we forgot to attach `GROQ_API_KEY`, we want HF + W&B to still work for Day-1 exploration — we don't need Groq until Day 6. Per-secret tolerance lets us proceed with what's available.

**What the output tells you**

```
HF_TOKEN set:       True
WANDB_API_KEY set:  True
GROQ_API_KEY set:   True
```

All three `True` → ready for Days 2 and 6. If any are `False`:
- **HF_TOKEN: False** → blocks Day 2 (can't push adapters). Fix now.
- **WANDB_API_KEY: False** → not blocking (can fall back to local logs). Fix when convenient.
- **GROQ_API_KEY: False** → not blocking until Day 6. Fix anytime before then.

**What could be improved**

- **Mask print output**: `print("HF_TOKEN ends in ...", os.environ["HF_TOKEN"][-4:])` so you can spot wrong-token issues without exposing the token.
- **Validate token shape**: HF tokens start with `hf_`, Groq tokens with `gsk_`. A simple `if HF_TOKEN and not HF_TOKEN.startswith("hf_"): warn(...)` catches paste errors.
- **Round-trip test**: call `huggingface_hub.whoami()` to confirm the HF token is *valid*, not just present. (We do something like this in the next cell.)"""))

cells.append(code("""# Hugging Face login — needed even for reading the OpenMed dataset
# (some HF mirrors return 401 without a token).
from huggingface_hub import login as hf_login

if os.environ["HF_TOKEN"]:
    hf_login(token=os.environ["HF_TOKEN"], add_to_git_credential=False)
    print("HF login OK")
else:
    print("[warning] HF_TOKEN not set — dataset download may fail or rate-limit")"""))

cells.append(md("""**What this does**

`huggingface_hub.login(token=...)` writes the token to `~/.huggingface/token` and registers it with the HF Hub client globally. From this point on, `load_dataset(...)` and `from_pretrained(...)` will use this token automatically — you don't need to pass `token=` to each call.

The `add_to_git_credential=False` keeps the token out of git's credential helper, which we don't need on Kaggle (we won't be pushing to HF via `git push`).

**Why explicit login when we already have the env var**

Two reasons:
1. **Some libraries read from disk, not env**. The `huggingface_hub` library checks `~/.huggingface/token` first.
2. **Defense in depth**. If a future cell strips env vars, the login persists for the rest of the session.

**What the output tells you**

`HF login OK` → token was syntactically accepted (32+ chars, starts with `hf_`). It does *not* prove the token is valid — only that login() didn't throw a parse error. The real proof comes when `load_dataset` succeeds in the next section.

**What could be improved**

- **Validate with `whoami()`**: `from huggingface_hub import whoami; print(whoami())` would print your HF username if the token is *actually* valid (network round-trip). One extra line, much higher confidence.
- **Catch HTTP errors**: in production, wrap `hf_login` in try/except `HfHubHTTPError` and downgrade to a warning if the network is flaky.
- **Token scope check**: HF Hub API can report a token's permissions (read/write). For our private-adapter push on Day 2, we need *write* — checking now would catch a read-only token before it bites us tomorrow."""))

# ─────────────────────────────────────────────────────────────────────────────
# Section 3 — dataset
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

cells.append(md("""**What this does**

`datasets.load_dataset(repo_id, split, token)` does three things:
1. Downloads the dataset's *parquet* shards from HF Hub (10 shards × ~250 MB each = ~2.5 GB total).
2. Caches them in `~/.cache/huggingface/datasets/` (so re-runs skip the download).
3. Returns a `Dataset` object that is **memory-mapped** — only the rows you actually access are loaded into RAM.

The `split="train"` argument selects the train portion. (This dataset only has a train split; Days 2–3 we'll create our own train/val/test slice.)

**Why parquet?**

Parquet is a *columnar* format: it stores all values of a single column together. Two big advantages here:
- **Fast random access**: reading row 463756 doesn't require parsing rows 0..463755.
- **Compression**: 506K rows × ~3 KB each as text = ~1.5 GB; compressed parquet shards total ~2.5 GB but include reasoning + content (the largest fields).

**What the output tells you**

```
Total rows: 506,150
Columns: ['messages']
```

Two key facts:
- **506K rows** is far more than we need (we'll use 3,300). We can afford strict filters later (e.g., "only rows where content < 1500 tokens") without running out of data.
- **Columns: `['messages']`** — a *single* column. Everything (user query, assistant answer, reasoning) is nested inside the `messages` list. This is OpenAI's chat format, which is what `tokenizer.apply_chat_template` consumes natively.

**What could be improved**

- **Stream instead of cache** (`load_dataset(..., streaming=True)`): no 2.5 GB on-disk cache, lazy loading. But you lose random-access (no `ds[i]` indexing), and we *do* want random access for our shuffled split. Stick with cached for now.
- **Pre-filter at load time**: `ds = ds.filter(lambda r: short_enough(r))` — saves us from materialising too-long rows we'd discard anyway. We'll do this on Day 2.
- **Verify dataset version**: the `Medical-Reasoning-SFT-GPT-OSS-120B-V2` repo could publish a new revision tomorrow that subtly changes the schema. Pin to a specific revision: `load_dataset(repo, revision="<commit_sha>")` for strict reproducibility."""))

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

cells.append(md("""**What this does**

Pulls one row (`ds[0]`) and prints each message's role + a 300-char preview of `content` and (for assistant) `reasoning_content`. The `m.get("content") or ""` is defensive — if a field is `None`, `.get()` returns `None`, and `or ""` converts it to an empty string so slicing `[:300]` doesn't crash.

**Why peek at the structure?**

To **validate our schema assumption**. Our formatters (`format_for_track_a`, `format_for_track_b`) assume:
- `row["messages"]` is a list of length ≥ 2
- The user message has `content`
- The assistant message has *both* `content` and `reasoning_content`

If any of these is wrong on the *real* dataset (vs the toy data in our pytest), training will fail in subtle ways. Better to discover now.

**What the output tells you**

You saw row 0 was:
- A grandparent asking about post-tonsillectomy thrombocytopenia (low platelets, bruises) in their grandchild.
- The assistant's `content` is a structured patient-friendly explanation.
- The assistant's `reasoning_content` walks through differentials: ITP, drug-induced, viral infection, marrow issues — exactly the chain-of-thought a clinician would think through.

This confirms three things:
1. **Schema matches**: 2 messages, user → assistant, both fields present.
2. **`reasoning_content` is real CoT** (not just the answer rewritten) — it's the model's intermediate "thinking" before the final answer.
3. **`content` is patient-facing** (warm tone, structured for laypeople) — meaning Track B output won't be "raw clinical bullet points," it'll be full prose responses.

**What could be improved**

- **Check multiple rows**: `for idx in [0, 1000, 100000, 500000]:` — the dataset might have inconsistent schemas across shards.
- **Count rows missing fields**: `n_missing_reasoning = sum(1 for r in ds if r['messages'][1].get('reasoning_content') is None)`. If non-zero, Track A would have empty rationales — we'd want to filter those out.
- **Print user message length too**: helps you anticipate token costs. Add `print("user content length:", len(content))`."""))

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

cells.append(md("""**What this does**

1. Loads the Qwen2.5-1.5B *tokenizer* (not the full model, just the tokenizer — ~2 MB download).
2. Samples 200 row indices reproducibly (`RandomState(42)` → same indices every run).
3. For each row, finds the assistant message and tokenizes both `reasoning_content` and `content`, recording the resulting token counts.
4. Computes mean / median / p95 / max for both, plus the fraction of rows with reasoning > 150 tokens.

`tok.encode(text, add_special_tokens=False)` returns a list of integer token IDs. We don't add special tokens (`<bos>`, `<eos>`) because we're measuring the raw text length, not a fully-formatted training example.

**Why we sample, not measure all 506K rows**

200 tokenizations take ~5 seconds. 506K would take ~3.5 minutes. The point of this cell is a quick sanity check — if we want a precise corpus statistic later, we can crank the sample size to 5000 (~2 min) and the percentile estimates tighten up.

**Why use the model's own tokenizer**

Token counts are model-specific. Qwen's tokenizer is different from Llama's, which is different from GPT-4's. Since our model is Qwen2.5-1.5B, *its* tokenizer is the only one whose counts predict actual sequence lengths during training/inference.

**What the output tells you**

You saw:

| Field | Mean | Median | p95 | Max |
|---|---:|---:|---:|---:|
| `reasoning_content` | 240 | 197 | 626 | 1,168 |
| `content` (final answer) | 1,888 | **1,954** | 3,542 | 19,255 |

Plus: **64.5% of rows have reasoning > 150 tokens**.

Two findings:

1. **Short-CoT cap is meaningful**: 64.5% of examples have reasoning longer than our 150-token cap, so the truncation actually *does* something to most rows. If the cap had affected only 5% of rows it would be a near no-op. ✓

2. **The `content` field is HUGE** — median 1,954 tokens. These are not concise terse answers; they're full structured patient writeups. Combined with a typical user query (~200 tok) and Track A's rationale + headers (~170 tok), a typical Track A example sums to **~2,320 tokens** — *over* our intended `max_seq_length=2048`.

   **This is the most important finding of Day 1.** Day 2's training will need either:
   - (a) **`max_seq_length=4096`** — uses ~50% more VRAM during training but fits everything, OR
   - (b) **filter to short rows** — keep only rows where total length < 1800 tokens, drops ~half the dataset, OR
   - (c) **truncate `content`** — caps the answer too, risks losing the conclusion which is usually at the end.

   Option (a) is the cleanest. We'll go with it.

**What could be improved**

- **Bigger sample**: `size=2000` for tighter p95 (~30 sec).
- **Combined-length distribution**: also tokenize the *whole* formatted Track A example and report the distribution of "user_msg + Clinical rationale (capped) + content" lengths. That's the *actual* sequence we'll train on; this cell measures fields in isolation.
- **Histogram, not just stats**: `import matplotlib.pyplot as plt; plt.hist(answer_lens, bins=50); plt.xlabel("tokens")` — visual is more diagnostic than 4 numbers.
- **Filter mask**: precompute an array `fits_in_2048 = (np.array(reasoning_lens) + np.array(answer_lens) + 200) < 2048` and report `np.mean(fits_in_2048)` — answers "what fraction would fit unmodified?" directly."""))

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

cells.append(md("""**What this does**

Iterates our 200 sampled indices in order. For each, tokenizes its reasoning and stops at the *first* row with > 200 tokens. The `assert` is a defensive guard — if our sample (which previously showed 64.5% over 150 tokens) miraculously contained zero rows over 200, we'd crash early instead of silently using a row that doesn't actually trigger truncation.

**Why > 200, not > 150?**

We want truncation to be *visible*. A row with reasoning of exactly 152 tokens would only get clipped by 2 tokens — you wouldn't notice. A row at 346 tokens (your case) clearly shows the cut.

**Why "first match" instead of, say, the longest row**

Determinism. `sample_indices` is seeded with RandomState(42); iterating in order gives a reproducible pick. Every Day-1 run finds the same row. (Picking the longest would also be reproducible but would always grab the most extreme example — which might not be representative.)

**What the output tells you**

```
Using row 463756 (reasoning = 346 tokens)
```

Row 463756 has 346-token reasoning. Our 150-token cap will cut it roughly in half — we'll see the truncation in the next cells.

**What could be improved**

- **Try several rows**, not just one. A loop over 3 long rows + 3 short rows would catch formatter bugs that only manifest at length extremes.
- **Pick a longer row**: `if rl > 600` would test our truncation more aggressively (closer to p95 of 626).
- **Random pick within range**: `eligible = [i for i in sample_indices if 200 < length(i) < 500]; pick = np.random.choice(eligible)` — gives a moderate-length representative example."""))

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

cells.append(md("""**What this does**

Imports our `format_for_track_a` from the cloned repo's `src/`, applies it to row 463756, and prints the formatted assistant turn (first 1500 chars + total tokenised length). The result of `format_for_track_a` is a `dict` with key `"messages"`, mirroring the input shape; we grab the last message (the assistant) for inspection.

The `short_cot_max_tokens=150` arg is the cap from our spec. Our formatter:
1. Tokenizes the reasoning (346 tokens here).
2. Truncates to 150 tokens.
3. Walks back to the last sentence boundary (`.`, `?`, `!`) within those 150.
4. Wraps in `Clinical rationale:\\n<truncated>\\n\\nFinal answer:\\n<content>`.

**Why this approach works for the assignment**

The assignment asks for *short CoT*. A simple "first 150 tokens" cut would chop mid-word. Sentence-boundary truncation produces grammatically valid output, which the model learns more cleanly. (Imagine training on `"...severe hypert"` — the model would learn that fine-tunes end with cut-off words.)

**What the output tells you**

You saw an output that started with:
```
Clinical rationale:
We need to respond empathetically, explain why birth control...
```
and ended with the full patient-facing response. The total length was **2,478 tokens**.

What this confirms:
- ✓ The `Clinical rationale:` and `Final answer:` headers were inserted correctly.
- ✓ The rationale was truncated (we can see the 150-token cap producing a coherent paragraph that ends on a period).
- ✓ The full `content` (the long patient-friendly answer) is preserved verbatim.

The 2,478-token total reinforces the seq-length finding from the previous cell — Track A examples routinely exceed our 2048 budget when content is long.

**What could be improved**

- **Print the rationale section in isolation**: right now it's at the top of `asst_a[:1500]`. Adding `rationale_section = asst_a.split("Final answer:")[0]; print(rationale_section)` would let you visually confirm the truncation is at a sentence boundary.
- **Compare lengths before vs after truncation**: `print(f"raw: {raw_reasoning_tokens}, capped: {short_cot_tokens}")` would give a direct numeric proof.
- **Show last 200 chars too** (`asst_a[-500:]`): so you can also visually confirm the answer ends naturally (not mid-sentence)."""))

cells.append(code("""# Apply the Track B formatter.
out_b = format_for_track_b(row)
asst_b = out_b["messages"][-1]["content"]

print("=" * 80)
print("TRACK B (final answer only, no reasoning)")
print("=" * 80)
print(asst_b[:1500])
print(f"\\n  -> assistant length (Track B): "
      f"{len(tok.encode(asst_b, add_special_tokens=False))} tokens")"""))

cells.append(md("""**What this does**

Applies `format_for_track_b` (the answer-only formatter) to the *same row* and prints the assistant turn.

**Why use the same row as Track A**

This is the *control*. The whole experiment is: same input, same model, same hyperparameters, *only* the assistant-output format differs. By running both formatters on identical rows here, we visually verify that Track B's output is *literally* a substring of Track A's (just without the rationale + headers).

**What the output tells you**

You saw the same patient-facing answer, **without** the `Clinical rationale:` block. Total length: **2,337 tokens**.

The delta:
- Track A: 2,478 tokens
- Track B: 2,337 tokens
- **Diff: 141 tokens**

That 141-token gap is *exactly* the experimental signal: it's the truncated Clinical rationale + headers. Track A pays a ~6% length tax for showing reasoning. Day 6's report will quantify whether that tax was worth it.

**What could be improved**

- **Compute the diff inline**: `print(f"Delta = {len_a - len_b} tokens")` — saves the reader from manual subtraction.
- **Verify Track B is a suffix of Track A's content**: `assert asst_b in asst_a` — proves the formatters operate on the same `content` field.
- **Histogram of deltas across many rows**: how does the rationale tax distribute? Is the average +120 tokens? +200? That number ends up in the report's "token cost" comparison."""))

cells.append(code("""# Sanity assertions — these MUST pass before we trust the formatters in training.
assert "Clinical rationale:" in asst_a
assert "Final answer:" in asst_a
assert asst_a.rstrip().endswith(row["messages"][-1]["content"].strip())

assert "Clinical rationale:" not in asst_b
assert "Final answer:" not in asst_b
assert asst_b == row["messages"][-1]["content"].strip()

print("formatter sanity checks passed")"""))

cells.append(md("""**What this does**

Six `assert` statements check the formatter invariants on a real row:

| Assert | What it guards against |
|---|---|
| `"Clinical rationale:" in asst_a` | Track A formatter forgot/lost the rationale header |
| `"Final answer:" in asst_a` | Track A formatter forgot/lost the answer header |
| `asst_a.rstrip().endswith(content.strip())` | Track A's final answer matches the *original* content (not truncated, not modified) |
| `"Clinical rationale:" not in asst_b` | Track B accidentally included the rationale header |
| `"Final answer:" not in asst_b` | Track B accidentally included the answer header |
| `asst_b == content.strip()` | Track B's output is *exactly* the content (no extra prefixes/suffixes) |

If any of these fails, we'd see `AssertionError` and stop. No silent failure paths.

**Why duplicate the pytest tests here?**

The pytest tests in `tests/test_data_formatting.py` use *toy* data (small synthetic rows). These notebook assertions use *real* data from OpenMed. Together they cover:
- pytest: edge cases (empty reasoning, exact-150-token boundary, etc.) — fast unit tests.
- notebook: real-world distribution shape — confirms toy assumptions match reality.

A formatter that passes pytest but fails the notebook assertion would tell us our toy data didn't represent reality (e.g., we assumed `content` was always a string, but real rows have `None`). Both layers protect us.

**What the output tells you**

```
formatter sanity checks passed
```

→ All six invariants hold on row 463756. ✓

**What could be improved**

- **Loop the asserts over multiple rows**: `for idx in sample_indices[:50]: row = ds[idx]; assert ...` — single-row pass doesn't prove the formatter works on the *distribution*, just on this one row.
- **Add safety-note position assertion**: if a row has a `Safety note:` block in `content`, Track A's output should keep it *after* `Final answer:`, not before. We don't currently check this.
- **Whitespace tolerance**: `asst_a.rstrip().endswith(...)` is sensitive to trailing newlines. A more robust check: `assert content_normalized in asst_a_normalized` where both are whitespace-collapsed."""))

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

cells.append(md("""**What this does**

Two steps:
1. **`get_chat_template(tok, chat_template="qwen-2.5")`** — Unsloth's helper attaches Qwen2.5's official chat template to the tokenizer. (HuggingFace's tokenizer.json *usually* includes a chat_template, but Unsloth's version has small fixes for training-time edge cases.)
2. **`apply_chat_template(messages, tokenize=False)`** — walks the `messages` list and produces a single string with role-tagged segments. `tokenize=False` returns the *string*; `tokenize=True` would return token IDs directly.

`add_generation_prompt=False` means: don't append a trailing `<|im_start|>assistant\\n` (which would tell the model "your turn"). We use `False` here because we're rendering a *complete* training example. We'll set it to `True` later (in inference) to prompt the model.

**Why this template specifically**

Qwen2.5's training data was wrapped with this exact template. The model learned to associate `<|im_start|>` with "new turn" and `<|im_end|>` with "turn boundary." If we trained with a *different* template — say Llama's `<s>[INST]...[/INST]</s>` — the model would have to relearn turn boundaries from scratch, wasting capacity and hurting fine-tune quality.

**The "#1 silent fine-tuning bug"**

Imagine training with template A and inferring with template B. The training loss goes down (model fits training data), but at inference the model sees `<s>[INST]...` (Llama format) instead of `<|im_start|>user\\n...` (Qwen format). It produces garbage. You blame the model, the LR, the data — but it's a one-line template mismatch. Hence: **same `get_chat_template` call in every notebook**.

**What the output tells you**

You saw:

```
<|im_start|>system
You are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>
<|im_start|>user
Hello, my name is XXXX...<|im_end|>
<|im_start|>assistant
Clinical rationale:
We need to respond empathetically...
```

Three things to note:
- **System message added automatically**: even though our `messages` list had no system message, Qwen's template prepends one. This is fine — Qwen models always get this preamble.
- **Track A's assistant turn starts with `Clinical rationale:`** — exactly our format.
- **Track B's assistant turn starts directly with the answer** — no rationale.

**What could be improved**

- **Tokenize and inspect IDs**: `ids = tok_chat.apply_chat_template(out_a["messages"], tokenize=True); print(ids[:30])` — confirms `<|im_start|>` etc. round-trip to special-token IDs (typically 151644 for Qwen2.5). If you see them tokenized as plain text byte-pair, the template isn't actually special.
- **Compare to inference template**: `add_generation_prompt=True` should produce a string ending in `<|im_start|>assistant\\n` *with no closing tag*. Print both training-time and inference-time renderings side by side.
- **Custom system prompt**: for medical use, you might prefer a system message like `"You are a medical-information assistant. Always recommend consulting a clinician for diagnosis or treatment decisions."` Qwen's default works for now."""))

# ─────────────────────────────────────────────────────────────────────────────
# Section 6 — memory math
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

cells.append(md("""**What this does**

`FastLanguageModel.from_pretrained` is Unsloth's wrapper around HuggingFace's `from_pretrained`. It returns `(model, tokenizer)` and applies several optimizations along the way:
- Detects the GPU (T4) and picks fp16 (bf16 isn't supported on T4 — this is what `dtype=None` auto-detects).
- Loads the **already-quantized** weights from `unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit` (the `bnb-4bit` suffix means the upload is pre-quantized — no on-the-fly quantization needed).
- Patches the model's attention forward pass with Unsloth's faster Triton kernels (gives ~2× faster training/inference vs vanilla HF).
- Sets up the model for **inference mode by default** (we'll call `for_training` later in Notebook 02 to switch).

The `max_seq_length=2048` is the *training/inference* max length — the model itself can handle more (Qwen2.5 supports 32k context), but we're capping for VRAM reasons.

We re-attach `get_chat_template` to the tokenizer because Unsloth's `from_pretrained` returns a fresh tokenizer object. Same template, applied once.

**Why the `unsloth/` repo, not `Qwen/`?**

The official `Qwen/Qwen2.5-1.5B-Instruct` is fp16 (~3 GB). Loading + on-the-fly 4-bit quantization works but adds ~30 seconds to load time. The `unsloth/` mirror is *already* 4-bit, so it loads in ~5 seconds.

**What the output tells you**

You saw the Unsloth banner:
```
==((====))==  Unsloth 2026.4.8: Fast Qwen2 patching. Transformers: 5.5.0.
   \\\\   /|    Tesla T4. Num GPUs = 2. Max memory: 14.563 GB. Platform: Linux.
O^O/ \\_/ \\    Torch: 2.10.0+cu128. CUDA: 7.5. CUDA Toolkit: 12.8. Triton: 3.6.0
\\        /    Bfloat16 = FALSE. FA [Xformers = 0.0.35. FA2 = False]
 "-____-"     Free license: http://github.com/unslothai/unsloth
```

Translation:
- **Tesla T4, Num GPUs = 2**: Kaggle gave you 2× T4. Unsloth defaults to using GPU 0; if Day 2 needs more, we can do data-parallel.
- **Max memory: 14.563 GB**: per GPU. Plenty.
- **CUDA: 7.5**: T4's compute capability (Turing arch) — limits us to fp16 (no bf16) and rules out FlashAttention-2 (needs Ampere+).
- **Bfloat16 = FALSE**: confirmed.
- **FA2 = False, Xformers = 0.0.35**: no FA2, but Xformers' efficient attention is enabled. ✓
- **`Free license`**: Unsloth's free tier (the 2× speed claim is real).

Plus `unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit does not have a padding token! Will use pad_token = <|PAD_TOKEN|>` — Unsloth added a padding token automatically. We need a pad token because batched training pads short sequences to the longest in the batch. ✓

**What could be improved**

- **Use the second T4**: `accelerate launch` with `device_map="balanced"` would split training across both GPUs — possibly 1.7× faster (not 2× due to sync overhead). Not needed for 1.5B but a Day-4 experiment if we have time.
- **Pre-warm the model**: the first `model.generate()` call has some Triton compile overhead. Add a 10-token throwaway generate now so the inference smoke test reports steady-state tok/s.
- **Increase max_seq_length to 4096**: based on Section 3's finding — but doing it here pre-allocates more KV cache. For Day 1 (no training), 2048 is fine; for Day 2 we'll bump it."""))

cells.append(code("""def _vram_used_gb():
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.memory_allocated() / 1e9

print(f"Allocated VRAM after model load: {_vram_used_gb():.2f} GB")
print(f"Total GPU VRAM:                  "
      f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
print(f"Device:                          {torch.cuda.get_device_name(0)}")"""))

cells.append(md("""**What this does**

Three queries via PyTorch's CUDA API:
- `torch.cuda.memory_allocated()` — bytes of GPU memory *currently* held by tensors PyTorch knows about.
- `torch.cuda.get_device_properties(0).total_memory` — the GPU's total physical VRAM.
- `torch.cuda.get_device_name(0)` — string name for confirmation.

We divide by `1e9` to convert bytes → gigabytes (decimal GB, the convention everyone uses).

**The subtle distinction: `memory_allocated` vs `memory_reserved`**

- **`memory_allocated`**: what tensors are *using* right now.
- **`memory_reserved`**: what PyTorch has *grabbed from CUDA* but not necessarily handed to a tensor (cache pool for fast re-allocation).

`nvidia-smi` shows `memory_reserved`. PyTorch's `memory_allocated` is usually a bit lower. For "is my model fitting?" the relevant number is `memory_reserved`; for "how big is my model?" `memory_allocated` is what you want.

**Why this matters**

If we know the base model uses 1.19 GB allocated and we have 15.6 GB total, we have ~14 GB headroom. Training will eat:
- LoRA adapter weights: ~70 MB (negligible)
- Adam optimizer state on LoRA: ~140 MB (8-bit AdamW; halved by paging)
- Activations: ~3–5 GB depending on `max_seq_length` and batch
- Gradients on LoRA: ~70 MB
- KV cache during generation: a few hundred MB

Total expected during training: ~5–8 GB. We have ~14 GB headroom → *plenty* of room.

**What the output tells you**

```
Allocated VRAM after model load: 1.19 GB
Total GPU VRAM:                  15.6 GB
Device:                          Tesla T4
```

1.19 GB is right on the theoretical 0.78 GB (4-bit weights) plus ~0.4 GB (tokenizer buffers, embedding lookups, internal Unsloth state). **You have 14.4 GB of VRAM available for training. Comfortable.**

**What could be improved**

- **Also print `memory_reserved`**: more accurate "is it fitting?" indicator. Add `torch.cuda.memory_reserved() / 1e9`.
- **Print compute capability**: `torch.cuda.get_device_capability(0)` — confirms T4 = (7, 5). Useful for understanding what kernels are available.
- **Profile a single forward pass** to see actual peak memory: `torch.cuda.reset_peak_memory_stats(); model(inputs); print(torch.cuda.max_memory_allocated()/1e9)` — gives you the real activation cost. Day-2 useful."""))

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

cells.append(md("""**What this does (line by line)**

| Line | What |
|---|---|
| `FastLanguageModel.for_inference(model)` | Unsloth flips the model to inference mode (disables dropout, enables KV cache, swaps in 2x-faster generation kernels) |
| `prompt_messages = [...]` | the user question, in chat format |
| `tokenizer.apply_chat_template(..., add_generation_prompt=True)` | renders to `<|im_start|>system…<|im_end|><|im_start|>user…<|im_end|><|im_start|>assistant\\n` — note the trailing `assistant\\n` with no closing tag — that signals "your turn, model" |
| `tokenizer(prompt_text, return_tensors="pt")` | turns the rendered string into PyTorch tensors of token IDs and attention masks |
| `.to("cuda")` | moves tensors to GPU |
| `with torch.no_grad():` | disables gradient tracking — saves memory and ~10% time during inference |
| `model.generate(...)` | runs autoregressive generation |
| `do_sample=False, temperature=0.0` | greedy decoding — deterministic, reproducible |
| `out_ids[0, prompt_len:]` | strips the prompt tokens, keeping only the model's *new* output |
| `tokenizer.decode(new_ids, skip_special_tokens=True)` | turns the new IDs back into text, dropping `<|im_end|>` and other markers |

**Why greedy decoding?**

Two reasons:
1. **Reproducibility**: same input → same output, every run. Critical for our 200-sample evaluation comparison on Day 5.
2. **Lower variance**: with sampling at temperature > 0, two runs can differ. We'd need N samples per question + majority-vote to get stable means. Greedy is one-shot.

The trade-off: greedy can produce "stuck" outputs where the model loops on a phrase. In practice, well-trained models rarely do this; if they do, it's a signal of a real bug (bad fine-tune, broken chat template) rather than randomness.

**Why measure tok/s?**

This becomes our **baseline number**. On Day 5, we'll measure tok/s of the fine-tuned Track A and Track B models on the same hardware. The fine-tune shouldn't slow inference (LoRA adds negligible compute), but if it does — say, due to a bug introducing a non-cached path — we'll catch it.

**What the output tells you**

You saw:
```
Generated 223 new tokens in 15.4s (14.5 tok/s)
For a 55-year-old non-diabetic patient with hypertension, the first-line treatment options typically include:
1. Lifestyle modifications: ...
2. Medications: ACE inhibitors, ARBs, Calcium channel blockers, Diuretics
...
```

Three things:
1. **End-to-end generation works** — model loads, tokenizes, generates, decodes. The pipeline is healthy.
2. **14.5 tok/s** is our baseline. Day-5 fine-tuned should match. (A T4's typical decoding speed for a 1.5B fp16 model is 20–30 tok/s. We're a bit slower — the 4-bit dequantization on each token adds overhead.)
3. **The answer is reasonable but generic**: the base model knows hypertension's first-line drugs (ACE/ARB/CCB/diuretic) and lifestyle advice. But it doesn't pick a specific one based on the patient's age/comorbidities — that's what the medical fine-tune is supposed to add.

This is your "before" picture. Day 5 will compare it to "after."

**What could be improved**

- **Multiple prompts, average tok/s**: a 1-prompt measurement is noisy. Run 3-5 prompts and report mean/std. The first prompt also includes Triton compile overhead, so a single-shot measurement understates real throughput.
- **Use a real test-set question**: the "hypertension treatment in a 55-year-old" prompt is ~clinical but not from OpenMed. Picking a real OpenMed test row would give a *direct* before/after comparison.
- **Print the prompt token count**: `print(f"prompt tokens: {prompt_len}")` — completes the latency picture (latency = first-token-time + per-decoded-token-time × n_new). Right now we only see the generation phase.
- **Catch and report `finish_reason`**: did generation hit `<|im_end|>` (eos) or `max_new_tokens` (truncation)? Add a check on `out_ids[0, -1]` vs `tokenizer.eos_token_id`. We log this on Day 5; doing it here would be a useful warmup."""))

# ─────────────────────────────────────────────────────────────────────────────
# Section 9 — Day-1 takeaways (filled in with real numbers)
# ─────────────────────────────────────────────────────────────────────────────
cells.append(md("""## 9. Day-1 takeaways (filled in from your run)

| What | Value | Implication |
|---|---|---|
| Dataset size | **506,150 rows** | We'll use 3,300 (3,000 train + 100 val + 200 test); plenty of slack to filter aggressively. |
| Median `reasoning_content` length | **197 tokens** | Below our 150-token cap for half the corpus; cap activates on the longer half. |
| Fraction reasoning > 150 tokens | **64.5%** | Short-CoT cap is meaningful — actually truncates ~2/3 of training rows. ✓ |
| Median `content` (final answer) length | **1,954 tokens** | **Critical**: with user msg (~200) + Track A rationale (~170) + content (~1954) ≈ 2,320 tokens, our `max_seq_length=2048` is too small. Day 2 needs to either bump to 4096 or filter rows. |
| GPU | Tesla T4 × 2 (Kaggle gave us 2 GPUs) | Plenty of headroom; second GPU is an optional Day-4 stretch (data-parallel). |
| VRAM after 4-bit base load | **1.19 GB** / 15.6 GB total | ~14 GB headroom for LoRA + Adam + activations during training. ✓ |
| Bfloat16 / FA2 supported on T4? | **No / No** | We'll use fp16 + Xformers efficient attention. Already configured. |
| Base-model inference (greedy) | **14.5 tok/s, 223 tokens in 15.4 s** | Reference number for Day-5 latency comparison. Fine-tuned model should match. |
| Track A / Track B token deltas (one row) | A=2,478 / B=2,337 → +141 (+6.0%) | Rough order of magnitude for "rationale tax." Day-5 reports a real distribution. |

### Decisions to carry into Day 2

1. **Increase `max_seq_length` to 4096** in `configs/experiment_config.yaml` (currently 2048). Without this, ~half our training rows would be truncated mid-answer.
2. **Don't pin GPU-stack versions** in `requirements.txt` (bitsandbytes / unsloth / torch). Pin only the higher-level libs that we control (transformers / trl / peft / datasets).
3. **Set `dtype=fp16` explicitly** on T4 (instead of `dtype=None`) to skip the auto-detect cost.
4. **Plan to log `versions.json`** to `outputs/<track>/training_meta.json` for reproducibility.

**Tomorrow (Day 2):** train Track B (the simpler ablation) end-to-end."""))

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
