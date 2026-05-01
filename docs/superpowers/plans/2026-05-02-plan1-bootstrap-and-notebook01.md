# Medical Reasoning LLM — Plan 1: Bootstrap + Notebook 01

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bootstrap the project structure (git, dirs, src/ modules with tests, pinned requirements, config) and build Notebook 01 — environment setup, dataset exploration, formatter sanity-checks, and a single inference smoke test on Kaggle.

**Architecture:** Notebook-first development for *learning*; reusable Python in `src/` with pytest tests so the same code runs in Kaggle and in the consolidated `train_sft.py`/`llm_judge.py` deliverables. Project is git-tracked locally; notebooks `git clone` the repo on Kaggle to access `src/` and `configs/`.

**Tech Stack:** Python 3.10+, Unsloth (QLoRA), TRL SFTTrainer, PEFT, Transformers, datasets, bitsandbytes, pytest, PyYAML, pandas. Kaggle GPU (T4 16 GB) for training notebooks; laptop/Colab for src/ tests + early exploration.

**Days covered:** Day 0 (bootstrap, ~1 h) + Day 1 (2 h, 2026-05-02).

**Spec reference:** `docs/superpowers/specs/2026-05-02-medical-reasoning-llm-design.md`

---

## File Structure (created in this plan)

```
d:\train a model\
├── .gitignore                                    # Task 0.2
├── README.md                                     # Task 0.9
├── requirements.txt                              # Task 0.3 (overwrite existing)
├── configs\
│   └── experiment_config.yaml                    # Task 0.4
├── src\
│   ├── __init__.py                               # Task 0.5
│   ├── data_formatting.py                        # Task 0.6 (with tests)
│   ├── inference.py                              # stub only — fleshed out in Plan 3
│   ├── metrics.py                                # stub only — fleshed out in Plan 3
│   └── safety_rubric.py                          # stub only — fleshed out in Plan 3
├── tests\
│   ├── __init__.py                               # Task 0.5
│   └── test_data_formatting.py                   # Task 0.6
├── notebooks\
│   └── 01_setup_and_data_exploration.ipynb       # Tasks 1.1–1.10
├── design_doc.md                                 # already exists
├── train_sft.py                                  # already exists, untouched in this plan
├── llm_judge.py                                  # already exists, untouched in this plan
└── outputs\                                      # gitignored
    └── .gitkeep                                  # Task 0.2
```

`d:\train a model\docs\superpowers\specs\…` and `d:\train a model\docs\superpowers\plans\…` already exist.

---

# Phase A — Bootstrap (Day 0, ~1 hour)

## Task 0.1: Initialize git repo and verify clean state

**Files:** none yet.

- [ ] **Step 1: Initialize git**

```bash
cd "d:/train a model"
git init -b main
git status
```

Expected: lists untracked files (`Assignment_*.pdf`, `design_doc.md`, `train_sft.py`, `llm_judge.py`, `requirements.txt`, `design_doc_template.md`, `docs/`).

- [ ] **Step 2: Configure git identity locally for this repo (only if not already set globally)**

```bash
git config user.name "Abhishek Kumar Singh"
git config user.email "<your-email>"
```

Replace `<your-email>` with your real email. We use `--local` (default) so this doesn't touch your global config.

---

## Task 0.2: Write `.gitignore` and create `outputs/.gitkeep`

**Files:**
- Create: `d:\train a model\.gitignore`
- Create: `d:\train a model\outputs\.gitkeep`

- [ ] **Step 1: Write `.gitignore`**

```gitignore
# Python
__pycache__/
*.pyc
*.pyo
.pytest_cache/
.venv/
venv/

# Notebooks
.ipynb_checkpoints/

# Outputs (large files, model adapters, eval CSVs)
outputs/
!outputs/.gitkeep

# Secrets and env
.env
*.key
.kaggle/
.huggingface/

# OS
.DS_Store
Thumbs.db
desktop.ini

# IDE
.idea/
.vscode/
*.swp
```

- [ ] **Step 2: Create empty `outputs/.gitkeep`**

```bash
mkdir -p "d:/train a model/outputs"
type nul > "d:/train a model/outputs/.gitkeep"
```

(The `type nul > path` is the Windows equivalent of `touch`. On bash, `touch` works.)

---

## Task 0.3: Pin `requirements.txt`

**Files:**
- Modify: `d:\train a model\requirements.txt` (overwrite)

The existing `requirements.txt` uses `>=` ranges. We change to **pinned versions** for reproducibility. These versions are known-compatible as of April 2026; if Kaggle preinstalls newer versions, the notebooks will pin via `pip install` calls.

- [ ] **Step 1: Overwrite `requirements.txt`**

```
# Medical Reasoning LLM — Pinned dependencies (April 2026)
# Train/eval reproducibility relies on these versions.

# --- Core training stack ---
unsloth==2025.4.7
transformers==4.49.0
trl==0.15.2
peft==0.14.0
datasets==3.2.0
accelerate==1.3.0
bitsandbytes==0.45.2

# --- Experiment tracking ---
wandb==0.19.4

# --- Evaluation ---
evaluate==0.4.3
rouge-score==0.1.2
sacrebleu==2.4.3
bert-score==0.3.13
sentence-transformers==3.4.1

# --- LLM-as-judge clients ---
openai==1.61.1            # works for Groq via base_url
google-genai==0.7.0       # used only if Gemini becomes a third judge

# --- General ---
pandas==2.2.3
numpy==1.26.4
PyYAML==6.0.2
tqdm==4.67.1

# --- Test framework ---
pytest==8.3.4
```

> If a Kaggle/Colab session shows incompatibilities (Unsloth often pulls a
> specific torch), let pip resolve them at install time and just record what
> got installed in `outputs/<track>/training_meta.json` — that's our reproducibility
> hatch.

---

## Task 0.4: Write `configs/experiment_config.yaml`

**Files:**
- Create: `d:\train a model\configs\experiment_config.yaml`

This is the single source of truth that every notebook reads.

- [ ] **Step 1: Create `configs/experiment_config.yaml`**

```yaml
# configs/experiment_config.yaml
# Single source of truth for the medical-reasoning fine-tuning experiment.
# DO NOT change values per-track — only the formatter ID changes.

seed: 42

model:
  name: unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit
  max_seq_length: 2048
  load_in_4bit: true
  chat_template: qwen-2.5

dataset:
  name: OpenMed/Medical-Reasoning-SFT-GPT-OSS-120B-V2
  split: train
  shuffle_seed: 42
  num_train: 3000
  num_val: 100
  num_test: 200
  split_order: shuffle_then_split_then_format
  short_cot_max_tokens: 150
  short_cot_truncate_at: sentence_boundary

training:
  epochs: 1.0
  learning_rate: 2.0e-4
  lr_scheduler_type: cosine
  warmup_ratio: 0.03
  weight_decay: 0.01
  optim: adamw_8bit
  per_device_train_batch_size: 2
  gradient_accumulation_steps: 8
  effective_batch_size: 16
  gradient_checkpointing: unsloth
  eval_strategy: steps
  eval_steps: 50
  save_strategy: steps
  save_steps: 50
  save_total_limit: 2
  logging_steps: 10
  packing: false
  assistant_only_loss: true

lora:
  r: 16
  alpha: 16
  dropout: 0.05
  bias: none
  target_modules:
    - q_proj
    - k_proj
    - v_proj
    - o_proj
    - gate_proj
    - up_proj
    - down_proj
  use_rslora: false

inference:
  temperature: 0.0
  do_sample: false
  max_new_tokens:
    track_A: 400
    track_B: 400
  repetition_penalty: 1.0
  batch_size: 1
  warmup_runs: 3
  log_finish_reason: true
  log_truncated_flag: true

outputs:
  save_adapters_only: true
  push_to_hub: true
  private_hub_repo: true

logging:
  report_to: wandb
  fallback_report_to: none
  save_local_json_log: true
  wandb_project: medical-reasoning-sft
  wandb_runs:
    trackA: trackA-v0
    trackB: trackB-v0

hub:
  username: <your-hf-username>      # FILL before first push_to_hub
  repos:
    trackA: qwen25-1.5b-medreason-trackA-v0
    trackB: qwen25-1.5b-medreason-trackB-v0
```

> The `<your-hf-username>` placeholder is the only thing you'll edit in this
> file during the project. Everything else is locked.

---

## Task 0.5: Create `src/` and `tests/` package skeletons

**Files:**
- Create: `d:\train a model\src\__init__.py`
- Create: `d:\train a model\tests\__init__.py`

- [ ] **Step 1: Make `src/` a package**

```python
# src/__init__.py
"""Medical Reasoning LLM — reusable utilities.

Contents
--------
data_formatting   Track A / Track B formatters and short-CoT truncation.
inference         Greedy generation with latency/token logging.
metrics           EM, ROUGE-L, BERTScore, sacreBLEU wrappers.
safety_rubric     Manual-audit data structures and CSV writer.
"""
```

- [ ] **Step 2: Make `tests/` a package**

```python
# tests/__init__.py
```

(Empty file. Just marks `tests/` as a package so pytest discovery works
cleanly even if we add cross-test imports later.)

---

## Task 0.6: Build `src/data_formatting.py` — TDD

This module is **the central piece of the entire experiment** because the
A-vs-B comparison hinges on it. So we test-drive it.

**What this module exports:**

- `truncate_to_n_tokens(text, tokenizer, max_tokens, truncate_at_sentence) -> str`
- `format_for_track_a(messages, tokenizer, short_cot_max_tokens) -> dict` (formatted row)
- `format_for_track_b(messages) -> dict`
- `extract_answer_for_scoring(prediction: str, track: str) -> str`

**Files:**
- Create: `d:\train a model\src\data_formatting.py`
- Create: `d:\train a model\tests\test_data_formatting.py`

- [ ] **Step 1: Write the test file (failing)**

```python
# tests/test_data_formatting.py
"""Unit tests for src.data_formatting.

We use the actual Qwen2.5-1.5B-Instruct tokenizer (small download, ~2 MB)
so token-boundary truncation is realistic. The tokenizer is module-scoped.
"""
import pytest
from transformers import AutoTokenizer

from src.data_formatting import (
    truncate_to_n_tokens,
    format_for_track_a,
    format_for_track_b,
    extract_answer_for_scoring,
)


@pytest.fixture(scope="module")
def tok():
    # Use the bare instruct model (not the bnb-4bit wrapper) — its tokenizer
    # is identical and downloads faster.
    return AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")


# ---------- truncate_to_n_tokens ----------

def test_truncate_returns_input_when_under_limit(tok):
    text = "Short clinical note."
    out = truncate_to_n_tokens(text, tok, max_tokens=100,
                               truncate_at_sentence=True)
    assert out == text


def test_truncate_respects_token_budget(tok):
    text = " ".join(["word"] * 500)
    out = truncate_to_n_tokens(text, tok, max_tokens=50,
                               truncate_at_sentence=False)
    assert len(tok.encode(out, add_special_tokens=False)) <= 50


def test_truncate_at_sentence_ends_on_period(tok):
    text = ("Patient has fever. Likely viral infection. "
            "Treat with rest and fluids. " * 10)
    out = truncate_to_n_tokens(text, tok, max_tokens=20,
                               truncate_at_sentence=True)
    assert out.endswith(".") or out.endswith("?") or out.endswith("!")
    assert len(tok.encode(out, add_special_tokens=False)) <= 20


def test_truncate_at_sentence_falls_back_when_no_boundary(tok):
    # No sentence boundary in budget → fall back to hard truncation.
    text = "wordwordword " * 200
    out = truncate_to_n_tokens(text, tok, max_tokens=10,
                               truncate_at_sentence=True)
    # Must not crash and must respect token cap.
    assert len(tok.encode(out, add_special_tokens=False)) <= 10
    assert len(out) > 0


# ---------- format_for_track_a ----------

def test_track_a_includes_clinical_rationale_header(tok):
    row = {
        "messages": [
            {"role": "user", "content": "What is hypertension?"},
            {"role": "assistant",
             "content": "It is high blood pressure.",
             "reasoning_content": "Hypertension is defined as systolic ≥130 mmHg."},
        ]
    }
    out = format_for_track_a(row, tok, short_cot_max_tokens=150)
    assistant = out["messages"][-1]["content"]
    assert "Clinical rationale:" in assistant
    assert "Final answer:" in assistant
    assert "It is high blood pressure." in assistant


def test_track_a_truncates_long_reasoning(tok):
    row = {
        "messages": [
            {"role": "user", "content": "Q?"},
            {"role": "assistant",
             "content": "A.",
             "reasoning_content": "Long reasoning. " * 200},
        ]
    }
    out = format_for_track_a(row, tok, short_cot_max_tokens=50)
    assistant = out["messages"][-1]["content"]
    # The rationale section between headers must be ≤50 tokens.
    rationale = assistant.split("Clinical rationale:")[1].split("Final answer:")[0]
    assert len(tok.encode(rationale, add_special_tokens=False)) <= 60  # +slack for headers


def test_track_a_passes_through_user_message(tok):
    row = {"messages": [
        {"role": "user", "content": "Original question"},
        {"role": "assistant", "content": "A", "reasoning_content": "R"},
    ]}
    out = format_for_track_a(row, tok, short_cot_max_tokens=150)
    assert out["messages"][0] == {"role": "user", "content": "Original question"}


# ---------- format_for_track_b ----------

def test_track_b_drops_reasoning():
    row = {"messages": [
        {"role": "user", "content": "Q?"},
        {"role": "assistant",
         "content": "It is high blood pressure.",
         "reasoning_content": "This should NOT appear."},
    ]}
    out = format_for_track_b(row)
    assistant = out["messages"][-1]["content"]
    assert assistant == "It is high blood pressure."
    assert "should NOT appear" not in assistant
    assert "Clinical rationale" not in assistant


def test_track_b_passes_through_user_message():
    row = {"messages": [
        {"role": "user", "content": "Original question"},
        {"role": "assistant", "content": "A", "reasoning_content": "R"},
    ]}
    out = format_for_track_b(row)
    assert out["messages"][0] == {"role": "user", "content": "Original question"}


# ---------- extract_answer_for_scoring ----------

def test_extract_track_a_pulls_text_after_final_answer():
    pred = ("Clinical rationale:\n1. fever\n2. viral\n\n"
            "Final answer:\nRest and fluids.")
    assert extract_answer_for_scoring(pred, "A") == "Rest and fluids."


def test_extract_track_a_strips_safety_note():
    pred = ("Clinical rationale:\n...\n\n"
            "Final answer:\nTake paracetamol 500mg.\n\n"
            "Safety note: Consult a physician.")
    assert extract_answer_for_scoring(pred, "A") == "Take paracetamol 500mg."


def test_extract_track_a_returns_full_text_if_no_final_answer_marker():
    # Defensive: if generation degenerated and didn't produce the marker,
    # return the prediction unchanged so EM still computes (will likely score 0).
    pred = "Some malformed output."
    assert extract_answer_for_scoring(pred, "A") == "Some malformed output."


def test_extract_track_b_strips_trailing_safety_note():
    pred = "Take paracetamol 500mg.\n\nSafety note: Consult a physician."
    assert extract_answer_for_scoring(pred, "B") == "Take paracetamol 500mg."


def test_extract_track_b_returns_unchanged_when_no_safety_note():
    pred = "Take paracetamol 500mg."
    assert extract_answer_for_scoring(pred, "B") == "Take paracetamol 500mg."
```

- [ ] **Step 2: Run tests to verify they fail with module-not-found**

```bash
cd "d:/train a model"
python -m pytest tests/test_data_formatting.py -v
```

Expected: `ModuleNotFoundError: No module named 'src.data_formatting'`. That's the
"red" of red-green-refactor.

- [ ] **Step 3: Write `src/data_formatting.py` to satisfy the tests**

```python
# src/data_formatting.py
"""Track A / Track B formatters and short-CoT truncation.

Track A = primary system: short clinical reasoning + final answer.
Track B = baseline ablation: final answer only.

Both formatters pass user messages through unchanged. Only the assistant
content differs.

The `extract_answer_for_scoring` function is used by the evaluation
pipeline to compare only the final answer against the gold reference,
not the rationale.
"""

from __future__ import annotations
import re
from typing import Any

# Sentence-end characters used for "truncate at sentence boundary".
_SENTENCE_END = (".", "?", "!")


def truncate_to_n_tokens(
    text: str,
    tokenizer: Any,
    max_tokens: int,
    truncate_at_sentence: bool = True,
) -> str:
    """Truncate `text` to at most `max_tokens` tokens (per `tokenizer`).

    If `truncate_at_sentence` is True, walk back to the last sentence-end
    within the budget. If no sentence-end exists in the budget, fall back
    to hard truncation.
    """
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) <= max_tokens:
        return text

    truncated = tokenizer.decode(ids[:max_tokens], skip_special_tokens=True)

    if not truncate_at_sentence:
        return truncated

    last_boundary = max(truncated.rfind(c) for c in _SENTENCE_END)
    if last_boundary == -1:
        return truncated  # no sentence boundary in budget — keep hard cut
    return truncated[: last_boundary + 1]


def format_for_track_a(
    row: dict,
    tokenizer: Any,
    short_cot_max_tokens: int = 150,
) -> dict:
    """Reshape an OpenMed row to Track-A output format.

    Output assistant content:
        Clinical rationale:
        <truncated reasoning_content>

        Final answer:
        <content>
    """
    out_messages: list[dict] = []
    for msg in row["messages"]:
        if msg["role"] == "user":
            out_messages.append({"role": "user", "content": msg["content"]})
            continue
        if msg["role"] == "assistant":
            content = (msg.get("content") or "").strip()
            reasoning = (msg.get("reasoning_content") or "").strip()
            short = truncate_to_n_tokens(
                reasoning, tokenizer, short_cot_max_tokens,
                truncate_at_sentence=True,
            )
            assistant = (
                f"Clinical rationale:\n{short}\n\n"
                f"Final answer:\n{content}"
            )
            out_messages.append({"role": "assistant", "content": assistant})
            continue
        # System / tool messages pass through unchanged.
        out_messages.append(msg)
    return {"messages": out_messages}


def format_for_track_b(row: dict) -> dict:
    """Reshape an OpenMed row to Track-B output format.

    Output assistant content = answer only (`content` field, stripped).
    `reasoning_content` is dropped entirely.
    """
    out_messages: list[dict] = []
    for msg in row["messages"]:
        if msg["role"] == "user":
            out_messages.append({"role": "user", "content": msg["content"]})
            continue
        if msg["role"] == "assistant":
            content = (msg.get("content") or "").strip()
            out_messages.append({"role": "assistant", "content": content})
            continue
        out_messages.append(msg)
    return {"messages": out_messages}


# Pre-compiled regexes for answer extraction.
_TRACK_A_FINAL = re.compile(r"Final answer:\s*\n?(?P<answer>.*?)(?:\n\nSafety note:.*)?$",
                            re.DOTALL)
_TRACK_B_SAFETY = re.compile(r"\n\nSafety note:.*$", re.DOTALL)


def extract_answer_for_scoring(prediction: str, track: str) -> str:
    """Return only the final answer text — used for EM/ROUGE/BERTScore.

    For Track A, return text after `Final answer:` (and before any
    `Safety note:` block).
    For Track B, strip any trailing `Safety note:` block.
    Defensive: if no marker is found, return the prediction unchanged so
    metrics can still be computed (will usually score 0, which surfaces
    the format failure in the report).
    """
    if track.upper().startswith("A"):
        m = _TRACK_A_FINAL.search(prediction)
        if m is None:
            return prediction.strip()
        return m.group("answer").strip()

    # Track B
    return _TRACK_B_SAFETY.sub("", prediction).strip()
```

- [ ] **Step 4: Run tests to verify all pass**

```bash
python -m pytest tests/test_data_formatting.py -v
```

Expected: 13 tests pass. If any fail, fix them before continuing.

- [ ] **Step 5: Commit**

```bash
cd "d:/train a model"
git add .gitignore outputs/.gitkeep requirements.txt configs/experiment_config.yaml src/__init__.py src/data_formatting.py tests/__init__.py tests/test_data_formatting.py
git commit -m "feat: bootstrap project — config, data_formatting with tests"
```

---

## Task 0.7: Stub `src/inference.py`, `src/metrics.py`, `src/safety_rubric.py`

These are fleshed out in Plans 2 and 3. We stub them now so imports work.

**Files:**
- Create: `d:\train a model\src\inference.py`
- Create: `d:\train a model\src\metrics.py`
- Create: `d:\train a model\src\safety_rubric.py`

- [ ] **Step 1: Stub `inference.py`**

```python
# src/inference.py
"""Greedy generation with latency / token logging.

Filled in during Plan 3 (Notebook 04). Stub now so imports work.
"""

def generate_with_logging(*args, **kwargs):
    raise NotImplementedError("Implemented in Plan 3.")
```

- [ ] **Step 2: Stub `metrics.py`**

```python
# src/metrics.py
"""EM, ROUGE-L, BERTScore, sacreBLEU wrappers.

Filled in during Plan 3 (Notebook 04). Stub now so imports work.
"""

def compute_em(*args, **kwargs):
    raise NotImplementedError("Implemented in Plan 3.")

def compute_rouge_l(*args, **kwargs):
    raise NotImplementedError("Implemented in Plan 3.")

def compute_bertscore(*args, **kwargs):
    raise NotImplementedError("Implemented in Plan 3.")
```

- [ ] **Step 3: Stub `safety_rubric.py`**

```python
# src/safety_rubric.py
"""Manual-audit data structures and CSV writer.

Filled in during Plan 3 (Notebook 05). Stub now so imports work.
"""

def make_audit_csv(*args, **kwargs):
    raise NotImplementedError("Implemented in Plan 3.")
```

- [ ] **Step 4: Commit**

```bash
git add src/inference.py src/metrics.py src/safety_rubric.py
git commit -m "chore: stub inference/metrics/safety_rubric modules"
```

---

## Task 0.8: Push to GitHub (private)

This lets Kaggle notebooks `git clone` the repo to access `src/` and `configs/`.

**Files:** none.

- [ ] **Step 1: Create a private GitHub repo**

In your browser: github.com → New → name `medical-reasoning-llm` → **Private** → no README/license/gitignore (we already have files locally) → Create.

- [ ] **Step 2: Add the remote and push**

```bash
git remote add origin https://github.com/<your-gh-username>/medical-reasoning-llm.git
git branch -M main
git push -u origin main
```

If 2FA is on, generate a personal access token at github.com/settings/tokens (classic, scope `repo`) and use it as the password when prompted.

- [ ] **Step 3: Verify clone works**

```bash
cd "$TEMP" && git clone https://github.com/<your-gh-username>/medical-reasoning-llm.git _verify
ls _verify/src
rm -rf _verify
```

Expected: prints `__init__.py data_formatting.py inference.py metrics.py safety_rubric.py`.

---

## Task 0.9: README — bootstrap notice

**Files:**
- Create: `d:\train a model\README.md`

A minimal README so visitors (and future-you) know what the project is.

- [ ] **Step 1: Write `README.md`**

```markdown
# Medical Reasoning LLM

Fine-tune a small open-weight LLM on `OpenMed/Medical-Reasoning-SFT-GPT-OSS-120B-V2`
and compare with-reasoning vs answer-only training. Learning artefact —
not a clinical product.

- **Spec:** [docs/superpowers/specs/2026-05-02-medical-reasoning-llm-design.md](docs/superpowers/specs/2026-05-02-medical-reasoning-llm-design.md)
- **Phase-1 design doc:** [design_doc.md](design_doc.md)
- **Plan 1 (current):** [docs/superpowers/plans/2026-05-02-plan1-bootstrap-and-notebook01.md](docs/superpowers/plans/2026-05-02-plan1-bootstrap-and-notebook01.md)

## Quick start (local, src/ tests)

```bash
pip install -r requirements.txt
pytest tests/ -v
```

## Quick start (Kaggle, training notebooks)

Inside a Kaggle notebook:

```python
!git clone https://github.com/<your-gh-username>/medical-reasoning-llm.git
%cd medical-reasoning-llm
!pip install -q -r requirements.txt
```

(Adapters and CSVs are gitignored under `outputs/`.)
```

- [ ] **Step 2: Commit and push**

```bash
git add README.md
git commit -m "docs: minimal README"
git push
```

---

# Phase B — Notebook 01: Setup + Data Exploration (Day 1, 2 h, 2026-05-02)

The notebook is built **section-by-section** as separate tasks. After each
task, you save the notebook (Ctrl-S in Jupyter) and commit.

> **Where to develop this notebook?** Author the notebook in **Kaggle**
> directly (so the GPU env is the same as later training notebooks).
> Each task below = "add cell(s) to the Kaggle notebook." After the
> notebook is complete, **download the .ipynb** and save it under
> `notebooks/01_setup_and_data_exploration.ipynb` in the repo, commit,
> and push.

> **Kaggle notebook setup (one-time):**
> - kaggle.com → New Notebook
> - Settings panel → Accelerator: **GPU T4 x1** → Internet: **On**
> - Save name: `medreason-01-setup-and-data-exploration`
> - Add Kaggle Secrets (icon: lock): `HF_TOKEN`, `WANDB_API_KEY`,
>   `GROQ_API_KEY` (Groq isn't used in this notebook but seeding it now
>   avoids round-tripping later).

---

## Task 1.1: Notebook header (markdown cell)

**Files:**
- Add cell to: Kaggle notebook `medreason-01-setup-and-data-exploration`

- [ ] **Step 1: Add a markdown cell at the top**

```markdown
# Notebook 01 — Setup and Data Exploration

**Day 1 of 7 (2026-05-02).** Goal: working Kaggle environment + first look at the
OpenMed medical-reasoning dataset + sanity-check that the Track A and Track B
formatters produce sensible outputs on real rows.

This notebook does **not** train anything. Training starts in Notebook 02.

## What you'll learn here
1. How tokenizers turn text into IDs and back, and what a chat template does.
2. The structure of the OpenMed dataset (`messages`, `content`, `reasoning_content`).
3. What 4-bit quantization does to model size, and why it matters on a 16 GB T4.
4. How Track A's "Clinical rationale + Final answer" output looks vs Track B's
   answer-only output.
5. A single end-to-end inference smoke test.

## Plan reference
`docs/superpowers/plans/2026-05-02-plan1-bootstrap-and-notebook01.md` — Phase B.
```

---

## Task 1.2: Clone repo and install dependencies

**Files:**
- Add cells to Kaggle notebook.

- [ ] **Step 1: Markdown cell — explain**

```markdown
## 1. Bootstrap: clone repo, install deps

Kaggle's preinstalled environment has *most* of what we need but not all.
We pin our exact versions from `requirements.txt`. Total install ≈ 2–4 min.
```

- [ ] **Step 2: Code cell — clone**

```python
# Clone the project so we can `import src.data_formatting` etc.
import subprocess, sys, os

REPO_URL = "https://github.com/<your-gh-username>/medical-reasoning-llm.git"
REPO_DIR = "/kaggle/working/medical-reasoning-llm"

if not os.path.exists(REPO_DIR):
    subprocess.run(["git", "clone", REPO_URL, REPO_DIR], check=True)

# Make src/ importable
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

print("Repo cloned to:", REPO_DIR)
print("Files:", os.listdir(REPO_DIR))
```

- [ ] **Step 3: Code cell — install pinned deps**

```python
!pip install -q -r {REPO_DIR}/requirements.txt
```

(Expect a few "package X is incompatible with Y" notices — Unsloth pulls a
specific torch. Read them but don't worry unless something fails to import
later.)

- [ ] **Step 4: Code cell — verify imports**

```python
import importlib, json
for pkg in ["unsloth", "transformers", "trl", "peft", "datasets",
            "bitsandbytes", "accelerate"]:
    m = importlib.import_module(pkg)
    print(f"  {pkg:20s} {getattr(m, '__version__', '?')}")
```

Expected output: each library prints a version. Save these — we'll record them
in `training_meta.json` later.

---

## Task 1.3: Load secrets (HF, W&B, Groq)

- [ ] **Step 1: Markdown cell**

```markdown
## 2. Secrets

We use Kaggle Secrets (icon: lock in the right sidebar). Three keys:
- `HF_TOKEN` — Hugging Face token with **write** access (needed in Day 2 to
  push private adapters).
- `WANDB_API_KEY` — Weights & Biases.
- `GROQ_API_KEY` — Groq (used by Notebook 05).

If a secret is missing, the corresponding step is skipped (we won't crash
the whole notebook).
```

- [ ] **Step 2: Code cell — load and check**

```python
from kaggle_secrets import UserSecretsClient

secrets = UserSecretsClient()

def _try_get(name):
    try:
        return secrets.get_secret(name)
    except Exception as e:
        print(f"  [skip] {name}: {e.__class__.__name__}")
        return None

os.environ["HF_TOKEN"]       = _try_get("HF_TOKEN")        or ""
os.environ["WANDB_API_KEY"]  = _try_get("WANDB_API_KEY")   or ""
os.environ["GROQ_API_KEY"]   = _try_get("GROQ_API_KEY")    or ""

print("HF_TOKEN set:    ", bool(os.environ["HF_TOKEN"]))
print("WANDB_API_KEY set:", bool(os.environ["WANDB_API_KEY"]))
print("GROQ_API_KEY set: ", bool(os.environ["GROQ_API_KEY"]))
```

Expected: all three `True`. If any are False, add them via Kaggle Secrets
panel before continuing.

- [ ] **Step 3: Code cell — HF login**

```python
from huggingface_hub import login as hf_login
if os.environ["HF_TOKEN"]:
    hf_login(token=os.environ["HF_TOKEN"], add_to_git_credential=False)
    print("HF login OK")
```

---

## Task 1.4: Load and explore the OpenMed dataset

- [ ] **Step 1: Markdown — explain dataset**

```markdown
## 3. Dataset — `OpenMed/Medical-Reasoning-SFT-GPT-OSS-120B-V2`

Synthetic medical QA distilled from GPT-OSS-120B. Each row is a
single-turn conversation:

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

We compare Track A (uses both `content` and a truncated `reasoning_content`)
vs Track B (uses only `content`). Same questions; different output formats.
```

- [ ] **Step 2: Code cell — load**

```python
from datasets import load_dataset

ds = load_dataset(
    "OpenMed/Medical-Reasoning-SFT-GPT-OSS-120B-V2",
    split="train",
    token=os.environ["HF_TOKEN"] or None,
)
print(f"Total rows: {len(ds):,}")
print(f"Columns: {ds.column_names}")
```

- [ ] **Step 3: Code cell — peek at one row**

```python
sample = ds[0]
print("Number of messages:", len(sample["messages"]))
for i, m in enumerate(sample["messages"]):
    print(f"\n--- message[{i}]  role={m['role']} ---")
    print("content       :", (m.get("content") or "")[:300], "...")
    if m["role"] == "assistant":
        print("reasoning_content:", (m.get("reasoning_content") or "")[:300], "...")
```

- [ ] **Step 4: Code cell — length statistics**

Critical check: does our 150-token short-CoT cap actually truncate? Or is
it a no-op?

```python
import numpy as np
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")

# Sample 200 rows for speed
sample_indices = np.random.RandomState(42).choice(len(ds), size=200, replace=False)

reasoning_lens, answer_lens = [], []
for i in sample_indices:
    row = ds[int(i)]
    asst = next(m for m in row["messages"] if m["role"] == "assistant")
    reasoning_lens.append(len(tok.encode(asst.get("reasoning_content") or "",
                                          add_special_tokens=False)))
    answer_lens.append(len(tok.encode(asst.get("content") or "",
                                       add_special_tokens=False)))

print(f"reasoning_content tokens — mean {np.mean(reasoning_lens):.0f},  "
      f"median {np.median(reasoning_lens):.0f}, p95 {np.percentile(reasoning_lens, 95):.0f}, "
      f"max {np.max(reasoning_lens):.0f}")
print(f"content (answer) tokens — mean {np.mean(answer_lens):.0f},      "
      f"median {np.median(answer_lens):.0f}, p95 {np.percentile(answer_lens, 95):.0f}, "
      f"max {np.max(answer_lens):.0f}")
print(f"\nFraction with reasoning > 150 tokens: "
      f"{np.mean(np.array(reasoning_lens) > 150):.1%}")
```

> **Why we measure this**: confirms our 150-token cap is meaningful (most
> reasoning blocks should be longer than 150). If <30% are >150, we'd
> reconsider the cap. The output of this cell goes in the report.

---

## Task 1.5: Apply formatters to a sample row (sanity check)

- [ ] **Step 1: Markdown**

```markdown
## 4. Formatters in action

Before training, we visually verify both formatters produce sensible
output on a real row. Pick one row with a long reasoning so we can see
the truncation actually trigger.
```

- [ ] **Step 2: Code cell — pick a long-reasoning row**

```python
# Find a row whose reasoning is > 200 tokens, so 150-token truncation matters.
for idx in sample_indices:
    row = ds[int(idx)]
    asst = next(m for m in row["messages"] if m["role"] == "assistant")
    rl = len(tok.encode(asst.get("reasoning_content") or "",
                         add_special_tokens=False))
    if rl > 200:
        long_idx = int(idx)
        break
print(f"Using row {long_idx} (reasoning = {rl} tokens)")
```

- [ ] **Step 3: Code cell — Track A formatter**

```python
from src.data_formatting import format_for_track_a, format_for_track_b

row = ds[long_idx]

# Track A
out_a = format_for_track_a(row, tok, short_cot_max_tokens=150)
asst_a = out_a["messages"][-1]["content"]
print("=" * 80)
print("TRACK A")
print("=" * 80)
print(asst_a[:1500])
print("...")
print(f"\n  → assistant length (Track A): "
      f"{len(tok.encode(asst_a, add_special_tokens=False))} tokens")
```

- [ ] **Step 4: Code cell — Track B formatter**

```python
out_b = format_for_track_b(row)
asst_b = out_b["messages"][-1]["content"]
print("=" * 80)
print("TRACK B")
print("=" * 80)
print(asst_b[:1500])
print(f"\n  → assistant length (Track B): "
      f"{len(tok.encode(asst_b, add_special_tokens=False))} tokens")
```

- [ ] **Step 5: Code cell — assertions**

```python
# Track A must include both headers and end with the answer text.
assert "Clinical rationale:" in asst_a
assert "Final answer:" in asst_a
assert asst_a.rstrip().endswith(row["messages"][-1]["content"].strip())

# Track B must NOT contain the rationale headers.
assert "Clinical rationale:" not in asst_b
assert "Final answer:" not in asst_b
assert asst_b == row["messages"][-1]["content"].strip()

print("✅ formatter sanity checks passed")
```

---

## Task 1.6: Apply chat template

- [ ] **Step 1: Markdown — explain chat templates**

```markdown
## 5. Chat templates

`tokenizer.apply_chat_template(messages, tokenize=False)` turns
`[{"role":"user","content":...}, {"role":"assistant","content":...}]` into
a single string the model knows how to parse. Qwen2.5 uses an
`<|im_start|>role\n...<|im_end|>` wrapping.

**Mismatch between the chat template at training and inference is
the #1 silent fine-tuning bug**, so we use the same Unsloth
`get_chat_template(tokenizer, "qwen-2.5")` everywhere.
```

- [ ] **Step 2: Code cell — install Unsloth template + render**

```python
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

print("=" * 80)
print("CHAT-TEMPLATED — TRACK B (first 1200 chars)")
print("=" * 80)
print(text_b[:1200])
```

You should see `<|im_start|>user\n…<|im_end|><|im_start|>assistant\n…<|im_end|>`.

---

## Task 1.7: Memory math cell (teaching)

- [ ] **Step 1: Markdown**

```markdown
## 6. Memory math — why 4-bit?

Qwen2.5-1.5B has ≈1.54B parameters. Memory cost depends on the dtype:

| Dtype | Bytes/param | Total weights memory |
|---|---|---|
| fp32 | 4 | ~6.2 GB |
| fp16 / bf16 | 2 | ~3.1 GB |
| int8 | 1 | ~1.5 GB |
| int4 (NF4) | 0.5 | ~0.78 GB |

But that's just **weights**. Training also needs activations, gradients,
and optimizer state. Adam's state is ~2× weights in fp32 →
~12 GB on top of weights. On a 16 GB T4 → fp16 fine-tuning is
infeasible.

**QLoRA's trick**: keep base weights frozen at 4-bit (~0.78 GB), train
only ~18M LoRA parameters on top (~70 MB in fp16). Adam state is on the
LoRA weights, not the base. Total VRAM expected: ~5–8 GB.

This is why the assignment uses QLoRA on a small GPU.
```

- [ ] **Step 2: Code cell — sanity print after model load (Task 1.8)**

(No code yet — combined into next task.)

---

## Task 1.8: Load the 4-bit model + tokenizer

- [ ] **Step 1: Markdown**

```markdown
## 7. Load `unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit`
```

- [ ] **Step 2: Code cell — load**

```python
import torch
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name      = "unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit",
    max_seq_length  = 2048,
    dtype           = None,        # auto-detect fp16/bf16
    load_in_4bit    = True,
)
tokenizer = get_chat_template(tokenizer, chat_template="qwen-2.5")
print("Model loaded.")
```

- [ ] **Step 3: Code cell — VRAM check**

```python
def _vram_used_gb():
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.memory_allocated() / 1e9

print(f"Allocated VRAM after model load: {_vram_used_gb():.2f} GB")
print(f"Total GPU VRAM: "
      f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
print(f"Device: {torch.cuda.get_device_name(0)}")
```

Expected: ~1.0–1.5 GB allocated for the 4-bit model on a fresh T4.

---

## Task 1.9: Inference smoke test (does the model talk?)

- [ ] **Step 1: Markdown**

```markdown
## 8. Smoke test: ask the *base* model a question

We're testing only that generation works end-to-end **before** any
fine-tuning. The base model has not seen our medical-reasoning training
data yet, so the answer may be generic. Inference should take 3–10
seconds.
```

- [ ] **Step 2: Code cell — generate**

```python
import time

FastLanguageModel.for_inference(model)   # ~2× faster generation

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
print(response)
```

> If this fails with OOM, restart the kernel and re-run from Task 1.2 —
> usually it's leftover state from a previous Kaggle session.

---

## Task 1.10: Save notebook, download, commit

- [ ] **Step 1: Markdown — final cell summarising findings**

```markdown
## 9. Day-1 takeaways

- Dataset has **<NN>** rows. Median reasoning is **<NN>** tokens; **<NN>%**
  exceed our 150-token cap.
- Model loads in **<X.X> GB** VRAM in 4-bit (out of 16 GB).
- Base-model inference works end-to-end; baseline tok/s ≈ **<NN>**.
- Track A formatter produces `Clinical rationale: … Final answer: …`
  outputs as expected; Track B formatter produces answer-only outputs.

Tomorrow (Day 2): train Track B (the simpler ablation) end-to-end.

Fill the `<NN>` placeholders with your actual numbers from earlier cells.
```

- [ ] **Step 2: Save the notebook in Kaggle (Ctrl-S), then "File → Download
  Notebook"**

- [ ] **Step 3: Place the downloaded `.ipynb` into the repo**

```bash
# On your laptop:
mkdir -p "d:/train a model/notebooks"
mv "$DOWNLOADS/medreason-01-setup-and-data-exploration.ipynb" \
   "d:/train a model/notebooks/01_setup_and_data_exploration.ipynb"
```

- [ ] **Step 4: Commit**

```bash
cd "d:/train a model"
git add notebooks/01_setup_and_data_exploration.ipynb
git commit -m "feat: notebook 01 — setup and data exploration (Day 1)"
git push
```

---

# Self-Review (run after writing the plan)

> This section is for me (the planner) to check the plan against the spec.
> Engineer can skip when executing.

**Spec coverage check (against `2026-05-02-medical-reasoning-llm-design.md`):**

| Spec section | Covered by tasks |
|---|---|
| §6 file layout | 0.2, 0.4, 0.5, 0.6, 0.7, 1.10 |
| §7 config | 0.4 (`configs/experiment_config.yaml`) |
| §8.2 final-answer extraction | 0.6 (`extract_answer_for_scoring` with tests) |
| §3 short-CoT truncation | 0.6 (`truncate_to_n_tokens` with tests) |
| §3 Track A/B formatters | 0.6 (`format_for_track_a`, `format_for_track_b` with tests) |
| §9 Day-1 schedule | 1.1–1.10 |
| Pinned requirements (correction #6 from review) | 0.3 |
| Private hub setup | 0.4 (`private_hub_repo: true`) |

Not covered in this plan (correctly — they belong to Plans 2 and 3):
- Training (Plan 2)
- Inference + metrics (Plan 3)
- LLM judge + audit (Plan 3)
- Final report (Plan 3)
- Consolidating `train_sft.py`/`llm_judge.py` (Plan 3)

**Placeholder scan:** All `<your-gh-username>` and `<your-hf-username>`
placeholders are explicitly called out as user-fill items in §14 of the
spec.

**Type consistency:** `format_for_track_a` and `format_for_track_b` both
return `dict` with key `"messages"`. `extract_answer_for_scoring` is
called consistently across tests. ✓

---

# Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-05-02-plan1-bootstrap-and-notebook01.md`.

Day 0 (bootstrap) is ~1 hour, mostly local setup. Day 1 is the 2-hour
Kaggle-side notebook work. Total: ~3 hours of clock time, but the
2-hour-per-day budget still holds because Day 0 is one-time setup.

Two execution options:

1. **Subagent-Driven (recommended)** — I dispatch a fresh subagent per
   task, review between tasks, fast iteration. Best when tasks are
   genuinely independent.

2. **Inline Execution** — Execute tasks in this session using
   executing-plans skill, batched with checkpoints for review.

For *this* learning project, I recommend **Inline Execution**: you
(Abhishek) are the engineer, the project is highly interactive, and you
want to learn the concepts as we go — not have a subagent run them
behind your back. We'll go task by task in real time.

**Which approach?**
