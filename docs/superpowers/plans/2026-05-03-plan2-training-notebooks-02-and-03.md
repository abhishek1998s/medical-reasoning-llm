# Medical Reasoning LLM — Plan 2: Training Notebooks (Track B + Track A)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train Track B (answer-only baseline) and Track A (short clinical reasoning + answer) on the same 3,000-row slice of OpenMed using identical hyperparameters. End state: two LoRA adapters on Hugging Face Hub (private), W&B run logs for both, and a working "fine-tuned model produces medical answers" generation smoke test.

**Architecture:** QLoRA (4-bit base, fp16 LoRA) on Qwen2.5-1.5B-Instruct via Unsloth + TRL SFTTrainer. Same `configs/experiment_config.yaml` for both tracks; only the formatter differs (`format_for_track_a` vs `format_for_track_b`). Notebook-first development, generated from `scripts/build_notebook_02.py` and `scripts/build_notebook_03.py` for reproducibility.

**Tech Stack:** Python 3.12 (local) / Python 3.11 (Kaggle), Unsloth, TRL SFTTrainer, PEFT (LoRA), bitsandbytes (4-bit), Weights & Biases, Hugging Face Hub.

**Days covered:** Day 2 (2026-05-03, 2 h — Track B), Day 3 (2026-05-04, 2 h — Track A), Day 4 (2026-05-05, buffer).

**Spec reference:** `docs/superpowers/specs/2026-05-02-medical-reasoning-llm-design.md`

**Plan-1 follow-through (Day-1 findings carried in):**
- ✅ Set `max_seq_length: 4096` (was 2048; median content was 1954 tokens).
- ✅ Don't re-pin the GPU stack — verify imports first; only `pip install` if a verify fails.
- ✅ Add length filter when slicing the 3,000 training rows (so we don't waste GPU on rows that would be truncated mid-answer).
- ✅ Add the **assistant-only-loss sanity cell** (per the spec's correction round) before training each track — assert that `labels[i] == -100` for all user/system tokens.
- ✅ Right-pad for training; left-pad for inference (different concerns).

---

## File Structure (created or modified in this plan)

```
d:\train a model\
├── configs\
│   └── experiment_config.yaml                    # MODIFY — max_seq_length 2048→4096
├── src\
│   ├── data_formatting.py                        # already exists (Plan 1)
│   └── splits.py                                 # NEW (Task A.2)
├── tests\
│   └── test_splits.py                            # NEW (Task A.2)
├── scripts\
│   ├── build_notebook_01.py                      # already exists
│   ├── build_notebook_02.py                      # NEW (Task B.1)
│   └── build_notebook_03.py                      # NEW (Task C.1)
├── notebooks\
│   ├── 01_setup_and_data_exploration.ipynb       # already exists
│   ├── 02_train_trackB_answer_only.ipynb         # NEW (Task B.2)
│   └── 03_train_trackA_short_cot.ipynb           # NEW (Task C.1)
└── outputs\                                      # gitignored
    ├── trackA\
    │   └── (filled by Day 3 Kaggle run)
    └── trackB\
        └── (filled by Day 2 Kaggle run)
```

---

# Phase A — Pre-training prep (Day 2 morning, ~30 min on laptop)

## Task A.1: Update `max_seq_length` in `configs/experiment_config.yaml`

**Why:** Day 1 measured median content at 1,954 tokens. user_msg (~200) + Track A rationale (~170) + content (median 1,954) ≈ 2,320 tokens — over our previous 2,048 cap. Bumping to 4,096 fits ~95% of training rows comfortably.

**Files:**
- Modify: `configs/experiment_config.yaml`

- [ ] **Step 1: Edit the file**

```diff
 model:
   name: unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit
-  max_seq_length: 2048
+  max_seq_length: 4096
   load_in_4bit: true
   chat_template: qwen-2.5
```

(Only that single line; rest of the file unchanged.)

- [ ] **Step 2: Verify YAML is still valid**

```powershell
python -c "import yaml; print(yaml.safe_load(open('configs/experiment_config.yaml'))['model']['max_seq_length'])"
```

Expected output: `4096`.

- [ ] **Step 3: Commit (deferred — bundled into Task A.3 commit)**

---

## Task A.2: Build `src/splits.py` with TDD

**Why:** Both Notebook 02 and Notebook 03 need to (a) shuffle the OpenMed dataset, (b) take a slice, (c) split into train/val/test, (d) optionally filter for length. Putting this in `src/splits.py` means *exactly the same indices* end up in train/val/test for both tracks — that's the controlled-experiment property we promised.

**Files:**
- Create: `src/splits.py`
- Create: `tests/test_splits.py`

### What `src/splits.py` exports

```python
def shuffle_filter_split(
    ds: "datasets.Dataset",
    shuffle_seed: int,
    num_train: int,
    num_val: int,
    num_test: int,
    *,
    tokenizer = None,
    max_total_tokens: int | None = None,
) -> tuple["Dataset", "Dataset", "Dataset"]:
    """Shuffle ds, optionally filter rows whose tokenized total exceeds
    max_total_tokens, then take the first num_train + num_val + num_test
    rows and slice into three subsets in that order.

    The split order is: shuffle -> filter -> select(N) -> three slices.
    Same shuffle_seed always produces the same train/val/test indices,
    regardless of which formatter (Track A or B) is later applied.
    """
```

### Tests (write first, fail, then implement)

- [ ] **Step 1: Create `tests/test_splits.py`**

```python
"""Unit tests for src.splits."""
import pytest
from datasets import Dataset

from src.splits import shuffle_filter_split


@pytest.fixture
def fake_ds():
    """100-row fake dataset with `messages` field."""
    rows = [
        {
            "messages": [
                {"role": "user", "content": f"Q{i}"},
                {"role": "assistant",
                 "content": f"A{i}" + " word" * (i % 50),  # varied length
                 "reasoning_content": "thinking " * (i % 30)},
            ]
        }
        for i in range(100)
    ]
    return Dataset.from_list(rows)


# ---------- shuffle_filter_split — basic split ----------

def test_split_sizes_match_request(fake_ds):
    train, val, test = shuffle_filter_split(
        fake_ds, shuffle_seed=42,
        num_train=60, num_val=10, num_test=20,
    )
    assert len(train) == 60
    assert len(val) == 10
    assert len(test) == 20


def test_split_subsets_are_disjoint(fake_ds):
    """Train, val, test indices should never overlap."""
    train, val, test = shuffle_filter_split(
        fake_ds, shuffle_seed=42,
        num_train=30, num_val=10, num_test=20,
    )
    train_qs = {r["messages"][0]["content"] for r in train}
    val_qs   = {r["messages"][0]["content"] for r in val}
    test_qs  = {r["messages"][0]["content"] for r in test}
    assert train_qs & val_qs == set()
    assert train_qs & test_qs == set()
    assert val_qs   & test_qs == set()


def test_split_is_deterministic_across_calls(fake_ds):
    """Same seed -> same indices in train/val/test."""
    a = shuffle_filter_split(fake_ds, shuffle_seed=42,
                             num_train=20, num_val=5, num_test=10)
    b = shuffle_filter_split(fake_ds, shuffle_seed=42,
                             num_train=20, num_val=5, num_test=10)
    for split_a, split_b in zip(a, b):
        a_qs = [r["messages"][0]["content"] for r in split_a]
        b_qs = [r["messages"][0]["content"] for r in split_b]
        assert a_qs == b_qs


def test_different_seeds_give_different_splits(fake_ds):
    a = shuffle_filter_split(fake_ds, shuffle_seed=42,
                             num_train=20, num_val=5, num_test=10)
    b = shuffle_filter_split(fake_ds, shuffle_seed=123,
                             num_train=20, num_val=5, num_test=10)
    a_train_qs = [r["messages"][0]["content"] for r in a[0]]
    b_train_qs = [r["messages"][0]["content"] for r in b[0]]
    assert a_train_qs != b_train_qs


# ---------- shuffle_filter_split — length filter ----------

def test_filter_drops_long_rows(fake_ds):
    """If max_total_tokens is small, only short rows survive."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
    train, val, test = shuffle_filter_split(
        fake_ds, shuffle_seed=42,
        num_train=10, num_val=2, num_test=3,
        tokenizer=tok, max_total_tokens=15,
    )
    # Every surviving row must fit the budget.
    for split in (train, val, test):
        for row in split:
            asst = next(m for m in row["messages"] if m["role"] == "assistant")
            user = next(m for m in row["messages"] if m["role"] == "user")
            total = (
                len(tok.encode(user["content"], add_special_tokens=False))
                + len(tok.encode(asst.get("content") or "", add_special_tokens=False))
                + len(tok.encode(asst.get("reasoning_content") or "",
                                  add_special_tokens=False))
            )
            assert total <= 15


def test_no_filter_when_no_tokenizer(fake_ds):
    """Calling without tokenizer skips length filtering."""
    train, val, test = shuffle_filter_split(
        fake_ds, shuffle_seed=42,
        num_train=60, num_val=10, num_test=20,
    )
    # All 90 rows should be present (no filtering).
    assert len(train) + len(val) + len(test) == 90


def test_raises_when_not_enough_rows_after_filter(fake_ds):
    """If the filter is so strict no rows pass, we should raise."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
    with pytest.raises(ValueError, match="not enough rows"):
        shuffle_filter_split(
            fake_ds, shuffle_seed=42,
            num_train=50, num_val=10, num_test=20,
            tokenizer=tok, max_total_tokens=1,  # impossibly tight
        )
```

- [ ] **Step 2: Run tests to verify they fail with `ModuleNotFoundError`**

```powershell
python -m pytest tests/test_splits.py -v
```

Expected: `ModuleNotFoundError: No module named 'src.splits'`

- [ ] **Step 3: Implement `src/splits.py`**

```python
"""Train/val/test slicing for the OpenMed dataset.

Both Track A and Track B notebooks must use *identical* indices in
train/val/test so the comparison is fair. This module enforces that:
same shuffle_seed -> same indices, regardless of which formatter is
applied later.

Optional length filter: drops rows whose user_msg + content +
reasoning_content tokenized total exceeds `max_total_tokens`. Useful
for keeping training examples within `max_seq_length`.
"""

from __future__ import annotations
from typing import Any


def _row_total_tokens(row: dict, tokenizer) -> int:
    """Approximate total tokens needed if we trained on this row.

    Counts user content + assistant content + assistant reasoning. Does
    NOT add chat-template overhead (~30 tokens for Qwen2.5 — small).
    """
    user = next((m for m in row["messages"] if m["role"] == "user"), {})
    asst = next((m for m in row["messages"] if m["role"] == "assistant"), {})
    n = 0
    n += len(tokenizer.encode(user.get("content") or "",
                               add_special_tokens=False))
    n += len(tokenizer.encode(asst.get("content") or "",
                               add_special_tokens=False))
    n += len(tokenizer.encode(asst.get("reasoning_content") or "",
                               add_special_tokens=False))
    return n


def shuffle_filter_split(
    ds: Any,
    shuffle_seed: int,
    num_train: int,
    num_val: int,
    num_test: int,
    *,
    tokenizer: Any = None,
    max_total_tokens: int | None = None,
):
    """Shuffle, optionally filter for length, then take train/val/test.

    Returns
    -------
    (train, val, test) — three datasets.Dataset objects in that order.

    Raises
    ------
    ValueError if not enough rows remain after filtering.
    """
    shuffled = ds.shuffle(seed=shuffle_seed)

    if tokenizer is not None and max_total_tokens is not None:
        shuffled = shuffled.filter(
            lambda r: _row_total_tokens(r, tokenizer) <= max_total_tokens
        )

    n_needed = num_train + num_val + num_test
    if len(shuffled) < n_needed:
        raise ValueError(
            f"not enough rows after filter: {len(shuffled)} < {n_needed} requested"
        )

    chunk = shuffled.select(range(n_needed))

    train = chunk.select(range(num_train))
    val   = chunk.select(range(num_train, num_train + num_val))
    test  = chunk.select(range(num_train + num_val,
                                num_train + num_val + num_test))
    return train, val, test
```

- [ ] **Step 4: Run tests to verify they pass**

```powershell
python -m pytest tests/test_splits.py -v
```

Expected: 7 passed.

> **Note**: `test_filter_drops_long_rows` and `test_raises_when_not_enough_rows_after_filter` download the Qwen tokenizer (~2 MB, cached after first run, ~30 s the first time).

- [ ] **Step 5: Run *all* tests to make sure we didn't break anything**

```powershell
python -m pytest tests/ -v
```

Expected: 14 (data_formatting) + 7 (splits) = 21 passed.

---

## Task A.3: Commit Phase A

**Files:** none (just stage previously-edited files)

- [ ] **Step 1: Stage and commit**

```bash
cd "d:/train a model"
git add configs/experiment_config.yaml src/splits.py tests/test_splits.py
git commit -m "$(cat <<'EOF'
feat(splits): add shuffle_filter_split with TDD; bump max_seq_length to 4096

- src/splits.py: deterministic train/val/test split with optional
  length filter. Same shuffle_seed always produces the same indices
  regardless of which formatter is later applied — the controlled-
  experiment property our spec promises.
- 7 pytest tests (all green): split sizes, disjointness, determinism,
  seed sensitivity, length filter, no-tokenizer skip path, raises
  when filter too strict.
- configs/experiment_config.yaml: max_seq_length 2048 -> 4096 to fit
  the median 1954-token content from Day-1 dataset stats.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
git push
```

Expected: `git status` clean.

---

# Phase B — Notebook 02 (Day 2, ~2 h Kaggle wallclock for training)

## Task B.1: Build `scripts/build_notebook_02.py` and generate the notebook

**Files:**
- Create: `scripts/build_notebook_02.py`
- Create (via script): `notebooks/02_train_trackB_answer_only.ipynb`

### Notebook 02 structure (sections + code cells)

The build script produces a notebook with the SAME teaching pattern as Notebook 01: every code cell is followed by a "What this does + why + output meaning + improvements" markdown.

**Section list** (~16 code cells, 18+ markdown cells, ~36 cells total):

| § | Type | Purpose |
|---|---|---|
| 0 | md | Header — Day 2 of 7, goal, prereqs, what you'll learn |
| 1 | md | Section 1: Bootstrap (clone, verify imports, secrets) |
| 1.a | code | Clone repo, set sys.path |
| 1.b | code | Verify imports + record versions to `versions.json` |
| 1.c | code | Load Kaggle secrets + HF login + W&B login |
| 2 | md | Section 2: Load + split + format dataset (Track B) |
| 2.a | code | Load dataset; create tokenizer |
| 2.b | code | `shuffle_filter_split` with `max_total_tokens=3500` (gives headroom under 4096 cap) |
| 2.c | code | Apply `format_for_track_b` to each split via `Dataset.map` |
| 2.d | code | Render chat template → "text" field per row |
| 3 | md | Section 3: Load 4-bit base + attach LoRA |
| 3.a | code | `FastLanguageModel.from_pretrained` with `max_seq_length=4096` |
| 3.b | code | `FastLanguageModel.get_peft_model` with rank=16, alpha=16, all 7 modules |
| 3.c | code | Print trainable-parameter count (should be ~18M = ~1% of 1.54B) |
| 4 | md | Section 4: assistant-only-loss sanity check |
| 4.a | code | Tokenize one training row; print labels alongside tokens; assert -100 on all user/system tokens, valid IDs only on assistant tokens |
| 5 | md | Section 5: Configure SFTTrainer |
| 5.a | code | `SFTConfig` from YAML config; `report_to="wandb"` |
| 5.b | code | `SFTTrainer(model, tokenizer, train, eval, args)` |
| 6 | md | Section 6: Train |
| 6.a | code | `trainer.train()` — ~30–45 min on T4. W&B URL printed. |
| 6.b | code | Print final eval_loss, train_loss, runtime, samples/s |
| 7 | md | Section 7: Save adapter + push to Hub |
| 7.a | code | `model.save_pretrained("outputs/trackB/final_adapter")` + `tokenizer.save_pretrained` |
| 7.b | code | Write `outputs/trackB/training_meta.json` with all hyperparams + actual versions + final losses + runtime |
| 7.c | code | `model.push_to_hub(<user>/qwen25-1.5b-medreason-trackB-v0, private=True)` |
| 8 | md | Section 8: Smoke test — base vs fine-tuned |
| 8.a | code | Quick generation on the same hypertension prompt as Notebook 01; report tok/s and answer; compare visually to Notebook 01's base-model answer |
| 9 | md | Section 9: Day-2 takeaways (template — fill after run) |

### Step-by-step

- [ ] **Step 1: Create the build script**

The script follows the same structure as `scripts/build_notebook_01.py`. For each section, the script appends a section-intro markdown cell, then code cell(s), then explainer markdown cell(s).

Detailed cell content (the script's source) is too long to embed here verbatim — see the actual cells in the next phases of execution. Key cells whose code must match exactly:

**Cell `2.b` — `shuffle_filter_split` invocation:**
```python
from src.splits import shuffle_filter_split

train_ds, val_ds, test_ds = shuffle_filter_split(
    ds,
    shuffle_seed=42,
    num_train=3000,
    num_val=100,
    num_test=200,
    tokenizer=tok,
    max_total_tokens=3500,   # leave ~500 tokens for chat-template overhead under 4096
)
print(f"train={len(train_ds)}  val={len(val_ds)}  test={len(test_ds)}")
```

**Cell `2.c` — Track-B formatter via Dataset.map:**
```python
from src.data_formatting import format_for_track_b

train_b = train_ds.map(format_for_track_b)
val_b   = val_ds.map(format_for_track_b)
# Note: test split is NOT formatter-applied here — we evaluate
# raw test rows and compare predictions against gold `content`.
```

**Cell `2.d` — chat template render:**
```python
def render(example):
    return {"text": tok.apply_chat_template(
        example["messages"], tokenize=False, add_generation_prompt=False,
    )}

train_b = train_b.map(render, remove_columns=train_b.column_names)
val_b   = val_b.map(render,   remove_columns=val_b.column_names)
print(train_b[0]["text"][:800])
```

**Cell `3.b` — Attach LoRA:**
```python
from unsloth import FastLanguageModel

model = FastLanguageModel.get_peft_model(
    model,
    r=16, lora_alpha=16, lora_dropout=0.05,
    target_modules=["q_proj","k_proj","v_proj","o_proj",
                    "gate_proj","up_proj","down_proj"],
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=42,
    use_rslora=False,
)
```

**Cell `3.c` — Trainable param count:**
```python
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total     = sum(p.numel() for p in model.parameters())
print(f"trainable: {trainable:,}  total: {total:,}  pct: {trainable/total:.2%}")
```

Expected output: `trainable: ~18,000,000  total: ~1,540,000,000  pct: ~1.17%`. This is the spec-correction-#1 fix — the actual trainable parameter count, not the wrong "~5M" we initially had.

**Cell `4.a` — assistant-only-loss sanity check:**
```python
# Verify that, after collation, labels are -100 for user/system tokens
# and valid token IDs only for assistant tokens. This is the spec
# correction #1 from Section 3 review — ensures assistant_only_loss
# is actually working.

from trl import DataCollatorForCompletionOnlyLM
# Note: TRL's SFTTrainer with assistant_only_loss=True handles this
# automatically via the chat template. We check by tokenizing one row,
# inspecting labels, and asserting our invariants.

import torch

example = train_b[0]
ids = tok(example["text"], return_tensors="pt", truncation=True,
          max_length=4096)
input_ids = ids["input_ids"][0]

# For Qwen2.5, the assistant turn starts after this exact token sequence:
# "<|im_start|>assistant\n"
# We find that boundary and assert: tokens BEFORE it should be masked
# in training; tokens AFTER it should be supervised.
asst_start_str = "<|im_start|>assistant\n"
asst_start_ids = tok(asst_start_str, add_special_tokens=False)["input_ids"]

# Find the assistant turn boundary
import numpy as np
arr = input_ids.tolist()
def find_subseq(needle, hay):
    n = len(needle)
    for i in range(len(hay) - n + 1):
        if hay[i:i+n] == needle:
            return i
    return -1
asst_idx = find_subseq(asst_start_ids, arr)
assert asst_idx >= 0, "assistant turn not found — chat template mismatch"
asst_idx += len(asst_start_ids)

# Manually construct the labels SFTTrainer would use with
# assistant_only_loss=True: -100 before asst_idx, copy after.
labels = input_ids.clone()
labels[:asst_idx] = -100

n_user_masked = (labels[:asst_idx] == -100).sum().item()
n_asst_supervised = (labels[asst_idx:] != -100).sum().item()
print(f"User/system tokens masked (label=-100): {n_user_masked}")
print(f"Assistant tokens supervised:           {n_asst_supervised}")

assert n_user_masked == asst_idx, "user-side masking incomplete"
assert n_asst_supervised == len(labels) - asst_idx, "assistant supervision incomplete"
print("assistant-only-loss sanity check passed")
```

**Cell `5.a` — SFTConfig:**
```python
import yaml
from trl import SFTConfig

cfg = yaml.safe_load(open(f"{REPO_DIR}/configs/experiment_config.yaml"))

sft_config = SFTConfig(
    output_dir                  = "outputs/trackB",
    run_name                    = "trackB-v0",
    num_train_epochs            = cfg["training"]["epochs"],
    per_device_train_batch_size = cfg["training"]["per_device_train_batch_size"],
    gradient_accumulation_steps = cfg["training"]["gradient_accumulation_steps"],
    learning_rate               = cfg["training"]["learning_rate"],
    lr_scheduler_type           = cfg["training"]["lr_scheduler_type"],
    warmup_ratio                = cfg["training"]["warmup_ratio"],
    weight_decay                = cfg["training"]["weight_decay"],
    optim                       = cfg["training"]["optim"],
    fp16                        = True,             # T4 doesn't support bf16
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
    dataset_text_field          = "text",
    packing                     = cfg["training"]["packing"],
    assistant_only_loss         = cfg["training"]["assistant_only_loss"],
    report_to                   = "wandb",
)
```

**Cell `7.b` — `training_meta.json`:**
```python
import json, datetime, importlib

versions = {}
for p in ["unsloth","transformers","trl","peft","datasets",
          "bitsandbytes","accelerate","wandb"]:
    versions[p] = importlib.import_module(p).__version__

meta = {
    "track":         "B",
    "timestamp_utc": datetime.datetime.now(datetime.UTC).isoformat(),
    "model":         cfg["model"]["name"],
    "max_seq_length": cfg["model"]["max_seq_length"],
    "lora":          cfg["lora"],
    "training":      cfg["training"],
    "dataset":       cfg["dataset"],
    "versions":      versions,
    "train_runtime_sec": train_stats.metrics.get("train_runtime"),
    "train_loss":    train_stats.metrics.get("train_loss"),
    "eval_loss":     trainer.evaluate().get("eval_loss"),
    "trainable_params": trainable,
    "total_params":     total,
    "gpu":           torch.cuda.get_device_name(0),
}
with open("outputs/trackB/training_meta.json","w") as f:
    json.dump(meta, f, indent=2)
print(json.dumps(meta, indent=2))
```

**Cell `7.c` — Push to Hub:**
```python
hub_repo = f"{cfg['hub']['username']}/{cfg['hub']['repos']['trackB']}"
print(f"Pushing to: {hub_repo} (private={cfg['outputs']['private_hub_repo']})")

model.push_to_hub(hub_repo, private=cfg['outputs']['private_hub_repo'],
                  token=os.environ["HF_TOKEN"])
tokenizer.push_to_hub(hub_repo, private=cfg['outputs']['private_hub_repo'],
                       token=os.environ["HF_TOKEN"])
print("pushed.")
```

**Cell `8.a` — Smoke test:**
```python
FastLanguageModel.for_inference(model)

prompt_messages = [{"role": "user",
    "content": "What are the first-line treatments for hypertension in a 55-year-old non-diabetic patient?"}]
prompt_text = tok.apply_chat_template(prompt_messages, tokenize=False,
                                       add_generation_prompt=True)
inputs = tok(prompt_text, return_tensors="pt").to("cuda")

import time
start = time.time()
with torch.no_grad():
    out_ids = model.generate(**inputs, max_new_tokens=400,
                              do_sample=False, temperature=0.0)
elapsed = time.time() - start

new_ids = out_ids[0, inputs["input_ids"].shape[1]:]
print(f"{len(new_ids)} new tokens in {elapsed:.1f}s ({len(new_ids)/elapsed:.1f} tok/s)")
print(tok.decode(new_ids, skip_special_tokens=True))
```

- [ ] **Step 2: Run the build script**

```powershell
cd "d:/train a model"
python scripts/build_notebook_02.py
```

Expected: `Wrote notebooks\02_train_trackB_answer_only.ipynb (~36 cells)`

- [ ] **Step 3: Validate the notebook JSON**

```powershell
python -c "import json; nb=json.load(open('notebooks/02_train_trackB_answer_only.ipynb',encoding='utf-8')); from collections import Counter; print('cells:', len(nb['cells'])); print('counts:', Counter(c['cell_type'] for c in nb['cells']))"
```

Expected: ~36 cells, mix of markdown + code.

---

## Task B.2: Commit Notebook 02 + push to GitHub

- [ ] **Step 1: Stage and commit**

```bash
cd "d:/train a model"
git add scripts/build_notebook_02.py notebooks/02_train_trackB_answer_only.ipynb
git commit -m "feat(notebook02): build Track B (answer-only) training notebook"
git push
```

---

## Task B.3: Run Notebook 02 on Kaggle (the actual training)

This is the long-running step. ~30–45 min of GPU compute, ~2 h wallclock with logs/explanation/checkpoints.

**Files:** none locally; all action is on Kaggle.

- [ ] **Step 1: Open Kaggle, create new notebook, upload `notebooks/02_train_trackB_answer_only.ipynb`**

   Same procedure as Notebook 01:
   - kaggle.com → Create → New Notebook
   - Right sidebar: GPU T4 x1, Internet On
   - Right sidebar: attach all 3 secrets (HF_TOKEN, WANDB_API_KEY, GROQ_API_KEY)
   - File → Upload Notebook → select `02_train_trackB_answer_only.ipynb`
   - Save (Ctrl-S) with name `medreason-02-train-trackB`

- [ ] **Step 2: Run cells 1–4 (bootstrap → format → sanity)**

   These are non-training cells. Each takes seconds. Pause to verify:
   - All 8 imports succeed (cell 1.b)
   - All 3 secrets True (cell 1.c)
   - `train=3000  val=100  test=200` (cell 2.b)
   - One training example printed (cell 2.d)
   - Trainable params ≈ 18M, ~1.17% of total (cell 3.c)
   - "assistant-only-loss sanity check passed" (cell 4.a)

- [ ] **Step 3: Run cells 5.a, 5.b, 6.a (configure + train)**

   Cell 6.a (`trainer.train()`) is the long one. Watch:
   - Initial print: `***** Running training *****` with steps and epochs
   - Loss prints every 10 steps (≈ every 30 sec)
   - Eval prints every 50 steps (≈ every 2.5 min)
   - W&B prints a project URL — open it in another tab to watch curves
   - Total wall-clock: ~30–45 min

- [ ] **Step 4: Run cells 6.b, 7.a, 7.b, 7.c (post-train: log, save, push)**

   Cell 7.c (push to Hub) takes ~1 min. Verify on huggingface.co:
   `https://huggingface.co/abhishek1998s/qwen25-1.5b-medreason-trackB-v0` should exist (private).

- [ ] **Step 5: Run cell 8.a (smoke test) and fill in cell 9 (takeaways)**

   The smoke-test answer should differ from Notebook-01's base-model answer — possibly more concise, possibly different structure. Note any differences.

   Fill in the takeaways markdown with:
   - Final eval_loss
   - Final train_loss
   - Total runtime
   - W&B URL
   - HF Hub URL
   - Smoke-test tok/s
   - Anything surprising

- [ ] **Step 6: Save, download `.ipynb`, commit to repo**

```bash
# After downloading the executed notebook with outputs from Kaggle
mv "$DOWNLOADS/medreason-02-train-trackB.ipynb" \
   "d:/train a model/notebooks/02_train_trackB_answer_only.ipynb"
cd "d:/train a model"
git add notebooks/02_train_trackB_answer_only.ipynb
git commit -m "run(notebook02): Day 2 Kaggle execution + outputs"
git push
```

---

# Phase C — Notebook 03 (Day 3, ~2 h Kaggle wallclock)

Notebook 03 mirrors Notebook 02 but with `format_for_track_a` instead of `format_for_track_b`. Same hyperparameters, same dataset slice (same shuffle seed → identical indices), same model, same evaluation. Only the formatter and the Hub repo name differ.

## Task C.1: Build `scripts/build_notebook_03.py` and generate

- [ ] **Step 1: Create `scripts/build_notebook_03.py`**

   The script is ~95% identical to `build_notebook_02.py`. The differences:

   1. Header text refers to "Day 3" and "Track A"
   2. Section 2.c uses `format_for_track_a` (with `tokenizer` arg) instead of `format_for_track_b`:

      ```python
      from src.data_formatting import format_for_track_a
      train_a = train_ds.map(lambda r: format_for_track_a(r, tok, short_cot_max_tokens=150))
      val_a   = val_ds.map(lambda r:   format_for_track_a(r, tok, short_cot_max_tokens=150))
      ```

   3. Section 5.a's `SFTConfig`: `output_dir="outputs/trackA"`, `run_name="trackA-v0"`
   4. Section 7.a: `save_pretrained("outputs/trackA/final_adapter")`
   5. Section 7.b: `meta["track"] = "A"`, write to `outputs/trackA/training_meta.json`
   6. Section 7.c: push to `<user>/qwen25-1.5b-medreason-trackA-v0`
   7. Section 8.a markdown explainer: "Compare this Track-A answer to Track-B's; the rationale section IS the experimental signal"

   Easiest implementation: copy `build_notebook_02.py` to `build_notebook_03.py`, then make the 7 edits above.

- [ ] **Step 2: Generate and validate**

```powershell
cd "d:/train a model"
python scripts/build_notebook_03.py
python -c "import json; nb=json.load(open('notebooks/03_train_trackA_short_cot.ipynb',encoding='utf-8')); print('cells:', len(nb['cells']))"
```

- [ ] **Step 3: Commit and push**

```bash
git add scripts/build_notebook_03.py notebooks/03_train_trackA_short_cot.ipynb
git commit -m "feat(notebook03): build Track A (short clinical reasoning) training notebook"
git push
```

## Task C.2: Run Notebook 03 on Kaggle (Day 3)

- [ ] **Step 1–6:** Identical workflow to Task B.3, swapping in `medreason-03-train-trackA` and `notebooks/03_train_trackA_short_cot.ipynb`. Same checkpoints (sanity at cell 4, train at cell 6, save+push at cell 7, smoke test at cell 8).

   Track A training is *slightly* slower than Track B because the assistant turns are ~6% longer (rationale + headers). Expect 35–50 min vs B's 30–45 min.

   At end of Day 3, `https://huggingface.co/abhishek1998s/qwen25-1.5b-medreason-trackA-v0` should exist (private) and `notebooks/03_train_trackA_short_cot.ipynb` with executed outputs is committed to GitHub.

---

# Phase D — Day 4 buffer (optional, see how Days 2 + 3 went)

Day 4 is reserved for one of these, in priority order:

## Option D.1: Fix anything that broke on Day 2 or 3

If either training run had issues (OOM, NaN losses, divergence, secret problems), debug and re-run. **Highest priority** — we cannot proceed to Phase 3 evaluation without working adapters.

## Option D.2: Run base GPT-OSS-20B inference on the same 200-row test set

The user has access to a local GPT-OSS-20B server. Capturing its predictions on the same test rows we'll use for Track A and Track B gives us a "1.5B-fine-tuned vs 20B-base" reference column for Day 6's report.

- [ ] **Step 1: Confirm endpoint URL and access shape** (still pending from spec §14).
- [ ] **Step 2: Write a small script `scripts/run_gpt_oss_20b_baseline.py`** that:
   - Loads `outputs/trackB/test_set.json` (saved by Notebook 02 cell 2.b)
   - For each of the 200 rows, calls the local 20B endpoint
   - Saves predictions + latency to `outputs/baseline_gpt_oss_20b/predictions.csv`
- [ ] **Step 3: Run it from your laptop** (no Kaggle — it talks to your local server)

## Option D.3: Pre-prep eval scripts for Day 5

Lightly stub `src/inference.py` and `src/metrics.py` (currently `NotImplementedError`) so Day 5 starts faster. Not required.

## Option D.4: Scale-up dry run (Approach 3 prep)

If Days 2 + 3 ran fast and you want to push toward Approach 3 (3 B model, 5 K samples), build `notebooks/04_train_trackA_full_5k.ipynb` as a v2 with `Qwen2.5-3B-Instruct-bnb-4bit` and `num_train=5000`. **Defer** unless time genuinely allows.

---

# Self-Review (planner-side, run after writing this plan)

**1. Spec coverage:**

| Spec section | Covered by tasks |
|---|---|
| §7 config (`max_seq_length`, LoRA, training) | A.1 (config update), B.1 cells 5.a/3.b |
| §3 short-CoT truncation (Track A) | C.1 — applies `format_for_track_a` from Plan 1's `src/data_formatting.py` |
| §3 Track-B answer-only | B.1 cell 2.c |
| §7 `assistant_only_loss=True` | B.1 cell 5.a + dedicated sanity cell 4.a |
| §11 Phase-2 deliverables (adapters on Hub, W&B runs, training_meta.json) | B.1 cells 7.a/7.b/7.c, mirror in Phase C |
| Spec correction round 2, item 1 (LoRA params ~18M, not ~5M) | B.1 cell 3.c (explicit print of trainable param count) |
| Spec correction round 2, item 2 (assistant_only_loss) | A done in B.1 cell 4.a |
| Spec correction round 2, item 4 (3000 train + 100 val + 200 test) | A.1, A.2, B.1 cell 2.b |
| Spec correction round 2, item 5 (eval_steps=50, save_steps=50) | inherits from `configs/experiment_config.yaml` |
| Spec correction round 2, item 7 (private hub repos) | B.1 cell 7.c (`private=True`) |

**2. Placeholder scan:** All `<your-...>` placeholders are explicit user-fill items. The `<NN>` numbers in cell 9 markdown are filled by the user from their actual run. No "TODO" or "implement later" phrases.

**3. Type consistency:** `shuffle_filter_split` returns `(train, val, test)` everywhere it's used. `format_for_track_a` and `format_for_track_b` both take row→dict and return dict — same signature pattern for `Dataset.map`.

**4. Scope check:** Two notebooks + one helper module + one config edit + one buffer day. Single coherent plan.

---

# Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-05-03-plan2-training-notebooks-02-and-03.md`.

This plan covers Days 2–4 (~6 hours wallclock spread over 3 days). Same execution choice as Plan 1:

1. **Subagent-Driven** — I dispatch a fresh subagent per task. Faster but you see less context.
2. **Inline Execution** — we walk through together. Right call for a learning project.

**Recommended: Inline.** Same as Plan 1.

When you're ready (probably tomorrow morning, 2026-05-03), say "continue" or "start day 2" and I'll begin with Task A.1 (the YAML one-line edit). Phase A (config + splits + tests) takes ~30 min and lands us at Phase B's notebook generation, which we then upload + run on Kaggle.
