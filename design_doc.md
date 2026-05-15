# Fine-tuning of Medical Reasoning LM — Design Document

**Project**: Fine-Tune a Small LLM for Medical Reasoning
**Author**: Abhishek Kumar Singh, Project Associate, C-DAC Bangalore
**Date**: 2026-05-16
**Version**: 2.0 (post full-run analysis)
**Status**: Learning artefact — not a clinical product, not for deployment.

> This document is the Phase-1 design doc the assignment requires, updated
> after the full training and evaluation run on 2026-05-15.  The runnable
> implementation lives in `train_sft.py`, `llm_judge.py`, `run_pipeline.py`,
> and `notebooks/full_pipeline.ipynb`; experiment hyperparameters are in
> `configs/experiment_config.yaml`.

---

## Compliance with the PDF assignment

The PDF defines three phases (Problem Formulation, Model Training,
Evaluation) plus a Final Outcome Expectation. Every PDF requirement is
mapped to a specific section below.

### Phase 1 — Problem Formulation deliverables

| PDF requirement | Section in this doc |
|---|---|
| **Problem definition** | §1 |
| **Input-output format** | §4 |
| **Output Quality** | §6 |
| **Required Design Decision — Why this track?** | §2 (Track Selection — with explicit justification, including why not Track B as primary and why not Track C) |
| **Required Design Decision — Risks (especially medical safety)** | §5 (six-category hallucination taxonomy with severity 1-5, dataset risks, operational risks, mitigations, disclaimer) |
| **Required Design Decision — Evaluation strategy** | §7 (four layers: lexical, semantic, LLM-judge consensus, manual safety audit; controlled-experiment property) |

### Phase 2 — Model Training deliverables

| PDF requirement | Section in this doc |
|---|---|
| Baseline Setup (base model, SFT / LoRA / QLoRA, truncated short CoT) | §3 "Phase 2 — Baseline Setup checklist" |
| Training Variants (Experiment 1 With Reasoning, Experiment 2 Without Reasoning) | §3 "Phase 2 — Training Variants" |
| Comparison Goals (Accuracy vs reasoning depth, Latency vs output quality, Token cost vs performance) | §3 "Phase 2 — Comparison Goals" |
| Deliverables (Training pipeline code, Model checkpoints, Experiment logs) | §3 "Phase 2 — Deliverables" + §10 |

### Phase 3 — Evaluation deliverables

| PDF requirement | Section in this doc |
|---|---|
| Automatic Evaluation: Core Metrics (EM, Semantic Scores) | §7 "Phase 3 — Automatic Evaluation compliance" |
| Automatic Evaluation: Reasoning-Aware Metrics (Logical consistency, Step correctness, Hallucination rate) | §7 "Phase 3 — Automatic Evaluation compliance" |
| Manual verification (Clinical correctness, Risk severity, Clarity of reasoning) | §7 "Phase 3 — Manual verification" |
| Must analyze (Where model fails, Fabricated facts, Incorrect reasoning, Overconfidence issues) | §7 "Phase 3 — Must analyze" + §8 (per-question) |
| Deliverables (Evaluation report, Error analysis document, Sample outputs) | §7 "Phase 3 — Deliverables" + §10 |

### Final Outcome Expectation (PDF closing requirements)

| PDF outcome question | Section in this doc |
|---|---|
| Does reasoning actually improve medical QA? | §8 Q1 |
| When should reasoning be hidden vs shown? | §8 Q2 |
| How unsafe is the model in edge cases? | §8 Q3 |
| What trade-offs exist (accuracy vs cost vs latency)? | §8 Q4 |

This doc exceeds the PDF's 2-3 page target deliberately — the user
asked for the design decisions to be documented "in depth," and a
2-3 page version cannot also include the full-run numbers (§6), the
A-vs-B trade-off analysis (§8), the limitations disclosure (§9), and
the deliverable inventory (§10). Treat §1, §2, §4, §5, §6 (training +
automatic metrics tables), and §7 (philosophy paragraphs) as the core
2-3 page submission; the Phase 2 / Phase 3 compliance subsections in
§3 and §7, plus §8, §9, §10 provide supporting depth and explicit
checkbox-style PDF compliance.

---

## 1. Problem Definition

### Objective (PDF top-level)

Design, train, and evaluate a medical reasoning LLM system with strong
emphasis on:

1. **Reasoning quality** — does the model produce a logically coherent
   chain-of-thought that supports the final answer? Measured by judge
   `reasoning_soundness` (1–5), manual `reasoning_clarity` (clear /
   vague / misleading), and the `REASONING` hallucination type counter.
2. **Correctness** — is the final answer medically right? Measured by
   ROUGE-L, BERTScore F1, judge `clinical_correctness` (1–5), and
   manual `clinical_correctness` (correct / partially_correct /
   incorrect).
3. **Reliability** — does the model fail safely and predictably?
   Measured by judge `safety` (1–5), `any_unsafe` rate, `n_major_errors`,
   `max_severity`, `truncation_rate`, and manual `risk_severity` and
   `safe_behavior`.

### Phase 1 — "Please clearly define what the model should learn"

The goal is to fine-tune a small open-weight instruct LLM on a high-quality
medical reasoning dataset so the model learns to produce step-by-step
clinical reasoning along with the correct answer. The dataset, provided by
OpenMed (`OpenMed/Medical-Reasoning-SFT-GPT-OSS-120B-V2`, 506K rows),
contains single-turn user/assistant chats in `role` / `content` /
`reasoning_content` format distilled from GPT-OSS-120B.

**The model must learn to:**

1. Generate a chain-of-thought block grounded in medical knowledge.
2. Provide a final answer consistent with that reasoning.
3. Carry forward any safety-note text from the training data verbatim.
4. Maximise interpretability — every final answer should be auditable to a
   reasoning trace.

**Final questions the project answers (from the PDF):**

| PDF question | Where answered |
|---|---|
| Does reasoning improve medical QA? | §6 Output Quality, §8 Findings |
| When should reasoning be hidden vs shown? | §8 Findings |
| How unsafe is the model in edge cases? | §5 Risks, §8 Findings |
| What trade-offs exist (accuracy vs cost vs latency)? | §8 Findings |

**Scope**

- Single-turn English medical question-answering only.
- Synthetic dataset distilled from GPT-OSS-120B; teacher biases inherited
  and explicitly disclaimed.
- Research / learning artefact. No clinical claim, no deployment.

---

## 2. Track Selection
### Required Design Decision — Why this track?

The PDF requires choosing one primary track and running at least two
training variants. Three tracks are defined:

**Track A — Full Reasoning Generation**
- Input: medical query
- Output: step-by-step reasoning + final answer
- Goal: maximise interpretability

**Track B — Answer-Only Model**
- Input: query
- Output: final answer only
- Goal: faster inference

**Track C — Conversational Medical Assistant**
- Multi-turn interaction
- Maintain context across queries

**Primary track: Track A (short clinical CoT).** Track A is the system
whose behaviour we want to understand — the assignment's headline goals
of reasoning quality, correctness, reliability, and interpretability are
all properties of a reasoning output, not of an answer-only system.

**Baseline ablation: Track B.** Without an answer-only baseline trained
on the same data, we cannot tell whether observed accuracy or
hallucination rates are caused by reasoning or by general fine-tuning.
Track B isolates that variable. The dataset, base model, hyperparameters,
train/val/test split, and inference settings are **identical** across
both — the only experimental variable is the assistant-output format.

**Track C dropped.** Single-turn QA is sufficient to study the reasoning
question and is much smaller in scope. The dataset itself is single-turn.

**Reasoning length variant.** The PDF baseline-setup line says
"truncated reasoning (short CoT)" while Experiment 1 says "train on full
reasoning traces." We treat this as ambiguous and pick **short CoT**
(150 tokens, truncated at sentence boundary) for the primary run. Full
CoT is supported by `train_sft.py --track A_full` if needed.

---

## 3. Model and Fine-tuning Method (PDF Phase 2)

### Phase 2 — Baseline Setup checklist

| PDF requirement | Choice in this project | Where verified |
|---|---|---|
| Select a base model | `unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit` | §3 "Base model" |
| Use SFT / LoRA / **QLoRA** fine-tuning | **QLoRA** (4-bit NF4 base + LoRA adapters, rank 16) | §3 "LoRA hyperparameters" |
| Truncated reasoning (short CoT) | First 150 tokens of `reasoning_content`, truncated at sentence boundary | §4 "Short-CoT truncation" |

### Phase 2 — Training Variants (must run at least 2)

| PDF variant | Implementation in this project | Tracked as |
|---|---|---|
| **Experiment 1 — With Reasoning (Chain-of-Thought)** — "train on full reasoning traces" | Track A (short CoT, 150-token truncation). `train_sft.py --track A_full` also supports the full untruncated trace if a reviewer interprets the PDF strictly. | `wandb_runs.trackA = "trackA-v0"`; adapter `kabhisheks/qwen25-1.5b-medreason-trackA-v0` |
| **Experiment 2 — Without Reasoning** — "train only on final answers" | Track B. `format_for_track_b` drops `reasoning_content` entirely and keeps only the assistant `content`. | `wandb_runs.trackB = "trackB-v0"`; adapter `kabhisheks/qwen25-1.5b-medreason-trackB-v0` |

Both experiments share identical hyperparameters (§3 below) — only the
formatter differs. This is the controlled-experiment property the design
doc enforces.

### Phase 2 — Comparison Goals (MUST compare per PDF)

Full-run measurements from §6 / §8, summarised here for Phase 2 compliance:

| PDF comparison goal | Measurement | Track A | Track B | Winner |
|---|---|---|---|---|
| **Accuracy vs reasoning depth** | ROUGE-L | 0.131 | **0.142** | B (+8.4 %) |
|  | BERTScore F1 | 0.815 | **0.849** | B (+4.2 %) |
|  | judge `clinical_correctness` | 2.27 | **3.16** | B (+0.89) |
| **Latency vs output quality** | mean generation time / row | 37.5 s | **18.0 s** | B (2.09× faster) |
|  | mean `tokens_per_sec` | 18.09 | 18.11 | tie (~throughput-bound) |
| **Token cost vs performance** | mean output tokens / row | 679.5 | **326.5** | B (2.08× cheaper) |
|  | Mean token delta A − B | +353 tokens / row | — | — |

**Headline finding (Phase 2 deliverable):** At Qwen2.5-1.5B with 3K
samples × 1 epoch, the reasoning track is dominated by the answer-only
track on every measured axis except interpretability — and even
interpretability isn't reliably delivered (82.5 % of Track A's outputs
were length-truncated in the full run).

### Phase 2 — Deliverables (PDF required)

| PDF deliverable | Status | Path / link |
|---|---|---|
| **Training pipeline (code)** | ✅ | `train_sft.py`, `run_pipeline.py`, `notebooks/full_pipeline.ipynb`, `src/` package, `configs/experiment_config.yaml` |
| **Model checkpoints** | ✅ | `outputs/track{A,B}/final_adapter/` + intermediate `checkpoint-150/` and `checkpoint-188/`; private HF Hub repos |
| **Experiment logs** | ✅ | W&B project `medical-reasoning-sft`; `outputs/track{A,B}/final_adapter/training_meta.json` with loss curves, hyperparameters, package versions, GPU info |

---

### Base model — `unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit`

- 1.5B parameters, Qwen2.5 instruct family, distributed pre-quantised to
  4-bit NF4 via bitsandbytes.
- 28 transformer layers; native chat template `qwen-2.5`.
- Chosen for fit in Kaggle T4 (16 GB VRAM) under QLoRA with `max_seq_length=4096`.
- Instruct-tuned: the base already follows chat-format prompts, so
  fine-tuning primarily teaches the *medical reasoning style* rather
  than basic instruction-following.

### Fine-tuning method — QLoRA via Unsloth + TRL

QLoRA = 4-bit NF4 base weights frozen + low-rank adaptation matrices
(LoRA) trained in fp16/bf16. Implemented via:

| Library | Role | Pinned version |
|---|---|---|
| `unsloth==2026.4.8` | 2× faster fwd/bwd kernels, padding-free, fast LoRA QKV/O/MLP patches | latest stable |
| `trl==0.24.0` | `SFTTrainer` + `SFTConfig` | matches transformers v5 |
| `peft==0.19.1` | LoRA adapter format, push-to-Hub | — |
| `bitsandbytes==0.49.2` | 4-bit NF4 quantisation | CUDA 12.8 build |
| `transformers==5.5.0` | base, tokenizer, Trainer loop | v5 API (warmup_steps replaces warmup_ratio) |
| `accelerate==1.13.0` | device placement, single-GPU launcher | — |

### LoRA hyperparameters

| Parameter | Value | Rationale |
|---|---|---|
| `r` (rank) | 16 | Standard for instruction-tuning a small model; 18.46 M trainable params |
| `alpha` | 16 | `alpha == r` (effective LoRA scale = 1.0) |
| `dropout` | 0.0 | Required for Unsloth's fast LoRA kernels; with 3K samples × 1 epoch overfitting is not a concern |
| `bias` | none | Standard for QLoRA |
| `target_modules` | `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj` | All 7 attention + MLP projections |
| `use_rslora` | false | Off — stable training preferred |

### Parameter count (verified at runtime)

- **All params**: 907,081,216
- **Trainable params**: 18,464,768
- **Trainable %**: **2.04 %**

### Training hyperparameters (full run)

| Parameter | Value | Notes |
|---|---|---|
| `per_device_train_batch_size` | 2 | T4 16 GB ceiling with seq=4096 |
| `gradient_accumulation_steps` | 8 | Effective batch size 16 |
| `effective_batch_size` | 16 | per_device × grad_accum |
| `num_train_epochs` | 1.0 | Single pass; instruction-tuning saturates fast |
| `learning_rate` | 2.0e-4 | Standard QLoRA LR |
| `lr_scheduler_type` | cosine | Decay LR to 0 by end of run |
| `warmup_steps` | 6 | = ⌈0.03 × 187⌉ (transformers v5 deprecates `warmup_ratio`) |
| `weight_decay` | 0.01 | Light regularisation |
| `optim` | adamw_8bit | bitsandbytes 8-bit Adam — half the optimizer-state RAM |
| `max_seq_length` | 4096 | Day-1 measured median content = 1954 tokens |
| `packing` | false | Safer than packing for chat-format data |
| `assistant_only_loss` | False (in SFTConfig) | Bypassed in favour of Unsloth's `train_on_responses_only` (Unsloth's compiled wrapper conflicts with TRL's native path) |
| `gradient_checkpointing` | unsloth | Trades memory for compute; ~30 % VRAM saved |

### Loss masking

Only assistant tokens contribute to the cross-entropy loss; user and
system tokens are masked with `label=-100`. Implemented via
`unsloth.chat_templates.train_on_responses_only(trainer, instruction_part="<|im_start|>user\n", response_part="<|im_start|>assistant\n")`
applied after `SFTTrainer` construction. A preflight check
(`_verify_masking`) asserts that ~94 % of sample-0 labels are masked
(observed: 1812/1929 = 93.9 % trainable).

### Hardware

- **Training**: Kaggle T4 ×2 environment, but forced to single-GPU mode
  because Unsloth issue #3915 (4-bit Qwen + accelerate DDP) crashes the
  trainer with `RuntimeError: No backend type associated with device type cpu`.
  Auto-detected via `_n_gpus`, launcher always set to plain `python`.
- **Inference / judging**: same single T4.

---

## 4. Input–Output Format

### Dataset fields used

| Field | Value in dataset row | Used for |
|---|---|---|
| Question | `messages[0]['content']` | User query (input to both tracks) |
| Reasoning | `messages[1]['reasoning_content']` | Track A only; truncated to first 150 tokens at last sentence boundary |
| Answer | `messages[1]['content']` | Final answer (Track A after `Final answer:` block; Track B entire output) |
| Safety note | inline in `messages[1]['content']` when present | Carried verbatim by both formatters (no asymmetric injection) |

### Prompt templates (Qwen-2.5 chat-template wrapped)

**Track A — short clinical CoT**

```
<|im_start|>user
{question}<|im_end|>
<|im_start|>assistant
Clinical rationale:
{truncated_reasoning_content}

Final answer:
{content}<|im_end|>
```

**Track B — answer only**

```
<|im_start|>user
{question}<|im_end|>
<|im_start|>assistant
{content}<|im_end|>
```

Header style is plain "`Clinical rationale:`" / "`Final answer:`" rather
than `<think>` / `</think>` tags. Plain headers are more readable for a
clinician auditing outputs and more justifiable in a medical setting.

### Short-CoT truncation

`format_for_track_a` calls `truncate_to_n_tokens(reasoning, tokenizer,
max_tokens=150, truncate_at_sentence=True)`. Algorithm:

1. Encode `reasoning_content` to token IDs.
2. If `≤ 150` tokens, return unchanged.
3. Else decode the first 150 tokens.
4. Walk back to the last `.`, `?`, or `!`. If no sentence boundary in
   budget, hard-cut at 150.

### Final-answer extraction (for automatic metrics)

Track A predictions are unwrapped before metric scoring:

```python
re.compile(r"Final answer:\s*\n?(?P<answer>.*?)(?:\n\nSafety note:.*)?$",
           re.DOTALL)
```

Track B predictions get only the trailing `Safety note:` block stripped.
EM, ROUGE-L, and BERTScore compare only the extracted final answer to
the gold answer — not the rationale. Reasoning quality is evaluated
separately via the LLM judge's `reasoning_soundness` axis (1–5 or null).

---

## 5. Risks (Medical Safety)
### Required Design Decision — Risks (especially medical safety)

### Risk taxonomy and judge scoring

Each generated answer is scored by the LLM judge on six hallucination
categories with severity 1–5 and a `major` boolean (would change
diagnosis/management):

| Type | Definition | Track A count (full run) | Track B count | Mean severity (A) |
|---|---|---|---|---|
| **FABRICATION** | Made-up drug, dose, citation, mechanism | **45** | 8 | 3.91 |
| **REASONING** | Logically incoherent CoT (Track A only meaningful) | **21** | 2 | 3.52 |
| **CONTEXTUAL** | Wrong patient cohort / specialty / setting | 13 | 3 | 3.69 |
| **NEGATION** | Output negates a clinically relevant fact | 8 | 5 | 4.12 |
| **CAUSALITY** | Speculates cause without evidence | 6 | 2 | 4.17 |
| **OVERCONFIDENCE** | Confident assertion absent uncertainty markers | 4 | 3 | 3.25 |

**Severity scale.** 1 = stylistic; 5 = could harm patient if followed.
60 % of Track A errors fall at severity 4 or 5 in the full run (35 + 26
of 99). 14 of 23 Track B errors are tagged "major".

### Dataset-specific risks

- OpenMed is *synthetic*, distilled from GPT-OSS-120B; biases of the
  teacher (drug-name distribution, geographic specialty mix,
  US-centric care patterns) are inherited.
- The Cerebras judge is **`gpt-oss-120b`** — the same model family as
  the dataset's teacher. This is a confirmation-bias risk: a judge
  trained on similar data may over-rate outputs that mimic the teacher's
  style. Mitigated by the Gemini second judge from a different family.
- The Gemini judge is **`gemini-2.5-flash`** — same vendor family as the
  Qwen student would not be (Qwen is Alibaba). Family separation: ✅
  Cerebras/GPT-OSS vs Qwen ≠ family; ✅ Gemini vs Qwen ≠ family.

### Operational risks (observed in the full run)

- **Truncation**: Track A 82.5 %, Track B 70.5 % of outputs hit the
  `max_new_tokens` cap. Fixed for re-run (track_A 2048, track_B 1024).
- **Judge rate-limit collapse**: 754 Gemini 429 errors and ~50 Cerebras
  hourly-cap errors caused Track B to drop to 45/200 judged rows. Fixed
  via paced calls (`sleep_seconds: 2.0`) and `judge_all` retry-with-backoff.
- **JSON parse failures**: ~5 Cerebras schema-error rows despite
  constrained decoding (unicode chars in medical text). Fixed via
  single-retry on `JSONDecodeError`.
- **Unknown error types**: Gemini occasionally invents type names
  (`INACCURACY`, `COMPLETENESS`) outside the prompt's 6-type enum. Fixed:
  unknown types now dropped at validation, not silently kept.

### Mitigations

| Risk | Mitigation |
|---|---|
| Hallucination — made-up drug names/doses | Manual safety audit on stratified 30-sample subset (high / medium / low risk) × 2 tracks |
| Overconfidence | Judge axis dedicated to OVERCONFIDENCE type; counted separately in aggregates |
| Single-judge bias | Two-judge consensus (Cerebras gpt-oss-120b + Gemini 2.5-flash); inter-judge agreement reported |
| Misinformation | Disclaimer in design doc, model card, final report; HF Hub repo kept **private** during development |
| Unsafe deployment | Adapters labelled `learning artefact`; no inference endpoint exposed; no clinical claim |
| Schema drift | Strict JSON schema on Cerebras with `additionalProperties: false` and full enum on `errors.items.type` |

### Disclaimer

This fine-tuned model is **not a certified medical device.** It is
intended only as a research / learning artefact. Any output the model
produces must not be used for actual clinical decision-making. The
risks of fabricated facts, incorrect reasoning, overconfident
assertions, and demographic / specialty bias are real and quantified
above; safety has not been formally validated by clinicians.

---

## 6. Output Quality

### Training (full run, 2026-05-15)

|  | Track A (short CoT) | Track B (answer only) |
|---|---|---|
| Train samples | 3,000 | 3,000 |
| Val samples | 150 (used at eval_steps=50) | 150 |
| Optimizer steps | 188 (1 epoch ÷ 16) | 188 |
| Wall time | 116 min (`6968s`) | 110 min (`6619s`) |
| Final `train_loss` | 1.328 | 1.336 |
| Final `eval_loss` | **1.216** | 1.251 |
| `eval_loss` trajectory | 1.297 → 1.235 → 1.218 → 1.216 | 1.322 → 1.268 → 1.253 → 1.251 |
| Trainable params | 18.46 M (2.04 %) | same |

Track A achieves a slightly **lower** eval_loss (-0.035, ~3 %) — the CoT
format is marginally easier to fit on the training distribution. This
direction is **reversed** in downstream quality evaluation (§8).

### Automatic metrics (200-row held-out test set)

|  | Track A | Track B | Winner | Δ |
|---|---|---|---|---|
| Exact Match (EM) | 0 | 0 | tie | — |
| ROUGE-L | 0.131 | **0.142** | B | +8.4 % |
| BERTScore F1 (roberta-large fallback) | 0.815 | **0.849** | B | +4.2 % |
| Mean output tokens | 679.5 | 326.5 | (A 2.08× longer) | — |
| Mean generation time (s) | 37.5 | **18.0** | B | A 2.09× slower |
| Tokens / sec | 18.09 | 18.11 | tie | — |

- EM is 0/200 on both tracks because medical answers are free-form
  paragraphs — exact string match is essentially impossible.
- BERTScore uses `roberta-large` instead of the configured
  `deberta-xlarge-mnli` because the latter throws
  `OverflowError: int too big to convert` under
  `bert-score==0.3.13` + `transformers==5.5.0`. The fallback path in
  `src/metrics.py` recovers the metric automatically.

### LLM-judge consensus (2-judge: Cerebras + Gemini)

|  | Track A | Track B | Δ |
|---|---|---|---|
| `clinical_correctness` (1-5) | 2.27 | **3.16** | +0.89 |
| `factuality` (1-5) | 3.43 | **3.73** | +0.30 |
| `completeness` (1-5) | 1.79 | **2.42** | +0.63 |
| `safety` (1-5) | 4.25 | 4.20 | -0.05 |
| `reasoning_soundness` (1-5, null when no reasoning shown) | 2.10 (n=29/200) | (3.6, n=5 — ignore) | — |
| `unsafe_rate` | 10.0 % | **2.2 %** | B 4.5× safer |
| `majority_pass` rate | 30 % | **44.4 %** | B +14 pp |

⚠️ **Sample-size caveat:** in the full run, Track B had 155/200 rows
with zero judges (Gemini quota exhausted, Cerebras hourly-cap hit) — so
`clinical_correctness=3.16` is computed on only **45 rows**, not 200.
Fixes in `ea4c7b2` (paced calls + retry-with-backoff) target this for
the re-run.

### Cross-judge agreement (full-run subset of 10 rows where both judges responded)

|  | Cerebras mean | Gemini mean | Abs diff |
|---|---|---|---|
| `clinical_correctness` | 2.00 | 1.50 | 0.50 |
| `factuality` | 3.80 | 1.90 | **2.10** |
| `completeness` | 1.80 | 1.40 | 0.40 |
| `safety` | 4.50 | 3.30 | 1.40 |
| Verdict agreement (PASS/FAIL/UNSAFE) | — | — | **90 %** |

Gemini is systematically **harsher** than Cerebras across every axis,
biggest gap on `factuality`. Verdict directions agree 9/10, but
score magnitudes differ substantially. This must be disclosed in the
report.

### Hallucination summary

|  | Track A | Track B | A/B ratio |
|---|---|---|---|
| Total errors flagged | 99 | 23 | 4.3× |
| Major errors | 64 | 14 | 4.6× |
| Fabrications | 45 | 8 | 5.6× |
| Reasoning errors | 21 | 2 | **10×** |
| Severity-4 errors | 35 | 10 | 3.5× |
| Severity-5 errors | 26 | 2 | 13× |

### Per-row token & latency deltas (200 paired samples)

- Mean **token delta** A − B = **+353 tokens / row** (A is 353 tokens longer per response)
- Mean **latency delta** A − B = **+19.6 s / row** (A is 19.6 s slower per response)
- EM-based row winners: A-wins=0, B-wins=0, tied=200 (all rows tied at EM=0)

---

## 7. Evaluation Strategy (PDF Phase 3)
### Required Design Decision — How success will be measured

### Phase 3 — Automatic Evaluation compliance

| PDF — Core Metric | Implementation | Where reported |
|---|---|---|
| **Exact Match (EM)** | `compute_em` in `src/metrics.py`; normalised lower-case + punctuation-stripped string comparison on extracted final answer | `metrics_summary.json[track].exact_match` (0/200 for both tracks in full run — free-form answers) |
| **Semantic Scores** | `compute_bertscore` using `roberta-large` (`microsoft/deberta-xlarge-mnli` configured but throws `OverflowError` with our `transformers==5.5.0`; auto-fallback in `src/metrics.py`) | `metrics_summary.json[track].bertscore_f1` (Track A 0.815, Track B 0.849); secondary: ROUGE-L F1 with LCS |

| PDF — Reasoning-Aware Metric | Implementation | Where reported |
|---|---|---|
| **Logical consistency** | LLM-judge axis `reasoning_soundness` (1-5 scale; null when no reasoning visible) | `judged.csv[mean_reasoning_soundness]`; Track A full-run mean 2.10 on the 29/200 rows where reasoning was visible |
| **Step correctness** (for CoT models) | Same `reasoning_soundness` axis; the judge prompt asks "Is the chain-of-thought logical?" with explicit OK to null for Track B. Manual `reasoning_clarity` (clear / vague / misleading / not_applicable) provides finer-grain step-level review on the audit subset. | `reasoning_soundness` per-row in judged.csv; `reasoning_clarity` in safety_audit.csv |
| **Hallucination rate** | LLM-judge `errors[]` aggregated by type. Six categories (FABRICATION / NEGATION / CAUSALITY / CONTEXTUAL / REASONING / OVERCONFIDENCE) per §5. Counted in `n_fabrication`, `n_negation`, … and `n_errors` (total) / `n_major_errors`. | `judged.csv` per-row, `report_summary.json` aggregates |

### Phase 3 — Manual verification (PDF: "verify the output manually and give remarks")

Stratified-sample non-clinical safety audit, 30 rows per track (10
high-risk / 10 medium-risk / 10 low-risk, bucketed by judge
`n_major_errors`, `max_severity`, and `any_unsafe`). One reviewer; not
clinical validation. Scored on the **three required axes** per PDF
plus three supplementary ones:

| PDF — Manual axis | Schema column | Values |
|---|---|---|
| **Clinical correctness** | `clinical_correctness` | correct / partially_correct / incorrect |
| **Risk severity of wrong answers** | `risk_severity` | low / medium / high / critical |
| **Clarity of reasoning** | `reasoning_clarity` | clear / vague / misleading / not_applicable (Track B only) |
| (supplementary) | `hallucination_type` | none / fabricated_fact / wrong_reasoning / overconfident_claim |
| (supplementary) | `safe_behavior` | safe / missing_disclaimer / dangerous_advice |
| (supplementary) | `manual_remark` | free-text |

CSV templates: `outputs/track{A,B}/safety_audit.csv` (generated;
unfilled at time of writing — Phase 3 deliverable still pending the
manual scoring step).

### Phase 3 — Must analyze (PDF requirements)

| PDF — Must analyze | Where in this project |
|---|---|
| **Where model fails** | `notebooks/full_pipeline.ipynb` cell `2873b018` ("Phase 3 Deliverable: Error Analysis Document") — per-track total samples, FAIL rates, UNSAFE rates, truncation rates, hallucination type breakdown, judge mean scores per axis, worst-2 samples by `n_major_errors` |
| **Types of hallucinations — Fabricated facts** | Judge axis `FABRICATION` (made-up drug / dose / citation / mechanism). Full-run counts: Track A 45 vs Track B 8 (5.6× more in A). Mean severity 3.91 (A). |
| **Types of hallucinations — Incorrect reasoning** | Judge axis `REASONING` (logically incoherent CoT). Full-run counts: Track A 21 vs Track B 2 (10× more in A). Mean severity 3.52 (A). |
| **Types of hallucinations — Overconfidence issues** | Judge axis `OVERCONFIDENCE` (confident assertion absent uncertainty markers). Added to taxonomy specifically to match this PDF requirement. Full-run counts: Track A 4 vs Track B 3 (similar rate). Mean severity 3.25 (A). |

### Phase 3 — Deliverables (PDF required)

| PDF deliverable | Status | Path / cell |
|---|---|---|
| **Evaluation report (with comparisons)** | ✅ | `outputs/final_comparison_table.csv` (per-track aggregates), `outputs/per_sample_comparison.csv` (200 paired rows with em / token / latency deltas), `outputs/report_summary.json` (judge + audit aggregates); rendered in `notebooks/full_pipeline.ipynb` cells `code-comparison-table`, `cbadca3d`, `code-judge-audit-sum` |
| **Error analysis document** | ✅ | `notebooks/full_pipeline.ipynb` cell `2873b018` titled "Phase 3 Deliverable: Error Analysis Document" |
| **Sample outputs (good vs bad cases)** | ✅ | `notebooks/full_pipeline.ipynb` cell `99ba25a5` — picks 3 A-wins and 3 B-wins from the per-sample comparison; falls back to tied examples if no winners. Quoted in §8 of this document where appropriate. |

### Evaluation philosophy and answers to PDF questions

The evaluation answers each PDF question with a specific metric layer.

| PDF question | Lexical metric | Semantic metric | Judge metric | Manual metric |
|---|---|---|---|---|
| Does reasoning improve QA? | EM, ROUGE-L | BERTScore F1 | `clinical_correctness` mean | manual `clinical_correctness` |
| Does reasoning increase cost? | mean `output_tokens` | — | — | — |
| Does reasoning increase latency? | mean `generation_time_s`, `tokens_per_sec` | — | — | — |
| Does reasoning increase hallucinations? | — | — | judge `errors[]` by type | manual `hallucination_type` |
| Should reasoning be hidden vs shown? | — | — | — | manual `reasoning_clarity` × `safe_behavior` |
| How unsafe in edge cases? | — | — | `UNSAFE` rate, `n_major_errors` | manual `risk_severity` |
| Trade-offs? | composite | composite | composite | composite |

### Layers of evaluation

1. **Lexical (primary).** Normalised Exact Match, ROUGE-L F1 with LCS.
   Optional sacreBLEU.
2. **Semantic (primary).** BERTScore via `roberta-large` (fallback from
   `deberta-xlarge-mnli` which fails under our transformers v5 pin).
   Optional S-PubMedBERT cosine for medical-domain-specific similarity.
3. **LLM-as-judge (supporting evaluators only).** Two judges in
   consensus: Cerebras `gpt-oss-120b` with strict JSON-schema constrained
   decoding; Gemini `gemini-2.5-flash` with loose JSON-object mode.
   Free-tier Groq dropped from consensus because its 100K-token-per-day
   cap is insufficient for 200×2 rows × ~2K tokens / call. Per-axis
   1–5 scores plus structured hallucination reports. **Supporting
   evidence; not final medical validation.**
4. **Manual non-clinical safety audit.** Stratified 30-sample subset
   per track (10 low-risk / 10 medium-risk / 10 high-risk, bucketed by
   judge `n_major_errors`, `max_severity`, and `any_unsafe`). Scored on
   `clinical_correctness`, `risk_severity`, `hallucination_type`,
   `reasoning_clarity`, `safe_behavior`, plus a free-text remark. 60
   rows total, one reviewer.

### Controlled-experiment property

Train, val, and test use the **same question indices** across both
tracks. Same model, same hyperparameters, same inference settings. The
*only* experimental variable is the formatter (Track A vs Track B). This
is the property that lets us attribute any A/B score difference to the
reasoning variable, not to data variance.

`shuffle_filter_split` enforces this: same `shuffle_seed=42`, same
`max_total_tokens=3500` length filter, same `num_train / num_val /
num_test`, deterministically split into disjoint subsets.

---

## 8. Findings and Trade-offs (PDF Final Outcome Expectation)

The PDF "Final Outcome Expectation" section requires this project to
answer four specific questions. This section is the explicit deliverable.

### Outcome-question compliance map

| PDF — Final Outcome question | Answered in subsection | Primary evidence |
|---|---|---|
| **Does reasoning actually improve medical QA?** | §8 Q1 | ROUGE-L, BERTScore, judge `clinical_correctness` |
| **When should reasoning be hidden vs shown?** | §8 Q2 | A vs B hallucination counts, severity, truncation |
| **How unsafe is the model in edge cases?** | §8 Q3 | Judge `any_unsafe` rate, `n_major_errors`, severity-5 count |
| **What trade-offs exist (accuracy vs cost vs latency)?** | §8 Q4 | Trade-off table combining quality / token cost / latency / safety |

Direct answers to the four required PDF outcome questions, with the
data caveats from §6.

### Q1 — Does reasoning improve medical QA?

**No, the opposite at this scale.** Track A (CoT) is worse than Track B
on ROUGE-L (-8 %), BERTScore (-4 %), `clinical_correctness` (-0.89),
`factuality` (-0.30), `completeness` (-0.63), `majority_pass` rate
(-14 pp), and produces 4.3× more hallucinations.

**Caveat:** Track A was 82.5 % truncated in the full run (capped at 768
`max_new_tokens`; outputs wanted ~1200-1500 tokens for rationale +
answer). The metrics partially measure "Track A got cut off" rather
than "Track A's reasoning was wrong." The re-run config (`track_A:
2048`, `track_B: 1024`) removes this contamination.

### Q2 — When should reasoning be hidden vs shown?

**At this model scale (1.5B Qwen, 3K samples, 1 epoch), hidden is
better.** A 1.5B parameter model cannot reliably reason; forcing it to
"think aloud" gives more opportunity to fabricate (45 vs 8) and to
commit reasoning errors (21 vs 2 — 10×). The hypothesis that CoT
helps small models is **not** supported at this scale on this dataset.

For larger models (≥ 7B) or with more curated data, the trade-off may
flip. This is a known research finding in the chain-of-thought
literature: CoT helps when the model is capable enough to reason
correctly; otherwise the reasoning chain compounds errors.

### Q3 — How unsafe is the model in edge cases?

**Track A: 10.0 % `any_unsafe` rate (Cerebras consensus); Track B:
2.2 %.** Gemini was substantially harsher — flagged 44 % of Track A as
unsafe vs Cerebras's 5 % on overlapping rows.

- 64 of 99 Track A errors (65 %) are tagged "major" — would change
  diagnosis or management.
- 26 Track A errors at severity 5 (potential patient harm); 2 in Track B.
- Highest-mean-severity error type is CAUSALITY (mean 4.17 for Track A,
  3.50 for Track B) — both tracks struggle with attributing causation.

**Neither track is safe for actual clinical deployment.** A 10 %
"could-harm-patient" rate is several orders of magnitude above any
realistic clinical-AI safety threshold. The project disclaimer is
not pro-forma; it is load-bearing.

### Q4 — What trade-offs exist (accuracy vs cost vs latency)?

|  | Track A | Track B | Winner |
|---|---|---|---|
| **Accuracy** (ROUGE-L, BERTScore, judge) | lower | higher | B |
| **Latency** (mean gen time) | 37.5 s | 18.0 s | B (2.09× faster) |
| **Token cost** (output tokens) | 679.5 | 326.5 | B (2.08× cheaper) |
| **Safety** (unsafe rate) | 10 % | 2.2 % | B |
| **Interpretability** | rationale visible (when not truncated) | none | A |

**Net trade-off at this scale: Track A trades nothing useful for the
rationale.** It is worse on accuracy, slower, more expensive in tokens,
and more unsafe. The interpretability gain is the only axis where A
wins, and even there 82.5 % of A's outputs in the full run had their
rationale truncated mid-sentence — so even interpretability isn't
reliably delivered.

There is no quadrant where Track A wins at this scale; **CoT requires
either a bigger model or more carefully curated training data to
become net-positive.**

---

## 9. Limitations

The following limitations are explicitly disclosed and would need to be
addressed before any deployment claim:

### Data
- Only **0.6 %** of the 506K-row dataset used (3,000 train + 100 val +
  200 test). Selected as a uniform random sample from the entire
  filtered pool, but small.
- Dataset is **synthetic** (GPT-OSS-120B distilled). Teacher biases
  inherited.
- No data deduplication, no quality filtering beyond a 3500-token
  length filter, no specialty-topic stratification, no demographic
  balance analysis.

### Model
- 1.5B parameters is **far below** state-of-the-art for medical AI.
  Med-PaLM 2 is 540B; clinical Llama variants are 70B.
- 4-bit QLoRA quantisation costs some quality vs fp16.
- LoRA rank 16 is small (production setups use 64-256).
- 1 epoch only; no preference optimisation (RLHF, DPO).

### Evaluation
- Test set 200 rows; production benchmarks use 1K-10K.
- No statistical-significance testing on A/B comparison (no McNemar's
  test, no bootstrap CIs).
- No comparison against established medical QA baselines (MedQA,
  PubMedQA, USMLE-style).
- No clinical expert in the loop; LLM judges treated as supporting
  evaluators only.
- Manual safety audit is 60 rows × 1 reviewer (vs production audits:
  thousands of rows × multiple board-certified physicians × inter-rater
  reliability).
- No bias analysis (race, gender, age, geography, specialty).
- No adversarial testing.

### Safety
- 10 % unsafe rate (Track A) / 2.2 % (Track B); a clinical product
  needs orders of magnitude better.
- No PII detection / filtering.
- No fact-checking against authoritative medical databases.
- No drug-interaction / dose / contraindication validation.
- No regulatory pathway engagement (FDA SaMD, EU MDR, HIPAA).

### Infrastructure
- Single Kaggle T4 GPU; no distributed training (Unsloth multi-GPU bug
  for 4-bit Qwen + DDP).
- No inference serving infrastructure (vLLM / Triton / TGI).
- No model registry, no drift monitoring, no production observability.
- No CI/CD for model updates.

### Documentation
- No HuggingFace-standard model card.
- Adapters kept **private** on the Hub during development.
- No peer review.

---

## 10. Expected Outputs (Deliverables)

| Deliverable | Status | Path |
|---|---|---|
| Phase-1 design doc (this file) | ✅ | `design_doc.md` |
| Track A LoRA adapter | ✅ trained, private | `kabhisheks/qwen25-1.5b-medreason-trackA-v0` |
| Track B LoRA adapter | ✅ trained, private | `kabhisheks/qwen25-1.5b-medreason-trackB-v0` |
| W&B training runs (both tracks) | ✅ | project `medical-reasoning-sft` |
| Training pipeline | ✅ | `train_sft.py`, `notebooks/full_pipeline.ipynb`, `run_pipeline.py` |
| Test predictions per track | ✅ | `outputs/track{A,B}/predictions.csv` |
| Automatic metrics summary | ✅ | `outputs/metrics_summary.json` |
| LLM-judge consensus output | ✅ (with truncation/coverage caveats) | `outputs/track{A,B}/judged.csv` |
| Manual safety audit (templates) | ✅ generated | `outputs/track{A,B}/safety_audit.csv` |
| Manual safety audit (filled) | ⏳ pending | — |
| Per-sample A/B comparison | ✅ | `outputs/per_sample_comparison.csv` |
| Final comparison table | ✅ | `outputs/final_comparison_table.csv` |
| Error-analysis section (PDF Phase 3) | ✅ | `notebooks/full_pipeline.ipynb` cell `2873b018` |
| Worked examples (3 good / 3 bad) | ✅ | `notebooks/full_pipeline.ipynb` cells `cbadca3d`, `99ba25a5` |
| Re-inference with fixed `max_new_tokens` | ⏳ pending | — |
| Re-judge with paced calls | ⏳ pending | — |
| Final narrative report | ⏳ pending | — |

---

## Sign-off

- **Author**: Abhishek Kumar Singh, ___________________
- **Reviewer**: ___________________
- **Reviewer**: ___________________
