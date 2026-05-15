# Fine-Tune a LLM for Medical Reasoning — Design Document

**Author**: Abhishek Kumar Singh, Project Associate, C-DAC Bangalore
**Date**: 16 May 2026
**Status**: Learning artefact — not a certified medical device, not for clinical deployment.

## Objective

Design, train, and evaluate a medical reasoning LLM with emphasis on
**reasoning quality**, **correctness**, and **reliability**. Dataset:
`OpenMed/Medical-Reasoning-SFT-GPT-OSS-120B-V2` (506,000 single-turn rows
distilled from GPT-OSS-120B; each row provides `content` and
`reasoning_content` on the assistant turn).

## Phase 1 — Problem Formulation

**Problem definition.** Given a medical query, the model must produce a
clinical chain-of-thought grounded in medical knowledge and a final
answer consistent with that reasoning, with each output auditable to
its rationale.

**Primary track: A — Full Reasoning Generation.** Input: medical query;
output: step-by-step reasoning + final answer. *Why Track A?* The
assignment's headline goals — reasoning quality, correctness,
reliability, interpretability — are properties of a reasoning output,
not an answer-only system. Track B (Answer-Only) is run as the
controlled-ablation baseline: identical dataset, base model,
hyperparameters, splits, and inference settings; the only experimental
variable is the assistant-output format. Track C (Conversational) is
dropped: single-turn QA is sufficient and the dataset is single-turn.

**Risks (especially medical safety).** Six hallucination categories
scored on severity 1–5 by the judge: FABRICATION (made-up drug, dose,
citation), NEGATION (negates a clinical fact), CAUSALITY (unfounded
cause), CONTEXTUAL (wrong cohort or specialty), REASONING (incoherent
CoT), OVERCONFIDENCE (assertion without uncertainty marker). The
dataset is synthetic and distilled from GPT-OSS-120B, so teacher biases
are inherited. Mitigations: two-judge consensus from independent model
families (Cerebras `gpt-oss-120b` with strict JSON-schema constrained
decoding + Gemini `2.5-flash`); stratified non-clinical safety audit on
30 samples per track; HF Hub adapters kept private during development;
explicit non-clinical disclaimer in every artefact.

**Evaluation strategy.** Four layers: (1) lexical — Exact Match,
ROUGE-L; (2) semantic — BERTScore F1; (3) LLM-judge consensus with
six-axis 1–5 scoring and structured hallucination reports; (4) manual
non-clinical safety audit. Controlled-experiment property: identical
train/val/test indices across both tracks (shuffle_seed=42), so any
A-vs-B score difference is attributable to the formatter variable.

**Input-output format.**

| Field | Source | Track A output | Track B output |
|---|---|---|---|
| Question | `messages[0].content` | preserved verbatim | preserved verbatim |
| Reasoning | `messages[1].reasoning_content` | first 150 tokens, truncated at last sentence boundary | dropped |
| Answer | `messages[1].content` | rendered after `Final answer:` block | sole assistant output |
| Safety note | inline in `content` when present | preserved verbatim | preserved verbatim |

**Output quality criteria.** Track A: rationale ≤150 tokens, ends with
`Final answer: …`. Both tracks: non-truncated ≥95% (post-fix
inference budget); no fabricated drugs or doses (manual audit);
appropriate safety disclaimer on high-risk topics; judge majority-pass
rate ≥50% target.

## Phase 2 — Model Training

**Baseline setup.** Base model: `unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit`
— 1.5B parameters, pre-quantised 4-bit NF4, fits Kaggle T4 (16 GB) at
sequence length 4096. Method: **QLoRA** — 4-bit base weights frozen,
LoRA adapters trained in bf16 on all 7 attention + MLP projections.
Truncated reasoning: first 150 tokens of `reasoning_content`, walked
back to the last sentence boundary. Stack: Unsloth 2026.4.8, TRL 0.24.0,
PEFT 0.19.1, transformers 5.5.0.

| Hyperparameter | Value |
|---|---|
| LoRA rank / alpha / dropout | 16 / 16 / 0.0 |
| All parameters / trainable / % | 907,081,216 / **18,464,768 / 2.04 %** |
| per_device_train_batch × grad_accum | 2 × 8 = effective batch 16 |
| Epochs | 1 |
| Learning rate / scheduler / warmup_steps | 2.0e-4 / cosine / 6 |
| Optimizer / weight decay | adamw_8bit / 0.01 |
| Sequence length | 4096 |
| Loss masking | assistant tokens only (Unsloth `train_on_responses_only`) |

**Training variants (full run, 2026-05-15).**

| | Experiment 1 — With Reasoning (Track A) | Experiment 2 — Without Reasoning (Track B) |
|---|---|---|
| Format | rationale (≤150 tok) + `Final answer:` block | final answer only |
| Train / val | 3,000 / 150 | 3,000 / 150 |
| Optimizer steps / wall time | 188 / 116 min | 188 / 110 min |
| Final `train_loss` / `eval_loss` | 1.328 / **1.216** | 1.336 / 1.251 |

**Required comparisons.**

| Comparison goal | Metric | Track A | Track B |
|---|---|---|---|
| Accuracy vs reasoning depth | ROUGE-L | 0.131 | **0.142** |
| | BERTScore F1 | 0.815 | **0.849** |
| | Judge `clinical_correctness` (1–5) | 2.27 | **3.16** |
| Latency vs output quality | mean generation time / row | 37.5 s | **18.0 s** |
| | tokens per second | 18.09 | 18.11 |
| Token cost vs performance | mean output tokens / row | 679.5 | **326.5** |

**Deliverables.** Training pipeline: `train_sft.py`, `run_pipeline.py`,
`notebooks/full_pipeline.ipynb`, `configs/experiment_config.yaml`,
`src/`. Model checkpoints: intermediate `checkpoint-150` / `checkpoint-188`
plus final adapters pushed to private HF Hub
(`kabhisheks/qwen25-1.5b-medreason-track{A,B}-v0`). Experiment logs:
W&B project `medical-reasoning-sft` and per-track `training_meta.json`
with loss curves and pinned package versions.

## Phase 3 — Evaluation

**Automatic evaluation — Core Metrics.** Exact Match = 0/200 on both
tracks; medical answers are free-form paragraphs so EM is intractable
but reported per PDF requirement. Semantic Scores: BERTScore F1 = 0.815
(Track A) / 0.849 (Track B), computed with `roberta-large` as automatic
fallback when `deberta-xlarge-mnli` fails under transformers v5.

**Automatic evaluation — Reasoning-aware metrics.** Logical consistency
and step correctness measured by judge `reasoning_soundness` (1–5; null
when no rationale visible). Track A mean = 2.10 on the 29/200 rows
where rationale was present and not truncated. Hallucination rate
(errors per row, consensus): Track A 0.495, Track B 0.115 — Track A
produces 4.3× more flagged errors.

**Manual verification.** `outputs/track{A,B}/safety_audit.csv` generated
with judge-driven high/medium/low risk stratification; rows scored on
`clinical_correctness`, `risk_severity`, `reasoning_clarity`,
`safe_behavior`, `manual_remark`. One non-clinical reviewer; scoring
pending at time of writing.

**Error analysis — where the model fails.**

| Hallucination type | Track A | Track B | Mean severity (A) |
|---|---|---|---|
| Fabricated facts (FABRICATION) | 45 | 8 | 3.91 |
| Incorrect reasoning (REASONING) | 21 | 2 | 3.52 |
| Contextual | 13 | 3 | 3.69 |
| Negation | 8 | 5 | 4.12 |
| Causality | 6 | 2 | 4.17 |
| Overconfidence issues | 4 | 3 | 3.25 |
| **Total / major / severity-5** | **99 / 64 / 26** | **23 / 14 / 2** | — |

Track A's longer outputs (mean 679 tokens vs 326) and explicit
chain-of-thought yield 5.6× more fabricated facts and 10× more
reasoning errors than Track B.

**Final Outcome — direct answers to the four PDF questions.**

1. *Does reasoning actually improve medical QA?* No at this scale.
   Track A loses on ROUGE-L (-8.4%), BERTScore (-4.2%), judge
   `clinical_correctness` (-0.89), majority-pass rate (-14 pp), and
   produces 4.3× more flagged errors. Caveat: 82.5% of Track A
   outputs hit the 768-token cap in this run, contaminating quality
   metrics; the post-run inference config raises Track A to 2048
   tokens to remove this confound.

2. *When should reasoning be hidden vs shown?* At Qwen2.5-1.5B with
   3K samples × 1 epoch, hidden is better. The 1.5B model cannot
   reliably reason; forcing CoT compounds errors (10× more REASONING,
   5.6× more FABRICATIONS than the answer-only variant). Larger models
   (≥7B) or more curated data may invert this result.

3. *How unsafe is the model in edge cases?* Track A 10.0%
   `any_unsafe` verdict rate, Track B 2.2% (Cerebras consensus); Gemini
   flagged 44% of Track A unsafe on overlapping rows. 64 of 99 Track A
   errors are "major" (would change diagnosis or management); 26 reach
   severity 5 ("could harm patient if followed") versus 2 in Track B.
   Neither track is safe for clinical deployment.

4. *What trade-offs exist (accuracy vs cost vs latency)?* Track A loses
   on every axis except interpretability: 2.09× slower, 2.08× more
   tokens per response, lower quality on every metric, less safe. The
   interpretability gain is undermined by 82.5% mid-sentence
   truncation. At this scale, the CoT track trades nothing useful for
   the rationale; CoT requires a larger model or more curated data to
   become net-positive.

**Deliverables.** Evaluation report:
`outputs/final_comparison_table.csv`,
`outputs/per_sample_comparison.csv` (200 paired rows),
`outputs/report_summary.json`. Error analysis document:
`notebooks/full_pipeline.ipynb` cell `2873b018`. Sample outputs (good
vs bad cases — three Track A wins / three Track B wins with tied
fallback): cell `99ba25a5`. All artefacts are reproducible from the
pinned versions in `requirements.txt` and the deterministic split in
`src/splits.py`.
