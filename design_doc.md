# Medical Reasoning LLM — Design Document (Phase 1 Deliverable)

**Project**: Fine-Tune a Small LLM for Medical Reasoning
**Author**: Abhishek Kumar Singh, Project Associate, C-DAC Bangalore
**Date**: 2026-05-02
**Version**: 1.0
**Status**: Learning artefact — not a clinical product, no deployment.

> The full implementation specification (configuration, data flow, day-by-day
> schedule, risks, deliverables) lives in
> `docs/superpowers/specs/2026-05-02-medical-reasoning-llm-design.md`.
> This document is the short Phase-1 design doc the assignment requires.

---

## 1. Problem Definition

Fine-tune a small open-weight instruct LLM on the OpenMed
`Medical-Reasoning-SFT-GPT-OSS-120B-V2` dataset and study, through a
controlled A/B comparison, whether **explicit short clinical reasoning**
in the assistant output improves accuracy, latency-quality trade-offs, and
hallucination/safety on **single-turn English medical question-answering**,
compared to direct-answer generation.

**Scope**

- Single-turn English QA only.
- Synthetic dataset distilled from GPT-OSS-120B; biases of the teacher are
  inherited — explicitly disclaimed.
- Research/learning artefact. No clinical claim, no deployment.

**Final questions the project will answer (mapped from the assignment)**

1. Does reasoning actually improve medical QA?
2. When should reasoning be hidden vs shown?
3. How unsafe is the model in edge cases?
4. What trade-offs exist between accuracy, cost, and latency?

---

## 2. Primary Track Choice and Justification

> **Primary system: Track A — Short clinical reasoning + final answer.**
> **Baseline ablation: Track B — Final answer only.**

The PDF requires choosing **one primary track** and running **at least two
training variants** (with reasoning, without reasoning). Track A is the
primary system the assignment is studying. Track B is the controlled
comparison. The dataset, base model, hyperparameters, train/val/test
split, and inference settings are **identical** across both — the only
experimental variable is the assistant-output format.

**Why Track A as primary.** The assignment's headline goals — *reasoning
quality, correctness, reliability, interpretability* — are all properties
of the reasoning output, not of an answer-only system. Track A is the
system whose behaviour we want to understand.

**Why Track B as ablation.** Without an answer-only baseline trained on
the same data, we cannot tell whether observed accuracy or hallucination
rates are caused by reasoning or by general fine-tuning. Track B isolates
that.

**Why not Track C (multi-turn).** Single-turn QA is sufficient to study
the reasoning question and is much smaller in scope; the dataset itself
is single-turn.

---

## 3. Input / Output Format

**Input (both tracks):**
```
messages = [{"role": "user", "content": "<medical question>"}]
```

**Track A output** — plain "Clinical rationale" headers, not `<think>`
tags (more readable, more justifiable for a medical setting):
```
Clinical rationale:
1. <key finding or symptom>
2. <likely mechanism or differential>
3. <safety concern, if any>

Final answer:
<concise answer>

Safety note: <inherited from training data when present>
```

**Track B output:**
```
<concise answer>

Safety note: <inherited from training data when present>
```

**Short-CoT truncation (PDF requirement).** Take the dataset's
`reasoning_content`, encode → keep first ≤150 tokens → decode → trim to
the last full sentence. Wrap in the Track A format above.

**Equal safety-note policy across tracks.** The safety-note text is
whatever the OpenMed `content` field already contains; both formatters
take that field verbatim. We do **not** inject any disclaimer asymmetrically
between tracks. The only experimental variable is the presence or absence
of the "Clinical rationale:" block.

**Final-answer extraction for scoring.** Automatic metrics (EM, ROUGE-L,
BERTScore, sacreBLEU) compare only the *extracted final answer*, not the
full prediction. For Track A, this means the text after `Final answer:`.
The reasoning is evaluated separately via judge `reasoning_soundness`,
manual `hallucination_type`, and manual `reasoning_clarity`.

---

## 4. Output Quality Criteria

| Track | Criterion | Threshold | Check |
|---|---|---|---|
| A | Clinical rationale length | ≤150 tokens | Tokenizer count |
| A | Output ends with `Final answer:\n…` | 100% | Regex |
| A, B | Answer length | ≤200 tokens (typical) | Tokenizer count |
| A, B | No length-truncated / runaway output | ≥95%; investigate truncated samples | `finish_reason` + `truncated` flag |
| A, B | No fabricated drug names | Manual flag | Audit `hallucination_type` |
| A, B | No misplaced/missing safety disclaimer on high-risk topics | Manual | 30-sample audit |

---

## 5. Risks and Mitigations (especially medical safety)

**Risk taxonomy**

| Type | Definition |
|---|---|
| Fabrication | Made-up drug, dose, citation, or mechanism |
| Negation | Output negates a clinically relevant fact |
| Causality | Speculates cause without evidence |
| Contextual | Mixes patient cohorts / specialties |
| Reasoning | Logically incoherent CoT (Track A only) |
| Overconfidence | Confident assertion absent uncertainty markers |

Each instance scored **major** (would change diagnosis or management) vs
**minor** (stylistic) with **severity 1–5** (5 = could harm patient).

**Dataset-specific risks**

- OpenMed is *synthetic*, distilled from GPT-OSS-120B; biases of the teacher
  are inherited.
- The local GPT-OSS-20B judge is the same model family as the teacher;
  it may show agreement bias toward synthetic-style outputs. Bias is
  reported via Cohen's κ vs Groq judge.

**Mitigations**

- Loud disclaimer in design doc, model card, and final report: this is a
  learning artefact, not a clinical tool.
- Two-judge LLM evaluation (Groq + local GPT-OSS-20B) with reported κ —
  treated as **supporting evaluators only**, not authoritative.
- Manual **non-clinical** safety audit on a stratified 30-sample subset
  for both tracks (60 rows total).
- Adapters on HF Hub kept **private** during development; flip to public
  only after the report is finalised.
- LLM judge prompt grades on `clinical_correctness`, `factuality`,
  `reasoning_soundness` (Track A only — `not_applicable` for Track B),
  `completeness`, `safety`.

---

## 6. Evaluation Strategy (How Success Is Measured)

The evaluation answers each PDF question with a specific metric.

| PDF Question | Metric we report |
|---|---|
| Does reasoning improve QA? | EM (extracted answer), ROUGE-L, BERTScore, judge `clinical_correctness`, manual `clinical_correctness` |
| Does reasoning increase cost? | mean `output_tokens`, mean `total_tokens` |
| Does reasoning increase latency? | mean `generation_time_s`, mean `tokens_per_sec` |
| Does reasoning increase hallucinations? | judge `errors[]` aggregated by type, manual `hallucination_type` distribution |
| Should reasoning be hidden vs shown? | manual `reasoning_clarity` × `safe_behavior` |
| How unsafe is the model in edge cases? | manual `risk_severity` distribution, judge `UNSAFE` rate |
| What trade-offs exist? | composite trade-off table: ΔAcc / ΔLatency / ΔToken-cost / ΔSafety |

**Layers of evaluation**

- **Lexical (primary)**: Exact Match (normalised), ROUGE-L.
  *Secondary*: sacreBLEU.
- **Semantic (primary)**: BERTScore using `microsoft/deberta-xlarge-mnli`
  — strong general semantic similarity; not medical-domain-specific.
  *Optional*: S-PubMedBERT cosine for medical-domain similarity.
- **LLM-as-judge (supporting evaluators only)**: Groq
  `llama-3.3-70b-versatile` and (optional) local GPT-OSS-20B with
  retry/backoff. Per-axis 1–5 scores plus structured hallucination
  reports. Supporting evidence; **not** final medical validation.
- **Manual non-clinical safety audit**: 30 stratified samples
  (10 low-risk / 10 medium / 10 high) × 2 tracks = 60 rows, scored on
  `clinical_correctness`, `risk_severity`, `hallucination_type`,
  `reasoning_clarity`, `safe_behavior`, free-text remark.

**Controlled-experiment property.** Train and test on *the same
question indices* across both tracks. Same hyperparameters. Same
inference settings. Only the formatter differs.

---

## 7. Expected Outputs

- Two LoRA adapters on Hugging Face Hub (private during development):
  `<user>/qwen25-1.5b-medreason-trackA-v0`,
  `<user>/qwen25-1.5b-medreason-trackB-v0`.
- W&B project `medical-reasoning-sft` with both training runs.
- 6 notebooks (setup, two training, inference+metrics,
  judge+manual-audit, report).
- Per-track CSVs: `predictions.csv`, `judged.csv`, `safety_audit.csv`.
- Final report (Notebook 06) with the comparison tables above and 6
  worked examples (3 good, 3 bad) for the assignment's required error
  analysis.
- Polished `train_sft.py` and `llm_judge.py` consolidated from the
  notebooks.

---

## 8. Open Questions for Reviewer

1. Are private HF Hub repos during development acceptable, or should
   adapters be released publicly with a strong limitations notice from
   the start?
2. For the optional big-base reference comparison (running 200 questions
   through unmodified GPT-OSS-20B), is this worth the time relative to
   spending it on a deeper manual audit?
3. Is the sample count (3K train / 100 val / 200 test) acceptable for
   submission, given hardware constraints?

---

**Sign-off**:
- Author: Abhishek Kumar Singh, ___________________
- Reviewer: ___________________
- Reviewer: ___________________
