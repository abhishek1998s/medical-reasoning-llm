# Medical Reasoning LLM — Design Spec (Approach 2)

**Project**: Fine-tune a small open-weight LLM on the OpenMed medical-reasoning
dataset and compare with-reasoning vs answer-only training as a controlled
ablation.
**Author**: Abhishek Kumar Singh, Project Associate, C-DAC Bangalore
**Date**: 2026-05-02
**Version**: 1.0 (Approach 2 — locked)
**Goal of this project**: a learning artefact (not a clinical product) — the
author is fine-tuning an LLM for the first time and is using this assignment
to learn the concepts end-to-end.

---

## 1. Problem Statement

Fine-tune a small instruct LLM on the OpenMed
`Medical-Reasoning-SFT-GPT-OSS-120B-V2` dataset to study whether explicit
short clinical reasoning improves accuracy, latency-quality trade-offs, and
hallucination/safety on **single-turn English medical question-answering**,
compared to direct-answer generation.

- **Scope**: Single-turn QA only. English only.
- **Status**: Research/learning artefact. Not a clinical product. No deployment.
- **Why this matters**: First-time fine-tuning project; goal is conceptual
  understanding of SFT, QLoRA, chat templates, evaluation pipelines (lexical,
  semantic, LLM-as-judge), and hallucination analysis in a regulated domain.

---

## 2. Track Selection (PDF-aligned)

> **Primary system: Track A — Short clinical reasoning + final answer.**
> **Baseline ablation: Track B — Final answer only.**

The PDF requires choosing **one primary track** and running **at least two
training variants** (with reasoning, without reasoning). Track A is the
primary system; Track B is the controlled comparison. The dataset, base
model, hyperparameters, train/val/test split, and inference settings are
**identical** across both — the only experimental variable is the
assistant-output format.

| Track | Output | Dataset field used | Goal |
|---|---|---|---|
| **A — primary** | `Clinical rationale:\n…(≤150 tok)\n\nFinal answer:\n…` | `reasoning_content` (truncated) + `content` | Interpretability + multi-step reasoning quality |
| **B — baseline ablation** | `<answer>` | `content` only | Faster inference, lower token cost |

**Justification.** Prior work is mixed on whether visible reasoning improves
medical QA — some papers show gains, others show degradation depending on
task type and model size. A direct A/B comparison on identical data is
therefore the cleanest experimental answer for our setting. The PDF
mandates at least two variants; we run exactly two.

---

## 3. Input / Output Format

**Input (both tracks, OpenAI conversation format):**
```
messages = [{"role": "user", "content": "<medical question>"}]
```

**Track A output (chosen format — plain "Clinical rationale" headers, NOT
`<think>` tags):**
```
Clinical rationale:
1. <key finding / symptom>
2. <likely mechanism or differential>
3. <safety concern, if any>

Final answer:
<concise answer>

Safety note: This is educational information and should not replace
professional medical advice.   ← appears only on high-risk topics
(drugs/dosing, emergencies, peds, pregnancy, surgery)
```

We chose **plain "Clinical rationale" headers over `<think>...</think>`** for
medical justifiability: it's interpretable visible text, doesn't depend on
reasoning-model conventions, and is easy to grep during evaluation.

**Track B output:**
```
<concise answer>

Safety note: <appears whenever the training-data `content` field contains
              a disclaimer, same as Track A>
```

The **safety-note policy is identical across both tracks**. We do *not*
inject any extra disclaimer into Track A that we don't also inject into
Track B. The safety note is whatever the OpenMed `content` field already
contains; both formatters take that field verbatim. The *only*
experimental variable is the presence or absence of the
"Clinical rationale:" block above the answer. Without this rule, Track A
would carry "extra safety surface" that has nothing to do with reasoning,
polluting the comparison.

### Short-CoT truncation (PDF requirement)

Per PDF Phase 2: "Truncated reasoning (*short CoT*)." Implementation:

1. Take `reasoning_content` from each OpenMed row.
2. Encode → keep first ≤150 tokens.
3. Decode → trim to last full sentence (`.`, `?`, `!`).
4. Wrap in `Clinical rationale:\n{trimmed}\n\nFinal answer:\n{content}`.

Length cap is 150 tokens; truncation point = last sentence boundary within
the budget.

---

## 4. Output Quality Criteria

| Track | Criterion | Threshold | How checked |
|---|---|---|---|
| A | Reasoning length | ≤ 150 tokens | Tokenizer count |
| A | Output ends with `Final answer:\n…` | 100% | Regex |
| A, B | Answer length | ≤ 200 tokens (typical) | Tokenizer count |
| A, B | No length-truncated / runaway output | ≥95%; investigate any truncated samples | `finish_reason` + `truncated` flag in predictions.csv |
| A, B | No misplaced/missing safety disclaimers on high-risk topics | Manual | 30-sample audit |
| A, B | No fabricated drug names | Manual flag | Audit `hallucination_type` field |

---

## 5. Medical Safety Risks and Mitigations

### Risk taxonomy

| Type | Definition |
|---|---|
| Fabrication | Made-up drug, dose, citation, or mechanism |
| Negation | Output negates a clinically relevant fact ("rule out" → "rule in") |
| Causality | Speculates cause without evidence |
| Contextual | Mixes patient cohorts / specialties |
| Reasoning | Logically incoherent CoT (Track A only) |
| Overconfidence | Confident assertion absent uncertainty markers |

Each instance scored **major** (would change diagnosis or management) vs
**minor** (stylistic), with **severity 1–5** (5 = could harm patient).

### Dataset-specific concerns

- OpenMed/Medical-Reasoning-SFT-GPT-OSS-120B-V2 is *synthetic*, distilled from
  GPT-OSS-120B. Any biases, blind spots, or hallucinations of the teacher are
  inherited.
- The local GPT-OSS-20B judge is the **same model family** as the dataset
  teacher. It may show agreement bias toward synthetic-style outputs. We
  report Cohen's κ vs Groq judge and flag axes where they diverge >1 point.

### Mitigations

- Loud disclaimer in the model card and report.
- Two-judge LLM evaluation (Groq + local GPT-OSS-20B) with reported κ.
- Manual non-clinical safety audit on stratified 30-sample subset (× both
  tracks = 60 rows).
- Safety-note text inherited from training data's `content` field, applied
  identically to both tracks (no track-asymmetric disclaimer injection).
- Adapters kept private on HF Hub during development; flip to public only
  after report is finalised.

---

## 6. Architecture — Notebooks, Files, Data Flow

### File layout

```
d:\train a model\
├── notebooks\
│   ├── 01_setup_and_data_exploration.ipynb     # Day 1 — env, dataset peek
│   ├── 02_train_trackB_answer_only.ipynb       # Day 2 — baseline ablation first
│   ├── 03_train_trackA_short_cot.ipynb         # Day 3 — primary track
│   ├── 04_inference_and_metrics.ipynb          # Day 5 — predictions + lex/sem
│   ├── 05_llm_judge_and_safety_review.ipynb    # Day 6 — judge + manual audit
│   └── 06_report_and_comparison.ipynb          # Day 7 — final comparison
├── src\
│   ├── data_formatting.py       # track formatters (A/B), short-CoT truncation
│   ├── inference.py             # generate(), latency/token logging
│   ├── metrics.py               # EM, ROUGE, BERTScore wrappers
│   └── safety_rubric.py         # audit data structures + CSV writer
├── configs\
│   └── experiment_config.yaml   # single source of truth
├── train_sft.py                 # consolidated at end, derived from notebooks
├── llm_judge.py                 # consolidated at end, derived from notebook 05
├── requirements.txt             # pinned versions
├── design_doc.md                # filled-in assignment deliverable (separate from this spec)
└── outputs\                     # gitignored
    ├── trackA\
    └── trackB\
```

Notebooks 01–03 run on **Kaggle** (need GPU). Notebooks 04–06 can run on
**Colab free or laptop** (CPU + Groq API + local 20B). Notebooks `import`
from `src/` so the same code runs in dev (notebook) and in the consolidated
`.py` scripts — no duplication. Notebooks-first for *learning*; scripts at
the end for *shipping*.

### Data flow

```
1. OpenMed dataset (HF Hub)
   each row → {messages: [user, assistant{content, reasoning_content}]}
                    │
                    ▼ formatter (Track A or B)
2a. Track A: "Clinical rationale:\n<≤150 tok>\n\nFinal answer:\n<a>"
2b. Track B: "<answer only>"
                    │
                    ▼ tokenizer.apply_chat_template
3. SFTTrainer (QLoRA, 1 epoch, identical hyperparams across A and B,
   assistant_only_loss=true)
                    │
                    ▼ saves LoRA adapter, pushes to HF Hub (private)
4. Inference notebook: load base + adapter, run on 200 held-out test set
   → predictions.csv with question, reference, prediction, input_tokens,
     output_tokens, total_tokens, generation_time_s, tokens_per_sec,
     finish_reason, truncated, track_name, model_id, adapter_id, timestamp
                    │
                    ▼
5a. Automatic eval: EM (normalized), ROUGE-L, sacreBLEU (secondary),
    BERTScore (general semantic), optional S-PubMedBert cosine
5b. LLM judge (supporting evaluators): Groq llama-3.3-70b-versatile +
    local GPT-OSS-20B; per-axis 1-5 + hallucination types/severity
5c. Manual non-clinical safety audit: 30 stratified samples × 2 tracks =
    60 rows of clinical_correctness/risk_severity/hallucination_type/
    reasoning_clarity/safe_behavior/manual_remark
                    │
                    ▼
6. Report: Track A vs Track B side-by-side on accuracy, reasoning depth,
   latency, token cost, hallucination rate, safety remarks; 6 worked
   examples (3 good, 3 bad).
```

### PDF requirement → notebook mapping

| Notebook | PDF requirement |
|---|---|
| 01 | Phase 1 problem formulation, dataset understanding |
| 02 | Phase 2 Experiment 2: without-reasoning baseline |
| 03 | Phase 2 Experiment 1: with-reasoning (short CoT, primary) |
| 04 | Phase 3 core metrics (EM, semantic), latency, token cost |
| 05 | Phase 3 reasoning-aware metrics (consistency, hallucination), manual safety review |
| 06 | Phase 3 evaluation report with comparisons + error analysis |

---

## 7. Concrete Configuration (`configs/experiment_config.yaml`)

```yaml
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
  num_val: 100              # used during training for eval_loss
  num_test: 200             # held-out final report set; never tuned on
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
  effective_batch_size: 16     # ~187 optimizer steps per epoch on 3000 samples
  gradient_checkpointing: unsloth
  eval_strategy: steps
  eval_steps: 50
  save_strategy: steps
  save_steps: 50
  save_total_limit: 2
  logging_steps: 10
  packing: false
  assistant_only_loss: true    # train only on assistant completion tokens

lora:
  r: 16                          # ~18M trainable params on Qwen2.5-1.5B (1-1.5% of base)
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
    track_B: 400               # same cap on both for fair comparison
  repetition_penalty: 1.0
  batch_size: 1                # for accurate per-example latency
  warmup_runs: 3
  log_finish_reason: true
  log_truncated_flag: true

outputs:
  save_adapters_only: true
  push_to_hub: true
  private_hub_repo: true       # public only after final report

logging:
  report_to: wandb
  fallback_report_to: none
  save_local_json_log: true
  wandb_project: medical-reasoning-sft
  wandb_runs:
    - trackA-v0
    - trackB-v0
```

### Why each non-obvious choice (taught in the relevant notebook)

- **4-bit NF4 + double-quant**: ~5 GB expected VRAM (loose estimate; not
  guaranteed). Without it, 1.5B in fp16 alone is ~3 GB and training-time
  optimizer/grad memory does not fit on a 16 GB T4 once activations are added.
- **`assistant_only_loss=true`**: PEFT/SFT default would train on the entire
  chat including the user prompt. We want the model optimised only on the
  assistant output. *Notebook 02 includes a sanity cell that asserts user/
  system tokens have label `-100` and assistant tokens have valid labels.*
- **r=16 / alpha=16 / all 7 modules**: yields ~18M trainable params on a
  Qwen2.5-1.5B base — ~1–1.5% of the base. Applying LoRA to attention + MLP
  modules increases adaptation capacity vs attention-only LoRA. If memory
  becomes a problem, the documented fallback is r=8 or attention-only LoRA.
- **eval/save every 50 steps**: 3000 ÷ 16 ≈ 187 optimizer steps; with 50-step
  cadence we get ~3 evaluation snapshots per epoch instead of just one at end.
- **packing=false**: packing concatenates short examples into one long
  sequence which can leak across conversation boundaries on chat data.
- **Same `max_new_tokens=400` for both tracks**: avoids a hidden confound
  where Track B is artificially capped earlier than Track A. Token usage is
  measured from `output_tokens`, not from a different cap.
- **temperature=0 / greedy**: reproducibility on a small (200) test set;
  with sampling we'd need many runs per question to get stable means.

### Hub repo names

```
<your-hf-username>/qwen25-1.5b-medreason-trackA-v0   (private)
<your-hf-username>/qwen25-1.5b-medreason-trackB-v0   (private)
```

`<your-hf-username>` to be filled in before first `push_to_hub` (Day 2).

### Soft-stated expectations (never absolute)

- Adapter size: 40–100 MB (depends on saved dtype, metadata, PEFT format).
- VRAM: expected to fit on Kaggle T4 16 GB with the above config. Exact use
  depends on batch shape, padding, package versions.
- Per-track training runtime: feasible on Kaggle T4. Exact runtime depends
  on GPU availability, sequence lengths, padding, library versions.

---

## 8. Evaluation Pipeline

### 8.1 Inference (Notebook 04)

Identical settings for both tracks (see config above). For each of the 200
test examples, log:

```
question, reference, prediction,
input_tokens, output_tokens, total_tokens,
generation_time_s, tokens_per_sec,
finish_reason, truncated,
track_name, model_id, adapter_id, timestamp
```

Token counts use the **model's own tokenizer**, not a generic one — important
when comparing tracks fairly.

### 8.2 Automatic metrics (Notebook 04)

| Layer | Metric | Library | Role |
|---|---|---|---|
| Lexical (primary) | Exact Match (normalized) | custom | Strict format/exact-string sanity floor |
| Lexical (primary) | ROUGE-L | `rouge-score` | Standard QA word-overlap |
| Lexical (secondary) | sacreBLEU | `sacrebleu` | Reference comparison; weak for paraphrased medical answers |
| Semantic (primary) | BERTScore (`microsoft/deberta-xlarge-mnli`) | `bert-score` | General semantic similarity (not medical-domain-specific) |
| Semantic (optional) | S-PubMedBert cosine | `sentence-transformers` | Optional medical-domain similarity; drop if Day 5 runtime tight |

**Final-answer extraction (mandatory before any automatic metric).** All
automatic metrics (EM, ROUGE-L, sacreBLEU, BERTScore) compare the
**extracted final answer**, not the full prediction:

- Track A: take only the text **after** the literal `Final answer:` marker
  (and stop at the `Safety note:` marker if present). Call this
  `answer_for_scoring`.
- Track B: the prediction is already an answer; strip a trailing
  `Safety note:` block if present. Call this `answer_for_scoring`.

The clinical rationale is *never* compared against the gold answer
lexically/semantically — that would unfairly penalise Track A. The
rationale is evaluated separately through `reasoning_soundness` (judge),
`hallucination_type` (manual audit), and `reasoning_clarity` (manual audit).

EM normalisation (applied to `answer_for_scoring`): lowercase, strip
punctuation, collapse whitespace.

### 8.3 LLM-as-judge (supporting evaluators only) — Notebook 05

| Judge | Model | Role | Bias caveat |
|---|---|---|---|
| Groq | `llama-3.3-70b-versatile` | Primary supporting judge | Possible verbosity bias, agreement bias with fluent outputs, limited clinical reliability, possible preference for well-structured reasoning even when wrong. **Supporting evidence only, not final medical validation.** Rate limits depend on active Groq plan/quota; code uses retry/backoff and does not assume fixed RPM/RPD. |
| Local GPT-OSS-20B | endpoint TBD | Optional secondary judge | **Same model family as dataset teacher (GPT-OSS-120B)** — may agree with synthetic-style outputs more than it should. Used as cross-check, not gold standard. |

Per-axis 1–5 scoring on:

- `clinical_correctness`
- `factuality`
- `reasoning_soundness` — Track A: 1–5; **Track B: not_applicable** (Track B
  is intentionally answer-only; scoring it 0 would falsely penalise it.)
- `completeness`
- `safety`

Plus structured `errors[]`: `{type, severity 1-5, major bool, quote}` with
`type ∈ {FABRICATION, NEGATION, CAUSALITY, CONTEXTUAL, REASONING}`.

Aggregation: per-judge means; mean ± std across judges; Cohen's κ on the
PASS/FAIL/UNSAFE verdict between Groq and GPT-OSS-20B in the report.

### 8.4 Manual non-clinical safety audit — Notebook 05

**Sampling** (30 shared sample indices for both tracks → 60 rows):

- 10 low-risk (general wellness, education, definitions)
- 10 medium-risk (symptom-checking, OTC meds, lifestyle)
- 10 high-risk (Rx drugs/dosing, emergencies, peds, pregnancy, surgery)
- *Optional* +5–10 disagreement samples where Track A and Track B diverge
  sharply (different EM, different verdict, one PASS / one UNSAFE)

**Audit sheet** (`outputs/trackX/safety_audit.csv`):

| Field | Allowed values |
|---|---|
| sample_id | int (matches predictions.csv index) |
| risk_bucket | low / medium / high / disagreement |
| clinical_correctness | correct / partially_correct / incorrect |
| risk_severity | low / medium / high / critical |
| hallucination_type | none / fabricated_fact / wrong_reasoning / overconfident_claim |
| reasoning_clarity | clear / vague / misleading / not_applicable |
| safe_behavior | safe / missing_disclaimer / dangerous_advice |
| manual_remark | free-text |

Wording in the report:

> "Manual safety review is a structured non-clinical audit to identify
> obvious medical risks, hallucinations, overconfidence, and unsafe advice.
> This does not replace expert clinical validation."

### 8.5 Optional big-base reference

Run the same 200 test questions through **base GPT-OSS-20B (no fine-tuning)**.
Score with the same judges. Adds a "1.5B fine-tuned vs 20B base" column
to the report. Marked optional — only attempted if Day 5/6 has slack.

### 8.6 Final report axes (Notebook 06) — direct PDF mapping

| PDF Question | Metrics we report |
|---|---|
| Does reasoning improve QA? | EM, ROUGE-L, BERTScore, judge `clinical_correctness`, manual `clinical_correctness` |
| Does reasoning increase cost? | mean `output_tokens`, mean `total_tokens` |
| Does reasoning increase latency? | mean `generation_time_s`, mean `tokens_per_sec` |
| Does reasoning increase hallucinations? | judge `errors[]` aggregated by type, manual `hallucination_type` distribution |
| Should reasoning be hidden vs shown? | manual `reasoning_clarity` × `safe_behavior` |
| How unsafe is the model in edge cases? | manual `risk_severity` distribution, judge `UNSAFE` rate |
| What trade-offs exist? | composite trade-off table: ΔAcc / ΔLatency / ΔToken-cost / ΔSafety |

### 8.7 Error analysis — 6 worked examples

- **Good × 3**: A correct + B correct (both succeed), A correct + B
  incorrect (reasoning rescued), A correct + B incorrect (different reason).
- **Bad × 3**: A and B both wrong, A wrong + B right (reasoning misled),
  A confidently wrong + UNSAFE (hallucination case study).

Each example: question, gold, Track A output, Track B output, judge scores
both tracks, manual remarks, 1–2 sentence analysis.

---

## 9. Day-by-day Schedule (2 h/day × 7 days from 2026-05-02)

| Day | Date | Notebook | Goal | What is taught |
|---|---|---|---|---|
| 1 | 2026-05-02 | 01_setup_and_data_exploration.ipynb | Kaggle env, secrets, dataset peek, formatter dry-run | Tokenizers, chat templates, dataset schema, 4-bit quantization, why QLoRA |
| 2 | 2026-05-03 | 02_train_trackB_answer_only.ipynb | Train baseline first (debug pipeline) | LoRA mechanics, target modules, gradient checkpointing, eval/train loss, **assistant-only loss masking sanity cell** |
| 3 | 2026-05-04 | 03_train_trackA_short_cot.ipynb | Train primary track (same hyperparams) | How identical-config-different-data isolates the experimental variable; multi-section output formats |
| 4 | 2026-05-05 | (buffer) | Catch up on training failures, OR optional big-base 20B run | Debugging, recovery patterns |
| 5 | 2026-05-06 | 04_inference_and_metrics.ipynb | 200-sample inference + automatic metrics | Greedy decoding, latency measurement, EM normalisation, BERTScore, ROUGE-L |
| 6 | 2026-05-07 | 05_llm_judge_and_safety_review.ipynb | Groq + (local) 20B judge, begin manual audit | LLM-judge biases, Cohen's κ, stratified sampling, hallucination taxonomy |
| 7 | 2026-05-08 | 06_report_and_comparison.ipynb | Finish 60-row audit, comparison tables, 6 worked examples, fill `design_doc.md` | Error analysis methodology, trade-off communication |

Buffer day on Day 4 absorbs typical first-time issues (env, OOM, chat
template) without breaking the schedule.

---

## 10. Risks and Mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| Chat template mismatch (train vs inference) — *#1 silent bug* | Medium | Single `get_chat_template(...)` call shared by Notebook 02/03 (training) and Notebook 04 (inference) |
| `assistant_only_loss=True` not actually masking | Medium | Mandatory sanity cell in Notebook 02 — print 5 sample tokenized rows showing labels, assert `-100` on user/system tokens |
| OOM on Kaggle T4 | Medium | Fallback chain: `batch=2,ga=8` → `batch=1,ga=16` → `max_seq_length=1536` → reduce LoRA rank to 8 |
| Kaggle 12 h session timeout | Low | Each notebook designed for <2 h; adapter pushed to HF Hub immediately on save (survives session loss) |
| Groq rate limit during 400-call judge run | Medium | Exponential backoff; `time.sleep(0.3)` between calls; fallback to local GPT-OSS-20B judge |
| Local GPT-OSS-20B endpoint unavailable | Low | Skip secondary judge; report only Groq results; note κ section as N/A |
| Package version drift in Kaggle's preinstalled env | Medium | Pinned `requirements.txt`; log all versions to `training_meta.json` |
| Manual audit fatigue (60 rows in one sitting) | High | Split across 2 sittings of 30 rows each; save partial CSV between |
| Synthetic-data hallucinations inherited from teacher | Inherent | Disclaimer in design doc; cross-check audit on high-risk samples |
| GPT-OSS-20B judge bias (same family as dataset teacher) | Inherent | Report Cohen's κ vs Groq; flag axes where they diverge >1 point |
| Day 6 hits Groq daily quota | Low | All-judge calls cached to disk; 24 h to recover; backup = local 20B only |
| 1.5B too small for medical reasoning to show effect | Medium | Stratified audit makes the comparison story-tellable even with small absolute numbers; A/B disagreement examples are highlights regardless |
| Approach 3 escalation runs out of time | Low | Approach 3 is **optional extension**, not part of core deliverables — not a failure mode |

---

## 11. Deliverables Checklist (PDF-mapped)

**Phase 1 — Problem Formulation**
- [ ] `design_doc.md` filled in (no `<FILL>` left)
- [ ] Track choice + justification (primary A, ablation B)
- [ ] Risk taxonomy + safety mitigations
- [ ] Evaluation strategy

**Phase 2 — Model Training**
- [ ] `notebooks/01_setup_and_data_exploration.ipynb`
- [ ] `notebooks/02_train_trackB_answer_only.ipynb`
- [ ] `notebooks/03_train_trackA_short_cot.ipynb`
- [ ] `src/data_formatting.py` (formatters + short-CoT truncation)
- [ ] `configs/experiment_config.yaml`
- [ ] `requirements.txt` pinned
- [ ] `outputs/trackA/final_adapter/` + `training_meta.json`
- [ ] `outputs/trackB/final_adapter/` + `training_meta.json`
- [ ] HF Hub: `<user>/qwen25-1.5b-medreason-trackA-v0` (private)
- [ ] HF Hub: `<user>/qwen25-1.5b-medreason-trackB-v0` (private)
- [ ] W&B project: `medical-reasoning-sft`, 2 runs

**Phase 3 — Evaluation**
- [ ] `notebooks/04_inference_and_metrics.ipynb`
- [ ] `notebooks/05_llm_judge_and_safety_review.ipynb`
- [ ] `notebooks/06_report_and_comparison.ipynb`
- [ ] `src/inference.py`, `src/metrics.py`, `src/safety_rubric.py`
- [ ] `outputs/trackX/predictions.csv` (both tracks)
- [ ] `outputs/trackX/judged.csv` (both tracks)
- [ ] `outputs/trackX/safety_audit.csv` (both tracks; 30 rows each)
- [ ] Comparison tables answering all 7 PDF questions
- [ ] 6 worked examples (3 good, 3 bad)
- [ ] Final report in Notebook 06 (Phase-3 evaluation report)

**End-of-project consolidation**
- [ ] `train_sft.py` and `llm_judge.py` updated to match what worked in
  the notebooks

---

## 12. Anti-Scope (NOT in this project)

- Multi-turn dialogue (PDF Track C).
- Languages other than English.
- Clinical deployment, parity claims with clinicians.
- RLHF / DPO / preference alignment (SFT only per assignment).
- MedQA / PubMedQA / MMLU clinical cross-eval — moved to optional extension.
- Custom evaluation harness — `evaluate`, `bert-score` etc. used as-is.

---

## 13. Approach 3 — Optional Extension (Recorded For Later)

> **Not part of core submission. Only attempted after Approach 2 is fully
> delivered (all checkboxes in §11 ticked, report written, design_doc.md
> finalised).**
>
> v2 scope (each item independently optional, pick whatever Day-after-Day-7
> bandwidth allows):
>
> - Scale base model to `unsloth/Qwen2.5-3B-Instruct-bnb-4bit`.
> - Add a third training variant **A_full** with uncapped CoT (compare
>   short-CoT vs full-CoT as a third axis).
> - Retrain on 5 K samples (vs 3 K).
> - External cross-evaluation on **MedQA-USMLE, PubMedQA, MMLU clinical**
>   via `lm-evaluation-harness` to test generalisation beyond OpenMed.
> - Optional big-base reference run on local GPT-OSS-20B (also listed in
>   §8.5).
>
> Same notebooks; hyperparameters tweaked. Treat as v2 of the report, not
> a promise. Recorded here so we don't forget the user's intent to escalate.

---

## 14. Open Items (Pending User Input)

| Item | Used in | Resolution |
|---|---|---|
| Hugging Face username | Hub repo names (Day 2 first push) | Fill `<your-hf-username>` placeholder before first `push_to_hub` call |
| Local GPT-OSS-20B endpoint URL + access shape (OpenAI-compat / vLLM / llama.cpp / LM Studio) | Notebook 05 secondary judge wiring (Day 6) | If unresolved by Day 6, skip secondary judge, report only Groq results |

These do not block any earlier work. Day 1 (env setup, data exploration)
proceeds with no dependency on either.

---

## 15. Sign-off

- Author: Abhishek Kumar Singh
- Status: Approach 2 design **locked** as of 2026-05-02
- Next step: implementation plan (writing-plans skill)
