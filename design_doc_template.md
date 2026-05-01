# Medical Reasoning LLM — Design Document

**Project**: Fine-Tuning a Small LLM for Medical Reasoning (Tracks A & B)
**Author**: Abhishek Kumar Singh, Project Associate, C-DAC Bangalore
**Date**: <FILL>
**Version**: 1.0

---

## 1. Problem Statement

This project fine-tunes a small open-weight instruct LLM on the OpenMed
`Medical-Reasoning-SFT-GPT-OSS-120B-V2` dataset to study whether explicit
chain-of-thought (CoT) reasoning improves accuracy, calibration, and safety on
single-turn English medical question-answering, compared to direct-answer
generation.

**Scope**: Single-turn QA only (no multi-turn dialogue). English only.
Research artefact, not a clinical product.

**Why this matters**: <FILL: 2–3 sentences on the value to your team — e.g.,
parallels to fraud-detection reasoning auditability, or motivation around
safer LLM deployment in regulated domains>

---

## 2. Track Selection — Both A and B (with a Short-CoT variant)

We deliberately train and evaluate two tracks (plus a short-CoT ablation) to
permit a controlled, single-axis comparison.

| Track | Output | Training data field used | Goal |
|---|---|---|---|
| **A — Full CoT** | `<think>{reasoning}</think>{answer}` | `reasoning_content` + `content` | Maximize interpretability and multi-step accuracy |
| **A short** | Same as A but reasoning truncated to ~150 tokens | first ~150 tok of `reasoning_content` + `content` | Test the "compact CoT" sweet spot reported by ReasonMed (arXiv 2506.09513) |
| **B — Answer only** | `{answer}` | `content` only | Faster inference, lower token-cost baseline |

**Justification**: The literature is split — Wei et al. 2022 and HuatuoGPT-o1
(arXiv 2412.18925) report CoT gains; Wu et al. 2025 (arXiv 2509.21933) report
that 86% of clinical-text-recall tasks degrade with CoT and incur 5–20×
latency; Selective-CoT (arXiv 2602.20130) argues for routing. A direct A/B
comparison on identical data is the cleanest experimental answer for our use
case.

---

## 3. Input / Output Format

**Input** (both tracks, OpenAI conversation format):
```
messages = [{"role": "user", "content": "<medical question>"}]
```

**Output Track A**:
```
<think>
Step 1: ...
Step 2: ...
</think>

Final Answer: <answer>
```

**Output Track B**:
```
<answer>
```

**Worked example**: <FILL — paste one real OpenMed sample reformatted for
both tracks side-by-side>

---

## 4. Output Quality Criteria

| Track | Criterion | Threshold | How checked |
|---|---|---|---|
| A | Reasoning length | ≤ 512 tokens (full) / ≤ 150 (short) | Tokenizer count |
| A | Reasoning ends with "Final Answer:" or `</think>` | 100% | Regex |
| A, B | No fabricated drug names | 0 unrecognized | RxNorm cross-check (optional) |
| A, B | Answer length | ≤ 200 tokens | Tokenizer count |
| A, B | EOS token present, no run-on generations | 100% | Generation logs |
| A, B | No misplaced/missing safety disclaimers | Manual | 30-sample audit |

---

## 5. Medical Safety Risks

**Risk taxonomy** (adopted verbatim from *Nature Digital Medicine* 2025,
`s41746-025-01670-7`, plus a Reasoning class for Track A):

| Type | Definition |
|---|---|
| **Fabrication** | Made-up drug, dose, citation, or mechanism |
| **Negation** | Output negates a clinically relevant fact ("rule out" → "rule in") |
| **Causality** | Speculates cause without evidence |
| **Contextual** | Mixes patient cohorts / specialties |
| **Reasoning** | Logically incoherent CoT (Track A only) |

Each instance scored **Major** (would change diagnosis or management) vs
**Minor** (stylistic), with **severity 1–5** (5 = could harm patient).

**Specific concerns for this dataset**:
- OpenMed/Medical-Reasoning-SFT-GPT-OSS-120B-V2 is *synthetic*, distilled from
  GPT-OSS-120B; any biases, blind spots, or hallucinations of the teacher are
  inherited.
- We never deploy the resulting model clinically.
- We never claim parity with clinician judgement.

**Mitigations**:
- Loud disclaimer in the model card and presentation.
- Three-judge LLM evaluation with reported inter-judge agreement (Cohen's κ).
- Manual audit on a stratified 30-sample subset.
- Explicit hallucination-rate reporting per Major/Minor × type bucket.

---

## 6. Evaluation Strategy

| Layer | Metric | Tool | Purpose |
|---|---|---|---|
| Lexical | Exact Match, ROUGE-L, sacreBLEU | `evaluate` | Sanity baseline |
| Semantic | BERTScore (deberta-xlarge), S-PubMedBert cosine | `bert_score`, `sentence-transformers` | Domain-aware similarity |
| Reasoning | LLM-judge step-correctness, faithfulness | Cerebras + Groq + Gemini consensus | Track A only |
| Clinical | LLM-judge correctness, safety | Same 3-judge | All tracks |
| External | MedQA-USMLE, PubMedQA, MMLU clinical (916 q) | `lm-evaluation-harness` | Cross-eval generalization |
| Manual | 30-sample stratified audit | Author | Sanity / inter-rater |
| Latency | tok/s, total inference time | Custom bench script | Cost trade-off |

Inter-judge agreement reported as Cohen's κ. Per-axis correctness reported as
mean ± std across the 3 judges.

---

## 7. Compute Plan

| Phase | Hardware | Software | Duration |
|---|---|---|---|
| Dev (Days 1–3) | Kaggle P100 16 GB | Unsloth + TRL + QLoRA | ~20–30 GPU-h |
| Final (Day 4) | C-DAC A30 / A100 | Same `train_sft.py`, scaled hyperparams | ~2 GPU-h |
| Eval | Laptop CPU + 3 free APIs | `llm_judge.py` | ~3 h wallclock |

**Models**:
- **Dev**: `unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit` (Apache-2.0, 32 K context,
  built-in `<think>` template support).
- **Final**: `unsloth/Llama-3.2-3B-Instruct-bnb-4bit` (matches the published
  medical-CoT precedent in arXiv 2510.05003) — fallback
  `Qwen2.5-3B-Instruct` if Llama license is an issue.

**Training method**: 4-bit QLoRA (NF4, double quantization, paged AdamW 8-bit),
rank-16 LoRA on `[q, k, v, o, gate, up, down]_proj`, cosine LR with 3% warmup.

---

## 8. Risks and Mitigations

| Risk | Mitigation |
|---|---|
| Free-tier API outage during eval | 3-provider rotation in `llm_judge.py` |
| Kaggle 30 h/week GPU quota | Subset to 5 K rows for dev; full set only on A100 |
| Chat-template mismatch (train vs inference) | Single `get_chat_template` call shared by `train_sft.py` and inference notebook |
| OOM on P100 | `max_seq_length=2048`, batch 1 + grad-accum 16, gradient checkpointing |
| Catastrophic forgetting | 1 epoch only; eval-loss early-stop |
| Single-shot A30/A100 booking failure | Day-3 Kaggle Track A on Qwen2.5-1.5B is the assignment-passing baseline |
| Single-judge bias | 3-judge consensus + κ |
| Synthetic-data hallucinations | Explicit disclaimer + cross-eval on real MedQA / PubMedQA |

---

## 9. Deliverables Checklist

- [ ] This design document
- [ ] `train_sft.py` + `requirements.txt`
- [ ] LoRA adapters on Hugging Face Hub: `trackB-v0`, `trackA-full-v0`,
  `trackA-short-v0`, `trackA-3B-final` (A30/A100 run)
- [ ] W&B project: `medical-reasoning-sft` with all 4 runs
- [ ] `llm_judge.py` + judged CSVs per track
- [ ] Phase-3 evaluation report
- [ ] Error analysis document with 6 worked examples (3 good, 3 bad)
- [ ] 10–12 slide presentation deck

---

## 10. Open Questions for Reviewer

1. <FILL — e.g., "Is the Llama-3.2 community license acceptable for the final
   scaled model, or must we stick with Apache-2.0 Qwen?">
2. <FILL — e.g., "Should final adapters be published on HF Hub publicly, or
   kept private to C-DAC?">
3. <FILL — e.g., "Do we want a clinician (radiology / internal medicine) on
   the manual audit panel?">

---

**Sign-off**:
- Author: Abhishek Kumar Singh, ___________________
- Reviewer (Mr. Ramesh Naidu L.): ___________________
- Reviewer (Dr. Janaki C H.): ___________________
