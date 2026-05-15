# Fine-Tune a LLM for Medical Reasoning

**Concise submission based on the full run archived in `notebooks/pipeline_outputs_20260515_131741.zip`**  
**Status:** research artifact only; not suitable for clinical deployment.

## Phase 1: Problem Formulation

The task is to fine-tune a small medical QA model on `OpenMed/Medical-Reasoning-SFT-GPT-OSS-120B-V2` so it can answer medical questions with high correctness, reliability, and interpretable reasoning. The dataset is single-turn and provides both a final answer and a teacher reasoning trace, so the cleanest primary formulation is **Track A: reasoning + final answer**, with **Track B: answer only** used as the required ablation baseline.

**Primary track choice.** Track A was selected because the assignment explicitly asks whether reasoning improves medical QA and when reasoning should be shown or hidden. That question cannot be answered with an answer-only system alone. In this implementation, Track A is a **short clinical rationale + final answer** format rather than unlimited chain-of-thought, because the base model is only 1.5B and the training/inference budget is constrained.

**Input-output design.**
- Input: one medical user query.
- Track A output: `Clinical rationale:` followed by a short rationale, then `Final answer:`.
- Track B output: final answer only.
- Both tracks are evaluated on the same held-out split.

**Main risks.**
- Medical hallucination: fabricated facts, incorrect anatomy, incorrect causal claims, or overconfident advice.
- Reasoning failure: a visible rationale can sound plausible while being clinically wrong.
- Dataset risk: the dataset is synthetic and distilled from a larger teacher model, so teacher bias can be inherited.

**Evaluation strategy.**
- Core metrics: Exact Match, ROUGE-L, BERTScore.
- Reasoning-aware checks: judge-rated clinical correctness, factuality, reasoning soundness, safety, and hallucination counts.
- Manual review: spot-check representative outputs for factuality, refusal behavior, and reasoning clarity.

## Phase 2: Model Training

The base model is `unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit`. Fine-tuning uses **QLoRA** with LoRA rank 16, alpha 16, dropout 0.0, sequence length 4096, learning rate `2e-4`, one epoch, and effective batch size 16. The full run used **3,000 train / 150 validation / 200 test** examples.

Two training variants were executed:

1. **Experiment 1: With Reasoning**
   Track A, trained on short visible rationale plus final answer.
2. **Experiment 2: Without Reasoning**
   Track B, trained on final answer only.

Training completed successfully for both tracks.

| Item | Track A | Track B |
|---|---:|---:|
| Train loss | 1.328 | 1.336 |
| Eval loss | 1.216 | 1.251 |
| Training time | 116.1 min | 110.3 min |

The training result is acceptable from an optimization standpoint: both adapters converged and produced usable outputs. The real question is whether visible reasoning improved answer quality enough to justify its additional cost.

## Phase 3: Evaluation

### Automatic results

| Metric | Track A | Track B |
|---|---:|---:|
| Exact Match | 0.000 | 0.000 |
| ROUGE-L | 0.131 | **0.142** |
| BERTScore F1 | 0.815 | **0.849** |
| Mean output tokens | 679.5 | **326.5** |
| Mean generation time | 37.5 s | **18.0 s** |
| Truncation rate | 82.5% | **70.5%** |

Track B is better on semantic accuracy, cheaper in tokens, and about twice as fast. Track A generates much longer outputs and is frequently cut off before finishing cleanly.

Judge-based evaluation points in the same direction, but the coverage is incomplete in the archived run and must be read as directional rather than final. Judged-row coverage is **130/200** for Track A and **45/200** for Track B. On those judged rows:

- Clinical correctness: **2.27/5** for Track A vs **3.16/5** for Track B.
- Factuality: **3.43/5** for Track A vs **3.73/5** for Track B.
- Safety: **4.25/5** for Track A vs **4.20/5** for Track B.
- Unsafe rate: **10.0%** for Track A vs **2.2%** for Track B.
- Pass rate: **30.0%** for Track A vs **44.4%** for Track B.

Hallucination counts are materially worse for Track A. In the archived judged outputs, Track A accumulated **45 fabrication errors** and **21 reasoning errors**, versus **8 fabrication** and **2 reasoning** errors for Track B.

### Manual verification remarks

Manual spot-checking of representative outputs shows a consistent pattern:

- On ambiguous questions, **Track B behaves better**. It usually asks for clarification or refuses to guess.
- **Track A often over-interprets the prompt**, writes a long rationale, and sometimes invents specifics that were not asked for.
- On direct factual questions, Track A frequently becomes verbose and repetitive, which increases truncation and reduces answer reliability.
- Neither model is safe enough for clinical use, but Track A is the riskier behavior because it exposes more unsupported reasoning.

Concrete examples from the archived run:
- For the anterior cerebral artery question, Track A produced an incorrect anatomy explanation and was flagged unsafe.
- For the intestinal inflammation cytokine question, Track A expanded into a long, error-prone list; Track B was still wrong, but less extreme.
- For under-specified study questions, Track B gave the more appropriate clarification-style answer.

### Final assessment

For this setup, **reasoning did not improve medical QA**. The answer-only model delivered better quality with lower latency and lower token cost. The evidence also suggests that reasoning should be **hidden, or not trained explicitly at this model scale**, unless a larger model, cleaner supervision, and stricter inference controls are used.

Two limitations remain before calling the evaluation fully complete:
- the manual safety audit CSVs are still blank in the archived outputs;
- judge coverage is partial, so the judge summary is not yet a full 200-row comparison.

Even with those limitations, the current result is clear: at **Qwen2.5-1.5B + 3K samples + 1 epoch**, visible reasoning increases cost and failure surface more than it improves answer quality.
