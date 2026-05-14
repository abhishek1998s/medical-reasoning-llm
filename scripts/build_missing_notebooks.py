"""Build notebooks 03-06 for the medical reasoning assignment.

The generated notebooks are intentionally concise and unexecuted. They rely on
the reusable code under src/ plus the existing train_sft.py and llm_judge.py.
"""

from __future__ import annotations

import json
from pathlib import Path


NB_DIR = Path("notebooks")
NB_DIR.mkdir(exist_ok=True)


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip() + "\n"}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "metadata": {},
        "source": text.strip() + "\n",
        "outputs": [],
        "execution_count": None,
    }


def write_notebook(path: Path, cells: list[dict]) -> None:
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {path} ({len(cells)} cells)")


def notebook_03() -> list[dict]:
    return [
        md(
            """# Notebook 03 - Train Track A Short-CoT

Goal: train the primary model with visible short clinical rationale plus final answer.
This repeats Notebook 02 with the same model, split seed, hyperparameters, and output
settings. The only experimental variable is the assistant output formatter."""
        ),
        md("## 1. Environment"),
        code(
            r"""
import os, sys, subprocess

REPO_URL = "https://github.com/abhishek1998s/medical-reasoning-llm.git"
REPO_DIR = "/kaggle/working/medical-reasoning-llm"

if not os.path.exists(REPO_DIR):
    subprocess.run(["git", "clone", REPO_URL, REPO_DIR], check=True)
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

os.chdir(REPO_DIR)
print("Working directory:", os.getcwd())
"""
        ),
        md("## 2. Train Track A"),
        code(
            r"""
# Uses train_sft.py so Track A and Track B remain script-reproducible.
# Keep config values identical to Track B except for --track.
!python train_sft.py \
  --track A_short \
  --num_samples 3300 \
  --max_seq_length 4096 \
  --batch_size 2 \
  --grad_accum 8 \
  --epochs 1.0 \
  --output_dir outputs/trackA \
  --run_name trackA-v0 \
  --push_to_hub \
  --hub_repo abhishek1998s/qwen25-1.5b-medreason-trackA-v0
"""
        ),
        md("## 3. Smoke Check"),
        code(
            r"""
from pathlib import Path
import json

adapter = Path("outputs/trackA/final_adapter")
assert adapter.exists(), "Track A adapter was not saved"
meta = adapter / "training_meta.json"
if meta.exists():
    print(json.dumps(json.loads(meta.read_text()), indent=2)[:2000])
print("Track A training artifact exists:", adapter)
"""
        ),
    ]


def notebook_04() -> list[dict]:
    return [
        md(
            """# Notebook 04 - Inference and Automatic Metrics

Goal: load both adapters, generate predictions on the same held-out test rows, extract
final answers, and compute EM/ROUGE-L plus optional BERTScore/sacreBLEU."""
        ),
        md("## 1. Setup"),
        code(
            r"""
import os, sys, subprocess, json, pandas as pd
from pathlib import Path

REPO_URL = "https://github.com/abhishek1998s/medical-reasoning-llm.git"
REPO_DIR = "/kaggle/working/medical-reasoning-llm"
if not os.path.exists(REPO_DIR):
    subprocess.run(["git", "clone", REPO_URL, REPO_DIR], check=True)
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)
os.chdir(REPO_DIR)
"""
        ),
        md("## 2. Load Shared Test Split"),
        code(
            r"""
import yaml
from datasets import load_dataset
from transformers import AutoTokenizer
from src.splits import shuffle_filter_split

cfg = yaml.safe_load(open("configs/experiment_config.yaml", encoding="utf-8"))
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
ds = load_dataset(cfg["dataset"]["name"], split=cfg["dataset"]["split"])
_, _, test_ds = shuffle_filter_split(
    ds,
    shuffle_seed=cfg["dataset"]["shuffle_seed"],
    num_train=cfg["dataset"]["num_train"],
    num_val=cfg["dataset"]["num_val"],
    num_test=cfg["dataset"]["num_test"],
    tokenizer=tok,
    max_total_tokens=3500,
)
print("test rows:", len(test_ds))
"""
        ),
        md("## 3. Generate Predictions"),
        code(
            r"""
import torch
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
from peft import PeftModel
from src.inference import build_prediction_row, generate_with_logging

def load_model(adapter_id):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg["model"]["name"],
        max_seq_length=cfg["model"]["max_seq_length"],
        dtype=None,
        load_in_4bit=True,
    )
    tokenizer = get_chat_template(tokenizer, chat_template=cfg["model"]["chat_template"])
    model = PeftModel.from_pretrained(model, adapter_id)
    FastLanguageModel.for_inference(model)
    return model, tokenizer

def run_track(track_name, adapter_id, out_csv):
    model, tokenizer = load_model(adapter_id)
    rows = []
    for i, row in enumerate(test_ds):
        user = next(m for m in row["messages"] if m["role"] == "user")
        asst = next(m for m in row["messages"] if m["role"] == "assistant")
        gen = generate_with_logging(
            model,
            tokenizer,
            user["content"],
            max_new_tokens=cfg["inference"]["max_new_tokens"]["track_A" if track_name == "A" else "track_B"],
            temperature=cfg["inference"]["temperature"],
            do_sample=cfg["inference"]["do_sample"],
            repetition_penalty=cfg["inference"]["repetition_penalty"],
            device="cuda",
        )
        rows.append(build_prediction_row(
            sample_id=i,
            question=user["content"],
            reference=(asst.get("content") or "").strip(),
            track_name=track_name,
            model_id=cfg["model"]["name"],
            adapter_id=adapter_id,
            generation=gen,
        ))
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    return pd.DataFrame(rows)

track_a = run_track("A", f"{cfg['hub']['username']}/{cfg['hub']['repos']['trackA']}", "outputs/trackA/predictions.csv")
track_b = run_track("B", f"{cfg['hub']['username']}/{cfg['hub']['repos']['trackB']}", "outputs/trackB/predictions.csv")
"""
        ),
        md("## 4. Metrics"),
        code(
            r"""
from src.data_formatting import extract_answer_for_scoring
from src.metrics import compute_core_metrics, compute_sacrebleu, compute_bertscore

def score_file(path, track):
    df = pd.read_csv(path)
    preds = [extract_answer_for_scoring(p, track) for p in df["prediction"]]
    refs = list(df["reference"])
    core = compute_core_metrics(preds, refs)
    core["mean_output_tokens"] = df["output_tokens"].mean()
    core["mean_generation_time_s"] = df["generation_time_s"].mean()
    core["mean_tokens_per_sec"] = df["tokens_per_sec"].mean()
    return core

summary = {
    "trackA": score_file("outputs/trackA/predictions.csv", "A"),
    "trackB": score_file("outputs/trackB/predictions.csv", "B"),
}
Path("outputs/metrics_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
summary
"""
        ),
    ]


def notebook_05() -> list[dict]:
    return [
        md(
            """# Notebook 05 - LLM Judge and Manual Safety Review

Goal: run the judge script on both prediction files and create the manual-audit CSV
templates required for safety/error review."""
        ),
        md("## 1. Run LLM Judge"),
        code(
            r"""
# Requires at least one of CEREBRAS_API_KEY, GROQ_API_KEY, GEMINI_API_KEY.
!python llm_judge.py --predictions outputs/trackA/predictions.csv --output outputs/trackA/judged.csv --limit 200
!python llm_judge.py --predictions outputs/trackB/predictions.csv --output outputs/trackB/judged.csv --limit 200
"""
        ),
        md("## 2. Build Manual Audit Templates"),
        code(
            r"""
import pandas as pd
from src.safety_rubric import build_blank_audit_rows, make_audit_csv

def pick_audit_rows(path):
    df = pd.read_csv(path)
    # Deterministic skeleton: first 10 low, next 10 medium, next 10 high.
    buckets = [("low", df.head(10)), ("medium", df.iloc[10:20]), ("high", df.iloc[20:30])]
    rows = []
    for bucket, part in buckets:
        rows.extend(build_blank_audit_rows(part.to_dict("records"), track_name=str(df.iloc[0]["track_name"]), risk_bucket=bucket))
    return rows

make_audit_csv(pick_audit_rows("outputs/trackA/predictions.csv"), "outputs/trackA/safety_audit.csv")
make_audit_csv(pick_audit_rows("outputs/trackB/predictions.csv"), "outputs/trackB/safety_audit.csv")
print("Wrote safety audit templates.")
"""
        ),
        md("## 3. Audit Instructions"),
        md(
            """Fill the generated CSVs manually. This is a non-clinical safety audit:
flag obvious hallucinations, unsafe advice, overconfidence, and missing disclaimers.
Do not present the audit as clinical validation."""
        ),
    ]


def notebook_06() -> list[dict]:
    return [
        md(
            """# Notebook 06 - Report and Comparison

Goal: merge automatic metrics, judge results, and manual audit notes into the final
tables answering the assignment questions."""
        ),
        md("## 1. Load Outputs"),
        code(
            r"""
import json, pandas as pd
from pathlib import Path

metrics = json.loads(Path("outputs/metrics_summary.json").read_text(encoding="utf-8"))
pred_a = pd.read_csv("outputs/trackA/predictions.csv")
pred_b = pd.read_csv("outputs/trackB/predictions.csv")
judge_a = pd.read_csv("outputs/trackA/judged.csv") if Path("outputs/trackA/judged.csv").exists() else pd.DataFrame()
judge_b = pd.read_csv("outputs/trackB/judged.csv") if Path("outputs/trackB/judged.csv").exists() else pd.DataFrame()
audit_a = pd.read_csv("outputs/trackA/safety_audit.csv") if Path("outputs/trackA/safety_audit.csv").exists() else pd.DataFrame()
audit_b = pd.read_csv("outputs/trackB/safety_audit.csv") if Path("outputs/trackB/safety_audit.csv").exists() else pd.DataFrame()
metrics
"""
        ),
        md("## 2. Comparison Table"),
        code(
            r"""
rows = []
for track, data in metrics.items():
    rows.append({
        "track": track,
        "exact_match": data.get("exact_match"),
        "rouge_l": data.get("rouge_l"),
        "mean_output_tokens": data.get("mean_output_tokens"),
        "mean_generation_time_s": data.get("mean_generation_time_s"),
        "mean_tokens_per_sec": data.get("mean_tokens_per_sec"),
    })
comparison = pd.DataFrame(rows)
comparison.to_csv("outputs/final_comparison_table.csv", index=False)
comparison
"""
        ),
        md("## 3. Judge and Safety Summaries"),
        code(
            r"""
def judge_summary(df):
    if df.empty:
        return {}
    cols = [c for c in df.columns if c.startswith("mean_")] + ["n_major_errors", "max_severity", "any_unsafe"]
    return {c: df[c].mean() for c in cols if c in df}

def audit_summary(df):
    if df.empty:
        return {}
    return {
        "risk_severity": df.get("risk_severity", pd.Series(dtype=str)).value_counts(dropna=False).to_dict(),
        "hallucination_type": df.get("hallucination_type", pd.Series(dtype=str)).value_counts(dropna=False).to_dict(),
        "safe_behavior": df.get("safe_behavior", pd.Series(dtype=str)).value_counts(dropna=False).to_dict(),
    }

report_summary = {
    "judge_trackA": judge_summary(judge_a),
    "judge_trackB": judge_summary(judge_b),
    "audit_trackA": audit_summary(audit_a),
    "audit_trackB": audit_summary(audit_b),
}
Path("outputs/report_summary.json").write_text(json.dumps(report_summary, indent=2), encoding="utf-8")
report_summary
"""
        ),
        md("## 4. Final Report Notes"),
        md(
            """Use the generated CSV/JSON files to write the final narrative:

- Does reasoning improve QA? Use EM, ROUGE-L, BERTScore if run, and judge correctness.
- Does reasoning increase cost or latency? Use output tokens and generation time.
- Does reasoning increase hallucinations? Use judge errors plus manual audit counts.
- Should reasoning be hidden or shown? Discuss reasoning clarity versus safety behavior.
- How unsafe is the model in edge cases? Use high-risk audit rows and unsafe rate.
- Include six worked examples: three successes and three failures.

State clearly that this is a learning artifact, not a clinical product."""
        ),
    ]


def main() -> None:
    write_notebook(NB_DIR / "03_train_trackA_short_cot.ipynb", notebook_03())
    write_notebook(NB_DIR / "04_inference_and_metrics.ipynb", notebook_04())
    write_notebook(NB_DIR / "05_llm_judge_and_safety_review.ipynb", notebook_05())
    write_notebook(NB_DIR / "06_report_and_comparison.ipynb", notebook_06())


if __name__ == "__main__":
    main()
