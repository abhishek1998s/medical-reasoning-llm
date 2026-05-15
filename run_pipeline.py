"""
run_pipeline.py — Full end-to-end medical-reasoning fine-tuning pipeline.

Combines Notebooks 02-06 into a single script for local GPU servers.

Stages (run all by default, or pick with --stages):
  train_b   — Train Track B  (answer only)
  train_a   — Train Track A  (short clinical CoT)
  infer     — Inference on test split for both adapters
  metrics   — Compute EM + ROUGE-L, write metrics_summary.json
  judge     — LLM-as-judge via Cerebras / Groq / Gemini API
  audit     — Build blank manual safety-audit CSV templates
  report    — Merge everything into final comparison tables

Usage examples:
  # Full pipeline, dry-run settings from config
  python run_pipeline.py

  # Only train then infer
  python run_pipeline.py --stages train_b,train_a,infer,metrics

  # Skip training (adapters already on Hub), just evaluate
  python run_pipeline.py --stages infer,metrics,judge,audit,report

  # Full training run (override config num_train, etc.)
  python run_pipeline.py --full

Required env vars (set in shell or .env):
  HF_TOKEN          — HuggingFace token with read+write scope
  WANDB_API_KEY     — Weights & Biases API key (set to 'none' to disable)
  CEREBRAS_API_KEY  — at least one judge key
  GROQ_API_KEY
  GEMINI_API_KEY
"""

import argparse
import gc
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml


# ============================================================
# Helpers
# ============================================================

def _load_cfg(repo_dir: Path) -> dict:
    return yaml.safe_load((repo_dir / "configs" / "experiment_config.yaml").read_text(encoding="utf-8"))


def _section(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print("=" * 60)


def _run(cmd: str, cwd: Path) -> None:
    """Run a shell command, raise on non-zero exit."""
    print(f"\n$ {cmd}\n")
    ret = subprocess.run(cmd, shell=True, cwd=str(cwd))
    if ret.returncode != 0:
        raise SystemExit(f"Command failed with code {ret.returncode}:\n  {cmd}")


# ============================================================
# Stage 1 & 2 — Training
# ============================================================

def train(track: str, cfg: dict, repo_dir: Path, full: bool) -> None:
    """Call train_sft.py for the given track."""
    hub_key = "trackA" if track == "A_short" else "trackB"
    run_key = "trackA" if track == "A_short" else "trackB"
    out_dir = f"outputs/track{'A' if track == 'A_short' else 'B'}"

    if full:
        num_samples = 3000
        max_seq     = 4096
        batch       = cfg["training"]["per_device_train_batch_size"]
        grad_accum  = 8
        max_rows    = None
    else:
        num_samples = cfg["dataset"]["num_train"]
        max_seq     = cfg["model"]["max_seq_length"]
        batch       = cfg["training"]["per_device_train_batch_size"]
        grad_accum  = cfg["training"]["gradient_accumulation_steps"]
        max_rows    = cfg["dataset"].get("max_rows")

    hub_repo = f"{cfg['hub']['username']}/{cfg['hub']['repos'][hub_key]}"
    run_name = cfg["logging"]["wandb_runs"][run_key]

    cmd = (
        f"python train_sft.py"
        f" --track {track}"
        f" --num_samples {num_samples}"
        f" --max_seq_length {max_seq}"
        f" --batch_size {batch}"
        f" --grad_accum {grad_accum}"
        f" --epochs {cfg['training']['epochs']}"
        f" --output_dir {out_dir}"
        f" --run_name {run_name}"
        f" --push_to_hub"
        f" --hub_repo {hub_repo}"
    )

    # Track A: pass short_cot_tokens from config
    if track == "A_short":
        cmd += f" --short_cot_tokens {cfg['dataset']['short_cot_max_tokens']}"

    # Dry-run: cap dataset scan
    if max_rows is not None:
        cmd += f" --max_rows {max_rows}"

    # Forward observability cadence from config so dry-run actually evals/saves/logs
    cmd += f" --eval_steps {cfg['training']['eval_steps']}"
    cmd += f" --save_steps {cfg['training']['save_steps']}"
    cmd += f" --logging_steps {cfg['training']['logging_steps']}"

    # Forward LoRA dropout (must be 0.0 to enable Unsloth's fast LoRA kernels)
    # and warmup_ratio so train_sft.py builds an absolute warmup_steps.
    cmd += f" --lora_dropout {cfg['lora']['dropout']}"
    cmd += f" --warmup_ratio {cfg['training']['warmup_ratio']}"

    _run(cmd, repo_dir)


# ============================================================
# Stage 3 — Inference
# ============================================================

def run_inference(cfg: dict, repo_dir: Path) -> None:
    """Load both adapters, generate predictions on the shared test split."""
    import pandas as pd
    import torch
    from datasets import load_dataset
    from peft import PeftModel
    from unsloth import FastLanguageModel
    from unsloth.chat_templates import get_chat_template

    from src.inference import build_prediction_row, generate_with_logging
    from src.splits import shuffle_filter_split

    hf_token = os.environ.get("HF_TOKEN") or None

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct", token=hf_token)

    ds = load_dataset(cfg["dataset"]["name"], split=cfg["dataset"]["split"],
                      token=hf_token)
    _, _, test_ds = shuffle_filter_split(
        ds,
        shuffle_seed=cfg["dataset"]["shuffle_seed"],
        num_train=cfg["dataset"]["num_train"],
        num_val=cfg["dataset"]["num_val"],
        num_test=cfg["dataset"]["num_test"],
        tokenizer=tok,
        max_total_tokens=3500,
        max_rows=cfg["dataset"].get("max_rows"),
    )
    print(f"  test rows: {len(test_ds)}")

    # Save frozen split indices for reproducibility
    split_path = Path("outputs/test_split_indices.json")
    split_path.parent.mkdir(parents=True, exist_ok=True)
    split_path.write_text(json.dumps({
        "indices": (
            test_ds._indices.to_pylist()
            if hasattr(test_ds, "_indices") and test_ds._indices is not None
            else list(range(len(test_ds)))
        ),
        "shuffle_seed": cfg["dataset"]["shuffle_seed"],
        "num_test": cfg["dataset"]["num_test"],
        "max_rows": cfg["dataset"].get("max_rows"),
    }, indent=2))
    print(f"  split indices saved -> {split_path}")

    def load_model(adapter_id: str):
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=cfg["model"]["name"],
            max_seq_length=cfg["model"]["max_seq_length"],
            dtype=None,
            load_in_4bit=True,
        )
        tokenizer = get_chat_template(tokenizer,
                                      chat_template=cfg["model"]["chat_template"])
        model = PeftModel.from_pretrained(model, adapter_id, token=hf_token)
        FastLanguageModel.for_inference(model)
        return model, tokenizer

    def run_track(track_name: str, adapter_id: str, out_csv: str) -> None:
        print(f"\n[{track_name}] loading adapter: {adapter_id}")
        model, tokenizer = load_model(adapter_id)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        rows = []
        cfg_track_key = "track_A" if track_name == "A" else "track_B"
        max_new = cfg["inference"]["max_new_tokens"][cfg_track_key]

        for i, row in enumerate(test_ds):
            user = next(m for m in row["messages"] if m["role"] == "user")
            asst = next(m for m in row["messages"] if m["role"] == "assistant")
            gen = generate_with_logging(
                model, tokenizer, user["content"],
                max_new_tokens=max_new,
                temperature=cfg["inference"]["temperature"],
                do_sample=cfg["inference"]["do_sample"],
                repetition_penalty=cfg["inference"]["repetition_penalty"],
                device=device,
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
            if (i + 1) % 10 == 0:
                print(f"  {i + 1}/{len(test_ds)} done")

        Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(out_csv, index=False)
        print(f"  saved -> {out_csv}")

        # Free GPU memory before loading the next adapter
        del model, tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    username = cfg["hub"]["username"]
    run_track("A", f"{username}/{cfg['hub']['repos']['trackA']}", "outputs/trackA/predictions.csv")
    run_track("B", f"{username}/{cfg['hub']['repos']['trackB']}", "outputs/trackB/predictions.csv")


# ============================================================
# Stage 4 — Metrics
# ============================================================

def run_metrics(cfg: dict) -> None:
    import pandas as pd
    from src.data_formatting import extract_answer_for_scoring
    from src.metrics import compute_core_metrics, compute_operational_stats

    def score_file(path: str, track: str) -> dict:
        df = pd.read_csv(path)
        preds = [extract_answer_for_scoring(p, track) for p in df["prediction"]]
        refs  = list(df["reference"])
        # try_bertscore=True: PDF Phase 3 requires "Semantic Scores" alongside
        # EM; BERTScore is the semantic metric. Silently skipped if bert-score
        # isn't installed (compute_core_metrics catches ImportError).
        core  = compute_core_metrics(preds, refs, try_bertscore=True)
        core["mean_output_tokens"]     = round(float(df["output_tokens"].mean()), 2)
        core["mean_generation_time_s"] = round(float(df["generation_time_s"].mean()), 3)
        core["mean_tokens_per_sec"]    = round(float(df["tokens_per_sec"].mean()), 2)
        return core

    summary = {
        "trackA": score_file("outputs/trackA/predictions.csv", "A"),
        "trackB": score_file("outputs/trackB/predictions.csv", "B"),
    }
    Path("outputs/metrics_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))

    # Inference diagnostics. compute_operational_stats may omit any key when
    # the underlying column is missing or non-numeric, so guard each format
    # with a numeric-only helper instead of passing the string fallback
    # straight into a `:.1%`/`:.3f` spec (which would ValueError).
    def _pct(v):
        return f"{v:.1%}" if isinstance(v, (int, float)) else "N/A"

    def _sec(v):
        return f"{v:.3f}s" if isinstance(v, (int, float)) else "N/A"

    for track_name, csv_path in [("A", "outputs/trackA/predictions.csv"),
                                   ("B", "outputs/trackB/predictions.csv")]:
        df = pd.read_csv(csv_path)
        stats = compute_operational_stats(df)
        print(f"\n=== Track {track_name} Inference Diagnostics ===")
        print(f"  truncation_rate:           {_pct(stats.get('truncation_rate'))}")
        print(f"  empty_prediction_rate:     {_pct(stats.get('empty_prediction_rate'))}")
        print(f"  finish_reason_dist:        {stats.get('finish_reason_dist', {})}")
        print(f"  output_tokens p50/p90/p99: "
              f"{stats.get('output_tokens_p50', '?')} / "
              f"{stats.get('output_tokens_p90', '?')} / "
              f"{stats.get('output_tokens_p99', '?')}")
        print(f"  gen_time p50/p90:          "
              f"{_sec(stats.get('generation_time_p50'))} / "
              f"{_sec(stats.get('generation_time_p90'))}")


# ============================================================
# Stage 5 — LLM Judge
# ============================================================

def run_judge(cfg: dict, repo_dir: Path) -> None:
    limit = cfg["dataset"]["num_test"]
    judge_cfg = cfg.get("judge", {})
    consensus = judge_cfg.get("consensus_mode", False)
    providers = judge_cfg.get("providers", ["cerebras", "groq", "gemini"])
    sleep_s = judge_cfg.get("sleep_seconds", 2.0)
    consensus_flag = " --all_judges" if consensus else ""
    judges_flag = " --judges " + ",".join(providers)
    sleep_flag = f" --sleep {sleep_s}"
    print(f"  judge mode: {'consensus (all judges per row)' if consensus else 'single-judge fallback chain'}")
    print(f"  providers : {providers}")
    print(f"  sleep_s   : {sleep_s}")
    for track in ("trackA", "trackB"):
        _run(
            f"python llm_judge.py"
            f" --predictions outputs/{track}/predictions.csv"
            f" --output      outputs/{track}/judged.csv"
            f" --limit       {limit}"
            f"{judges_flag}"
            f"{consensus_flag}"
            f"{sleep_flag}",
            repo_dir,
        )


# ============================================================
# Stage 6 — Safety Audit templates
# ============================================================

def run_audit() -> None:
    import pandas as pd
    from src.safety_rubric import build_blank_audit_rows, make_audit_csv

    def pick_audit_rows(pred_path: str, judged_path: str) -> list:
        pred_df = pd.read_csv(pred_path)
        if len(pred_df) == 0:
            raise ValueError(f"predictions file is empty: {pred_path}")

        track_name = str(pred_df.iloc[0]["track_name"])

        try:
            judge_df = pd.read_csv(judged_path)
            df = pred_df.merge(
                judge_df[["sample_id", "any_unsafe", "majority_pass",
                           "n_major_errors", "max_severity", "n_errors"]],
                on="sample_id", how="left",
            )
            has_judge = True
        except (FileNotFoundError, KeyError):
            df = pred_df.copy()
            has_judge = False

        if has_judge:
            mask_high   = (df.get("any_unsafe", False) == True) | \
                          (df.get("n_major_errors", 0) > 0) | \
                          (df.get("max_severity", 0) >= 4)
            mask_medium = (~mask_high) & (
                          (df.get("majority_pass", True) == False) |
                          (df.get("n_errors", 0) > 0) |
                          (df.get("truncated", False) == True)
            )
        else:
            mask_high   = df.get("truncated", pd.Series([False] * len(df))) == True
            mask_medium = pd.Series([False] * len(df))
            print(f"  [{track_name}] judge output not found — used truncation as risk signal")

        mask_low = ~mask_high & ~mask_medium
        high_rows   = df[mask_high]
        medium_rows = df[mask_medium]
        low_rows    = df[mask_low].sample(frac=1, random_state=42)

        # For very small test sets (dry-run), ensure at least 1 row per bucket
        if len(high_rows) == 0 and len(df) > 0:
            high_rows   = df.iloc[[0]]
            medium_rows = df.iloc[1:2] if len(df) > 1 else medium_rows
            low_rows    = df.iloc[2:]  if len(df) > 2 else low_rows

        rows = []
        for bucket, part in [("high", high_rows), ("medium", medium_rows), ("low", low_rows)]:
            if part.empty:
                continue
            rows.extend(build_blank_audit_rows(
                part.to_dict("records"),
                track_name=track_name,
                risk_bucket=bucket,
            ))

        print(f"  [{track_name}] high={len(high_rows)}  medium={len(medium_rows)}  low={len(low_rows)} rows selected")
        return rows

    for track in ("trackA", "trackB"):
        make_audit_csv(
            pick_audit_rows(
                f"outputs/{track}/predictions.csv",
                f"outputs/{track}/judged.csv",
            ),
            f"outputs/{track}/safety_audit.csv",
        )
        print(f"  wrote outputs/{track}/safety_audit.csv")

    print("\nFill in the CSVs manually, then re-run with --stages report")


# ============================================================
# Stage 7 — Report
# ============================================================

_COMPARABLE_AXES = [
    "mean_clinical_correctness", "mean_factuality",
    "mean_completeness", "mean_safety",
]


def run_report() -> None:
    import pandas as pd
    from src.data_formatting import extract_answer_for_scoring

    metrics = json.loads(Path("outputs/metrics_summary.json").read_text(encoding="utf-8"))

    def _load_csv(path: str) -> pd.DataFrame:
        p = Path(path)
        if p.exists() and p.stat().st_size > 0:
            return pd.read_csv(p)
        print(f"  [missing] {path}")
        return pd.DataFrame()

    judge_a = _load_csv("outputs/trackA/judged.csv")
    judge_b = _load_csv("outputs/trackB/judged.csv")
    audit_a = _load_csv("outputs/trackA/safety_audit.csv")
    audit_b = _load_csv("outputs/trackB/safety_audit.csv")

    # Core comparison table
    rows = []
    for track, data in metrics.items():
        rows.append({
            "track":                  track,
            "exact_match":            data.get("exact_match"),
            "rouge_l":                data.get("rouge_l"),
            "mean_output_tokens":     data.get("mean_output_tokens"),
            "mean_generation_time_s": data.get("mean_generation_time_s"),
            "mean_tokens_per_sec":    data.get("mean_tokens_per_sec"),
            "n":                      data.get("n"),
        })
    comparison = pd.DataFrame(rows).set_index("track")
    comparison.to_csv("outputs/final_comparison_table.csv")
    print("\n=== Core Comparison ===")
    print(comparison.to_string())

    # Per-sample merged table with deltas
    pa = _load_csv("outputs/trackA/predictions.csv")
    pb = _load_csv("outputs/trackB/predictions.csv")
    if not pa.empty and not pb.empty:
        def _em_col(df: pd.DataFrame, track: str) -> list:
            preds = [extract_answer_for_scoring(p, track) for p in df["prediction"]]
            refs  = list(df["reference"])
            return [1.0 if p.strip().lower() == r.strip().lower() else 0.0
                    for p, r in zip(preds, refs)]

        pa = pa.copy()
        pb = pb.copy()
        pa["em"] = _em_col(pa, "A")
        pb["em"] = _em_col(pb, "B")

        merged = pa[["sample_id", "question", "reference",
                     "prediction", "em", "output_tokens", "generation_time_s"]].merge(
            pb[["sample_id", "prediction", "em", "output_tokens", "generation_time_s"]],
            on="sample_id", suffixes=("_A", "_B"),
        )
        merged["em_delta_A_minus_B"]      = merged["em_A"]             - merged["em_B"]
        merged["token_delta_A_minus_B"]   = merged["output_tokens_A"]  - merged["output_tokens_B"]
        merged["latency_delta_A_minus_B"] = merged["generation_time_s_A"] - merged["generation_time_s_B"]
        merged.to_csv("outputs/per_sample_comparison.csv", index=False)
        print(f"\nPer-sample table: {len(merged)} rows -> outputs/per_sample_comparison.csv")

        a_better = int((merged["em_delta_A_minus_B"] >  0).sum())
        b_better = int((merged["em_delta_A_minus_B"] <  0).sum())
        tied     = int((merged["em_delta_A_minus_B"] == 0).sum())
        trunc_a  = int(pa.get("truncated", pd.Series([False] * len(pa))).sum())
        trunc_b  = int(pb.get("truncated", pd.Series([False] * len(pb))).sum())

        print(f"\n=== Case Breakdown ===")
        print(f"  A better  (EM A=1, B=0): {a_better}")
        print(f"  B better  (EM B=1, A=0): {b_better}")
        print(f"  Tied:                    {tied}")
        print(f"  Truncated A / B:         {trunc_a} / {trunc_b}")

        # Worked examples: 3 A-wins, 3 B-wins
        def _show(row: pd.Series, idx: int) -> None:
            print(f"\n--- Example {idx} ---")
            print(f"Q:         {str(row['question'])[:300]}")
            print(f"Reference: {str(row['reference'])[:200]}")
            print(f"Track A:   {str(row['prediction_A'])[:300]}")
            print(f"Track B:   {str(row['prediction_B'])[:300]}")
            print(f"EM A={row['em_A']:.0f}  EM B={row['em_B']:.0f}  "
                  f"tokens A={row['output_tokens_A']}  B={row['output_tokens_B']}")

        a_wins = merged[merged["em_delta_A_minus_B"] >  0].head(3)
        b_wins = merged[merged["em_delta_A_minus_B"] <  0].head(3)

        print("\n===== 3 Cases where Track A (CoT) wins =====")
        for i, (_, row) in enumerate(a_wins.iterrows(), 1):
            _show(row, i)

        print("\n===== 3 Cases where Track B (no CoT) wins =====")
        for i, (_, row) in enumerate(b_wins.iterrows(), 1):
            _show(row, i)

    # Judge + audit summary
    def judge_summary(df: pd.DataFrame) -> dict:
        if df.empty:
            return {"note": "no judge results yet"}
        out = {}
        for col in _COMPARABLE_AXES:
            if col in df.columns and df[col].notna().any():
                out[col] = round(float(df[col].mean()), 3)
        rs_col = "mean_reasoning_soundness"
        if rs_col in df.columns and df[rs_col].notna().any():
            n_valid = int(df[rs_col].notna().sum())
            out[rs_col]           = round(float(df[rs_col].mean()), 3)
            out[f"{rs_col}_n"]    = n_valid
        for col in ["n_major_errors", "max_severity", "n_errors"]:
            if col in df.columns:
                out[col] = round(float(df[col].mean()), 3)
        if "any_unsafe" in df.columns:
            out["unsafe_rate"] = round(float(df["any_unsafe"].mean()), 3)
        if "majority_pass" in df.columns:
            out["pass_rate"]   = round(float(df["majority_pass"].mean()), 3)
        return out

    def audit_summary(df: pd.DataFrame) -> dict:
        if df.empty:
            return {"note": "no audit results yet"}
        return {
            "risk_severity":      df["risk_severity"].value_counts(dropna=False).to_dict()      if "risk_severity"      in df.columns else {},
            "hallucination_type": df["hallucination_type"].value_counts(dropna=False).to_dict() if "hallucination_type" in df.columns else {},
            "safe_behavior":      df["safe_behavior"].value_counts(dropna=False).to_dict()      if "safe_behavior"      in df.columns else {},
        }

    report_summary = {
        "judge_trackA": judge_summary(judge_a),
        "judge_trackB": judge_summary(judge_b),
        "audit_trackA": audit_summary(audit_a),
        "audit_trackB": audit_summary(audit_b),
    }
    Path("outputs/report_summary.json").write_text(
        json.dumps(report_summary, indent=2), encoding="utf-8"
    )
    print("\n=== Report Summary ===")
    print(json.dumps(report_summary, indent=2))
    print("\nSaved:")
    print("  outputs/final_comparison_table.csv")
    print("  outputs/per_sample_comparison.csv")
    print("  outputs/report_summary.json")


# ============================================================
# Main
# ============================================================

ALL_STAGES = ["train_b", "train_a", "infer", "metrics", "judge", "audit", "report"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--stages",
        default=",".join(ALL_STAGES),
        help=f"Comma-separated stages to run. Default: all. Options: {', '.join(ALL_STAGES)}",
    )
    ap.add_argument(
        "--full",
        action="store_true",
        help="Override config to full-run values (num_train=3000, max_seq=4096, grad_accum=8)",
    )
    args = ap.parse_args()

    stages = [s.strip() for s in args.stages.split(",") if s.strip()]
    bad = [s for s in stages if s not in ALL_STAGES]
    if bad:
        ap.error(f"Unknown stages: {bad}. Valid: {ALL_STAGES}")

    repo_dir = Path(__file__).parent.resolve()
    if str(repo_dir) not in sys.path:
        sys.path.insert(0, str(repo_dir))

    cfg = _load_cfg(repo_dir)

    # HF login if token present
    hf_token = os.environ.get("HF_TOKEN", "")
    if hf_token:
        try:
            from huggingface_hub import login as hf_login
            hf_login(token=hf_token, add_to_git_credential=False)
            print("HF login OK")
        except Exception as e:
            print(f"HF login failed: {e}")

    # WandB config
    wandb_key = os.environ.get("WANDB_API_KEY", "")
    if not wandb_key or wandb_key.lower() == "none":
        os.environ["WANDB_DISABLED"] = "true"
    else:
        os.environ["WANDB_PROJECT"] = cfg["logging"]["wandb_project"]

    print(f"\nPipeline stages: {stages}")
    print(f"Mode: {'FULL' if args.full else 'DRY-RUN'}")
    print(f"num_train: {3000 if args.full else cfg['dataset']['num_train']}")

    # ---- Run stages ----

    if "train_b" in stages:
        _section("Stage 1: Train Track B (answer only)")
        train("B", cfg, repo_dir, args.full)

    if "train_a" in stages:
        _section("Stage 2: Train Track A (short clinical CoT)")
        train("A_short", cfg, repo_dir, args.full)

    if "infer" in stages:
        _section("Stage 3: Inference (both tracks)")
        run_inference(cfg, repo_dir)

    if "metrics" in stages:
        _section("Stage 4: Automatic Metrics")
        run_metrics(cfg)

    if "judge" in stages:
        _section("Stage 5: LLM Judge")
        run_judge(cfg, repo_dir)

    if "audit" in stages:
        _section("Stage 6: Safety Audit Templates")
        run_audit()

    if "report" in stages:
        _section("Stage 7: Report and Comparison")
        run_report()

    print("\n✅ Pipeline complete.")


if __name__ == "__main__":
    main()
