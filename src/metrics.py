"""Automatic evaluation metrics for the medical-reasoning A/B study.

The public helpers accept plain Python sequences so they work in notebooks,
scripts, and tests. Optional heavyweight metrics import their dependencies
only when called.
"""

from __future__ import annotations

import re
import string
from collections.abc import Sequence
from dataclasses import dataclass
from statistics import mean
from typing import Any


_PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def normalize_answer(text: str) -> str:
    """Lowercase, remove punctuation, and collapse whitespace."""
    text = str(text).lower().translate(_PUNCT_TABLE)
    return re.sub(r"\s+", " ", text).strip()


def exact_match(prediction: str, reference: str) -> int:
    """Return 1 when normalized strings match exactly, else 0."""
    return int(normalize_answer(prediction) == normalize_answer(reference))


def compute_em(predictions: Sequence[str], references: Sequence[str]) -> dict[str, Any]:
    """Compute exact-match score over aligned prediction/reference lists."""
    _validate_parallel(predictions, references)
    scores = [exact_match(p, r) for p, r in zip(predictions, references)]
    return {"exact_match": mean(scores) if scores else 0.0, "n": len(scores)}


def _lcs_len(a: list[str], b: list[str]) -> int:
    prev = [0] * (len(b) + 1)
    for token_a in a:
        cur = [0]
        for j, token_b in enumerate(b, start=1):
            if token_a == token_b:
                cur.append(prev[j - 1] + 1)
            else:
                cur.append(max(prev[j], cur[-1]))
        prev = cur
    return prev[-1]


def rouge_l_score(prediction: str, reference: str) -> float:
    """Compute ROUGE-L F1 with normalized whitespace tokenization."""
    pred_tokens = normalize_answer(prediction).split()
    ref_tokens = normalize_answer(reference).split()
    if not pred_tokens or not ref_tokens:
        return 0.0
    lcs = _lcs_len(pred_tokens, ref_tokens)
    precision = lcs / len(pred_tokens)
    recall = lcs / len(ref_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def compute_rouge_l(predictions: Sequence[str], references: Sequence[str]) -> dict[str, Any]:
    """Compute mean ROUGE-L F1 without requiring external packages."""
    _validate_parallel(predictions, references)
    scores = [rouge_l_score(p, r) for p, r in zip(predictions, references)]
    return {"rouge_l": mean(scores) if scores else 0.0, "n": len(scores)}


def compute_sacrebleu(predictions: Sequence[str], references: Sequence[str]) -> dict[str, Any]:
    """Compute corpus sacreBLEU if sacrebleu is installed."""
    _validate_parallel(predictions, references)
    try:
        import sacrebleu
    except ImportError as exc:
        raise ImportError("Install sacrebleu to use compute_sacrebleu.") from exc

    score = sacrebleu.corpus_bleu(list(predictions), [list(references)])
    return {"sacrebleu": float(score.score), "n": len(predictions)}


def compute_bertscore(
    predictions: Sequence[str],
    references: Sequence[str],
    *,
    model_type: str = "microsoft/deberta-xlarge-mnli",
    lang: str = "en",
    device: str | None = None,
) -> dict[str, Any]:
    """Compute mean BERTScore precision/recall/F1 if bert-score is installed."""
    _validate_parallel(predictions, references)
    try:
        from bert_score import score
    except ImportError as exc:
        raise ImportError("Install bert-score to use compute_bertscore.") from exc

    p, r, f1 = score(
        list(predictions),
        list(references),
        lang=lang,
        model_type=model_type,
        device=device,
        verbose=False,
    )
    return {
        "bertscore_precision": float(p.mean().item()),
        "bertscore_recall": float(r.mean().item()),
        "bertscore_f1": float(f1.mean().item()),
        "n": len(predictions),
        "model_type": model_type,
    }


@dataclass(frozen=True)
class MetricBundle:
    """Container used by notebooks to display one row per track."""

    exact_match: float
    rouge_l: float
    n: int

    def as_dict(self) -> dict[str, Any]:
        return {"exact_match": self.exact_match, "rouge_l": self.rouge_l, "n": self.n}


def compute_core_metrics(
    predictions: Sequence[str],
    references: Sequence[str],
    *,
    try_bertscore: bool = False,
) -> dict[str, Any]:
    """Compute the dependency-free metrics used in quick local checks.

    Parameters
    ----------
    try_bertscore : bool
        When True, attempt to import ``bert_score`` and add ``bertscore_f1``
        to the returned dict.  If the package is not installed or scoring
        fails for any reason, the key is silently omitted.
    """
    em = compute_em(predictions, references)
    rouge = compute_rouge_l(predictions, references)
    result = MetricBundle(
        exact_match=em["exact_match"],
        rouge_l=rouge["rouge_l"],
        n=em["n"],
    ).as_dict()

    if try_bertscore:
        try:
            bs = compute_bertscore(predictions, references)
            result["bertscore_f1"] = bs["bertscore_f1"]
        except Exception:
            pass  # not installed or scoring failed — silently skip

    return result


def compute_operational_stats(df: Any) -> dict[str, Any]:
    """Compute operational / generation-quality statistics from a predictions CSV.

    Parameters
    ----------
    df : pandas.DataFrame
        The predictions DataFrame, typically loaded from the CSV written by
        the inference script.  Expected columns (all optional — missing
        columns are silently skipped):

        - ``truncated``       : bool — whether the generation was truncated
        - ``prediction``      : str  — the generated answer text
        - ``finish_reason``   : str  — e.g. ``"stop"``, ``"length"``
        - ``output_tokens``   : int  — number of tokens in the generation
        - ``generation_time_s``: float — wall-clock seconds for the call

    Returns
    -------
    dict with a subset of the following keys depending on which columns
    are present:

    - ``truncation_rate``        : float
    - ``empty_prediction_rate``  : float
    - ``finish_reason_dist``     : dict[str, int]
    - ``output_tokens_p50``      : float
    - ``output_tokens_p90``      : float
    - ``output_tokens_p99``      : float
    - ``generation_time_p50``    : float
    - ``generation_time_p90``    : float
    """
    stats: dict[str, Any] = {}

    if len(df) == 0:
        return stats

    # truncation_rate
    if "truncated" in df.columns:
        try:
            stats["truncation_rate"] = float(df["truncated"].astype(bool).mean())
        except Exception:
            pass

    # empty_prediction_rate
    if "prediction" in df.columns:
        try:
            empty = df["prediction"].isna() | (df["prediction"].astype(str).str.strip() == "")
            stats["empty_prediction_rate"] = float(empty.mean())
        except Exception:
            pass

    # finish_reason_dist
    if "finish_reason" in df.columns:
        try:
            stats["finish_reason_dist"] = df["finish_reason"].value_counts().to_dict()
        except Exception:
            pass

    # output_tokens percentiles
    if "output_tokens" in df.columns:
        try:
            col = df["output_tokens"].dropna()
            stats["output_tokens_p50"] = float(col.quantile(0.50))
            stats["output_tokens_p90"] = float(col.quantile(0.90))
            stats["output_tokens_p99"] = float(col.quantile(0.99))
        except Exception:
            pass

    # generation_time_s percentiles
    if "generation_time_s" in df.columns:
        try:
            col = df["generation_time_s"].dropna()
            stats["generation_time_p50"] = float(col.quantile(0.50))
            stats["generation_time_p90"] = float(col.quantile(0.90))
        except Exception:
            pass

    return stats


def _validate_parallel(predictions: Sequence[str], references: Sequence[str]) -> None:
    if len(predictions) != len(references):
        raise ValueError(
            f"predictions and references must have the same length: "
            f"{len(predictions)} != {len(references)}"
        )
