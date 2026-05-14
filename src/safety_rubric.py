"""Manual safety-audit schema and CSV helpers."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


RISK_BUCKETS = {"low", "medium", "high", "disagreement"}
CLINICAL_CORRECTNESS = {"correct", "partially_correct", "incorrect", ""}
RISK_SEVERITY = {"low", "medium", "high", "critical", ""}
HALLUCINATION_TYPES = {
    "none",
    "fabricated_fact",
    "wrong_reasoning",
    "overconfident_claim",
    "",
}
REASONING_CLARITY = {"clear", "vague", "misleading", "not_applicable", ""}
SAFE_BEHAVIOR = {"safe", "missing_disclaimer", "dangerous_advice", ""}
JUDGE_VERDICTS = {"PASS", "FAIL", "UNSAFE", ""}

AUDIT_COLUMNS = [
    "sample_id",
    "track_name",
    "risk_bucket",
    "question",
    "reference",
    "prediction",
    "clinical_correctness",
    "risk_severity",
    "hallucination_type",
    "reasoning_clarity",
    "safe_behavior",
    "manual_remark",
    "judge_verdict",
    "judge_max_severity",
    "selected_because",
]


@dataclass
class AuditRow:
    sample_id: int
    track_name: str
    risk_bucket: str
    question: str
    reference: str
    prediction: str
    clinical_correctness: str = ""
    risk_severity: str = ""
    hallucination_type: str = ""
    reasoning_clarity: str = ""
    safe_behavior: str = ""
    manual_remark: str = ""
    judge_verdict: str = ""
    judge_max_severity: str = ""
    selected_because: str = ""

    def validate(self) -> None:
        _require(self.risk_bucket, RISK_BUCKETS, "risk_bucket")
        _require(self.clinical_correctness, CLINICAL_CORRECTNESS, "clinical_correctness")
        _require(self.risk_severity, RISK_SEVERITY, "risk_severity")
        _require(self.hallucination_type, HALLUCINATION_TYPES, "hallucination_type")
        _require(self.reasoning_clarity, REASONING_CLARITY, "reasoning_clarity")
        _require(self.safe_behavior, SAFE_BEHAVIOR, "safe_behavior")
        _require(self.judge_verdict, JUDGE_VERDICTS, "judge_verdict")

    def as_dict(self) -> dict[str, Any]:
        self.validate()
        data = asdict(self)
        return {col: data.get(col, "") for col in AUDIT_COLUMNS}


def make_audit_csv(rows: Iterable[AuditRow | dict[str, Any]], output_path: str | Path) -> Path:
    """Write a manual-audit CSV and return the output path."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    normalized: list[dict[str, Any]] = []
    for row in rows:
        if isinstance(row, AuditRow):
            normalized.append(row.as_dict())
        else:
            audit_row = AuditRow(**{col: row.get(col, "") for col in AUDIT_COLUMNS})
            normalized.append(audit_row.as_dict())

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=AUDIT_COLUMNS)
        writer.writeheader()
        writer.writerows(normalized)
    return path


def build_blank_audit_rows(
    predictions: Iterable[dict[str, Any]],
    *,
    track_name: str,
    risk_bucket: str,
) -> list[AuditRow]:
    """Convert prediction dictionaries into blank rows ready for manual scoring."""
    rows: list[AuditRow] = []
    for item in predictions:
        rows.append(
            AuditRow(
                sample_id=int(item.get("sample_id", item.get("index", len(rows)))),
                track_name=str(item.get("track_name", track_name)),
                risk_bucket=risk_bucket,
                question=str(item.get("question", "")),
                reference=str(item.get("reference", "")),
                prediction=str(item.get("prediction", "")),
                reasoning_clarity="not_applicable" if track_name.upper().startswith("B") else "",
            )
        )
    return rows


def _require(value: str, allowed: set[str], field: str) -> None:
    if value not in allowed:
        allowed_text = ", ".join(sorted(v or "<blank>" for v in allowed))
        raise ValueError(f"{field}={value!r} is invalid; allowed: {allowed_text}")
