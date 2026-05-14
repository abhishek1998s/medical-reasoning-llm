import csv

import pytest

from src.safety_rubric import AuditRow, AUDIT_COLUMNS, build_blank_audit_rows, make_audit_csv


def test_audit_row_validates_allowed_values():
    row = AuditRow(
        sample_id=1,
        track_name="A",
        risk_bucket="high",
        question="Q",
        reference="R",
        prediction="P",
        clinical_correctness="partially_correct",
        risk_severity="high",
        hallucination_type="wrong_reasoning",
        reasoning_clarity="misleading",
        safe_behavior="missing_disclaimer",
    )
    assert row.as_dict()["risk_bucket"] == "high"


def test_audit_row_rejects_bad_value():
    row = AuditRow(
        sample_id=1,
        track_name="A",
        risk_bucket="urgent",
        question="Q",
        reference="R",
        prediction="P",
    )
    with pytest.raises(ValueError, match="risk_bucket"):
        row.as_dict()


def test_make_audit_csv_writes_expected_header(tmp_path):
    out = make_audit_csv(
        [
            AuditRow(
                sample_id=2,
                track_name="B",
                risk_bucket="low",
                question="Q",
                reference="R",
                prediction="P",
                reasoning_clarity="not_applicable",
            )
        ],
        tmp_path / "audit.csv",
    )
    with out.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    assert reader.fieldnames == AUDIT_COLUMNS
    assert rows[0]["sample_id"] == "2"
    assert rows[0]["reasoning_clarity"] == "not_applicable"


def test_build_blank_audit_rows_marks_track_b_reasoning_not_applicable():
    rows = build_blank_audit_rows(
        [{"sample_id": 7, "question": "Q", "reference": "R", "prediction": "P"}],
        track_name="B",
        risk_bucket="medium",
    )
    assert rows[0].reasoning_clarity == "not_applicable"
