import pytest

from src.metrics import (compute_core_metrics, compute_em, compute_rouge_l,
                         normalize_answer, count_reasoning_steps,
                         compute_avg_reasoning_steps)


def test_normalize_answer_removes_case_punctuation_and_extra_space():
    assert normalize_answer("  High-Blood Pressure!  ") == "highblood pressure"


def test_compute_em_scores_normalized_matches():
    out = compute_em(["High blood pressure.", "diabetes"], ["high blood pressure", "asthma"])
    assert out == {"exact_match": 0.5, "n": 2}


def test_compute_rouge_l_scores_partial_overlap():
    out = compute_rouge_l(["ace inhibitor first line"], ["ace inhibitor or arb"])
    assert 0.0 < out["rouge_l"] < 1.0
    assert out["n"] == 1


def test_core_metrics_combines_dependency_free_metrics():
    out = compute_core_metrics(["a b c"], ["a b c"])
    assert out["exact_match"] == 1.0
    assert out["rouge_l"] == 1.0
    assert out["n"] == 1


def test_metric_length_mismatch_raises():
    with pytest.raises(ValueError, match="same length"):
        compute_em(["one"], ["one", "two"])


# ---------- count_reasoning_steps / compute_avg_reasoning_steps ----------

def test_reasoning_steps_track_b_always_zero():
    """Track B (answer-only) has no rationale block — always 0."""
    assert count_reasoning_steps("Clinical rationale:\n1. x\n\nFinal answer:\ny", "B") == 0


def test_reasoning_steps_counts_numbered_list():
    pred = ("Clinical rationale:\n1. First consideration.\n2. Second one.\n"
            "3. Third point.\n\nFinal answer:\nThe answer.")
    assert count_reasoning_steps(pred, "A") == 3


def test_reasoning_steps_counts_bulleted_list():
    pred = ("Clinical rationale:\n- point a\n- point b\n\nFinal answer:\nx")
    assert count_reasoning_steps(pred, "A") == 2


def test_reasoning_steps_falls_back_to_sentences():
    pred = ("Clinical rationale:\nThe patient has symptom X. This suggests Y. "
            "Therefore Z.\n\nFinal answer:\nDiagnosis.")
    assert count_reasoning_steps(pred, "A") == 3


def test_reasoning_steps_no_rationale_block_is_zero():
    assert count_reasoning_steps("Just a plain answer, no markers.", "A") == 0


def test_reasoning_steps_truncated_output_still_counts():
    """Length-truncated Track A output never reaches 'Final answer:'."""
    pred = "Clinical rationale:\n1. a\n2. b\n3. c"
    assert count_reasoning_steps(pred, "A") == 3


def test_reasoning_steps_handles_non_string():
    assert count_reasoning_steps(None, "A") == 0
    assert count_reasoning_steps(float("nan"), "A") == 0
    assert count_reasoning_steps(123, "A") == 0


def test_compute_avg_reasoning_steps():
    preds = [
        "Clinical rationale:\n1. a\n2. b\n\nFinal answer:\nx",       # 2 steps
        "Clinical rationale:\nOne sentence only.\n\nFinal answer:\ny",  # 1 step
    ]
    out = compute_avg_reasoning_steps(preds, "A")
    assert out["avg_reasoning_steps"] == 1.5
    assert out["n"] == 2


def test_compute_avg_reasoning_steps_track_b_is_zero():
    preds = ["any answer", "another answer"]
    out = compute_avg_reasoning_steps(preds, "B")
    assert out["avg_reasoning_steps"] == 0.0
    assert out["n"] == 2


def test_compute_avg_reasoning_steps_empty_list():
    out = compute_avg_reasoning_steps([], "A")
    assert out["avg_reasoning_steps"] == 0.0
    assert out["n"] == 0
