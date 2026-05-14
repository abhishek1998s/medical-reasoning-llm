import pytest

from src.metrics import compute_core_metrics, compute_em, compute_rouge_l, normalize_answer


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
