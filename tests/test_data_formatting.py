"""Unit tests for src.data_formatting.

We use the actual Qwen2.5-1.5B-Instruct tokenizer so token-boundary
truncation is realistic. The tokenizer is module-scoped (downloaded
once, ~2 MB).
"""
import pytest
from transformers import AutoTokenizer

from src.data_formatting import (
    truncate_to_n_tokens,
    format_for_track_a,
    format_for_track_b,
    extract_answer_for_scoring,
)


@pytest.fixture(scope="module")
def tok():
    # Use the bare instruct model — its tokenizer is identical to the
    # bnb-4bit wrapper but downloads faster.
    return AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")


# ---------- truncate_to_n_tokens ----------

def test_truncate_returns_input_when_under_limit(tok):
    text = "Short clinical note."
    out = truncate_to_n_tokens(text, tok, max_tokens=100,
                               truncate_at_sentence=True)
    assert out == text


def test_truncate_respects_token_budget(tok):
    text = " ".join(["word"] * 500)
    out = truncate_to_n_tokens(text, tok, max_tokens=50,
                               truncate_at_sentence=False)
    assert len(tok.encode(out, add_special_tokens=False)) <= 50


def test_truncate_at_sentence_ends_on_period(tok):
    text = ("Patient has fever. Likely viral infection. "
            "Treat with rest and fluids. " * 10)
    out = truncate_to_n_tokens(text, tok, max_tokens=20,
                               truncate_at_sentence=True)
    assert out.endswith(".") or out.endswith("?") or out.endswith("!")
    assert len(tok.encode(out, add_special_tokens=False)) <= 20


def test_truncate_at_sentence_falls_back_when_no_boundary(tok):
    # No sentence boundary in budget -> fall back to hard truncation.
    text = "wordwordword " * 200
    out = truncate_to_n_tokens(text, tok, max_tokens=10,
                               truncate_at_sentence=True)
    # Must not crash and must respect token cap.
    assert len(tok.encode(out, add_special_tokens=False)) <= 10
    assert len(out) > 0


# ---------- format_for_track_a ----------

def test_track_a_includes_clinical_rationale_header(tok):
    row = {
        "messages": [
            {"role": "user", "content": "What is hypertension?"},
            {"role": "assistant",
             "content": "It is high blood pressure.",
             "reasoning_content":
                 "Hypertension is defined as systolic >= 130 mmHg."},
        ]
    }
    out = format_for_track_a(row, tok, short_cot_max_tokens=150)
    assistant = out["messages"][-1]["content"]
    assert "Clinical rationale:" in assistant
    assert "Final answer:" in assistant
    assert "It is high blood pressure." in assistant


def test_track_a_truncates_long_reasoning(tok):
    row = {
        "messages": [
            {"role": "user", "content": "Q?"},
            {"role": "assistant",
             "content": "A.",
             "reasoning_content": "Long reasoning. " * 200},
        ]
    }
    out = format_for_track_a(row, tok, short_cot_max_tokens=50)
    assistant = out["messages"][-1]["content"]
    rationale = (assistant.split("Clinical rationale:")[1]
                          .split("Final answer:")[0])
    # +slack for whitespace tokens around the rationale.
    assert len(tok.encode(rationale, add_special_tokens=False)) <= 60


def test_track_a_passes_through_user_message(tok):
    row = {"messages": [
        {"role": "user", "content": "Original question"},
        {"role": "assistant", "content": "A", "reasoning_content": "R"},
    ]}
    out = format_for_track_a(row, tok, short_cot_max_tokens=150)
    assert out["messages"][0] == {"role": "user",
                                  "content": "Original question"}


# ---------- format_for_track_b ----------

def test_track_b_drops_reasoning():
    row = {"messages": [
        {"role": "user", "content": "Q?"},
        {"role": "assistant",
         "content": "It is high blood pressure.",
         "reasoning_content": "This should NOT appear."},
    ]}
    out = format_for_track_b(row)
    assistant = out["messages"][-1]["content"]
    assert assistant == "It is high blood pressure."
    assert "should NOT appear" not in assistant
    assert "Clinical rationale" not in assistant


def test_track_b_passes_through_user_message():
    row = {"messages": [
        {"role": "user", "content": "Original question"},
        {"role": "assistant", "content": "A", "reasoning_content": "R"},
    ]}
    out = format_for_track_b(row)
    assert out["messages"][0] == {"role": "user",
                                  "content": "Original question"}


# ---------- extract_answer_for_scoring ----------

def test_extract_track_a_pulls_text_after_final_answer():
    pred = ("Clinical rationale:\n1. fever\n2. viral\n\n"
            "Final answer:\nRest and fluids.")
    assert extract_answer_for_scoring(pred, "A") == "Rest and fluids."


def test_extract_track_a_strips_safety_note():
    pred = ("Clinical rationale:\n...\n\n"
            "Final answer:\nTake paracetamol 500mg.\n\n"
            "Safety note: Consult a physician.")
    assert (extract_answer_for_scoring(pred, "A")
            == "Take paracetamol 500mg.")


def test_extract_track_a_returns_full_text_if_no_final_answer_marker():
    # Defensive: degenerate output -> return prediction unchanged so EM
    # can still compute (will likely score 0, surfacing the format failure).
    pred = "Some malformed output."
    assert extract_answer_for_scoring(pred, "A") == "Some malformed output."


def test_extract_track_b_strips_trailing_safety_note():
    pred = "Take paracetamol 500mg.\n\nSafety note: Consult a physician."
    assert (extract_answer_for_scoring(pred, "B")
            == "Take paracetamol 500mg.")


def test_extract_track_b_returns_unchanged_when_no_safety_note():
    pred = "Take paracetamol 500mg."
    assert extract_answer_for_scoring(pred, "B") == "Take paracetamol 500mg."
