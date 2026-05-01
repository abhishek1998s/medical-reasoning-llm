"""Unit tests for src.splits."""
import pytest
from datasets import Dataset

from src.splits import shuffle_filter_split


@pytest.fixture
def fake_ds():
    """100-row fake dataset with `messages` field of varied length."""
    rows = [
        {
            "messages": [
                {"role": "user", "content": f"Q{i}"},
                {"role": "assistant",
                 "content": f"A{i}" + " word" * (i % 50),
                 "reasoning_content": "thinking " * (i % 30)},
            ]
        }
        for i in range(100)
    ]
    return Dataset.from_list(rows)


# ---------- shuffle_filter_split — basic split ----------

def test_split_sizes_match_request(fake_ds):
    train, val, test = shuffle_filter_split(
        fake_ds, shuffle_seed=42,
        num_train=60, num_val=10, num_test=20,
    )
    assert len(train) == 60
    assert len(val) == 10
    assert len(test) == 20


def test_split_subsets_are_disjoint(fake_ds):
    """Train, val, test indices should never overlap."""
    train, val, test = shuffle_filter_split(
        fake_ds, shuffle_seed=42,
        num_train=30, num_val=10, num_test=20,
    )
    train_qs = {r["messages"][0]["content"] for r in train}
    val_qs   = {r["messages"][0]["content"] for r in val}
    test_qs  = {r["messages"][0]["content"] for r in test}
    assert train_qs & val_qs == set()
    assert train_qs & test_qs == set()
    assert val_qs   & test_qs == set()


def test_split_is_deterministic_across_calls(fake_ds):
    """Same seed -> same indices in train/val/test."""
    a = shuffle_filter_split(fake_ds, shuffle_seed=42,
                             num_train=20, num_val=5, num_test=10)
    b = shuffle_filter_split(fake_ds, shuffle_seed=42,
                             num_train=20, num_val=5, num_test=10)
    for split_a, split_b in zip(a, b):
        a_qs = [r["messages"][0]["content"] for r in split_a]
        b_qs = [r["messages"][0]["content"] for r in split_b]
        assert a_qs == b_qs


def test_different_seeds_give_different_splits(fake_ds):
    a = shuffle_filter_split(fake_ds, shuffle_seed=42,
                             num_train=20, num_val=5, num_test=10)
    b = shuffle_filter_split(fake_ds, shuffle_seed=123,
                             num_train=20, num_val=5, num_test=10)
    a_train_qs = [r["messages"][0]["content"] for r in a[0]]
    b_train_qs = [r["messages"][0]["content"] for r in b[0]]
    assert a_train_qs != b_train_qs


# ---------- shuffle_filter_split — length filter ----------

def test_filter_drops_long_rows(fake_ds):
    """If max_total_tokens is small, only short rows survive."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
    # Request small splits — the fake_ds has only ~10 rows under 30 tokens.
    train, val, test = shuffle_filter_split(
        fake_ds, shuffle_seed=42,
        num_train=3, num_val=1, num_test=2,
        tokenizer=tok, max_total_tokens=30,
    )
    # Every surviving row must fit the budget.
    for split in (train, val, test):
        for row in split:
            asst = next(m for m in row["messages"] if m["role"] == "assistant")
            user = next(m for m in row["messages"] if m["role"] == "user")
            total = (
                len(tok.encode(user["content"], add_special_tokens=False))
                + len(tok.encode(asst.get("content") or "",
                                  add_special_tokens=False))
                + len(tok.encode(asst.get("reasoning_content") or "",
                                  add_special_tokens=False))
            )
            assert total <= 30


def test_no_filter_when_no_tokenizer(fake_ds):
    """Calling without tokenizer skips length filtering."""
    train, val, test = shuffle_filter_split(
        fake_ds, shuffle_seed=42,
        num_train=60, num_val=10, num_test=20,
    )
    # All 90 rows should be present (no filtering).
    assert len(train) + len(val) + len(test) == 90


def test_raises_when_not_enough_rows_after_filter(fake_ds):
    """If the filter is so strict no rows pass, we should raise."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
    with pytest.raises(ValueError, match="not enough rows"):
        shuffle_filter_split(
            fake_ds, shuffle_seed=42,
            num_train=50, num_val=10, num_test=20,
            tokenizer=tok, max_total_tokens=1,  # impossibly tight
        )
