"""Train/val/test slicing for the OpenMed dataset.

Both Track A and Track B notebooks must use *identical* indices in
train/val/test so the comparison is fair. This module enforces that:
same shuffle_seed -> same indices, regardless of which formatter is
applied later.

Optional length filter: drops rows whose user_msg + content +
reasoning_content tokenized total exceeds `max_total_tokens`. Useful
for keeping training examples within `max_seq_length`.
"""

from __future__ import annotations
from typing import Any


def _row_total_tokens(row: dict, tokenizer) -> int:
    """Approximate total tokens needed if we trained on this row.

    Counts user content + assistant content + assistant reasoning. Does
    NOT add chat-template overhead (~30 tokens for Qwen2.5 — small).
    """
    user = next((m for m in row["messages"] if m["role"] == "user"), {})
    asst = next((m for m in row["messages"] if m["role"] == "assistant"), {})
    n = 0
    n += len(tokenizer.encode(user.get("content") or "",
                               add_special_tokens=False))
    n += len(tokenizer.encode(asst.get("content") or "",
                               add_special_tokens=False))
    n += len(tokenizer.encode(asst.get("reasoning_content") or "",
                               add_special_tokens=False))
    return n


def shuffle_filter_split(
    ds: Any,
    shuffle_seed: int,
    num_train: int,
    num_val: int,
    num_test: int,
    *,
    tokenizer: Any = None,
    max_total_tokens: int | None = None,
):
    """Shuffle, optionally filter for length, then take train/val/test.

    Returns
    -------
    (train, val, test) — three datasets.Dataset objects in that order.

    Raises
    ------
    ValueError if not enough rows remain after filtering.
    """
    shuffled = ds.shuffle(seed=shuffle_seed)

    if tokenizer is not None and max_total_tokens is not None:
        shuffled = shuffled.filter(
            lambda r: _row_total_tokens(r, tokenizer) <= max_total_tokens
        )

    n_needed = num_train + num_val + num_test
    if len(shuffled) < n_needed:
        raise ValueError(
            f"not enough rows after filter: {len(shuffled)} < {n_needed} requested"
        )

    chunk = shuffled.select(range(n_needed))

    train = chunk.select(range(num_train))
    val   = chunk.select(range(num_train, num_train + num_val))
    test  = chunk.select(range(num_train + num_val,
                                num_train + num_val + num_test))
    return train, val, test
