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
import json
import os
from typing import Any


def _row_total_tokens(row: dict, tokenizer, cot_budget: int | None = None) -> int:
    """Approximate total tokens needed if we trained on this row.

    Counts user content + assistant content + assistant reasoning. Does
    NOT add chat-template overhead (~30 tokens for Qwen2.5 — small).

    Parameters
    ----------
    cot_budget : int | None
        When set, caps the reasoning token count at this value instead of
        counting the full reasoning length.  This prevents over-filtering
        rows for Track A short-CoT runs where reasoning is truncated at
        inference time anyway.
    """
    user = next((m for m in row["messages"] if m["role"] == "user"), {})
    asst = next((m for m in row["messages"] if m["role"] == "assistant"), {})
    n = 0
    n += len(tokenizer.encode(user.get("content") or "",
                               add_special_tokens=False))
    n += len(tokenizer.encode(asst.get("content") or "",
                               add_special_tokens=False))
    reasoning_tokens = len(tokenizer.encode(asst.get("reasoning_content") or "",
                                             add_special_tokens=False))
    if cot_budget is not None:
        reasoning_tokens = min(cot_budget, reasoning_tokens)
    n += reasoning_tokens
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
    max_rows: int | None = None,
    cot_budget: int | None = None,
    save_indices_path: str | None = None,
    load_indices_path: str | None = None,
):
    """Shuffle, optionally filter for length, then take train/val/test.

    Parameters
    ----------
    max_rows : int | None
        If set, cap the dataset to this many rows *before* shuffling and
        filtering.  Use for dry-run / smoke-test runs where you only want
        to verify the pipeline works, not train on the full dataset.
        Set to None (default) to use the full dataset.
    cot_budget : int | None
        When set, estimate Track A token length as
        ``min(cot_budget, len(encode(reasoning)))`` instead of the full
        reasoning length.  This avoids over-filtering rows for Track A
        short-CoT runs.  Ignored when ``tokenizer`` or
        ``max_total_tokens`` is None.
    save_indices_path : str | None
        If provided, save the selected sample indices as a JSON file
        ``{"train": [...], "val": [...], "test": [...]}`` after computing
        the split.  Lets callers reproduce the exact split later without
        re-running the shuffling/filtering logic.
    load_indices_path : str | None
        If provided *and* the file exists, skip shuffling/filtering and
        select rows directly from ``ds`` using the stored indices.  Lets
        NB04 load a frozen split artifact instead of recomputing.

    Returns
    -------
    (train, val, test) — three datasets.Dataset objects in that order.

    Raises
    ------
    ValueError if not enough rows remain after filtering.
    """
    # --- Fast path: load a frozen split from disk -----------------------
    if load_indices_path is not None and os.path.exists(load_indices_path):
        with open(load_indices_path, "r", encoding="utf-8") as fh:
            indices = json.load(fh)
        train = ds.select(indices["train"])
        val   = ds.select(indices["val"])
        test  = ds.select(indices["test"])
        return train, val, test

    # --- Normal path: shuffle, filter, split ----------------------------
    if max_rows is not None:
        ds = ds.select(range(min(len(ds), max_rows)))

    shuffled = ds.shuffle(seed=shuffle_seed)

    if tokenizer is not None and max_total_tokens is not None:
        shuffled = shuffled.filter(
            lambda r: _row_total_tokens(r, tokenizer, cot_budget=cot_budget)
            <= max_total_tokens
        )

    n_needed = num_train + num_val + num_test
    if len(shuffled) < n_needed:
        raise ValueError(
            f"not enough rows after filter: {len(shuffled)} < {n_needed} requested"
        )

    chunk = shuffled.select(range(n_needed))

    train = chunk.select(range(num_train))
    val   = chunk.select(range(num_train, num_train + num_val))
    # datasets.select(range(n, n)) raises IndexError when start == len(dataset),
    # even for an empty range.  Guard explicitly.
    test  = (chunk.select(range(num_train + num_val, num_train + num_val + num_test))
             if num_test > 0 else chunk.select([]))

    # --- Optionally persist indices for reproducibility -----------------
    if save_indices_path is not None:
        indices = {
            "train": train["__index_level_0__"]
                     if "__index_level_0__" in (train.column_names or [])
                     else list(range(num_train)),
            "val":   val["__index_level_0__"]
                     if "__index_level_0__" in (val.column_names or [])
                     else list(range(num_train, num_train + num_val)),
            "test":  test["__index_level_0__"]
                     if "__index_level_0__" in (test.column_names or [])
                     else list(range(num_train + num_val,
                                     num_train + num_val + num_test)),
        }
        # Derive the original dataset indices from the shuffled chunk
        # by using the shuffled dataset's internal indices if available.
        try:
            shuffled_indices = list(shuffled._indices.to_pylist()
                                    if shuffled._indices is not None
                                    else range(len(shuffled)))
            chunk_indices = shuffled_indices[:n_needed]
            indices = {
                "train": chunk_indices[:num_train],
                "val":   chunk_indices[num_train:num_train + num_val],
                "test":  chunk_indices[num_train + num_val:
                                        num_train + num_val + num_test],
            }
        except Exception:
            pass  # fall back to the positional indices already set above

        os.makedirs(os.path.dirname(os.path.abspath(save_indices_path)),
                    exist_ok=True)
        with open(save_indices_path, "w", encoding="utf-8") as fh:
            json.dump(indices, fh)

    return train, val, test
