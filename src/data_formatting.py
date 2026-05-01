"""Track A / Track B formatters and short-CoT truncation.

Track A = primary system: short clinical reasoning + final answer.
Track B = baseline ablation: final answer only.

Both formatters pass user messages through unchanged. Only the assistant
content differs.

`extract_answer_for_scoring` is used by the evaluation pipeline to
compare only the final answer against the gold reference, never the
rationale.
"""

from __future__ import annotations
import re
from typing import Any

# Sentence-end characters used by "truncate at sentence boundary".
_SENTENCE_END = (".", "?", "!")


def truncate_to_n_tokens(
    text: str,
    tokenizer: Any,
    max_tokens: int,
    truncate_at_sentence: bool = True,
) -> str:
    """Truncate `text` to at most `max_tokens` tokens (per `tokenizer`).

    If `truncate_at_sentence` is True, walk back to the last sentence-end
    within the budget. If no sentence-end exists in the budget, fall back
    to hard truncation.
    """
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) <= max_tokens:
        return text

    truncated = tokenizer.decode(ids[:max_tokens], skip_special_tokens=True)

    if not truncate_at_sentence:
        return truncated

    last_boundary = max(truncated.rfind(c) for c in _SENTENCE_END)
    if last_boundary == -1:
        return truncated  # no sentence boundary in budget — keep hard cut
    return truncated[: last_boundary + 1]


def format_for_track_a(
    row: dict,
    tokenizer: Any,
    short_cot_max_tokens: int = 150,
) -> dict:
    """Reshape an OpenMed row to Track-A output format.

    Output assistant content:
        Clinical rationale:
        <truncated reasoning_content>

        Final answer:
        <content>
    """
    out_messages: list[dict] = []
    for msg in row["messages"]:
        if msg["role"] == "user":
            out_messages.append({"role": "user", "content": msg["content"]})
            continue
        if msg["role"] == "assistant":
            content = (msg.get("content") or "").strip()
            reasoning = (msg.get("reasoning_content") or "").strip()
            short = truncate_to_n_tokens(
                reasoning, tokenizer, short_cot_max_tokens,
                truncate_at_sentence=True,
            )
            assistant = (
                f"Clinical rationale:\n{short}\n\n"
                f"Final answer:\n{content}"
            )
            out_messages.append({"role": "assistant", "content": assistant})
            continue
        # System / tool messages pass through unchanged.
        out_messages.append(msg)
    return {"messages": out_messages}


def format_for_track_b(row: dict) -> dict:
    """Reshape an OpenMed row to Track-B output format.

    Output assistant content = answer only (`content` field, stripped).
    `reasoning_content` is dropped entirely.
    """
    out_messages: list[dict] = []
    for msg in row["messages"]:
        if msg["role"] == "user":
            out_messages.append({"role": "user", "content": msg["content"]})
            continue
        if msg["role"] == "assistant":
            content = (msg.get("content") or "").strip()
            out_messages.append({"role": "assistant", "content": content})
            continue
        out_messages.append(msg)
    return {"messages": out_messages}


# Pre-compiled regexes for answer extraction.
_TRACK_A_FINAL = re.compile(
    r"Final answer:\s*\n?(?P<answer>.*?)(?:\n\nSafety note:.*)?$",
    re.DOTALL,
)
_TRACK_B_SAFETY = re.compile(r"\n\nSafety note:.*$", re.DOTALL)


def extract_answer_for_scoring(prediction: str, track: str) -> str:
    """Return only the final answer text — used for EM/ROUGE/BERTScore.

    For Track A, return text after `Final answer:` (and before any
    `Safety note:` block).
    For Track B, strip any trailing `Safety note:` block.
    Defensive: if no marker is found, return the prediction unchanged so
    metrics can still compute (will usually score 0, surfacing the format
    failure in the report).
    """
    if track.upper().startswith("A"):
        m = _TRACK_A_FINAL.search(prediction)
        if m is None:
            return prediction.strip()
        return m.group("answer").strip()

    # Track B
    return _TRACK_B_SAFETY.sub("", prediction).strip()
