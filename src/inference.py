"""Inference helpers with token, latency, and truncation logging."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class GenerationRecord:
    prediction: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    generation_time_s: float
    tokens_per_sec: float
    finish_reason: str
    truncated: bool

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def render_user_prompt(tokenizer: Any, question: str) -> str:
    """Render a single-turn user prompt with the tokenizer's chat template."""
    messages = [{"role": "user", "content": question}]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def generate_with_logging(
    model: Any,
    tokenizer: Any,
    question: str,
    *,
    max_new_tokens: int = 400,
    temperature: float = 0.0,
    do_sample: bool = False,
    repetition_penalty: float = 1.0,
    device: str | None = None,
) -> dict[str, Any]:
    """Generate one answer and return prediction plus evaluation metadata.

    The helper intentionally handles one example at a time so latency metrics
    match the assignment's per-question comparison.
    """
    prompt_text = render_user_prompt(tokenizer, question)
    inputs = tokenizer(prompt_text, return_tensors="pt")
    if device is not None:
        inputs = inputs.to(device)
    elif hasattr(model, "device"):
        inputs = inputs.to(model.device)

    input_len = int(inputs["input_ids"].shape[1])
    start = time.perf_counter()
    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature,
        repetition_penalty=repetition_penalty,
    )
    elapsed = time.perf_counter() - start

    new_ids = output_ids[0, input_len:]
    prediction = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
    output_tokens = int(new_ids.shape[0])
    eos_id = getattr(tokenizer, "eos_token_id", None)
    ended_on_eos = bool(eos_id is not None and output_tokens > 0 and int(new_ids[-1]) == eos_id)
    truncated = output_tokens >= max_new_tokens and not ended_on_eos
    finish_reason = "stop" if ended_on_eos else ("length" if truncated else "unknown")

    return GenerationRecord(
        prediction=prediction,
        input_tokens=input_len,
        output_tokens=output_tokens,
        total_tokens=input_len + output_tokens,
        generation_time_s=elapsed,
        tokens_per_sec=(output_tokens / elapsed) if elapsed > 0 else 0.0,
        finish_reason=finish_reason,
        truncated=truncated,
    ).as_dict()


def build_prediction_row(
    *,
    sample_id: int,
    question: str,
    reference: str,
    track_name: str,
    model_id: str,
    adapter_id: str,
    generation: dict[str, Any],
) -> dict[str, Any]:
    """Merge one generation record with stable CSV columns."""
    return {
        "sample_id": sample_id,
        "question": question,
        "reference": reference,
        "track_name": track_name,
        "model_id": model_id,
        "adapter_id": adapter_id,
        **generation,
    }
