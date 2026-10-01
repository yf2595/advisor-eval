"""Chat Completions helpers (reasoning effort for GPT-5 / o-series, vLLM routing)."""

from __future__ import annotations

import os
import time
from typing import Any

from experiment import models


def uses_reasoning_effort(model: str) -> bool:
    """True if the model accepts ``reasoning_effort`` on Chat Completions."""
    spec = models.local_spec(model)
    if spec is not None:
        return bool(spec.get("reasoning_effort"))
    m = model.lower().strip()
    if m.startswith("gpt-4"):
        return False
    if m.startswith("gpt-5"):
        return True
    if m.startswith(("o1", "o3", "o4")):
        return True
    return False


def default_reasoning_effort() -> str:
    return os.environ.get("REASONING_EFFORT", "low").strip() or "low"


def chat_completions_create(client: Any, *, model: str, **kwargs: Any) -> Any:
    """``client.chat.completions.create`` with decoding params for reasoning models."""
    from experiment import trace

    spec = models.local_spec(model)
    if spec is not None:
        client = models.client_for(model, client)
        if spec.get("reasoning_effort"):
            kwargs.setdefault("reasoning_effort", spec["reasoning_effort"])
        if spec.get("extra_body"):
            kwargs.setdefault("extra_body", spec["extra_body"])
    elif uses_reasoning_effort(model):
        kwargs.setdefault("reasoning_effort", default_reasoning_effort())
        # GPT-5 + reasoning_effort rejects temperature=0.0; omit so the API default applies.
        if kwargs.get("temperature") == 0.0:
            kwargs.pop("temperature", None)

    messages = kwargs.get("messages") or []
    last = messages[-1].get("content", "") if messages else ""
    params = {k: v for k, v in kwargs.items() if k != "messages"}
    trace.event(
        "llm_request",
        f"{model} msgs={len(messages)} {params}",
        model=model,
        params=params,
        n_messages=len(messages),
        last_message=trace.preview(last, 2000),
    )
    t0 = time.perf_counter()
    try:
        resp = client.chat.completions.create(model=models.served_name(model), **kwargs)
    except Exception as exc:  # noqa: BLE001
        trace.event(
            "llm_error",
            f"{model} after {time.perf_counter() - t0:.1f}s: {type(exc).__name__}: {trace.preview(exc, 200)}",
            model=model,
            latency_s=round(time.perf_counter() - t0, 3),
            error=f"{type(exc).__name__}: {exc}",
        )
        raise
    latency = time.perf_counter() - t0
    usage = getattr(resp, "usage", None)
    details = getattr(usage, "completion_tokens_details", None) if usage else None
    content = resp.choices[0].message.content if resp.choices else ""
    trace.event(
        "llm_response",
        f"{model} {latency:.1f}s in={getattr(usage, 'prompt_tokens', None)} "
        f"out={getattr(usage, 'completion_tokens', None)} -> {trace.preview(content, 160)}",
        model=model,
        latency_s=round(latency, 3),
        prompt_tokens=getattr(usage, "prompt_tokens", None),
        completion_tokens=getattr(usage, "completion_tokens", None),
        reasoning_tokens=getattr(details, "reasoning_tokens", None),
        finish_reason=resp.choices[0].finish_reason if resp.choices else None,
        content=trace.preview(content, 4000),
    )
    return resp
