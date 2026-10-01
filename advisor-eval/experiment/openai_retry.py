"""Sync OpenAI chat completion with retries (429/5xx)."""

from __future__ import annotations

import time
from typing import Any, Callable


def chat_completions_with_retry(
    create_fn: Callable[..., Any],
    *,
    max_retries: int = 3,
    **kwargs: Any,
) -> Any:
    delay = 1.0
    last_exc: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            return create_fn(**kwargs)
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            msg = str(exc).lower()
            retryable = any(x in msg for x in ("429", "500", "502", "503", "504", "rate", "timeout"))
            if not retryable or attempt >= max_retries:
                raise
            time.sleep(delay)
            delay = min(delay * 2, 30.0)
    raise last_exc  # type: ignore[misc]
