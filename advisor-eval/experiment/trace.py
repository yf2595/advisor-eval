"""Per-task call tracing: LLM calls, HTTP requests, tool calls.

Events go to ``runs/<run_name>/trace.jsonl`` (one JSON object per line) and,
unless ``TRACE_CONSOLE=0``, a one-line summary is printed to the console.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from contextvars import ContextVar
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_trace_path: ContextVar[Path | None] = ContextVar("trace_path", default=None)
_task_id: ContextVar[str | None] = ContextVar("trace_task_id", default=None)
_task_t0: ContextVar[float | None] = ContextVar("trace_task_t0", default=None)
_lock = threading.Lock()

PREVIEW_CHARS = int(os.environ.get("TRACE_PREVIEW_CHARS", "300"))


def preview(text: Any, limit: int | None = None) -> str:
    s = "" if text is None else str(text)
    s = s.replace("\r", " ").replace("\n", " | ")
    lim = PREVIEW_CHARS if limit is None else limit
    return s if len(s) <= lim else s[:lim] + f"... (+{len(s) - lim} chars)"


def start_task(run_dir: Path, task_id: str) -> None:
    _trace_path.set(Path(run_dir) / "trace.jsonl")
    _task_id.set(task_id)
    _task_t0.set(time.perf_counter())


def end_task() -> None:
    _task_id.set(None)
    _task_t0.set(None)


def _console(line: str) -> None:
    if os.environ.get("TRACE_CONSOLE", "1") == "0":
        return
    try:
        from tqdm import tqdm

        tqdm.write(line)
    except UnicodeEncodeError:
        enc = getattr(sys.stdout, "encoding", None) or "ascii"
        print(line.encode(enc, errors="replace").decode(enc, errors="replace"), flush=True)


def event(kind: str, summary: str = "", **fields: Any) -> None:
    """Record a trace event. Never raises: tracing must not break a task."""
    try:
        _event(kind, summary, **fields)
    except Exception:  # noqa: BLE001
        pass


def _event(kind: str, summary: str, **fields: Any) -> None:
    t0 = _task_t0.get()
    elapsed = round(time.perf_counter() - t0, 2) if t0 is not None else None
    tid = _task_id.get()
    row = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "task_id": tid,
        "t_task_s": elapsed,
        "kind": kind,
        **fields,
    }
    path = _trace_path.get()
    if path is not None:
        with _lock:
            with path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, default=str) + "\n")
    stamp = f"{elapsed:7.1f}s" if elapsed is not None else "       "
    _console(f"    {stamp} {kind:<14} {summary}")
