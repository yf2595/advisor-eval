"""Structured run output under runs/<run_name>/."""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from logger import TaskLog


class StructuredRunLogger:
    def __init__(self, run_dir: Path, run_name: str):
        self.run_dir = run_dir
        self.run_name = run_name
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.tasks_path = self.run_dir / "tasks.jsonl"
        self.messages_path = self.run_dir / "messages.jsonl"
        self.meta_path = self.run_dir / "meta.json"
        self._completed_ids: set[str] = set()
        if self.tasks_path.exists():
            for line in self.tasks_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    row = json.loads(line)
                    tid = row.get("task_id")
                    if tid and row.get("status") not in ("error",):
                        self._completed_ids.add(tid)

    def is_completed(self, task_id: str) -> bool:
        return task_id in self._completed_ids

    def write_meta(self, meta: dict[str, Any]) -> None:
        self.meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    def log_task(self, row: dict[str, Any]) -> None:
        with self.tasks_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        tid = row.get("task_id")
        if tid and row.get("status") != "error":
            self._completed_ids.add(tid)

    def log_message(self, row: dict[str, Any]) -> None:
        with self.messages_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")

    @staticmethod
    def task_log_to_structured(
        log: TaskLog,
        *,
        benchmark: str,
        level_or_type: str | None,
        status: str = "ok",
        terminated_by: str = "answer",
        executor_in: int = 0,
        executor_out: int = 0,
        advisor_in: int = 0,
        advisor_out: int = 0,
    ) -> dict[str, Any]:
        actions = []
        for entry in log.tool_trace or []:
            actions.append({
                "step": entry.get("step"),
                "type": "tool" if entry.get("tool") not in ("executor_api", "advisor_api", "json_parse") else "system",
                "tool": entry.get("tool"),
                "args_normalized": str(entry.get("input", ""))[:500],
                "is_error": not entry.get("success", True),
            })
        return {
            "task_id": log.task_id,
            "benchmark": benchmark,
            "level_or_type": level_or_type,
            "status": status,
            "correct": log.correct if status == "ok" else None,
            "pred": log.prediction,
            "gold": log.ground_truth,
            "n_steps": log.steps,
            "n_tool_calls": log.tool_calls,
            "n_tool_errors": log.tool_errors,
            "n_messages": log.advisor_calls,
            "advisor_in_tokens": advisor_in,
            "advisor_out_tokens": advisor_out,
            "executor_in_tokens": executor_in,
            "executor_out_tokens": executor_out,
            "cost_usd": log.cost_total,
            "latency_s": log.latency_total_s,
            "terminated_by": terminated_by,
            "actions": actions,
            "legacy": log.to_dict(),
        }
