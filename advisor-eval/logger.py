"""JSONL task logger and cost/latency tracker."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class CostTracker:
    """Converts token counts into dollar costs using per-model rates."""

    rates: dict[str, dict[str, float]]

    def compute(self, model: str, prompt_tokens: int, completion_tokens: int) -> float:
        r = self.rates.get(model, {})
        input_rate = r.get("input", 0.0)
        output_rate = r.get("output", 0.0)
        return prompt_tokens * input_rate + completion_tokens * output_rate


@dataclass
class StepLog:
    step: int
    executor_latency_s: float = 0.0
    advisor_latency_s: float = 0.0
    executor_prompt_tokens: int = 0
    executor_completion_tokens: int = 0
    advisor_prompt_tokens: int = 0
    advisor_completion_tokens: int = 0
    confidence: float | None = None
    advisor_called: bool = False


@dataclass
class TaskLog:
    task_id: str
    dataset: str
    method: str
    question: str = ""
    prediction: str | None = None
    ground_truth: str = ""
    correct: bool = False

    cost_executor: float = 0.0
    cost_advisor: float = 0.0
    cost_total: float = 0.0

    latency_total_s: float = 0.0
    latency_executor_s: float = 0.0
    latency_advisor_s: float = 0.0

    steps: int = 0
    advisor_calls: int = 0
    advisor_guidance: list[dict[str, Any]] = field(default_factory=list)
    confidence_scores: list[float] = field(default_factory=list)
    step_latencies: list[float] = field(default_factory=list)
    tool_calls: int = 0
    tool_errors: int = 0
    recovery_success: bool = False
    advisor_after_error_rate: float = 0.0
    advisor_calls_after_error: int = 0
    dead_end_count: int = 0
    tool_trace: list[dict[str, Any]] = field(default_factory=list)
    repeated_query_violations: int = 0
    blocked_host_rehits: int = 0
    advisor_followed_first_step_count: int = 0
    advisor_first_step_total: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)
    run_metadata: dict[str, Any] = field(default_factory=dict)

    cached: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "dataset": self.dataset,
            "method": self.method,
            "question": self.question,
            "metadata": self.metadata,
            "prediction": self.prediction,
            "ground_truth": self.ground_truth,
            "correct": self.correct,
            "cost_total": round(self.cost_total, 8),
            "cost_executor": round(self.cost_executor, 8),
            "cost_advisor": round(self.cost_advisor, 8),
            "latency_total_s": round(self.latency_total_s, 4),
            "latency_executor_s": round(self.latency_executor_s, 4),
            "latency_advisor_s": round(self.latency_advisor_s, 4),
            "steps": self.steps,
            "advisor_calls": self.advisor_calls,
            "advisor_guidance": self.advisor_guidance,
            "advisor_calls_after_error": self.advisor_calls_after_error,
            "confidence_scores": [round(c, 4) for c in self.confidence_scores],
            "step_latencies": [round(s, 4) for s in self.step_latencies],
            "tool_calls": self.tool_calls,
            "tool_errors": self.tool_errors,
            "recovery_success": self.recovery_success,
            "advisor_after_error_rate": round(self.advisor_after_error_rate, 4),
            "dead_end_count": self.dead_end_count,
            "tool_trace": self.tool_trace,
            "repeated_query_violations": self.repeated_query_violations,
            "blocked_host_rehits": self.blocked_host_rehits,
            "advisor_followed_first_step_count": self.advisor_followed_first_step_count,
            "advisor_first_step_total": self.advisor_first_step_total,
            "run_metadata": self.run_metadata,
            "cached": self.cached,
        }


class TaskLogger:
    """Accumulates per-task results and writes them as JSONL."""

    def __init__(
        self,
        dataset: str,
        method: str,
        policy: str,
        results_dir: str = "results",
    ):
        self.dataset = dataset
        self.method = method
        self.policy = policy
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        filename = f"{dataset}_{method}_{policy}_{ts}.jsonl"
        self.filepath = self.results_dir / filename
        self._logs: list[TaskLog] = []

    def log(self, task_log: TaskLog) -> None:
        self._logs.append(task_log)
        with open(self.filepath, "a", encoding="utf-8") as f:
            f.write(json.dumps(task_log.to_dict()) + "\n")

    @property
    def logs(self) -> list[TaskLog]:
        return list(self._logs)

    @staticmethod
    def read_jsonl(filepath: str | Path) -> list[dict]:
        results = []
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    results.append(json.loads(line))
        return results

    @staticmethod
    def read_results_dir(results_dir: str | Path) -> list[dict]:
        """Read all JSONL files in a directory into a flat list."""
        results_dir = Path(results_dir)
        all_results = []
        for fp in sorted(results_dir.glob("*.jsonl")):
            all_results.extend(TaskLogger.read_jsonl(fp))
        return all_results
