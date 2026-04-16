"""Core evaluation loop: cheap-only, strong-only, and advisor-augmented runs."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from executor import ExecutorAgent, CallStats
from advisor import AdvisorAgent, AdvisorCallStats
from policies import EscalationPolicy
from logger import CostTracker, TaskLog


def _normalise_number(text: str) -> str:
    """Strip formatting from a numeric string for exact-match comparison."""
    text = text.strip().rstrip(".")
    text = text.replace(",", "").replace("$", "").replace("%", "")
    # Remove trailing zeroes after decimal
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text


def _extract_answer_from_text(text: str) -> str | None:
    match = re.search(r"FINAL ANSWER:\s*(.+)", text, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return None


def check_correct(prediction: str | None, ground_truth: str, dataset: str) -> bool:
    if prediction is None:
        return False

    if dataset == "gsm8k":
        return _normalise_number(prediction) == _normalise_number(ground_truth)

    # HotpotQA: case-insensitive exact match
    return prediction.strip().lower() == ground_truth.strip().lower()


class DiskCache:
    """Simple hash-based disk cache for API responses."""

    def __init__(self, cache_dir: str = ".cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _key(model: str, messages: list[dict]) -> str:
        blob = json.dumps({"model": model, "messages": messages}, sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()

    def get(self, model: str, messages: list[dict]) -> dict | None:
        key = self._key(model, messages)
        path = self.cache_dir / f"{key}.json"
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
        return None

    def put(self, model: str, messages: list[dict], data: dict) -> None:
        key = self._key(model, messages)
        path = self.cache_dir / f"{key}.json"
        path.write_text(json.dumps(data), encoding="utf-8")


class Evaluator:
    def __init__(self, config: dict[str, Any], cache: DiskCache | None = None):
        models = config["models"]
        run_cfg = config["run"]

        self.executor = ExecutorAgent(
            model=models["executor"],
            temperature=run_cfg["temperature"],
            seed=run_cfg["seed"],
        )
        self.advisor = AdvisorAgent(
            model=models["advisor"],
            temperature=run_cfg["temperature"],
            seed=run_cfg["seed"],
        )
        self.max_steps = run_cfg["max_steps"]
        self.cost_tracker = CostTracker(rates=config["costs"])
        self.executor_model = models["executor"]
        self.advisor_model = models["advisor"]
        self.cache = cache

    # ------------------------------------------------------------------
    # Cheap-only baseline
    # ------------------------------------------------------------------

    def run_cheap_only(self, task: dict) -> TaskLog:
        """One-shot solve with the cheap executor model."""
        messages = [
            {"role": "system", "content": self.executor.system_prompt()},
            {"role": "user", "content": task["question"]},
        ]

        result = self.executor.step(messages)
        answer = result.answer

        # If no FINAL ANSWER marker, take last number for gsm8k
        if answer is None and task.get("dataset", "gsm8k") == "gsm8k":
            nums = re.findall(r"-?\d[\d,]*\.?\d*", result.text)
            if nums:
                answer = nums[-1].replace(",", "")

        cost_exec = self.cost_tracker.compute(
            self.executor_model,
            result.stats.prompt_tokens,
            result.stats.completion_tokens,
        )

        dataset = task.get("dataset", "gsm8k")
        return TaskLog(
            task_id=task["id"],
            dataset=dataset,
            method="cheap_only",
            question=task["question"],
            prediction=answer,
            ground_truth=task["answer"],
            correct=check_correct(answer, task["answer"], dataset),
            cost_executor=cost_exec,
            cost_total=cost_exec,
            latency_total_s=result.stats.latency_s,
            latency_executor_s=result.stats.latency_s,
            steps=1,
            step_latencies=[result.stats.latency_s],
        )

    # ------------------------------------------------------------------
    # Strong-only baseline
    # ------------------------------------------------------------------

    def run_strong_only(self, task: dict) -> TaskLog:
        """One-shot solve with the strong advisor model."""
        text, stats = self.advisor.solve_directly(task["question"])
        answer = _extract_answer_from_text(text)

        if answer is None and task.get("dataset", "gsm8k") == "gsm8k":
            nums = re.findall(r"-?\d[\d,]*\.?\d*", text)
            if nums:
                answer = nums[-1].replace(",", "")

        cost_adv = self.cost_tracker.compute(
            self.advisor_model,
            stats.prompt_tokens,
            stats.completion_tokens,
        )

        dataset = task.get("dataset", "gsm8k")
        return TaskLog(
            task_id=task["id"],
            dataset=dataset,
            method="strong_only",
            question=task["question"],
            prediction=answer,
            ground_truth=task["answer"],
            correct=check_correct(answer, task["answer"], dataset),
            cost_advisor=cost_adv,
            cost_total=cost_adv,
            latency_total_s=stats.latency_s,
            latency_advisor_s=stats.latency_s,
            steps=1,
            step_latencies=[stats.latency_s],
        )

    # ------------------------------------------------------------------
    # Advisor-augmented run
    # ------------------------------------------------------------------

    def run_advisor(self, task: dict, policy: EscalationPolicy, policy_name: str) -> TaskLog:
        """Multi-step executor loop with advisor escalation."""
        dataset = task.get("dataset", "gsm8k")
        messages = [
            {"role": "system", "content": self.executor.system_prompt()},
            {"role": "user", "content": task["question"]},
        ]

        total_exec_latency = 0.0
        total_adv_latency = 0.0
        total_exec_prompt = 0
        total_exec_completion = 0
        total_adv_prompt = 0
        total_adv_completion = 0
        advisor_calls = 0
        confidence_scores: list[float] = []
        step_latencies: list[float] = []
        prev_answers: list[str] = []
        answer: str | None = None

        for step_idx in range(self.max_steps):
            # Executor step
            result = self.executor.step(messages)
            step_lat = result.stats.latency_s
            total_exec_latency += result.stats.latency_s
            total_exec_prompt += result.stats.prompt_tokens
            total_exec_completion += result.stats.completion_tokens

            messages.append({"role": "assistant", "content": result.text})

            # Build state/result dicts for policy evaluation
            if result.answer:
                prev_answers.append(result.answer)

            policy_result: dict[str, Any] = {
                "text": result.text,
                "answer": result.answer,
                "wants_advisor": result.wants_advisor,
                "done": result.done,
            }
            policy_state: dict[str, Any] = {
                "prev_answers": prev_answers,
                "step": step_idx,
                "messages": messages,
            }

            # Self-eval confidence (if policy needs it)
            if policy_name.startswith("self_eval") and result.answer:
                confidence, conf_stats = self.executor.evaluate_confidence(
                    task["question"], result.answer
                )
                policy_result["confidence"] = confidence
                confidence_scores.append(confidence)
                total_exec_latency += conf_stats.latency_s
                total_exec_prompt += conf_stats.prompt_tokens
                total_exec_completion += conf_stats.completion_tokens
                step_lat += conf_stats.latency_s

            # Check if done BEFORE escalation
            if result.done:
                answer = result.answer
                step_latencies.append(step_lat)
                break

            # Escalation check
            if policy.should_escalate(step_idx, policy_result, policy_state):
                guidance, adv_stats = self.advisor.advise(messages)
                total_adv_latency += adv_stats.latency_s
                total_adv_prompt += adv_stats.prompt_tokens
                total_adv_completion += adv_stats.completion_tokens
                advisor_calls += 1
                step_lat += adv_stats.latency_s

                messages = self.advisor.integrate_advice(messages, guidance)

            step_latencies.append(step_lat)

            # Prompt executor to continue if not done
            if not result.done:
                messages.append({
                    "role": "user",
                    "content": "Continue working toward the answer. "
                    "When ready, output 'FINAL ANSWER: <value>'.",
                })

        # If loop ended without explicit answer, try to extract from last output
        if answer is None:
            last_text = messages[-1].get("content", "") if messages else ""
            answer = _extract_answer_from_text(last_text)
            if answer is None and dataset == "gsm8k":
                nums = re.findall(r"-?\d[\d,]*\.?\d*", last_text)
                if nums:
                    answer = nums[-1].replace(",", "")

        cost_exec = self.cost_tracker.compute(
            self.executor_model, total_exec_prompt, total_exec_completion
        )
        cost_adv = self.cost_tracker.compute(
            self.advisor_model, total_adv_prompt, total_adv_completion
        )

        return TaskLog(
            task_id=task["id"],
            dataset=dataset,
            method=f"advisor_{policy_name}",
            question=task["question"],
            prediction=answer,
            ground_truth=task["answer"],
            correct=check_correct(answer, task["answer"], dataset),
            cost_executor=cost_exec,
            cost_advisor=cost_adv,
            cost_total=cost_exec + cost_adv,
            latency_total_s=total_exec_latency + total_adv_latency,
            latency_executor_s=total_exec_latency,
            latency_advisor_s=total_adv_latency,
            steps=len(step_latencies),
            advisor_calls=advisor_calls,
            confidence_scores=confidence_scores,
            step_latencies=step_latencies,
        )
