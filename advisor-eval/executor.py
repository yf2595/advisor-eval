"""Executor agent – the cheap model that drives tasks end-to-end."""

import re
import time
from dataclasses import dataclass, field

from openai import OpenAI

EXECUTOR_SYSTEM_PROMPT = """\
You are a precise problem-solving agent. Solve the given task step by step.

You have access to a stronger advisor model. When you need strategic guidance, \
include the marker [REQUEST_ADVISOR] on its own line in your response. Use it:
- Before committing to a complex approach you are unsure about.
- When you are stuck or an approach is not converging.
- When considering a change of strategy.
Do NOT request the advisor for straightforward steps you can handle yourself.

When you have a final answer, output it on a line starting with "FINAL ANSWER: " \
followed by ONLY the answer value (a number, a name, etc.). No extra text after it.

Think step by step. Be concise.\
Before finalizing, verify answer format exactly matches the question.\
"""

CONFIDENCE_PROMPT = """\
Rate your confidence (0.0–1.0) that the following answer is correct for the \
given question. Output ONLY a single decimal number, nothing else.

Question: {question}
Your answer: {answer}\
"""


@dataclass
class CallStats:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_s: float = 0.0


@dataclass
class StepResult:
    text: str = ""
    answer: str | None = None
    wants_advisor: bool = False
    done: bool = False
    stats: CallStats = field(default_factory=CallStats)


class ExecutorAgent:
    def __init__(self, model: str, temperature: float = 0.0, seed: int = 42):
        self.client = OpenAI()
        self.model = model
        self.temperature = temperature
        self.seed = seed

    def step(self, messages: list[dict]) -> StepResult:
        """Run one executor step. Returns StepResult with latency and token counts."""
        t0 = time.perf_counter()
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            seed=self.seed,
            max_completion_tokens=2048,
        )
        latency = time.perf_counter() - t0

        choice = response.choices[0].message
        text = choice.content or ""
        usage = response.usage

        stats = CallStats(
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
            latency_s=latency,
        )

        answer = self._extract_answer(text)
        wants_advisor = "[REQUEST_ADVISOR]" in text
        done = answer is not None

        return StepResult(
            text=text,
            answer=answer,
            wants_advisor=wants_advisor,
            done=done,
            stats=stats,
        )

    def evaluate_confidence(self, question: str, answer: str) -> tuple[float, CallStats]:
        """Ask the executor to rate its own confidence. Returns (score, stats)."""
        prompt = CONFIDENCE_PROMPT.format(question=question, answer=answer)
        t0 = time.perf_counter()
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            seed=self.seed,
            max_completion_tokens=16,
        )
        latency = time.perf_counter() - t0

        text = response.choices[0].message.content or ""
        usage = response.usage
        stats = CallStats(
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
            latency_s=latency,
        )

        try:
            score = float(text.strip())
            score = max(0.0, min(1.0, score))
        except ValueError:
            nums = re.findall(r"[\d.]+", text)
            score = float(nums[0]) if nums else 0.5

        return score, stats

    @staticmethod
    def _extract_answer(text: str) -> str | None:
        """Look for 'FINAL ANSWER: <value>' in executor output."""
        match = re.search(r"FINAL ANSWER:\s*(.+)", text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return None

    @staticmethod
    def system_prompt() -> str:
        return EXECUTOR_SYSTEM_PROMPT
