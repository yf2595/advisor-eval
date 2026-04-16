"""Advisor agent – the strong model that provides strategic guidance only."""

import time
from dataclasses import dataclass

from openai import OpenAI

ADVISOR_SYSTEM_PROMPT = """\
You are a strategic advisor assisting a weaker executor model. You see the \
executor's full reasoning chain.

Your job:
1. Identify where the executor's reasoning went wrong or could be improved.
2. Provide concise guidance: a corrected plan, a course correction, or a \
   stop signal if the answer looks correct.
3. Use enumerated steps, not lengthy explanations.

Rules:
- Do NOT produce the final answer yourself.
- Do NOT repeat the executor's work. Only add what is missing or wrong.
- Keep your response under 500 tokens.\
"""


@dataclass
class AdvisorCallStats:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_s: float = 0.0


class AdvisorAgent:
    def __init__(self, model: str, temperature: float = 0.0, seed: int = 42):
        self.client = OpenAI()
        self.model = model
        self.temperature = temperature
        self.seed = seed

    def advise(self, conversation: list[dict]) -> tuple[str, AdvisorCallStats]:
        """Provide strategic guidance given the full conversation context.

        Returns (guidance_text, stats).
        """
        messages = [
            {"role": "system", "content": ADVISOR_SYSTEM_PROMPT},
            *conversation,
            {
                "role": "user",
                "content": (
                    "Review the executor's reasoning above. Provide concise "
                    "strategic guidance to help it reach the correct answer. "
                    "Do NOT give the final answer yourself."
                ),
            },
        ]

        t0 = time.perf_counter()
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            seed=self.seed,
            max_tokens=512,
        )
        latency = time.perf_counter() - t0

        text = response.choices[0].message.content or ""
        usage = response.usage

        stats = AdvisorCallStats(
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
            latency_s=latency,
        )
        return text, stats

    @staticmethod
    def integrate_advice(messages: list[dict], guidance: str) -> list[dict]:
        """Inject advisor guidance into the conversation so the executor sees it."""
        return messages + [
            {
                "role": "user",
                "content": (
                    f"[ADVISOR GUIDANCE]\n{guidance}\n[/ADVISOR GUIDANCE]\n\n"
                    "Continue solving the task using this guidance. "
                    "When you have the final answer, output it on a line "
                    "starting with 'FINAL ANSWER: '."
                ),
            }
        ]

    def solve_directly(self, question: str) -> tuple[str, AdvisorCallStats]:
        """Use the strong model to solve a task in one shot (for baseline).

        Returns (answer_text, stats).
        """
        messages = [
            {
                "role": "system",
                "content": (
                    "Solve the following task step by step. "
                    "Output your final answer on a line starting with "
                    "'FINAL ANSWER: ' followed by ONLY the answer value."
                ),
            },
            {"role": "user", "content": question},
        ]

        t0 = time.perf_counter()
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            seed=self.seed,
            max_tokens=2048,
        )
        latency = time.perf_counter() - t0

        text = response.choices[0].message.content or ""
        usage = response.usage

        stats = AdvisorCallStats(
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
            latency_s=latency,
        )
        return text, stats
