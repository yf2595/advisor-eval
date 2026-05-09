"""Advisor agent – the strong model that provides strategic guidance only."""

import time
from dataclasses import dataclass

from openai import OpenAI

ADVISOR_SYSTEM_PROMPT = """\
You are the strategic advisor to a weaker tool-using executor.
You are consulted ONLY when the executor is blocked; act fast and concrete.
Prioritize plan repair and next best action over broad exploration.

Output EXACTLY this structure (no preamble, no filler, <=350 tokens):
DIAGNOSIS: one sentence naming the specific blocker (wrong tool, bad
query, misread tool output, stuck in a loop, formatting mistake, budget
pressure).
NEXT: up to 5 numbered steps. Each step names ONE tool and the exact
input to try, or says "finalize with <FORMAT>" where FORMAT describes
the answer shape (e.g. "comma-separated list with spaces", "digits
only, divided by 1000", "phrase without INT./scene directive"). Every
step must be NEW (never repeat a query already tried verbatim).
The first NEXT step must be unambiguous and immediately executable.
AVOID: up to 3 short bullets listing pitfalls or dead-ends already
seen in the trace (URL types that 403, queries that returned nothing,
misleading sources).

Hard rules:
- Never reveal or restate the final answer, value, number, name, or list.
- Never reproduce tool outputs or write the executor's JSON schema.
- If the executor is about to emit a final answer that is clearly hedged
  or empty (e.g. "I can't find", "unknown", "N/A", ""), NEXT must force
  ONE new source/tool before finalizing. If the executor already has a
  concrete-looking answer (a number, a name, a specific phrase) do NOT
  order more searches -- only coach the format, e.g. "finalize with
  <FORMAT>".
- If the trace shows the same (tool, query) 2+ times, NEXT MUST
  propose a different tool or clearly different keywords.
- If a host has already failed (403/404/empty) 2+ times, do NOT tell the
  executor to fetch that host again; pick a different host or switch to
  web_search / wiki_search / arxiv_search.
- If the question requires a strict format (comma-list, unit, rounding,
  name-only), DIAGNOSIS MUST flag it and the last NEXT step MUST be a
  "finalize with <FORMAT>" instruction matching the question literally.
- Prefer concrete tool parameters over generic advice (exact query/url/id/json).
- If parse/format issues are visible, include one NEXT step that explains
  exactly how to parse the observed tool response into the target answer.
- Stay under 350 tokens. Use short bullets. No filler, no preamble.\
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
            max_completion_tokens=400,
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
    def integrate_advice(
        messages: list[dict],
        guidance: str,
        format_hint: str | None = None,
    ) -> list[dict]:
        """Inject advisor guidance into the conversation so the executor sees it.

        `format_hint` lets callers pass a dataset-specific reminder about how
        to emit the final answer (JSON schema for GAIA, 'FINAL ANSWER:' prefix
        for QA tasks). If omitted, a directive default is used that orders
        the executor to act on step 1 of the NEXT list immediately.
        """
        default_hint = (
            "On your NEXT turn: execute step 1 of the advisor's NEXT list "
            "exactly, unless it is clearly impossible. Do NOT repeat any "
            "query listed in AVOID or any query you already tried. Keep "
            "the same response format you were asked to use."
        )
        suffix = format_hint.strip() if format_hint else default_hint
        return messages + [
            {
                "role": "user",
                "content": (
                    f"[ADVISOR GUIDANCE]\n{guidance}\n[/ADVISOR GUIDANCE]\n\n"
                    f"{suffix}"
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
            max_completion_tokens=2048,
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
