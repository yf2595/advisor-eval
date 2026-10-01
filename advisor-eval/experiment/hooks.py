"""Per-condition settings passed to the GAIA / HotpotQA agent loops."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"

ANSWER_HINT_PREFIX = "A stronger model proposes this answer; you may use or ignore it."


@dataclass
class ExperimentHooks:
    """How the advisor channel behaves in one condition."""

    speaker: str = "advisor"  # advisor | executor (Self-DNA)
    message_format: str = "dna_compact"
    advisor_system_prompt: str | None = None
    advisor_max_completion_tokens: int = 400
    action_handoff: bool = False
    answer_hint: bool = False
    loop_guards: bool = False
    dry_run: bool = False

    @classmethod
    def legacy(cls) -> ExperimentHooks:
        return cls()

    def resolve_advisor_system_prompt(self) -> str | None:
        """None means use ``advisor.ADVISOR_SYSTEM_PROMPT`` (= prompts/dna_sentences.txt)."""
        if self.advisor_system_prompt is not None:
            return self.advisor_system_prompt
        if self.message_format == "dna_sentences":
            return None
        return load_prompt_file(f"{self.message_format}.txt")

    def integrate_format_hint(self, default_hint: str) -> str:
        if self.answer_hint:
            return f"{ANSWER_HINT_PREFIX}\n\n{default_hint}"
        return default_hint


def load_prompt_file(name: str) -> str:
    path = PROMPTS_DIR / name
    return path.read_text(encoding="utf-8").strip()


def policy_name_from_condition(policy: str, tau: float) -> str:
    if policy == "hybrid":
        return f"failure_or_conf_t{tau}"
    if policy == "monitor":
        return "failure_based"
    if policy == "report":
        return f"self_eval_t{tau}"
    if policy == "self":
        return "model_driven"
    if policy == "random":
        return "random_prob"
    return policy


def parse_dna_fields(text: str) -> dict[str, str | None]:
    """Best-effort parse of DIAGNOSE / AVOID / NEXT from a compact (JSON) or sentence message."""
    stripped = (text or "").strip()
    if stripped.startswith("{"):
        try:
            obj = json.loads(stripped)
        except json.JSONDecodeError:
            obj = None
        if isinstance(obj, dict):
            return {k: (str(obj[k]) if obj.get(k) is not None else None) for k in ("diagnose", "avoid", "next")}

    labels = r"DIAGNOS(?:IS|E)|NEXT|AVOID"

    def block(label: str) -> str | None:
        m = re.search(rf"(?:{label}):\s*(.*?)(?=\n(?:{labels}):|\Z)", stripped, re.S | re.I)
        return m.group(1).strip() if m else None

    return {
        "diagnose": block(r"DIAGNOS(?:IS|E)"),
        "avoid": block("AVOID"),
        "next": block("NEXT"),
    }
