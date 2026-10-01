"""Load YAML experiment conditions (configs/conditions/*.yaml)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from experiment.hooks import ExperimentHooks, load_prompt_file, policy_name_from_condition

METHODS = ("cheap", "advisor", "strong")

MESSAGE_FORMATS = (
    "dna_compact",
    "dna_sentences",
    "dna_compact_dn",
    "dna_compact_an",
    "dna_compact_da",
    "dna_compact_n",
    "dna_compact_d",
    "dna_compact_a",
    "freeform",
    "freeform_lenmatched",
    "action_handoff",
    "answer_hint",
)

# Free-form, length-matched: capped to the length of a full-sentence DNA message.
FREEFORM_LENMATCHED_MAX_TOKENS = 172
FREEFORM_LENMATCHED_WORDS = 130


@dataclass
class ConditionSpec:
    condition_id: str
    benchmark: str
    executor: str
    advisor: str
    policy: str = "hybrid"
    tau: float = 0.75
    message_format: str = "dna_compact"
    speaker: str = "advisor"
    method: str = "advisor"
    seed: int = 42
    loop_guards: bool = False

    def __post_init__(self) -> None:
        if self.method not in METHODS:
            raise ValueError(f"{self.condition_id}: method must be one of {METHODS}")
        if self.message_format not in MESSAGE_FORMATS:
            raise ValueError(f"{self.condition_id}: unknown message_format '{self.message_format}'")
        if self.speaker not in ("advisor", "executor"):
            raise ValueError(f"{self.condition_id}: speaker must be 'advisor' or 'executor'")

    @property
    def dataset_name(self) -> str:
        if self.benchmark in ("hotpot", "hotpotqa", "hotpotqa_fullwiki"):
            return "hotpotqa_fullwiki"
        return self.benchmark

    @property
    def policy_name(self) -> str:
        if self.method != "advisor":
            return "none"
        return policy_name_from_condition(self.policy, self.tau)

    @property
    def run_name(self) -> str:
        return self.condition_id

    def to_hooks(self, *, dry_run: bool = False) -> ExperimentHooks:
        fmt = self.message_format
        prompt_text: str | None = None
        max_tok = 400
        if fmt == "freeform_lenmatched":
            prompt_text = (
                load_prompt_file("freeform.txt")
                + f"\n\nBe concise; at most ~{FREEFORM_LENMATCHED_WORDS} words."
            )
            max_tok = FREEFORM_LENMATCHED_MAX_TOKENS
        elif fmt != "dna_sentences":
            prompt_text = load_prompt_file(f"{fmt}.txt")

        return ExperimentHooks(
            speaker=self.speaker,
            message_format=fmt,
            advisor_system_prompt=prompt_text,
            advisor_max_completion_tokens=max_tok,
            action_handoff=fmt == "action_handoff",
            answer_hint=fmt == "answer_hint",
            loop_guards=self.loop_guards,
            dry_run=dry_run,
        )


def load_condition(path: str | Path) -> ConditionSpec:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return ConditionSpec(
        condition_id=data["condition_id"],
        benchmark=data["benchmark"],
        executor=data["executor"],
        advisor=data["advisor"],
        policy=data.get("policy", "hybrid"),
        tau=float(data.get("tau", 0.75)),
        message_format=data.get("message_format", "dna_compact"),
        speaker=data.get("speaker", "advisor"),
        method=data.get("method", "advisor"),
        seed=int(data.get("seed", 42)),
        loop_guards=bool(data.get("loop_guards", False)),
    )


def load_all_conditions(dir_path: str | Path) -> list[ConditionSpec]:
    return [load_condition(fp) for fp in sorted(Path(dir_path).glob("*.yaml"))]
