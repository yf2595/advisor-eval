"""Escalation policies that decide when the executor should consult the advisor."""

from __future__ import annotations

import random
from typing import Any, Callable, Protocol


class EscalationPolicy(Protocol):
    def should_escalate(
        self,
        step: int,
        result: dict[str, Any],
        state: dict[str, Any],
    ) -> bool: ...


class FixedIntervalPolicy:
    """Escalate every N steps."""

    def __init__(self, interval: int = 3):
        self.interval = interval

    def should_escalate(self, step: int, result: dict[str, Any], state: dict[str, Any]) -> bool:
        return step % self.interval == 0


class RandomPolicy:
    """Escalate with a fixed probability each step."""

    def __init__(self, prob: float = 0.3):
        self.prob = prob

    def should_escalate(self, step: int, result: dict[str, Any], state: dict[str, Any]) -> bool:
        return random.random() < self.prob


class FailureBasedPolicy:
    """Escalate when the executor's output looks like a failure."""

    def should_escalate(self, step: int, result: dict[str, Any], state: dict[str, Any]) -> bool:
        text = result.get("text", "")
        if not text.strip():
            return True

        prev_answers = state.get("prev_answers", [])
        current = result.get("answer")
        if current and prev_answers and current == prev_answers[-1]:
            return True

        return False


class SelfEvalPolicy:
    """Escalate when executor's self-rated confidence is below a threshold."""

    def __init__(self, threshold: float = 0.6):
        self.threshold = threshold

    def should_escalate(self, step: int, result: dict[str, Any], state: dict[str, Any]) -> bool:
        confidence = result.get("confidence", 1.0)
        return confidence < self.threshold


class ModelDrivenPolicy:
    """Escalate when the executor itself requests the advisor.

    This is the primary policy aligned with Anthropic's advisor pattern:
    the executor model decides when it needs help by emitting a
    [REQUEST_ADVISOR] marker in its output.
    """

    def should_escalate(self, step: int, result: dict[str, Any], state: dict[str, Any]) -> bool:
        return result.get("wants_advisor", False)


POLICY_REGISTRY: dict[str, Callable[..., EscalationPolicy]] = {
    "fixed_interval": FixedIntervalPolicy,
    "random_prob": RandomPolicy,
    "failure_based": FailureBasedPolicy,
    "self_eval": SelfEvalPolicy,
    "model_driven": ModelDrivenPolicy,
}


def get_policy(name: str, config: dict[str, Any]) -> EscalationPolicy:
    """Instantiate a policy by name with config-driven parameters."""
    if name not in POLICY_REGISTRY:
        raise ValueError(f"Unknown policy '{name}'. Choose from: {list(POLICY_REGISTRY.keys())}")

    policy_cfg = config.get("policies", {})

    if name == "fixed_interval":
        return FixedIntervalPolicy(interval=policy_cfg.get("fixed_interval", 3))
    elif name == "random_prob":
        return RandomPolicy(prob=policy_cfg.get("random_prob", 0.3))
    elif name == "self_eval":
        return SelfEvalPolicy(threshold=policy_cfg.get("threshold", 0.6))
    elif name == "failure_based":
        return FailureBasedPolicy()
    elif name == "model_driven":
        return ModelDrivenPolicy()
    else:
        return POLICY_REGISTRY[name]()
