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
    """Escalate only on genuine failures: empty output or no answer produced."""

    def should_escalate(self, step: int, result: dict[str, Any], state: dict[str, Any]) -> bool:
        text = result.get("text", "")
        if not text.strip():
            return True

        if result.get("done") and result.get("answer") is None:
            return True

        if result.get("tool_error", False) or result.get("parse_error", False):
            return True

        # Escalate when repeated dead-ends suggest the executor is stuck.
        if state.get("dead_end_count", 0) >= 2:
            return True

        return False


class SelfEvalPolicy:
    """Escalate when executor's self-rated confidence is below a threshold."""

    def __init__(self, threshold: float = 0.6):
        self.threshold = threshold

    def should_escalate(self, step: int, result: dict[str, Any], state: dict[str, Any]) -> bool:
        confidence = result.get("confidence", 1.0)
        return confidence < self.threshold


class FailureOrLowConfidencePolicy:
    """Escalate if failure-based signals OR confidence is below a threshold (logical OR)."""

    def __init__(self, threshold: float = 0.75):
        self._failure = FailureBasedPolicy()
        self.threshold = threshold

    def should_escalate(self, step: int, result: dict[str, Any], state: dict[str, Any]) -> bool:
        if self._failure.should_escalate(step, result, state):
            return True
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
    if name.startswith("self_eval_t"):
        try:
            threshold = float(name.split("self_eval_t", 1)[1])
        except ValueError as exc:
            raise ValueError(
                "Invalid self_eval threshold policy name. "
                "Use format self_eval_t<value>, e.g. self_eval_t0.25"
            ) from exc
        return SelfEvalPolicy(threshold=threshold)

    if name.startswith("failure_or_conf_t"):
        try:
            threshold = float(name.split("failure_or_conf_t", 1)[1])
        except ValueError as exc:
            raise ValueError(
                "Invalid failure_or_conf policy name. "
                "Use format failure_or_conf_t<value>, e.g. failure_or_conf_t0.75"
            ) from exc
        return FailureOrLowConfidencePolicy(threshold=threshold)

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
