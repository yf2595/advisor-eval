"""Model registry: routes locally served (vLLM) models to their OpenAI-compatible endpoints.

Models listed under ``local_models`` in ``config.yaml`` are called through
their own ``base_url``; every other model name goes to the OpenAI API.
"""

from __future__ import annotations

import os
from typing import Any

from openai import OpenAI

_LOCAL_MODELS: dict[str, dict[str, Any]] = {}
_CLIENTS: dict[str, OpenAI] = {}


def configure_models(config: dict[str, Any]) -> None:
    _LOCAL_MODELS.clear()
    _CLIENTS.clear()
    _LOCAL_MODELS.update(config.get("local_models") or {})


def local_spec(model: str) -> dict[str, Any] | None:
    return _LOCAL_MODELS.get(model)


def is_local(model: str) -> bool:
    return model in _LOCAL_MODELS


def client_for(model: str, default: Any) -> Any:
    """Client for ``model``: a cached vLLM client for local models, else ``default``."""
    spec = _LOCAL_MODELS.get(model)
    if spec is None:
        return default
    if model not in _CLIENTS:
        env_name = spec.get("base_url_env")
        base_url = (os.environ.get(env_name) if env_name else None) or spec["base_url"]
        _CLIENTS[model] = OpenAI(
            base_url=base_url,
            api_key=os.environ.get("VLLM_API_KEY", "EMPTY"),
            timeout=float(os.environ.get("OPENAI_TIMEOUT_S", "180")),
            max_retries=2,
        )
    return _CLIENTS[model]


def served_name(model: str) -> str:
    spec = _LOCAL_MODELS.get(model)
    return spec.get("served_name", model) if spec else model
