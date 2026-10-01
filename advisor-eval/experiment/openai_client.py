"""Shared OpenAI client (request timeout so runs do not hang indefinitely)."""

from __future__ import annotations

import os

from openai import OpenAI


def make_openai_client() -> OpenAI:
    timeout_s = float(os.environ.get("OPENAI_TIMEOUT_S", "180"))
    # Runs with only vLLM-served models need no OpenAI key; requests that do reach
    # the OpenAI API fail with an authentication error.
    api_key = os.environ.get("OPENAI_API_KEY") or "OPENAI_API_KEY-not-set"
    return OpenAI(api_key=api_key, timeout=timeout_s, max_retries=2)
