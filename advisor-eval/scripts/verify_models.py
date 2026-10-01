#!/usr/bin/env python3
"""Send one tiny chat completion to every model used in the paper (OpenAI API and local vLLM)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import yaml

from experiment.env_loader import load_dotenv_if_present

load_dotenv_if_present()

from experiment import models
from experiment.openai_chat import chat_completions_create
from experiment.openai_client import make_openai_client

OPENAI_MODELS = ("gpt-5.4-nano", "gpt-4.1-mini", "gpt-4.1", "gpt-5.4")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--local-only", action="store_true", help="Only check the vLLM-served models")
    p.add_argument("--openai-only", action="store_true", help="Only check the OpenAI API models")
    args = p.parse_args()

    config = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    models.configure_models(config)
    targets: list[str] = []
    if not args.local_only:
        targets += list(OPENAI_MODELS)
    if not args.openai_only:
        targets += list(config.get("local_models") or {})

    client = None if args.local_only else make_openai_client()
    failed = 0
    for model in targets:
        try:
            resp = chat_completions_create(
                client,
                model=model,
                messages=[{"role": "user", "content": "Reply with exactly: OK"}],
                temperature=0.0,
                seed=42,
                max_completion_tokens=64,
            )
            print(f"{model}: OK ({(resp.choices[0].message.content or '').strip()[:40]!r})")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"{model}: FAIL {type(exc).__name__}: {str(exc)[:160]}")
    print(json.dumps({"checked": len(targets), "failed": failed}))
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
