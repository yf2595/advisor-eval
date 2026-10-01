"""Load advisor-eval/.env without committing secrets."""

from __future__ import annotations

from pathlib import Path


def load_dotenv_if_present() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    root = Path(__file__).resolve().parent.parent
    load_dotenv(root / ".env", override=True)
    import os

    for var in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "MAX_TOTAL_USD"):
        v = os.environ.get(var)
        if v is not None:
            os.environ[var] = v.strip()
