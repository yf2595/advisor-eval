#!/usr/bin/env python3
"""Verify prompts/dna_sentences.txt matches advisor.ADVISOR_SYSTEM_PROMPT byte for byte."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from advisor import ADVISOR_SYSTEM_PROMPT  # noqa: E402


def main() -> int:
    text = (ROOT / "prompts" / "dna_sentences.txt").read_text(encoding="utf-8")
    if text != ADVISOR_SYSTEM_PROMPT:
        print("FAIL: prompts/dna_sentences.txt differs from advisor.ADVISOR_SYSTEM_PROMPT")
        return 1
    print("OK: prompts/dna_sentences.txt is identical to advisor.ADVISOR_SYSTEM_PROMPT")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
