#!/usr/bin/env python3
"""Build the DNA-act ablation prompts (paper Table 12) from prompts/dna_compact.txt.

Each variant keeps the compact DNA prompt verbatim and changes only the
output-format line, keeping the listed acts and dropping the others.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PROMPTS = ROOT / "prompts"
BASE = (PROMPTS / "dna_compact.txt").read_text(encoding="utf-8")
FORMAT_LINE = next(line for line in BASE.splitlines() if line.startswith("Output ONLY a JSON object"))

ACTS = {"d": "diagnose", "a": "avoid", "n": "next"}
SUBSETS = ("dn", "an", "da", "n", "d", "a")


def format_line(keys: list[str]) -> str:
    fields = ", ".join(f'"{k}": str' for k in keys)
    dropped = [k for k in ACTS.values() if k not in keys]
    each = "with each value at most 25 tokens" if len(keys) > 1 else "with the value at most 25 tokens"
    omit = " Do NOT include " + " or ".join(f'"{k}"' for k in dropped) + "." if dropped else ""
    return f"Output ONLY a JSON object {{{fields}}} {each}.{omit} No markdown fences."


def main() -> None:
    for subset in SUBSETS:
        keys = [ACTS[c] for c in "dan" if c in subset]
        text = BASE.replace(FORMAT_LINE, format_line(keys))
        (PROMPTS / f"dna_compact_{subset}.txt").write_text(text, encoding="utf-8")
    print(f"Wrote {len(SUBSETS)} ablation prompts to {PROMPTS}")


if __name__ == "__main__":
    main()
