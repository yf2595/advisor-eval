#!/usr/bin/env python3
"""Help/harm event counts vs cheap-only baseline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable


def load_tasks(path: Path) -> dict[str, dict]:
    rows = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("status") == "error":
            continue
        tid = r.get("task_id")
        if tid:
            rows[tid] = r
    return rows


def help_harm(cheap: dict[str, dict], advised: dict[str, dict]) -> dict[str, int]:
    help_n = harm_n = 0
    for tid in set(cheap) & set(advised):
        c = bool(cheap[tid].get("correct"))
        a = bool(advised[tid].get("correct"))
        if not c and a:
            help_n += 1
        if c and not a:
            harm_n += 1
    acc = sum(1 for t in advised if advised[t].get("correct")) / max(len(advised), 1)
    msgs = sum(int(advised[t].get("n_messages") or advised[t].get("advisor_calls") or 0) for t in advised)
    return {
        "n": len(advised),
        "accuracy_pct": round(100 * acc, 2),
        "help": help_n,
        "harm": harm_n,
        "advisor_messages": msgs,
    }


def find_run(runs_dir: Path, substring: str) -> Path | None:
    for p in sorted(runs_dir.glob("*/tasks.jsonl")):
        if substring in p.parent.name:
            return p
    return None


def main() -> None:
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--cheap", type=Path, required=True)
    p.add_argument("--advised", type=Path, required=True)
    args = p.parse_args()
    stats = help_harm(load_tasks(args.cheap), load_tasks(args.advised))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
