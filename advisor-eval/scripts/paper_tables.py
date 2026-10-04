#!/usr/bin/env python3
"""Compute the paper's metrics from runs/ and write one CSV per paper table.

For every condition in configs/paper_tables.yaml that has a run under
runs/<condition_id>/, this reports accuracy, help / harm against the
executor-alone run with the same benchmark, executor and seed, the exact
McNemar p-value, message counts, tokens per message, cost, latency, tool
errors, recovery efficiency, and the GAIA easy / hard split.

Output: results/paper_tables/<table>.csv (missing runs are listed and skipped).

GAIA runs are not distributed with the repository (dataset terms), so GAIA
rows appear only after running the GAIA conditions locally; see the README.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import yaml

from experiment.condition import ConditionSpec, load_condition
from scripts.help_harm import help_harm, load_tasks
from scripts.stats import mcnemar_exact

RUNS = ROOT / "runs"
OUT = ROOT / "results" / "paper_tables"

COLUMNS = [
    "condition_id", "benchmark", "executor", "advisor", "method", "policy", "tau",
    "message_format", "speaker", "seed", "n", "accuracy", "help", "harm", "mcnemar_p",
    "advisor_messages", "tokens_per_message", "cost_per_task_usd", "cost_reduction_vs_gpt54_pct",
    "latency_s", "tool_errors", "recovery_efficiency", "acc_easy", "acc_hard",
]

_ENCODER = None


def count_tokens(text: str) -> int:
    global _ENCODER
    if _ENCODER is None:
        import tiktoken

        _ENCODER = tiktoken.get_encoding("o200k_base")
    return len(_ENCODER.encode(text or ""))


def baseline_for(spec: ConditionSpec, specs: dict[str, ConditionSpec]) -> ConditionSpec | None:
    for other in specs.values():
        if (
            other.method == "cheap"
            and other.benchmark == spec.benchmark
            and other.executor == spec.executor
            and other.seed == spec.seed
        ):
            return other
    return None


def gpt54_alone_for(spec: ConditionSpec, specs: dict[str, ConditionSpec]) -> ConditionSpec | None:
    for other in specs.values():
        if other.method == "strong" and other.advisor == "gpt-5.4" and other.benchmark == spec.benchmark:
            return other
    return None


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def accuracy(tasks: dict[str, dict], keep=lambda t: True) -> float | None:
    rows = [t for t in tasks.values() if keep(t)]
    return 100 * sum(1 for t in rows if t.get("correct")) / len(rows) if rows else None


def summarize(spec: ConditionSpec, specs: dict[str, ConditionSpec], cache: dict[str, dict]) -> dict | None:
    tasks = cache.get(spec.condition_id)
    if not tasks:
        return None
    acc = accuracy(tasks)
    row: dict = {
        "condition_id": spec.condition_id,
        "benchmark": spec.benchmark,
        "executor": spec.executor,
        "advisor": spec.advisor,
        "method": spec.method,
        "policy": spec.policy if spec.method == "advisor" else "",
        "tau": spec.tau if spec.method == "advisor" else "",
        "message_format": spec.message_format if spec.method == "advisor" else "",
        "speaker": spec.speaker if spec.method == "advisor" else "",
        "seed": spec.seed,
        "n": len(tasks),
        "accuracy": round(acc, 1),
        "cost_per_task_usd": round(mean([float(t.get("cost_usd") or 0) for t in tasks.values()]), 5),
        "latency_s": round(mean([float(t.get("latency_s") or 0) for t in tasks.values()]), 2),
        "tool_errors": sum(int(t.get("n_tool_errors") or 0) for t in tasks.values()),
    }

    if spec.benchmark == "gaia":
        def level(t: dict) -> str:
            return str(t.get("level_or_type") or "")

        easy = accuracy(tasks, lambda t: level(t) == "1")
        hard = accuracy(tasks, lambda t: level(t) in ("2", "3"))
        row["acc_easy"] = round(easy, 1) if easy is not None else ""
        row["acc_hard"] = round(hard, 1) if hard is not None else ""

    if spec.method == "advisor":
        msgs_path = RUNS / spec.condition_id / "messages.jsonl"
        texts = []
        if msgs_path.exists():
            done = set(tasks)
            for line in msgs_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    m = json.loads(line)
                    if m.get("task_id") in done:
                        texts.append(m.get("message_text") or "")
        n_msgs = sum(int(t.get("n_messages") or 0) for t in tasks.values())
        row["advisor_messages"] = n_msgs
        if texts:
            row["tokens_per_message"] = round(mean([count_tokens(x) for x in texts]), 1)
        base = baseline_for(spec, specs)
        if base is not None and cache.get(base.condition_id):
            hh = help_harm(cache[base.condition_id], tasks)
            row["help"], row["harm"] = hh["help"], hh["harm"]
            row["mcnemar_p"] = f"{mcnemar_exact(hh['help'], hh['harm']):.4g}"
            base_acc = accuracy(cache[base.condition_id])
            if n_msgs:
                row["recovery_efficiency"] = round((acc - base_acc) / n_msgs, 3)

    ref = gpt54_alone_for(spec, specs)
    if ref is not None and cache.get(ref.condition_id) and row["cost_per_task_usd"] > 0:
        ref_cost = mean([float(t.get("cost_usd") or 0) for t in cache[ref.condition_id].values()])
        if ref_cost:
            row["cost_reduction_vs_gpt54_pct"] = round(100 * (1 - row["cost_per_task_usd"] / ref_cost), 1)
    return row


def main() -> None:
    tables = yaml.safe_load((ROOT / "configs" / "paper_tables.yaml").read_text(encoding="utf-8"))
    specs = {p.stem: load_condition(p) for p in sorted((ROOT / "configs" / "conditions").glob("*.yaml"))}
    cache = {
        cid: load_tasks(RUNS / cid / "tasks.jsonl")
        for cid in specs
        if (RUNS / cid / "tasks.jsonl").exists()
    }
    OUT.mkdir(parents=True, exist_ok=True)
    missing: set[str] = set()
    for table, ids in tables.items():
        rows = []
        for cid in ids:
            row = summarize(specs[cid], specs, cache)
            if row is None:
                missing.add(cid)
            else:
                rows.append(row)
        path = OUT / f"{table}.csv"
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
        print(f"{table}: {len(rows)}/{len(ids)} conditions -> {path.relative_to(ROOT)}")
    if missing:
        print(f"\n{len(missing)} conditions have no run yet, e.g. {sorted(missing)[:5]}")
        n_gaia = sum(1 for cid in missing if specs[cid].benchmark == "gaia")
        if n_gaia:
            print(
                f"{n_gaia} of them are GAIA conditions. GAIA runs are not redistributed; "
                "run them locally with your own Hugging Face access (see README, 'GAIA data')."
            )


if __name__ == "__main__":
    main()
