#!/usr/bin/env python3
"""Run the full HotpotQA fullwiki benchmark matrix (baselines + advisor policies).

Uses models from config.yaml (executor=gpt-5.4-nano, advisor=gpt-5.4 by default).
Writes a manifest JSON listing all result JSONL paths for reporting.

Usage:
  python run_hotpot_fullwiki_matrix.py
  python run_hotpot_fullwiki_matrix.py --samples 100 --config config.yaml
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from run import load_config, run_single_experiment


DATASET = "hotpotqa_fullwiki"

MATRIX: list[tuple[str, str]] = [
    ("cheap", "none"),
    ("strong", "none"),
    ("advisor", "failure_based"),
    ("advisor", "model_driven"),
    ("advisor", "self_eval_t0.25"),
    ("advisor", "self_eval_t0.5"),
    ("advisor", "self_eval_t0.75"),
    ("advisor", "failure_or_conf_t0.75"),
]


def main() -> None:
    parser = argparse.ArgumentParser(description="HotpotQA fullwiki full matrix")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--samples", type=int, default=None)
    parser.add_argument("--cache", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.samples is not None:
        config.setdefault("datasets", {})["hotpotqa_fullwiki_samples"] = args.samples

    samples = config.get("datasets", {}).get("hotpotqa_fullwiki_samples", 300)

    from evaluator import DiskCache

    cache = None
    if args.cache or config.get("cache", {}).get("enabled", False):
        cache_dir = config.get("cache", {}).get("directory", ".cache")
        cache = DiskCache(cache_dir)

    from run import _register_sweep_and_ablation_policies

    _register_sweep_and_ablation_policies()

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    manifest_name = f"hotpotqa_fullwiki_matrix_manifest_{ts}.json"
    manifest_path = Path("results") / manifest_name
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    entries: list[dict[str, str]] = []
    print(f"\n>>> HotpotQA fullwiki matrix: {samples} tasks × {len(MATRIX)} runs\n")

    for method, policy in MATRIX:
        label = f"{method}_{policy}" if policy != "none" else method
        print(f"\n{'=' * 60}\nRUN: {label}\n{'=' * 60}")
        fp = run_single_experiment(
            config,
            DATASET,
            method,
            policy,
            samples,
            cache,
        )
        entries.append({
            "method": method,
            "policy": policy,
            "label": label,
            "jsonl": fp,
        })

    manifest = {
        "dataset": DATASET,
        "samples": samples,
        "created_utc": ts,
        "executor": config.get("models", {}).get("executor"),
        "advisor_model": config.get("models", {}).get("advisor"),
        "runs": entries,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\n>>> Manifest written: {manifest_path}")
    print(">>> Next: python run.py --analyze results/")


if __name__ == "__main__":
    main()
