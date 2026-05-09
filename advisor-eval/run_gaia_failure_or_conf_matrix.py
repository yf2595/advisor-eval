#!/usr/bin/env python3
"""GAIA matrix: failure_based vs failure_or_conf_t0.75 on three executors (nano, 4.1-mini, 4.1).

Usage:
  python run_gaia_failure_or_conf_matrix.py
  python run_gaia_failure_or_conf_matrix.py --config my.yaml --samples 10
"""

from __future__ import annotations

import argparse

from evaluator import DiskCache
from run import load_config, run_single_experiment

EXECUTORS = (
    "gpt-5.4-nano",
    "gpt-4.1-mini",
    "gpt-4.1",
)
POLICIES = (
    "failure_based",
    "failure_or_conf_t0.75",
)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--samples", type=int, default=None, help="Override gaia sample count")
    p.add_argument("--cache", action="store_true", help="Enable response cache from config")
    args = p.parse_args()

    config = load_config(args.config)
    n = args.samples
    if n is None:
        n = int(config.get("datasets", {}).get("gaia_samples", 165))

    cache = None
    if args.cache or config.get("cache", {}).get("enabled", False):
        cache = DiskCache(config.get("cache", {}).get("directory", ".cache"))

    for executor_model in EXECUTORS:
        config.setdefault("models", {})["executor"] = executor_model
        for policy in POLICIES:
            run_single_experiment(
                config,
                dataset_name="gaia",
                method="advisor",
                policy_name=policy,
                samples=n,
                cache=cache,
            )


if __name__ == "__main__":
    main()
