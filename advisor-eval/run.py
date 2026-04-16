#!/usr/bin/env python3
"""CLI entry point for the Advisor Strategy Evaluation framework."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml
from tqdm import tqdm

from datasets_loader import load_dataset_by_name
from evaluator import Evaluator, DiskCache
from logger import TaskLogger
from policies import get_policy, POLICY_REGISTRY
from analysis import analyse_results


def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_single_experiment(
    config: dict,
    dataset_name: str,
    method: str,
    policy_name: str,
    samples: int,
    cache: DiskCache | None = None,
) -> str:
    """Run one experiment and return the path to the JSONL result file."""
    tasks = load_dataset_by_name(dataset_name, n=samples)
    for t in tasks:
        t["dataset"] = dataset_name

    evaluator = Evaluator(config, cache=cache)
    logger = TaskLogger(
        dataset=dataset_name,
        method=method,
        policy=policy_name,
    )

    policy = None
    if method == "advisor":
        policy = get_policy(policy_name, config)

    for task in tqdm(tasks, desc=f"{dataset_name}/{method}/{policy_name}"):
        if method == "cheap":
            log = evaluator.run_cheap_only(task)
        elif method == "strong":
            log = evaluator.run_strong_only(task)
        elif method == "advisor":
            log = evaluator.run_advisor(task, policy, policy_name)
        else:
            raise ValueError(f"Unknown method '{method}'")

        logger.log(log)

    correct = sum(1 for l in logger.logs if l.correct)
    total = len(logger.logs)
    print(f"\n  => {dataset_name}/{method}/{policy_name}: "
          f"{correct}/{total} correct ({100*correct/total:.1f}%)")
    print(f"     Results: {logger.filepath}")
    return str(logger.filepath)


def run_all_experiments(config: dict, samples: int, cache: DiskCache | None = None) -> None:
    """Run Experiments 1-5 as described in the plan."""
    datasets = ["gsm8k", "hotpotqa"]
    result_files: list[str] = []

    # Experiment 1: Core comparison (cheap / strong / advisor model_driven)
    print("\n" + "=" * 60)
    print("EXPERIMENT 1: Core Comparison")
    print("=" * 60)
    for ds in datasets:
        for method, pol in [("cheap", "none"), ("strong", "none"), ("advisor", "model_driven")]:
            fp = run_single_experiment(config, ds, method, pol, samples, cache)
            result_files.append(fp)

    # Experiment 2: Policy comparison
    print("\n" + "=" * 60)
    print("EXPERIMENT 2: Policy Comparison")
    print("=" * 60)
    for ds in datasets:
        for pol in POLICY_REGISTRY:
            fp = run_single_experiment(config, ds, "advisor", pol, samples, cache)
            result_files.append(fp)

    # Experiment 3: Threshold sweep for self_eval
    print("\n" + "=" * 60)
    print("EXPERIMENT 3: Cost/Latency vs Performance (threshold sweep)")
    print("=" * 60)
    original_threshold = config.get("policies", {}).get("threshold", 0.6)
    for threshold in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        config.setdefault("policies", {})["threshold"] = threshold
        for ds in datasets:
            fp = run_single_experiment(
                config, ds, "advisor", f"self_eval_t{threshold}", samples, cache
            )
            result_files.append(fp)
    config["policies"]["threshold"] = original_threshold

    # Experiment 5: Ablation (multi-step executor, no advisor calls)
    print("\n" + "=" * 60)
    print("EXPERIMENT 5: Ablation (no advisor)")
    print("=" * 60)
    for ds in datasets:
        fp = run_single_experiment(config, ds, "advisor", "no_advisor", samples, cache)
        result_files.append(fp)

    # Analysis
    print("\n" + "=" * 60)
    print("ANALYSIS")
    print("=" * 60)
    analyse_results("results")


class NoAdvisorPolicy:
    """Ablation policy: never escalate."""
    def should_escalate(self, step, result, state):
        return False


def _register_sweep_and_ablation_policies():
    """Register dynamic policies used by --all."""
    for threshold in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        name = f"self_eval_t{threshold}"
        from policies import SelfEvalPolicy
        POLICY_REGISTRY[name] = lambda t=threshold: SelfEvalPolicy(threshold=t)

    POLICY_REGISTRY["no_advisor"] = NoAdvisorPolicy


def main():
    parser = argparse.ArgumentParser(
        description="Advisor Strategy Evaluation Framework"
    )
    parser.add_argument("--dataset", choices=["gsm8k", "hotpotqa"], default="gsm8k")
    parser.add_argument("--method", choices=["cheap", "strong", "advisor"], default="advisor")
    parser.add_argument("--policy", default="model_driven",
                        help="Escalation policy (for advisor method)")
    parser.add_argument("--samples", type=int, default=50)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--all", action="store_true",
                        help="Run all experiments (1-5)")
    parser.add_argument("--analyze", type=str, default=None, metavar="DIR",
                        help="Run analysis on existing results directory")
    parser.add_argument("--cache", action="store_true",
                        help="Enable disk caching of API responses")
    args = parser.parse_args()

    config = load_config(args.config)

    cache = None
    if args.cache or config.get("cache", {}).get("enabled", False):
        cache_dir = config.get("cache", {}).get("directory", ".cache")
        cache = DiskCache(cache_dir)

    _register_sweep_and_ablation_policies()

    if args.analyze:
        analyse_results(args.analyze)
        return

    if args.all:
        run_all_experiments(config, args.samples, cache)
        return

    run_single_experiment(
        config,
        dataset_name=args.dataset,
        method=args.method,
        policy_name=args.policy,
        samples=args.samples,
        cache=cache,
    )


if __name__ == "__main__":
    main()
