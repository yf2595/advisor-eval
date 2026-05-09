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
from analysis import analyse_results, load_results, check_gaia_smoke_gates


def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# Hardcoded fallbacks used only if config.yaml is missing the entry.
_DATASET_SAMPLE_FALLBACKS = {
    "gsm8k": 200,
    "hotpotqa": 300,
    "hotpotqa_fullwiki": 300,
    "gaia": 165,
}


def resolve_samples(samples: int | None, dataset: str, config: dict) -> int:
    """Return the sample count for `dataset`, preferring the explicit CLI value.

    If `samples` is None, fall back to `config['datasets']['<dataset>_samples']`,
    then to a hardcoded per-dataset default (HotpotQA=300, GAIA=165, GSM8K=200).
    """
    if samples is not None:
        return int(samples)
    cfg_n = config.get("datasets", {}).get(f"{dataset}_samples")
    if cfg_n is not None:
        return int(cfg_n)
    return _DATASET_SAMPLE_FALLBACKS.get(dataset, 50)


def run_single_experiment(
    config: dict,
    dataset_name: str,
    method: str,
    policy_name: str,
    samples: int,
    cache: DiskCache | None = None,
) -> str:
    """Run one experiment and return the path to the JSONL result file."""
    gaia_cfg = config.get("gaia", {})
    hpfw_cfg = config.get("hotpotqa_fullwiki", {})
    tasks = load_dataset_by_name(
        dataset_name,
        n=samples,
        config={
            "gaia_split": gaia_cfg.get("split", "validation"),
            "gaia_dataset_id": gaia_cfg.get("dataset_id", "gaia-benchmark/GAIA"),
            "gaia_config_name": gaia_cfg.get("config_name"),
            "gaia_local_file": gaia_cfg.get("local_file"),
            "hotpotqa_fullwiki_split": hpfw_cfg.get("split", "validation"),
            "hotpotqa_fullwiki_level": hpfw_cfg.get("level", "hard"),
            "hotpotqa_fullwiki_seed": hpfw_cfg.get("seed", 42),
            "hotpotqa_fullwiki_stratify": hpfw_cfg.get("stratify", True),
            "hotpotqa_fullwiki_local_file": hpfw_cfg.get("local_file"),
            "hotpotqa_fullwiki_dataset_id": hpfw_cfg.get("dataset_id", "hotpot_qa"),
            "hotpotqa_fullwiki_config_name": hpfw_cfg.get("config_name", "fullwiki"),
            "hotpotqa_fullwiki_write_manifest": hpfw_cfg.get("write_manifest", True),
            "hotpotqa_fullwiki_manifest_dir": hpfw_cfg.get("manifest_dir"),
        },
    )
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

    from logger import TaskLog  # local import to avoid circular deps in tests

    for task in tqdm(tasks, desc=f"{dataset_name}/{method}/{policy_name}"):
        try:
            if method == "cheap":
                log = evaluator.run_cheap_only(task)
            elif method == "strong":
                log = evaluator.run_strong_only(task)
            elif method == "advisor":
                log = evaluator.run_advisor(task, policy, policy_name)
            else:
                raise ValueError(f"Unknown method '{method}'")
        except Exception as exc:  # noqa: BLE001
            # Guarantee the matrix continues even if the API rejects a prompt
            # or a transient failure slips past the inner handlers.
            print(f"  ! task {task.get('id')} failed: {type(exc).__name__}: {exc}")
            log = TaskLog(
                task_id=task.get("id", ""),
                dataset=dataset_name,
                method=f"{method}_{policy_name}",
                question=task.get("question", ""),
                metadata=task.get("metadata", {}) or {},
                prediction=None,
                ground_truth=task.get("answer", ""),
                correct=False,
                tool_trace=[{
                    "step": 0,
                    "tool": "task_runner",
                    "input": "",
                    "success": False,
                    "error": f"{type(exc).__name__}: {exc}"[:400],
                }],
            )

        logger.log(log)

    correct = sum(1 for l in logger.logs if l.correct)
    total = len(logger.logs)
    print(f"\n  => {dataset_name}/{method}/{policy_name}: "
          f"{correct}/{total} correct ({100*correct/total:.1f}%)")
    print(f"     Results: {logger.filepath}")
    return str(logger.filepath)


def run_all_experiments(
    config: dict,
    samples: int,
    cache: DiskCache | None = None,
    datasets: list[str] | None = None,
    advisor_only: bool = False,
) -> None:
    """Run Experiments 1-5 as described in the plan."""
    datasets = datasets or ["hotpotqa"]
    result_files: list[str] = []

    if not advisor_only:
        # Experiment 1: Core comparison (cheap / strong baselines only).
        # model_driven and fixed_interval policies are intentionally skipped.
        print("\n" + "=" * 60)
        print("EXPERIMENT 1: Core Comparison")
        print("=" * 60)
        for ds in datasets:
            for method, pol in [("cheap", "none"), ("strong", "none")]:
                fp = run_single_experiment(config, ds, method, pol, samples, cache)
                result_files.append(fp)
    else:
        print("\n" + "=" * 60)
        print("Skipping Experiment 1 (cheap/strong baselines) -- advisor-only mode")
        print("=" * 60)

    # Experiment 2: Policy comparison (skip model_driven and fixed_interval).
    BASE_POLICIES = ["random_prob", "failure_based", "self_eval"]
    print("\n" + "=" * 60)
    print("EXPERIMENT 2: Policy Comparison")
    print("=" * 60)
    for ds in datasets:
        for pol in BASE_POLICIES:
            fp = run_single_experiment(config, ds, "advisor", pol, samples, cache)
            result_files.append(fp)

    # Experiment 3: Threshold sweep for self_eval
    print("\n" + "=" * 60)
    print("EXPERIMENT 3: Cost/Latency vs Performance (threshold sweep)")
    print("=" * 60)
    original_threshold = config.get("policies", {}).get("threshold", 0.6)
    for threshold in [0.3, 0.5, 0.7, 0.9]:
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


def run_gaia_smoke_trio(
    config: dict,
    samples: int,
    cache: DiskCache | None = None,
) -> None:
    """Run the fixed GAIA smoke trio for rapid quality gating."""
    dataset = "gaia"
    print("\n" + "=" * 60)
    print("GAIA SMOKE TRIO: cheap / strong / advisor(model_driven)")
    print("=" * 60)
    run_single_experiment(config, dataset, "cheap", "none", samples, cache)
    run_single_experiment(config, dataset, "strong", "none", samples, cache)
    run_single_experiment(config, dataset, "advisor", "model_driven", samples, cache)


class NoAdvisorPolicy:
    """Ablation policy: never escalate."""
    def should_escalate(self, step, result, state):
        return False


def _register_sweep_and_ablation_policies():
    """Register dynamic policies used by --all."""
    from policies import SelfEvalPolicy
    for threshold in [0.3, 0.5, 0.7, 0.9]:
        name = f"self_eval_t{threshold}"
        POLICY_REGISTRY[name] = lambda t=threshold: SelfEvalPolicy(threshold=t)

    POLICY_REGISTRY["no_advisor"] = NoAdvisorPolicy


def main():
    parser = argparse.ArgumentParser(
        description="Advisor Strategy Evaluation Framework"
    )
    parser.add_argument("--dataset", choices=["gsm8k", "hotpotqa", "hotpotqa_fullwiki", "gaia"], default="gsm8k")
    parser.add_argument("--method", choices=["cheap", "strong", "advisor"], default="advisor")
    parser.add_argument("--policy", default="model_driven",
                        help="Escalation policy (advisor). Includes failure_or_conf_t0.25|0.5|0.75 "
                             "(failure OR confidence<threshold; GAIA uses heuristic confidence).")
    parser.add_argument("--samples", type=int, default=None,
                        help="Number of tasks to evaluate. Defaults to the "
                             "dataset-specific value in config.yaml under "
                             "`datasets.<dataset>_samples` (HotpotQA=300, "
                             "GAIA=165, GSM8K=200).")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--executor-model", type=str, default=None,
                        help="Override executor model from config")
    parser.add_argument("--advisor-model", type=str, default=None,
                        help="Override advisor model from config")
    parser.add_argument("--all", action="store_true",
                        help="Run all experiments (1-5)")
    parser.add_argument("--advisor-only", action="store_true",
                        help="With --all, skip cheap/strong baselines and run "
                             "only advisor experiments (2, 3, 5)")
    parser.add_argument("--analyze", type=str, default=None, metavar="DIR",
                        help="Run analysis on existing results directory")
    parser.add_argument("--cache", action="store_true",
                        help="Enable disk caching of API responses")
    parser.add_argument("--gaia-split", type=str, default=None,
                        help="GAIA split name (e.g. validation, test)")
    parser.add_argument("--gaia-dataset-id", type=str, default=None,
                        help="HuggingFace dataset id for GAIA")
    parser.add_argument("--gaia-config-name", type=str, default=None,
                        help="HuggingFace GAIA config name (e.g. 2023_all)")
    parser.add_argument("--max-tool-calls", type=int, default=None,
                        help="Maximum allowed tool calls per GAIA task")
    parser.add_argument("--gaia-local-file", type=str, default=None,
                        help="Optional local GAIA JSONL file")
    parser.add_argument("--hotpotqa-fullwiki-local-file", type=str, default=None,
                        help="Local JSONL export for HotpotQA fullwiki (optional)")
    parser.add_argument("--hotpotqa-fullwiki-seed", type=int, default=None,
                        help="Stratified sampling seed for hotpotqa_fullwiki")
    parser.add_argument("--hotpotqa-fullwiki-split", type=str, default=None,
                        help="Dataset split (default: validation / dev)")
    parser.add_argument("--check-smoke-gates", type=str, default=None, metavar="DIR",
                        help="Check GAIA smoke gates on existing results dir")
    parser.add_argument("--gaia-smoke-trio", action="store_true",
                        help="Run fixed GAIA trio: cheap, strong, advisor(model_driven)")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.executor_model:
        config.setdefault("models", {})["executor"] = args.executor_model
    if args.advisor_model:
        config.setdefault("models", {})["advisor"] = args.advisor_model
    config.setdefault("gaia", {})
    if args.gaia_split is not None:
        config["gaia"]["split"] = args.gaia_split
    if args.gaia_dataset_id is not None:
        config["gaia"]["dataset_id"] = args.gaia_dataset_id
    if args.gaia_config_name is not None:
        config["gaia"]["config_name"] = args.gaia_config_name
    if args.max_tool_calls is not None:
        config["gaia"]["max_tool_calls"] = args.max_tool_calls
    if args.gaia_local_file is not None:
        config["gaia"]["local_file"] = args.gaia_local_file
    config.setdefault("hotpotqa_fullwiki", {})
    if args.hotpotqa_fullwiki_local_file is not None:
        config["hotpotqa_fullwiki"]["local_file"] = args.hotpotqa_fullwiki_local_file
    if args.hotpotqa_fullwiki_seed is not None:
        config["hotpotqa_fullwiki"]["seed"] = args.hotpotqa_fullwiki_seed
    if args.hotpotqa_fullwiki_split is not None:
        config["hotpotqa_fullwiki"]["split"] = args.hotpotqa_fullwiki_split

    cache = None
    if args.cache or config.get("cache", {}).get("enabled", False):
        cache_dir = config.get("cache", {}).get("directory", ".cache")
        cache = DiskCache(cache_dir)

    _register_sweep_and_ablation_policies()

    if args.analyze:
        analyse_results(args.analyze)
        return

    if args.check_smoke_gates:
        df = load_results(args.check_smoke_gates)
        gates = check_gaia_smoke_gates(df)
        if gates is None:
            raise SystemExit("Could not compute GAIA smoke gates from provided results.")
        print(f"strong > cheap: {gates['strong_beats_cheap']}")
        print(f"advisor >= cheap: {gates['advisor_at_least_cheap']}")
        if not all(gates.values()):
            raise SystemExit(1)
        return

    if args.all:
        # Respect --dataset when running the full matrix; otherwise default to hotpotqa.
        datasets_to_run = [args.dataset] if args.dataset else None
        # When running multiple datasets in --all mode the same N is used for
        # all of them, so we resolve against the dataset selected on the CLI.
        n = resolve_samples(args.samples, args.dataset, config)
        run_all_experiments(
            config,
            n,
            cache,
            datasets=datasets_to_run,
            advisor_only=args.advisor_only,
        )
        return

    if args.gaia_smoke_trio:
        run_gaia_smoke_trio(
            config=config,
            samples=resolve_samples(args.samples, "gaia", config),
            cache=cache,
        )
        return

    run_single_experiment(
        config,
        dataset_name=args.dataset,
        method=args.method,
        policy_name=args.policy,
        samples=resolve_samples(args.samples, args.dataset, config),
        cache=cache,
    )


if __name__ == "__main__":
    main()
