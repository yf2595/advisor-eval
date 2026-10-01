#!/usr/bin/env python3
"""Run a YAML-defined experiment condition into runs/<condition_id>/ (resumable)."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from tqdm import tqdm

from datasets_loader import load_dataset_by_name
from evaluator import Evaluator, DiskCache
from experiment.condition import ConditionSpec, load_condition
from experiment import trace
from experiment.hooks import parse_dna_fields
from experiment.run_logger import StructuredRunLogger
from logger import TaskLog
from policies import get_policy
from experiment.env_loader import load_dotenv_if_present

load_dotenv_if_present()

PROJECT_ROOT = Path(__file__).resolve().parent

_DATASET_SAMPLE_FALLBACKS = {"hotpotqa_fullwiki": 300, "gaia": 165}


def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_samples(samples: int | None, dataset: str, config: dict) -> int:
    """Explicit ``--limit`` wins, then ``datasets.<dataset>_samples`` in config.yaml."""
    if samples is not None:
        return int(samples)
    cfg_n = config.get("datasets", {}).get(f"{dataset}_samples")
    if cfg_n is not None:
        return int(cfg_n)
    return _DATASET_SAMPLE_FALLBACKS.get(dataset, 50)


def _apply_run_env(config: dict) -> None:
    effort = (config.get("run") or {}).get("reasoning_effort")
    if effort:
        os.environ["REASONING_EFFORT"] = str(effort)


def _git_hash() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def _estimate_batch_cost(spec: ConditionSpec, n_tasks: int, runs_root: Path) -> float:
    """Rough estimate from finished runs with the same method and models."""
    per_task: list[float] = []
    for meta_path in runs_root.glob("*/meta.json"):
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        same = (
            meta.get("method") == spec.method
            and meta.get("executor") == spec.executor
            and meta.get("advisor") == spec.advisor
            and meta.get("benchmark") == spec.benchmark
        )
        if same and meta.get("n_tasks") and meta.get("total_cost_usd") is not None:
            per_task.append(float(meta["total_cost_usd"]) / int(meta["n_tasks"]))
    mean = sum(per_task) / len(per_task) if per_task else 0.02
    return mean * n_tasks


def _budget_cap() -> float:
    try:
        return float(os.environ.get("MAX_TOTAL_USD", "150"))
    except ValueError:
        return 150.0


def _dataset_config(config: dict) -> dict[str, Any]:
    gaia_cfg = config.get("gaia", {})
    hpfw_cfg = config.get("hotpotqa_fullwiki", {})
    return {
        "gaia_split": gaia_cfg.get("split", "validation"),
        "gaia_dataset_id": gaia_cfg.get("dataset_id", "gaia-benchmark/GAIA"),
        "gaia_config_name": gaia_cfg.get("config_name"),
        "gaia_local_file": gaia_cfg.get("local_file"),
        "gaia_task_ids_file": gaia_cfg.get("task_ids_file"),
        "hotpotqa_fullwiki_split": hpfw_cfg.get("split", "validation"),
        "hotpotqa_fullwiki_level": hpfw_cfg.get("level", "hard"),
        "hotpotqa_fullwiki_seed": hpfw_cfg.get("seed", 42),
        "hotpotqa_fullwiki_stratify": hpfw_cfg.get("stratify", True),
        "hotpotqa_fullwiki_local_file": hpfw_cfg.get("local_file"),
        "hotpotqa_fullwiki_dataset_id": hpfw_cfg.get("dataset_id", "hotpot_qa"),
        "hotpotqa_fullwiki_config_name": hpfw_cfg.get("config_name", "fullwiki"),
        "hotpotqa_fullwiki_write_manifest": hpfw_cfg.get("write_manifest", True),
        "hotpotqa_fullwiki_manifest_dir": hpfw_cfg.get("manifest_dir"),
    }


def run_condition(
    spec: ConditionSpec,
    config: dict,
    *,
    limit: int | None = None,
    dry_run: bool = False,
    cache: DiskCache | None = None,
    runs_root: Path | None = None,
) -> Path:
    _apply_run_env(config)
    runs_root = runs_root or (PROJECT_ROOT / "runs")
    run_dir = runs_root / spec.run_name
    logger = StructuredRunLogger(run_dir, spec.run_name)

    meta: dict = {}

    dataset = spec.dataset_name
    samples = resolve_samples(limit, dataset, config)
    tasks = load_dataset_by_name(dataset, n=samples, config=_dataset_config(config))
    for t in tasks:
        t["dataset"] = dataset

    pending = [t for t in tasks if not logger.is_completed(t["id"])]

    if dry_run:
        sample = pending[: min(5, len(pending))]
        print(f"[DRY RUN] condition={spec.condition_id} method={spec.method} seed={spec.seed}")
        print(f"Would run {len(pending)} tasks on {dataset}; sample IDs: {[t['id'] for t in sample]}")
        hooks = spec.to_hooks(dry_run=True)
        print(
            f"executor={spec.executor} advisor={spec.advisor} policy={spec.policy_name} "
            f"speaker={hooks.speaker} format={hooks.message_format}"
        )
        sp = hooks.resolve_advisor_system_prompt()
        preview = (sp or "(advisor.ADVISOR_SYSTEM_PROMPT = prompts/dna_sentences.txt)")[:400]
        print(f"Advisor system prompt preview:\n{preview}...\n")
        meta["dry_run_sample"] = [t["id"] for t in sample]
        meta["ended_utc"] = datetime.now(timezone.utc).isoformat()
        logger.write_meta(meta)
        return run_dir

    est = _estimate_batch_cost(spec, len(pending), runs_root)
    n_total = len(tasks)
    n_already = n_total - len(pending)
    print(
        f"Progress: {n_already}/{n_total} done, {len(pending)} remaining "
        f"(est. ${est:.2f} for this batch, cap ${_budget_cap():.0f})",
        flush=True,
    )
    if est > _budget_cap():
        raise SystemExit(f"Refusing run: estimate ${est:.2f} exceeds MAX_TOTAL_USD={_budget_cap()}")

    config = dict(config)
    config["models"] = {"executor": spec.executor, "advisor": spec.advisor}
    config["run"] = {**config.get("run", {}), "seed": spec.seed}
    hooks = spec.to_hooks(dry_run=dry_run)
    evaluator = Evaluator(config, cache=cache)
    evaluator.set_experiment(hooks)

    meta.update({
        "condition_id": spec.condition_id,
        "run_name": spec.run_name,
        "benchmark": spec.benchmark,
        "dataset": dataset,
        "method": spec.method,
        "executor": spec.executor,
        "advisor": spec.advisor,
        "policy": spec.policy if spec.method == "advisor" else None,
        "policy_name": spec.policy_name,
        "tau": spec.tau,
        "message_format": spec.message_format,
        "speaker": spec.speaker,
        "seed": spec.seed,
        "loop_guards": spec.loop_guards,
        "git_hash": _git_hash(),
        "decoding_params": config.get("run", {}),
        "local_models": config.get("local_models", {}),
        "prices": config.get("costs", {}),
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "dry_run": dry_run,
        "n_tasks": len(tasks),
    })
    logger.write_meta(meta)

    policy = get_policy(spec.policy_name, config) if spec.method == "advisor" else None
    total_cost = 0.0
    msg_idx = 0

    pbar: Any = None
    if not os.environ.get("NO_TQDM"):
        pbar = tqdm(
            total=n_total,
            initial=n_already,
            desc=spec.condition_id,
            unit="task",
            file=sys.stdout,
            dynamic_ncols=True,
            mininterval=0.2,
        )

    for step, task in enumerate(pending, start=n_already + 1):
        if pbar is not None:
            pbar.set_postfix_str(task["id"][:8], refresh=False)
        tqdm.write(f"[{step}/{n_total}] start {task['id']}  Q: {trace.preview(task.get('question'), 100)}")
        trace.start_task(run_dir, task["id"])
        trace.event("task_start", f"{task['id']}", question=task.get("question", ""),
                    gold=task.get("answer", ""), metadata=task.get("metadata", {}))
        status = "ok"
        try:
            if spec.method == "cheap":
                log = evaluator.run_cheap_only(task)
            elif spec.method == "strong":
                log = evaluator.run_strong_only(task)
            else:
                log = evaluator.run_advisor(task, policy, spec.policy_name)
        except Exception as exc:  # noqa: BLE001
            status = "error"
            log = TaskLog(
                task_id=task["id"],
                dataset=dataset,
                method=f"{spec.method}_{spec.policy_name}",
                question=task.get("question", ""),
                metadata=task.get("metadata", {}) or {},
                prediction=None,
                ground_truth=task.get("answer", ""),
                correct=False,
            )
            tqdm.write(f"[{step}/{n_total}] ERROR {task['id']}: {exc}")

        level_or_type = None
        if isinstance(task.get("metadata"), dict):
            level_or_type = task["metadata"].get("level") or task["metadata"].get("type")

        row = StructuredRunLogger.task_log_to_structured(
            log,
            benchmark=spec.benchmark,
            level_or_type=level_or_type,
            status=status,
        )
        logger.log_task(row)
        total_cost += log.cost_total
        trace.event("task_end", f"{status} pred={trace.preview(log.prediction, 60)} gold={trace.preview(log.ground_truth, 40)} "
                                f"steps={log.steps} tools={log.tool_calls} ${log.cost_total:.4f}",
                    status=status, pred=log.prediction, correct=log.correct,
                    steps=log.steps, tool_calls=log.tool_calls, cost_usd=log.cost_total)
        trace.end_task()
        if pbar is not None:
            pbar.update(1)
        tqdm.write(
            f"[{step}/{n_total}] done  {task['id']}  {status}  "
            f"${log.cost_total:.4f}  {'correct' if log.correct else 'wrong'}"
        )

        for g in log.advisor_guidance or []:
            msg_idx += 1
            logger.log_message({
                "task_id": log.task_id,
                "msg_idx": msg_idx,
                "step": g.get("step"),
                "trigger": g.get("trigger"),
                "message_text": g.get("guidance"),
                "parsed_fields": g.get("parsed_fields") or parse_dna_fields(g.get("guidance", "")),
                "next_action_after": g.get("next_tool_hint"),
                "handoff_executed": g.get("handoff_executed", False),
            })

    if pbar is not None:
        pbar.close()

    all_rows = [json.loads(line) for line in logger.tasks_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    meta["ended_utc"] = datetime.now(timezone.utc).isoformat()
    meta["total_cost_usd"] = round(sum(float(r.get("cost_usd") or 0) for r in all_rows), 4)
    logger.write_meta(meta)
    print(f"Run complete: {run_dir} (${total_cost:.4f} this session)")
    return run_dir


def main() -> None:
    os.chdir(PROJECT_ROOT)
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--condition", required=True,
                   help="Condition id (configs/conditions/<id>.yaml) or path to a condition YAML")
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--limit", type=int, default=None, help="Run only the first N tasks")
    p.add_argument("--dry-run", action="store_true", help="Print the setup without calling any model")
    p.add_argument("--cache", action="store_true")
    args = p.parse_args()

    config = load_config(args.config)
    condition_path = Path(args.condition)
    if not condition_path.is_file():
        condition_path = PROJECT_ROOT / "configs" / "conditions" / f"{args.condition}.yaml"
    spec = load_condition(condition_path)
    cache = None
    if args.cache or config.get("cache", {}).get("enabled"):
        cache = DiskCache(config.get("cache", {}).get("directory", ".cache"))
    run_condition(spec, config, limit=args.limit, dry_run=args.dry_run, cache=cache)


if __name__ == "__main__":
    main()
