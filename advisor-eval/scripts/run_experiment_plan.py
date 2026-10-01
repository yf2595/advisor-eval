#!/usr/bin/env python3
"""Run many conditions in sequence (resumable) with a cumulative spend cap.

Examples:
  python scripts/run_experiment_plan.py --table table1_initiation
  python scripts/run_experiment_plan.py --only "gaia_nano_*" --limit 5
  python scripts/run_experiment_plan.py --all --dry-run
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import yaml

from experiment.condition import load_condition
from experiment.env_loader import load_dotenv_if_present
from run_condition import load_config, run_condition


def select_conditions(args: argparse.Namespace) -> list[Path]:
    cond_dir = ROOT / "configs" / "conditions"
    all_paths = {p.stem: p for p in sorted(cond_dir.glob("*.yaml"))}
    ids: list[str] = []
    if args.all:
        ids = list(all_paths)
    for table in args.table or []:
        tables = yaml.safe_load((ROOT / "configs" / "paper_tables.yaml").read_text(encoding="utf-8"))
        if table not in tables:
            raise SystemExit(f"Unknown table '{table}'. Choose from: {list(tables)}")
        ids.extend(tables[table])
    if args.only:
        def matches(cid: str) -> bool:
            return any(fnmatch.fnmatch(cid, pattern) for pattern in args.only)
        # With --table/--all, --only narrows the selection; on its own it selects.
        ids = [c for c in ids if matches(c)] if ids else [c for c in all_paths if matches(c)]
    seen: set[str] = set()
    return [all_paths[c] for c in ids if not (c in seen or seen.add(c))]


def is_complete(run_dir: Path) -> bool:
    meta_path = run_dir / "meta.json"
    tasks_path = run_dir / "tasks.jsonl"
    if not meta_path.exists() or not tasks_path.exists():
        return False
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    n_expected = int(meta.get("n_tasks") or 0)
    n_ok = sum(
        1 for line in tasks_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and json.loads(line).get("status") != "error"
    )
    return bool(meta.get("ended_utc")) and n_expected > 0 and n_ok >= n_expected


def main() -> None:
    os.chdir(ROOT)
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    load_dotenv_if_present()

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--table", action="append", help="Paper table key from configs/paper_tables.yaml")
    p.add_argument("--only", action="append", help="Glob over condition ids, e.g. 'hotpot_*_self*'; narrows --table/--all if given")
    p.add_argument("--all", action="store_true", help="Every condition in configs/conditions/")
    p.add_argument("--limit", type=int, default=None, help="Run only the first N tasks of each condition")
    p.add_argument("--dry-run", action="store_true", help="Print the setup of each condition; no model calls")
    args = p.parse_args()

    paths = select_conditions(args)
    if not paths:
        raise SystemExit("No conditions selected (use --table, --only or --all).")

    config = load_config(str(ROOT / "config.yaml"))
    cap = float(os.environ.get("MAX_TOTAL_USD", "150"))
    spent = 0.0
    print(f"{len(paths)} conditions selected", flush=True)

    for path in paths:
        spec = load_condition(path)
        if args.dry_run:
            run_condition(spec, config, limit=args.limit or 5, dry_run=True)
            continue
        run_dir = ROOT / "runs" / spec.run_name
        if args.limit is None and is_complete(run_dir):
            print(f"=== {spec.condition_id}: already complete, skip")
            continue
        print(f"\n=== Running {spec.condition_id} ===", flush=True)
        before = 0.0
        if (run_dir / "meta.json").exists():
            before = float(json.loads((run_dir / "meta.json").read_text(encoding="utf-8")).get("total_cost_usd") or 0)
        run_condition(spec, config, limit=args.limit, dry_run=False)
        after = float(json.loads((run_dir / "meta.json").read_text(encoding="utf-8")).get("total_cost_usd") or 0)
        spent += max(after - before, 0.0)
        print(f"Cumulative spend this session ~${spent:.2f} / ${cap:.0f}")
        if spent > cap:
            print("MAX_TOTAL_USD reached; stopping.")
            break


if __name__ == "__main__":
    main()
