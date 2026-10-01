#!/usr/bin/env python3
"""5-task dry runs (no API) for every condition YAML."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from experiment.condition import load_condition
from run_condition import load_config, run_condition


def main() -> None:
    config = load_config(str(ROOT / "config.yaml"))
    cond_dir = ROOT / "configs" / "conditions"
    for fp in sorted(cond_dir.glob("*.yaml")):
        spec = load_condition(fp)
        run_condition(spec, config, limit=5, dry_run=True)
    print("Dry runs complete.")


if __name__ == "__main__":
    main()
