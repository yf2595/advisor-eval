#!/usr/bin/env python3
"""Export frozen task ID lists for GAIA (165) and HotpotQA fullwiki (300)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import yaml

from datasets_loader import load_dataset_by_name


def main() -> None:
    config = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    gaia_cfg = config.get("gaia", {})
    hpfw = config.get("hotpotqa_fullwiki", {})
    n_gaia = int(config.get("datasets", {}).get("gaia_samples", 165))
    n_hot = int(config.get("datasets", {}).get("hotpotqa_fullwiki_samples", 300))

    gaia_tasks = load_dataset_by_name(
        "gaia",
        n=n_gaia,
        config={
            "gaia_split": gaia_cfg.get("split", "validation"),
            "gaia_dataset_id": gaia_cfg.get("dataset_id", "gaia-benchmark/GAIA"),
            "gaia_config_name": gaia_cfg.get("config_name"),
            "gaia_local_file": gaia_cfg.get("local_file"),
        },
    )
    hot_tasks = load_dataset_by_name(
        "hotpotqa_fullwiki",
        n=n_hot,
        config={
            "hotpotqa_fullwiki_split": hpfw.get("split", "validation"),
            "hotpotqa_fullwiki_level": hpfw.get("level", "hard"),
            "hotpotqa_fullwiki_seed": hpfw.get("seed", 42),
            "hotpotqa_fullwiki_stratify": hpfw.get("stratify", True),
            "hotpotqa_fullwiki_local_file": hpfw.get("local_file"),
            "hotpotqa_fullwiki_dataset_id": hpfw.get("dataset_id", "hotpot_qa"),
            "hotpotqa_fullwiki_config_name": hpfw.get("config_name", "fullwiki"),
            "hotpotqa_fullwiki_write_manifest": True,
            "hotpotqa_fullwiki_manifest_dir": str(ROOT / "data"),
        },
    )

    data_dir = ROOT / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    gaia_out = {
        "benchmark": "gaia",
        "n": len(gaia_tasks),
        "task_ids": [t["id"] for t in gaia_tasks],
    }
    hot_out = {
        "benchmark": "hotpotqa_fullwiki",
        "n": len(hot_tasks),
        "seed": hpfw.get("seed", 42),
        "task_ids": [t["id"] for t in hot_tasks],
    }
    (data_dir / "task_ids_gaia.json").write_text(json.dumps(gaia_out, indent=2), encoding="utf-8")
    (data_dir / "task_ids_hotpot.json").write_text(json.dumps(hot_out, indent=2), encoding="utf-8")
    print(f"Wrote {len(gaia_tasks)} GAIA ids and {len(hot_tasks)} Hotpot ids")


if __name__ == "__main__":
    main()
