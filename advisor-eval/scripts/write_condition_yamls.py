#!/usr/bin/env python3
"""Generate configs/conditions/*.yaml for every configuration in the paper.

Also writes configs/paper_tables.yaml, which lists the conditions behind each
paper table (used by scripts/run_experiment_plan.py --table and
scripts/paper_tables.py).

Condition ids: <benchmark>_<pair>_<variant>[_s<seed>], e.g. gaia_nano_hybrid,
hotpot_qwen_self_s43, gaia_oss_hybrid_dna_compact_dn, gaia_gpt54_alone.
"""

from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "configs" / "conditions"

BENCHMARKS = ("gaia", "hotpot")

# Same-family executor-advisor pairs, named by their executor (paper Section 4).
PAIRS = {
    "nano": ("gpt-5.4-nano", "gpt-5.4"),
    "mini": ("gpt-4.1-mini", "gpt-5.4"),
    "gpt41": ("gpt-4.1", "gpt-5.4"),
    "qwen": ("qwen3.5-9b", "qwen3.6-27b"),
    "oss": ("gpt-oss-20b", "gpt-oss-120b"),
}
OPENAI_PAIRS = ("nano", "mini", "gpt41")
OPEN_WEIGHT_PAIRS = ("qwen", "oss")
SEEDED_PAIRS = ("nano", "qwen", "oss")  # main comparison repeated with seeds 43, 44

# Cross-family pairs (paper Tables 3 and 8).
CROSS_PAIRS = {
    "qwen-x-gpt54": ("qwen3.5-9b", "gpt-5.4"),
    "oss-x-gpt54": ("gpt-oss-20b", "gpt-5.4"),
    "nano-x-qwen27b": ("gpt-5.4-nano", "qwen3.6-27b"),
}

# Advisor alone: each pair's advisor solving the task by itself.
ADVISOR_ALONE = {
    "gpt54": "gpt-5.4",
    "qwen27b": "qwen3.6-27b",
    "oss120b": "gpt-oss-120b",
}
PAIR_ADVISOR_ALONE = {"nano": "gpt54", "mini": "gpt54", "gpt41": "gpt54", "qwen": "qwen27b", "oss": "oss120b"}

ABLATION_FORMATS = (
    "dna_compact_dn", "dna_compact_an", "dna_compact_da",
    "dna_compact_n", "dna_compact_d", "dna_compact_a", "freeform",
)
CHANNEL_FORMATS = ("freeform", "freeform_lenmatched", "dna_sentences", "action_handoff", "answer_hint")

conditions: dict[str, dict] = {}
tables: dict[str, list[str]] = {}


def add(table: str, cid: str, **fields) -> str:
    spec = {"condition_id": cid, **fields}
    if cid in conditions and conditions[cid] != spec:
        raise ValueError(f"Conflicting definitions for {cid}")
    conditions[cid] = spec
    tables.setdefault(table, [])
    if cid not in tables[table]:
        tables[table].append(cid)
    return cid


def cheap(table: str, bench: str, pair: str, seed: int = 42) -> str:
    executor, advisor = PAIRS[pair]
    cid = f"{bench}_{pair}_cheap" + (f"_s{seed}" if seed != 42 else "")
    return add(table, cid, benchmark=bench, executor=executor, advisor=advisor,
               method="cheap", seed=seed)


def advised(
    table: str,
    bench: str,
    pair: str,
    policy: str,
    *,
    tau: float = 0.75,
    fmt: str = "dna_compact",
    speaker: str = "advisor",
    seed: int = 42,
    pairs: dict[str, tuple[str, str]] = PAIRS,
) -> str:
    executor, advisor = pairs[pair]
    variant = f"report_t{tau}" if policy == "report" else policy
    if speaker == "executor":
        variant += "_selfdna"
    elif fmt != "dna_compact":
        variant += f"_{fmt}"
    cid = f"{bench}_{pair}_{variant}" + (f"_s{seed}" if seed != 42 else "")
    return add(table, cid, benchmark=bench, executor=executor, advisor=advisor, method="advisor",
               policy=policy, tau=tau, message_format=fmt, speaker=speaker, seed=seed, loop_guards=False)


def alone(table: str, bench: str, name: str) -> str:
    model = ADVISOR_ALONE[name]
    return add(table, f"{bench}_{name}_alone", benchmark=bench, executor=model, advisor=model,
               method="strong", seed=42)


def build() -> None:
    # Tables 1, 5, 6, 9, 10, 11: initiation policies (seed 42).
    for bench in BENCHMARKS:
        for pair in PAIRS:
            t = "table1_initiation"
            cheap(t, bench, pair)
            for policy in ("random", "monitor", "hybrid", "self"):
                advised(t, bench, pair, policy)
            advised(t, bench, pair, "report", tau=0.75)
            alone(t, bench, PAIR_ADVISOR_ALONE[pair])
        for pair in OPENAI_PAIRS if bench == "gaia" else PAIRS:
            for tau in ((0.25, 0.5) if bench == "gaia" else (0.5,)):
                advised("table5_6_report_thresholds", bench, pair, "report", tau=tau)

    # Table 7: three seeds for the main comparison.
    for bench in BENCHMARKS:
        for pair in SEEDED_PAIRS:
            for seed in (42, 43, 44):
                cheap("table7_seeds", bench, pair, seed)
                advised("table7_seeds", bench, pair, "hybrid", seed=seed)
                advised("table7_seeds", bench, pair, "self", seed=seed)

    # Table 2: speaker and channel comparison (hybrid initiation).
    for bench in BENCHMARKS:
        for pair in SEEDED_PAIRS:
            t = "table2_channels"
            cheap(t, bench, pair)
            advised(t, bench, pair, "hybrid")
            advised(t, bench, pair, "hybrid", speaker="executor")
            for fmt in CHANNEL_FORMATS:
                advised(t, bench, pair, "hybrid", fmt=fmt)
            alone(t, bench, PAIR_ADVISOR_ALONE[pair])

    # Tables 12, 13: DNA-act ablation (hybrid initiation).
    for bench in BENCHMARKS:
        for pair in PAIRS if bench == "gaia" else SEEDED_PAIRS:
            t = "table12_ablation"
            cheap(t, bench, pair)
            advised(t, bench, pair, "hybrid")
            for fmt in ABLATION_FORMATS:
                advised(t, bench, pair, "hybrid", fmt=fmt)

    # Tables 3, 8: cross-family advisor swaps.
    for bench in BENCHMARKS:
        t = "table3_8_cross_family"
        for pair in ("qwen", "oss", "nano"):
            cheap(t, bench, pair)
            advised(t, bench, pair, "hybrid")
            advised(t, bench, pair, "self")
        for cross in CROSS_PAIRS:
            advised(t, bench, cross, "hybrid", pairs=CROSS_PAIRS)
            advised(t, bench, cross, "self", pairs=CROSS_PAIRS)
        for name in ADVISOR_ALONE:
            alone(t, bench, name)


def main() -> None:
    build()
    OUT.mkdir(parents=True, exist_ok=True)
    for old in OUT.glob("*.yaml"):
        old.unlink()
    for cid, spec in sorted(conditions.items()):
        (OUT / f"{cid}.yaml").write_text(yaml.safe_dump(spec, sort_keys=False), encoding="utf-8")
    (ROOT / "configs" / "paper_tables.yaml").write_text(
        yaml.safe_dump(tables, sort_keys=False), encoding="utf-8"
    )
    print(f"Wrote {len(conditions)} conditions to {OUT}")
    for t, ids in tables.items():
        print(f"  {t}: {len(ids)} conditions")


if __name__ == "__main__":
    main()
