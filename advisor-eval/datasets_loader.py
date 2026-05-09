"""Dataset loading utilities for GSM8K, HotpotQA, and GAIA."""

import json
import re
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from datasets import load_dataset
from datasets.exceptions import DatasetNotFoundError


def _parse_gsm8k_answer(answer_text: str) -> str:
    """Extract the final numeric answer from a GSM8K solution string.

    GSM8K answers end with '#### <number>'. We strip commas so that
    '12,345' becomes '12345' for exact-match comparison.
    """
    match = re.search(r"####\s*(.+)", answer_text)
    if match:
        return match.group(1).strip().replace(",", "")
    # Fallback: last number in the string
    nums = re.findall(r"-?\d[\d,]*\.?\d*", answer_text)
    if nums:
        return nums[-1].replace(",", "")
    return answer_text.strip()


def load_gsm8k(n: int | None = None) -> list[dict]:
    """Load GSM8K test split, return normalised task dicts.

    Each dict: {"id": str, "question": str, "answer": str}
    """
    ds = load_dataset("openai/gsm8k", "main", split="test")
    tasks = []
    for i, row in enumerate(ds):
        if n is not None and i >= n:
            break
        tasks.append({
            "id": f"gsm8k_{i}",
            "question": row["question"],
            "answer": _parse_gsm8k_answer(row["answer"]),
        })
    return tasks


def load_hotpotqa(n: int | None = None) -> list[dict]:
    """Load HotpotQA validation split (distractor), return normalised task dicts.

    Each dict: {"id": str, "question": str, "answer": str}
    """
    ds = load_dataset("hotpot_qa", "distractor", split="validation")
    tasks = []
    for i, row in enumerate(ds):
        if n is not None and i >= n:
            break
        tasks.append({
            "id": row.get("id", f"hotpotqa_{i}"),
            "question": row["question"],
            "answer": row["answer"],
        })
    return tasks


def _serialize_supporting_facts(raw: Any) -> Any:
    """Make supporting_facts JSON-serializable for manifests/logs."""
    if raw is None:
        return None
    if isinstance(raw, dict):
        return {str(k): _serialize_supporting_facts(v) for k, v in raw.items()}
    if isinstance(raw, list):
        return [_serialize_supporting_facts(x) for x in raw]
    return raw


def load_hotpotqa_fullwiki(
    n: int | None = None,
    *,
    split: str = "validation",
    level: str = "hard",
    seed: int = 42,
    stratify: bool = True,
    local_file: str | None = None,
    dataset_id: str = "hotpot_qa",
    config_name: str = "fullwiki",
    write_manifest: bool = True,
    manifest_dir: str | None = None,
) -> list[dict]:
    """Load HotpotQA fullwiki split, keep *level* only, optional stratified sample by *type*.

    The model only sees *question*; *answer* and *supporting_facts* are for scoring / analysis.
    """
    if local_file:
        path = Path(local_file)
        if not path.exists():
            raise FileNotFoundError(f"HotpotQA local file not found: {local_file}")
        rows: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
        ds = rows
    else:
        try:
            ds = load_dataset(dataset_id, config_name, split=split)
        except DatasetNotFoundError as exc:
            raise RuntimeError(
                f"Could not load {dataset_id} config={config_name} split={split}. "
                "Check network and dataset id, or pass a local JSONL via local_file."
            ) from exc

    hard: list[dict[str, Any]] = []
    for i, row in enumerate(ds):
        row_dict = dict(row)
        lev = str(row_dict.get("level", "")).strip().lower()
        if lev != str(level).strip().lower():
            continue
        qid = str(row_dict.get("id", f"hotpotqa_fullwiki_{i}"))
        qtype = str(row_dict.get("type", "unknown") or "unknown")
        hard.append({
            "id": qid,
            "question": str(row_dict.get("question", "")).strip(),
            "answer": str(row_dict.get("answer", "")).strip(),
            "row": row_dict,
            "type": qtype,
        })

    if n is None or n >= len(hard):
        chosen = hard
    elif n <= 0:
        chosen = []
    elif not stratify:
        chosen = hard[:n]
    else:
        by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for h in hard:
            by_type[h["type"]].append(h)
        types_sorted = sorted(by_type.keys())
        n_types = len(types_sorted)
        if n_types == 0:
            chosen = []
        else:
            base, rem = divmod(n, n_types)
            quotas = {t: base + (1 if j < rem else 0) for j, t in enumerate(types_sorted)}
            rng = random.Random(seed)
            chosen = []
            for t in types_sorted:
                pool = by_type[t][:]
                rng.shuffle(pool)
                q = quotas[t]
                chosen.extend(pool[:q])
            # Redistribute if some types are too small
            if len(chosen) < n:
                taken_ids = {h["id"] for h in chosen}
                rest = [h for h in hard if h["id"] not in taken_ids]
                rng.shuffle(rest)
                for h in rest:
                    if len(chosen) >= n:
                        break
                    chosen.append(h)
            chosen = chosen[:n]

    tasks: list[dict] = []
    for h in chosen:
        row_dict = h["row"]
        meta: dict[str, Any] = {
            "type": h["type"],
            "level": str(row_dict.get("level", level)),
            "source_subset": "fullwiki_hard",
            "dataset_id": dataset_id,
            "config_name": config_name,
            "source_split": split,
            "supporting_facts": _serialize_supporting_facts(row_dict.get("supporting_facts")),
        }
        tasks.append({
            "id": h["id"],
            "question": h["question"],
            "answer": h["answer"],
            "metadata": meta,
        })

    if write_manifest and tasks:
        base_dir = Path(manifest_dir) if manifest_dir else Path(__file__).resolve().parent / "data"
        base_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = base_dir / f"hotpotqa_fullwiki_manifest_seed{seed}.json"
        type_counts = Counter(t["metadata"]["type"] for t in tasks)
        manifest = {
            "dataset_id": dataset_id,
            "config_name": config_name,
            "split": split,
            "level_filter": level,
            "seed": seed,
            "stratified": stratify,
            "target_n": n,
            "actual_n": len(tasks),
            "type_counts": dict(sorted(type_counts.items())),
            "task_ids": [t["id"] for t in tasks],
        }
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return tasks


def _pick_first(row: dict, keys: list[str], default: str = "") -> str:
    """Return first present key from a row, coerced to string."""
    for key in keys:
        if key in row and row[key] is not None:
            return str(row[key]).strip()
    return default


def load_gaia(
    n: int | None = None,
    split: str = "validation",
    dataset_id: str = "gaia-benchmark/GAIA",
    dataset_config_name: str | None = None,
    local_file: str | None = None,
) -> list[dict]:
    """Load GAIA split and return normalised task dicts.

    Each dict: {"id": str, "question": str, "answer": str, "metadata": dict}
    """
    if local_file:
        path = Path(local_file)
        if not path.exists():
            raise FileNotFoundError(f"GAIA local file not found: {local_file}")
        rows = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
        ds = rows
    else:
        try:
            if dataset_config_name:
                ds = load_dataset(dataset_id, dataset_config_name, split=split)
            else:
                ds = load_dataset(dataset_id, split=split)
        except DatasetNotFoundError as exc:
            raise RuntimeError(
                "GAIA dataset is gated on HuggingFace. "
                "Authenticate with `huggingface-cli login` and request access, "
                "or pass --gaia-local-file to a local JSONL export."
            ) from exc
    tasks = []
    for i, row in enumerate(ds):
        if n is not None and i >= n:
            break

        row_dict = dict(row)
        task_id = _pick_first(row_dict, ["task_id", "id", "instance_id"], f"gaia_{i}")
        question = _pick_first(
            row_dict,
            ["Question", "question", "problem", "prompt", "query"],
            "",
        )
        answer = _pick_first(
            row_dict,
            ["Final answer", "final_answer", "answer", "target"],
            "",
        )

        metadata = {
            "level": _pick_first(row_dict, ["Level", "level"], ""),
            "source_split": split,
            "file_name": _pick_first(row_dict, ["file_name", "File name"], ""),
            "file_path": _pick_first(row_dict, ["file_path", "File path"], ""),
            "dataset_id": dataset_id,
            "dataset_config_name": dataset_config_name or "",
        }

        tasks.append({
            "id": task_id,
            "question": question,
            "answer": answer,
            "metadata": metadata,
        })
    return tasks


def load_dataset_by_name(
    name: str,
    n: int | None = None,
    config: dict | None = None,
) -> list[dict]:
    """Dispatch loader by dataset name."""
    config = config or {}
    loaders = {
        "gsm8k": load_gsm8k,
        "hotpotqa": load_hotpotqa,
        "hotpotqa_fullwiki": lambda k: load_hotpotqa_fullwiki(
            n=k,
            split=config.get("hotpotqa_fullwiki_split", "validation"),
            level=config.get("hotpotqa_fullwiki_level", "hard"),
            seed=config.get("hotpotqa_fullwiki_seed", 42),
            stratify=config.get("hotpotqa_fullwiki_stratify", True),
            local_file=config.get("hotpotqa_fullwiki_local_file"),
            dataset_id=config.get("hotpotqa_fullwiki_dataset_id", "hotpot_qa"),
            config_name=config.get("hotpotqa_fullwiki_config_name", "fullwiki"),
            write_manifest=config.get("hotpotqa_fullwiki_write_manifest", True),
            manifest_dir=config.get("hotpotqa_fullwiki_manifest_dir"),
        ),
        "gaia": lambda k: load_gaia(
            n=k,
            split=config.get("gaia_split", "validation"),
            dataset_id=config.get("gaia_dataset_id", "gaia-benchmark/GAIA"),
            dataset_config_name=config.get("gaia_config_name"),
            local_file=config.get("gaia_local_file"),
        ),
    }
    if name not in loaders:
        raise ValueError(f"Unknown dataset '{name}'. Choose from: {list(loaders.keys())}")
    return loaders[name](n)
