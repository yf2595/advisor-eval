"""Dataset loading utilities for GSM8K and HotpotQA via HuggingFace."""

import re
from datasets import load_dataset


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


def load_dataset_by_name(name: str, n: int | None = None) -> list[dict]:
    """Dispatch loader by dataset name."""
    loaders = {
        "gsm8k": load_gsm8k,
        "hotpotqa": load_hotpotqa,
    }
    if name not in loaders:
        raise ValueError(f"Unknown dataset '{name}'. Choose from: {list(loaders.keys())}")
    return loaders[name](n)
