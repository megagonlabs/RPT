from __future__ import annotations

import argparse
from datetime import date, datetime
import json
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
from urllib.request import urlopen

from datasets import load_dataset

from rpt.paths import HOTPOTQA_DATA_DIR, LIVEBENCH_MATH_DATA_DIR, XBRL_FORMULA_DATA_DIR


HOTPOTQA_SOURCE = ("hotpot_qa", "distractor", "validation")
HOTPOTQA_SPLITS = {
    "train": {"size": 300, "seed": 0},
    "test": {"size": 500, "seed": 1},
    "dev": {"size": 300, "seed": 2},
}

LIVEBENCH_MATH_SOURCE = ("livebench/math", "test")
LIVEBENCH_MATH_SPLIT_SEED = 0

ACE_FINANCE_RAW_BASE = "https://raw.githubusercontent.com/ace-agent/ace/main/eval/finance/data"
XBRL_FORMULA_FILES = {
    "train": "formula_train_subset_500.jsonl",
    "val": "formula_val_subset_300.jsonl",
    "test": "formula_test.jsonl",
}


def json_default(obj: Any) -> str:
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def write_jsonl(path: Path, items: Sequence[Dict[str, Any]], *, force: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not force:
        with path.open(encoding="utf-8") as f:
            n_rows = sum(1 for _ in f)
        print(f"exists: {path} ({n_rows} rows)")
        return
    with path.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False, default=json_default) + "\n")
    print(f"wrote:  {path} ({len(items)} rows)")


def write_bytes(path: Path, data: bytes, *, force: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not force:
        print(f"exists: {path}")
        return
    path.write_bytes(data)
    print(f"wrote:  {path}")


def sample_hotpotqa(
    source_rows: Sequence[Dict[str, Any]],
    *,
    sample_n: int,
    seed: int,
    excluding_ids: Optional[Sequence[str]] = None,
) -> List[Dict[str, Any]]:
    idxs = list(range(len(source_rows)))
    random.Random(seed).shuffle(idxs)
    excluded = set(excluding_ids or [])
    items: List[Dict[str, Any]] = []
    for idx in idxs:
        row = dict(source_rows[int(idx)])
        if row.get("id") in excluded:
            continue
        items.append(row)
        if len(items) >= sample_n:
            break
    if len(items) != sample_n:
        raise RuntimeError(f"Expected {sample_n} HotpotQA rows, got {len(items)}")
    return items


def ensure_disjoint(named_splits: Dict[str, Sequence[Dict[str, Any]]]) -> None:
    seen: Dict[str, str] = {}
    for split_name, rows in named_splits.items():
        for row in rows:
            row_id = str(row.get("id") or row.get("question_id"))
            if row_id in seen:
                raise RuntimeError(f"Duplicate row id {row_id} in {seen[row_id]} and {split_name}")
            seen[row_id] = split_name


def prepare_hotpotqa(out_dir: Path, *, force: bool) -> None:
    dataset_name, config_name, split_name = HOTPOTQA_SOURCE
    source = load_dataset(dataset_name, config_name, split=split_name)
    source_rows = [dict(row) for row in source]

    train = sample_hotpotqa(
        source_rows,
        sample_n=HOTPOTQA_SPLITS["train"]["size"],
        seed=HOTPOTQA_SPLITS["train"]["seed"],
    )
    train_ids = [row["id"] for row in train]
    test = sample_hotpotqa(
        source_rows,
        sample_n=HOTPOTQA_SPLITS["test"]["size"],
        seed=HOTPOTQA_SPLITS["test"]["seed"],
        excluding_ids=train_ids,
    )
    test_ids = [row["id"] for row in test]
    dev = sample_hotpotqa(
        source_rows,
        sample_n=HOTPOTQA_SPLITS["dev"]["size"],
        seed=HOTPOTQA_SPLITS["dev"]["seed"],
        excluding_ids=train_ids + test_ids,
    )

    ensure_disjoint({"train": train, "dev": dev, "test": test})
    write_jsonl(out_dir / "train.jsonl", train, force=force)
    write_jsonl(out_dir / "dev.jsonl", dev, force=force)
    write_jsonl(out_dir / "test.jsonl", test, force=force)


def extract_primary_turn(row: Dict[str, Any]) -> str:
    turns = row.get("turns")
    if isinstance(turns, str):
        return turns.strip()
    if isinstance(turns, Sequence) and turns:
        return str(turns[0]).strip()
    question = row.get("question")
    if isinstance(question, str):
        return question.strip()
    raise ValueError(f"Unable to extract question text for row {row.get('question_id') or row.get('id')}")


def extract_answer_format_hint(format_block: Optional[str]) -> Optional[str]:
    if not format_block:
        return None

    for pattern in [
        r"(?:^|\n)\s*Answer\s*:\s*(.+?)(?:\n|$)",
        r"(?:^|\n)\s*Exact Answer\s*:\s*(.+?)(?:\n|$)",
    ]:
        match = re.search(pattern, format_block, flags=re.IGNORECASE | re.DOTALL)
        if match:
            hint = match.group(1).strip().strip('"').strip("'")
            if hint:
                return hint

    compact = re.sub(r"\s+", " ", format_block).strip()
    compact = re.sub(r"<\s*Detailed reasoning\s*>", "", compact, flags=re.IGNORECASE).strip()
    compact = re.sub(r"^Answer\s*:\s*", "", compact, flags=re.IGNORECASE).strip()
    compact = compact.strip('"').strip("'").strip()
    for prefix in [
        r"^Your final answer should(?: be)?(?: STRICTLY)? in the format:\s*",
        r"^Return your final answer in the format:\s*",
        r"^Format your final answer as:\s*",
        r"^<\s*Detailed reasoning\s*>\s*",
        r"^Answer\s*:\s*",
    ]:
        compact = re.sub(prefix, "", compact, flags=re.IGNORECASE).strip()
    return compact or None


def split_question_and_format_hint(question: str) -> Tuple[str, Optional[str]]:
    patterns = [
        r"\n+\s*(Your final answer should be STRICTLY in the format:.*)\Z",
        r"\n+\s*(Your final answer should be in the format:.*)\Z",
        r"\n+\s*(Return your final answer in the format:.*)\Z",
        r"\n+\s*(Format your final answer as:.*)\Z",
    ]
    stripped_question = question.strip()
    format_block: Optional[str] = None
    for pattern in patterns:
        match = re.search(pattern, stripped_question, flags=re.IGNORECASE | re.DOTALL)
        if match:
            format_block = match.group(1).strip()
            stripped_question = stripped_question[: match.start()].strip()
            break
    return stripped_question, extract_answer_format_hint(format_block)


def normalize_livebench_math_row(row: Dict[str, Any]) -> Dict[str, Any]:
    normalized = dict(row)
    raw_question = extract_primary_turn(row)
    stripped_question, answer_format_hint = split_question_and_format_hint(raw_question)
    normalized["id"] = str(row.get("question_id") or row.get("id"))
    normalized["raw_question"] = raw_question
    normalized["question"] = stripped_question
    normalized["answer_format_hint"] = answer_format_hint
    normalized["answer"] = str(row.get("ground_truth", "")).strip()
    return normalized


def equal_split_counts(total_items: int) -> Tuple[int, int, int]:
    base, remainder = divmod(total_items, 3)
    counts = [base, base, base]
    for idx in range(remainder):
        counts[idx] += 1
    return counts[0], counts[1], counts[2]


def prepare_livebench_math(out_dir: Path, *, force: bool) -> None:
    dataset_name, split_name = LIVEBENCH_MATH_SOURCE
    rows = [normalize_livebench_math_row(dict(row)) for row in load_dataset(dataset_name, split=split_name)]
    random.Random(LIVEBENCH_MATH_SPLIT_SEED).shuffle(rows)
    n_train, n_val, _ = equal_split_counts(len(rows))
    train = rows[:n_train]
    val = rows[n_train : n_train + n_val]
    test = rows[n_train + n_val :]
    ensure_disjoint({"train": train, "val": val, "test": test})
    write_jsonl(out_dir / "train.jsonl", train, force=force)
    write_jsonl(out_dir / "val.jsonl", val, force=force)
    write_jsonl(out_dir / "test.jsonl", test, force=force)


def download_url(url: str) -> bytes:
    with urlopen(url) as response:
        return response.read()


def prepare_xbrl_formula(out_dir: Path, *, force: bool) -> None:
    for split_name, source_name in XBRL_FORMULA_FILES.items():
        url = f"{ACE_FINANCE_RAW_BASE}/{source_name}"
        write_bytes(out_dir / f"{split_name}.jsonl", download_url(url), force=force)


def resolve_output_dirs(data_root: Optional[Path]) -> Tuple[Path, Path, Path]:
    if data_root is None:
        return HOTPOTQA_DATA_DIR, LIVEBENCH_MATH_DATA_DIR, XBRL_FORMULA_DATA_DIR
    root = data_root.expanduser()
    return root / "hotpotqa", root / "livebench_math", root / "xbrl_formula"


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare cached JSONL splits for RPT without redistributing data files.")
    parser.add_argument(
        "--dataset",
        choices=["all", "hotpotqa", "livebench_math", "xbrl_formula"],
        default="all",
        help="Dataset to prepare.",
    )
    parser.add_argument("--data_root", type=Path, default=None, help="Optional output root. Defaults to RPT_DATA_ROOT or ./data.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing JSONL files.")
    args = parser.parse_args()

    hotpot_dir, livebench_dir, xbrl_dir = resolve_output_dirs(args.data_root)
    selected = {args.dataset} if args.dataset != "all" else {"hotpotqa", "livebench_math", "xbrl_formula"}

    if "hotpotqa" in selected:
        prepare_hotpotqa(hotpot_dir, force=args.force)
    if "livebench_math" in selected:
        prepare_livebench_math(livebench_dir, force=args.force)
    if "xbrl_formula" in selected:
        prepare_xbrl_formula(xbrl_dir, force=args.force)

    print("Data preparation complete.")


if __name__ == "__main__":
    main()
