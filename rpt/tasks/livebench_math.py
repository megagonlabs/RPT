"""
Reflective Prompt Tuning (RPT) for LiveBench Math.

This script mirrors the existing RPT workflow in this repo, adapted to the
public Hugging Face LiveBench math subset.

Key choices:
- Loads `livebench/math` from Hugging Face.
- Uses the public `test` split as the source pool, then shuffles it with
  Python's `random.Random(split_seed)` and partitions it into train/val/test.
- Defaults `split_seed=0`, per the requested dataset shuffle.
- Uses task-specific scoring aligned with the public LiveBench math result
  processors, while keeping RPT-specific logging, confidence, and critique
  tracking.

Requirements:
  pip install openai datasets pydantic tqdm

Env:
  export OPENAI_API_KEY="..."

Example:
  python -m rpt.tasks.livebench_math --prepare_only
  python -m rpt.tasks.livebench_math --iters 10 --target_model gpt-4.1
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from decimal import Decimal, InvalidOperation
from fractions import Fraction
import json
import math
import multiprocessing as mp
import os
import random
import re
import traceback
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple
import warnings

from datasets import load_dataset
from openai import OpenAI
from pydantic import BaseModel, Field
from tqdm import tqdm

import lark
import sympy
from sympy.parsing.latex import parse_latex

from rpt.common import JsonlLogger, extract_json_object_text, json_default
from rpt.paths import LIVEBENCH_MATH_DATA_DIR

try:
    from rpt.analysis.cluster_fusion import ClusterFusionConfig, PartitionConfig, run_clusterfusion

    HAS_CLUSTER_FUSION = True
except Exception:
    HAS_CLUSTER_FUSION = False
    ClusterFusionConfig = None
    PartitionConfig = None
    run_clusterfusion = None


client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

LIVEBENCH_MATH_DATASET = "livebench/math"
LIVEBENCH_MATH_SOURCE_SPLIT = "test"
DEFAULT_SPLIT_SEED = 0
DEFAULT_DATA_DIR = str(LIVEBENCH_MATH_DATA_DIR)
TARGET_REQUEST_TIMEOUT_SECS = 180.0
SCORING_TIMEOUT_SECS = 90
TIMEOUT_SUBPROCESS_TASKS = {"AMPS_Hard", "integrals_with_game"}
TIMEOUT_MP_START_METHOD = (
    "forkserver" if "forkserver" in mp.get_all_start_methods() else "spawn"
)
class TargetAnswer(BaseModel):
    answer: str
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Model confidence that the final answer is correct.",
    )
    reasoning: str = Field(description="Brief reasoning for the answer.")


class PromptProgram(BaseModel):
    system: str
    instruction: str
    enforce_json_only: bool = True
    # max_reasoning_sentences: int = 4

    def render(self, question: str, answer_format_hint: Optional[str] = None) -> Tuple[str, str]:
        instruction_parts = [self.instruction.strip()]
        if answer_format_hint:
            instruction_parts.append(
                "For this question, the `answer` field must follow this format exactly:\n"
                f"{answer_format_hint}"
            )
        if self.enforce_json_only:
            instruction_parts.append("Output only valid JSON that matches the required schema.")
        # instruction_parts.append(
        #     f"Keep the reasoning to at most {self.max_reasoning_sentences} sentences."
        # )
        user = (
            "\n".join(part for part in instruction_parts if part).strip()
            + "\n\nQuestion:\n"
            + question
            + "\n"
        )
        return self.system, user

def parse_target_answer_from_text(raw_text: str) -> Optional["TargetAnswer"]:
    json_text = extract_json_object_text(raw_text)
    if not json_text:
        return None

    try:
        payload = json.loads(json_text)
    except Exception:
        return None

    if not isinstance(payload, dict):
        return None

    try:
        return TargetAnswer.model_validate(payload)
    except Exception:
        return None


class EvalMetrics(BaseModel):
    n: int
    task_score: float
    avg_confidence: float
    brier: float
    format_error_rate: float
    per_task_scores: Dict[str, float] = Field(default_factory=dict)


class FailureModeItem(BaseModel):
    label: str = Field(description="2-6 words, consistent across similar errors")
    definition: str = Field(description="Explanation of the failure mode")
    why: str = Field(description="Brief explanation for this example")
    basis: str = Field(description="Evidence from the trace showing the failure")


class FailureCritique(BaseModel):
    failure_modes: List[FailureModeItem] = Field(default_factory=list)


class FailureModeTopic(BaseModel):
    name: str
    definition: str
    examples: List[str] = Field(default_factory=list)


class TraceInsights(BaseModel):
    failure_modes: List[FailureModeTopic] = Field(default_factory=list)


class EvalItemTrace(BaseModel):
    idx: int
    row_id: str
    task: Optional[str] = None
    subtask: Optional[str] = None
    question: str
    gold: str
    pred: Optional[str]
    score: float
    full_credit: bool
    confidence: Optional[float]
    reasoning: Optional[str]
    error_type: str


class EvalReport(BaseModel):
    iteration: int = 0
    prompt_program: Dict[str, Any]
    metrics: EvalMetrics
    insights: Optional[TraceInsights] = None


class PromptPatch(BaseModel):
    system: Optional[str] = Field(
        default=None,
        description="Complete replacement system message. Omit to leave unchanged.",
    )
    instruction: Optional[str] = Field(
        default=None,
        description=(
            "Complete replacement instruction text for the target model. "
            "This will not be merged; the target model will see this text verbatim, "
            "so it must be clean target-facing prompt text rather than optimizer-facing "
            "changelog text. Do not include meta-edit language like 'revised', "
            "'targeted', 'this patch', 'observed failures', 'merge these edits', "
            "'apply this patch', 'apply the following', 'keep all other invariants unchanged', "
            "'preserve all other behavior', 'insert start', or 'replace the current rule'."
        ),
    )
    enforce_json_only: Optional[bool] = Field(
        default=None,
        description="JSON-only output contract flag. Omit to leave unchanged.",
    )
    # max_reasoning_sentences: Optional[int] = None
    rationale: str = Field(
        description=(
            "Explanation for the optimizer log only. Put meta-edit explanations here, "
            "not inside system or instruction."
        )
    )


class LoopDecision(BaseModel):
    action: Literal["patch", "stop"]
    patch: Optional[PromptPatch] = None
    stop_reason: Optional[str] = None


PATCH_META_MARKERS = (
    "apply this patch",
    "apply these targeted",
    "apply the following",
    "merge/override",
    "current promptprogram",
    "keep all other invariants unchanged",
    "keep all other behavior unchanged",
    "preserve all other behavior",
    "insert start",
    "insert end",
    "replace the current rule",
    "observed failures",
    "existing instruction text",
    "[revised",
    "targeted fixes",
    "merge these edits",
)

PATCH_DECISION_MAX_RETRIES = 5
PARSE_COLLAPSE_FORMAT_ERROR_RATE_THRESHOLD = 0.5
PARSE_COLLAPSE_FORMAT_ERROR_RATE_DELTA_THRESHOLD = 0.25


def load_jsonl_items(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def write_jsonl_items(path: str, items: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False, default=json_default) + "\n")


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


def extract_answer_format_hint(format_block: Optional[str]) -> Optional[str]:
    if not format_block:
        return None

    answer_only_patterns = [
        r"(?:^|\n)\s*Answer\s*:\s*(.+?)(?:\n|$)",
        r"(?:^|\n)\s*Exact Answer\s*:\s*(.+?)(?:\n|$)",
    ]
    for pattern in answer_only_patterns:
        match = re.search(pattern, format_block, flags=re.IGNORECASE | re.DOTALL)
        if match:
            hint = match.group(1).strip().strip('"').strip("'")
            if hint:
                return hint

    answer_line_match = re.search(
        r"(?:^|\n)\s*Answer\s*:\s*(.+?)(?:\n|$)",
        format_block,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if answer_line_match:
        hint = answer_line_match.group(1).strip()
        if hint:
            return hint

    exact_answer_line_match = re.search(
        r"(?:^|\n)\s*Exact Answer\s*:\s*(.+?)(?:\n|$)",
        format_block,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if exact_answer_line_match:
        hint = exact_answer_line_match.group(1).strip()
        if hint:
            return hint

    compact = re.sub(r"\s+", " ", format_block).strip()
    if not compact:
        return None

    compact = re.sub(
        r"<\s*Detailed reasoning\s*>",
        "",
        compact,
        flags=re.IGNORECASE,
    ).strip()
    compact = re.sub(
        r"^Answer\s*:\s*",
        "",
        compact,
        flags=re.IGNORECASE,
    ).strip()
    compact = compact.strip('"').strip("'").strip()

    compact = re.sub(
        r"^Your final answer should(?: be)?(?: STRICTLY)? in the format:\s*",
        "",
        compact,
        flags=re.IGNORECASE,
    )
    compact = re.sub(
        r"^Return your final answer in the format:\s*",
        "",
        compact,
        flags=re.IGNORECASE,
    )
    compact = re.sub(
        r"^Format your final answer as:\s*",
        "",
        compact,
        flags=re.IGNORECASE,
    )
    compact = re.sub(
        r"^<\s*Detailed reasoning\s*>\s*",
        "",
        compact,
        flags=re.IGNORECASE,
    ).strip()
    compact = re.sub(
        r"^Answer\s*:\s*",
        "",
        compact,
        flags=re.IGNORECASE,
    ).strip()
    compact = compact.strip()
    return compact or None


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


def load_livebench_math_items() -> List[Dict[str, Any]]:
    ds = load_dataset(LIVEBENCH_MATH_DATASET, split=LIVEBENCH_MATH_SOURCE_SPLIT)
    return [normalize_livebench_math_row(dict(row)) for row in ds]


def compute_equal_split_counts(total_items: int) -> Tuple[int, int, int]:
    base, remainder = divmod(total_items, 3)
    counts = [base, base, base]
    for i in range(remainder):
        counts[i] += 1
    return counts[0], counts[1], counts[2]


def partition_items_equally(
    items: List[Dict[str, Any]],
    split_seed: int) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    shuffled = list(items)
    random.Random(split_seed).shuffle(shuffled)
    n_train, n_val, n_test = compute_equal_split_counts(len(shuffled))
    train_end = n_train
    val_end = n_train + n_val
    return shuffled[:train_end], shuffled[train_end:val_end], shuffled[val_end:]


def ensure_disjoint_split_ids(named_splits: Dict[str, List[Dict[str, Any]]]) -> None:
    seen: Dict[str, str] = {}
    for split_name, items in named_splits.items():
        for item in items:
            row_id = str(item.get("id"))
            prev = seen.get(row_id)
            if prev is not None:
                raise ValueError(f"Found overlapping row id {row_id} in both {prev} and {split_name}.")
            seen[row_id] = split_name


def split_paths(base_dir: str, split_seed: int) -> Tuple[str, str, str]:
    train_path = os.path.join(base_dir, "train.jsonl")
    val_path = os.path.join(base_dir, "val.jsonl")
    test_path = os.path.join(base_dir, "test.jsonl")
    return train_path, val_path, test_path


def load_or_create_livebench_math_splits(
    *,
    base_dir: str = DEFAULT_DATA_DIR,
    split_seed: int = DEFAULT_SPLIT_SEED) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    train_path, val_path, test_path = split_paths(base_dir, split_seed)

    if all(os.path.exists(path) for path in (train_path, val_path, test_path)):
        train_items = load_jsonl_items(train_path)
        val_items = load_jsonl_items(val_path)
        test_items = load_jsonl_items(test_path)
        ensure_disjoint_split_ids(
            {
                "train": train_items,
                "val": val_items,
                "test": test_items,
            }
        )
        return train_items, val_items, test_items

    all_items = load_livebench_math_items()
    train_items, val_items, test_items = partition_items_equally(all_items, split_seed=split_seed)
    ensure_disjoint_split_ids(
        {
            "train": train_items,
            "val": val_items,
            "test": test_items,
        }
    )
    write_jsonl_items(train_path, train_items)
    write_jsonl_items(val_path, val_items)
    write_jsonl_items(test_path, test_items)
    return train_items, val_items, test_items


def normalize_ws(text: str) -> str:
    return " ".join(str(text).strip().split())


def _strip_markdown_wrappers(text: str) -> str:
    out = text.strip()
    out = out.strip("`")
    while out.startswith("**") and out.endswith("**") and len(out) >= 4:
        out = out[2:-2].strip()
    while out.startswith("*") and out.endswith("*") and len(out) >= 2:
        out = out[1:-1].strip()
    return out


def _strip_answer_prefix(text: str) -> str:
    return re.sub(
        r"^\s*(?:final\s+answer|exact\s+answer|answer)\s*[:\-]\s*",
        "",
        text,
        flags=re.IGNORECASE,
    ).strip()


def _extract_latex_command_argument(text: str, command: str) -> Optional[str]:
    marker = f"\\{command}" + "{"
    start = text.rfind(marker)
    if start == -1:
        return None
    i = start + len(marker)
    depth = 1
    chars: List[str] = []
    while i < len(text):
        ch = text[i]
        if ch == "{":
            depth += 1
            chars.append(ch)
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return "".join(chars).strip()
            chars.append(ch)
        else:
            chars.append(ch)
        i += 1
    return None


def _strip_outer_math_delimiters(text: str) -> str:
    out = text.strip()
    wrappers = [
        ("$$", "$$"),
        ("$", "$"),
        ("\\(", "\\)"),
        ("\\[", "\\]"),
        ("(", ")"),
        ("[", "]"),
    ]
    changed = True
    while changed:
        changed = False
        for left, right in wrappers:
            if out.startswith(left) and out.endswith(right) and len(out) > len(left) + len(right):
                out = out[len(left) : -len(right)].strip()
                changed = True
    return out


def canonicalize_answer(text: Optional[str]) -> str:
    if text is None:
        return ""

    out = normalize_ws(text)
    out = _strip_markdown_wrappers(out)
    out = _strip_answer_prefix(out)

    for command in ("boxed", "fbox"):
        extracted = _extract_latex_command_argument(out, command)
        if extracted:
            out = extracted

    out = out.replace("\\left", "").replace("\\right", "")
    out = out.replace("−", "-").replace("–", "-")
    out = _strip_outer_math_delimiters(out)
    out = out.rstrip(".")
    out = normalize_ws(out)
    return out.casefold()


def last_boxed_only_string(string: str) -> Optional[str]:
    idx = string.rfind("\\boxed")

    if "\\boxed " in string:
        return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        return None
    return string[idx : right_brace_idx + 1].replace("$", "").replace("fbox", "boxed")


def remove_boxed(s: str) -> str:
    if "\\boxed " in s:
        left = "\\boxed "
        if s.startswith(left):
            return s[len(left) :]
    left = "\\boxed{"
    if s.startswith(left) and s.endswith("}"):
        return s[len(left) : -1]
    return s


def parse_fraction_like(text: str) -> Optional[float]:
    match = re.fullmatch(r"([+-]?\d+)\s*/\s*([+-]?\d+)", text)
    if not match:
        return None
    numerator = int(match.group(1))
    denominator = int(match.group(2))
    if denominator == 0:
        return None
    return float(Fraction(numerator, denominator))


def parse_latex_fraction(text: str) -> Optional[float]:
    match = re.fullmatch(r"\\frac\s*\{([+-]?\d+)\}\s*\{([+-]?\d+)\}", text)
    if not match:
        return None
    numerator = int(match.group(1))
    denominator = int(match.group(2))
    if denominator == 0:
        return None
    return float(Fraction(numerator, denominator))


def parse_numeric_answer(text: str) -> Optional[float]:
    candidate = canonicalize_answer(text)
    if not candidate:
        return None

    fraction_value = parse_fraction_like(candidate)
    if fraction_value is not None:
        return fraction_value

    latex_fraction_value = parse_latex_fraction(candidate)
    if latex_fraction_value is not None:
        return latex_fraction_value

    numeric_candidate = candidate.replace(",", "")
    if "%" in numeric_candidate:
        return None
    if not re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[+-]?\d+)?", numeric_candidate):
        return None
    try:
        return float(Decimal(numeric_candidate))
    except (InvalidOperation, ValueError):
        return None


def split_answer_list(text: str) -> Optional[List[str]]:
    if ";" in text:
        parts = [part.strip() for part in text.split(";")]
    elif "," in text:
        parts = [part.strip() for part in text.split(",")]
    else:
        return None
    if len(parts) <= 1:
        return None
    return parts


def _run_with_timeout_worker(func: Any, args: Tuple[Any, ...], conn: Any) -> None:
    try:
        conn.send((True, func(*args)))
    except BaseException as exc:  # pragma: no cover - pass through
        conn.send((False, f"{type(exc).__name__}: {exc}"))
    finally:
        conn.close()


def run_with_timeout(func: Any, args: Tuple[Any, ...] = (), timeout: int = 8) -> Any:
    # Avoid `fork` from ThreadPoolExecutor workers; it can strand child Python
    # processes with the same argv as the parent script.
    ctx = mp.get_context(TIMEOUT_MP_START_METHOD)
    parent_conn, child_conn = ctx.Pipe(duplex=False)
    process = ctx.Process(target=_run_with_timeout_worker, args=(func, args, child_conn))
    process.daemon = True

    try:
        process.start()
        child_conn.close()
        process.join(timeout)

        if process.is_alive():
            process.kill()
            process.join(5)
            raise TimeoutError("Operation timed out")

        if not parent_conn.poll(1):
            raise RuntimeError("Timed operation exited without returning a result")

        ok, payload = parent_conn.recv()
        if ok:
            return payload
        raise RuntimeError(f"Timed operation failed: {payload}")
    finally:
        parent_conn.close()
        if process.is_alive():
            process.kill()
            process.join(1)
        close = getattr(process, "close", None)
        if callable(close):
            close()


def can_spawn_timeout_worker() -> bool:
    return not bool(getattr(mp.current_process(), "daemon", False))


def timed_call(func: Any, args: Tuple[Any, ...] = (), timeout: int = 8) -> Any:
    # When scoring already runs inside a timeout worker (for example AMPS_Hard),
    # spawning another subprocess triggers "daemonic processes are not allowed
    # to have children". In that case, fall back to a direct call and let the
    # outer timeout guard the whole scoring path.
    if not can_spawn_timeout_worker():
        return func(*args)
    return run_with_timeout(func, args=args, timeout=timeout)


def normalize_final_answer(final_answer: str) -> str:
    final_answer = final_answer.split("=")[-1]
    final_answer = re.sub(r"(.*?)(\$)(.*?)(\$)(.*)", "$\\3$", final_answer)
    final_answer = re.sub(r"(\\text\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\textbf\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\overline\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\boxed\{)(.*)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(frac)([^{])(.)", "frac{\\2}{\\3}", final_answer)
    final_answer = re.sub(r"(sqrt)([^{\[])", "sqrt{\\2}", final_answer)
    final_answer = final_answer.replace("$", "")
    if final_answer.replace(",", "").isdigit():
        final_answer = final_answer.replace(",", "")
    return final_answer


def normalize_symbolic_latex(expr: str) -> str:
    expr = expr.strip()
    expr = expr.replace("\\left", "")
    expr = expr.replace("\\right", "")
    expr = expr.replace("\\!", "")
    expr = expr.replace("\\,", "")
    expr = expr.replace("\\;", "")
    expr = expr.replace("\\:", "")
    expr = expr.replace("\\quad", " ")
    expr = expr.replace("\\qquad", " ")
    expr = re.sub(r"\s+", " ", expr)

    # Add explicit multiplication for common implicit LaTeX forms that
    # SymPy's parser frequently rejects: `4\sqrt{5}i`, `2^{a} 5^{b}`,
    # `\sin(x)\cos(y)`, `\sqrt{3}(x-1)^2`, `28 x^3`, etc.
    patterns = [
        r"(?<=\d)(?=\\[A-Za-z]+)",
        r"(?<=\d)(?=[A-Za-z])",
        r"(?<=[\}\]])(?=\\[A-Za-z]+|[A-Za-z0-9(])",
        r"(?<=\))(?=\\[A-Za-z]+|[A-Za-z0-9(])",
        r"(?<=\d)\s+(?=\\[A-Za-z]+|\d|[A-Za-z(])",
        r"(?<=[\}\]])\s+(?=\\[A-Za-z]+|[A-Za-z0-9(])",
        r"(?<=\))\s+(?=\\[A-Za-z]+|[A-Za-z0-9(])",
    ]
    for pattern in patterns:
        expr = re.sub(pattern, "*", expr)
    return expr


def parse_symbolic_latex(expr: str) -> List[sympy.Expr]:
    expr = expr.strip()
    normalized_expr = normalize_symbolic_latex(expr)
    parse_candidates = [expr]
    if normalized_expr != expr:
        parse_candidates.append(normalized_expr)

    escaped_candidates: List[str] = []
    for candidate in parse_candidates:
        unescaped = candidate.replace("\\\\", "\\")
        if unescaped != candidate:
            escaped_candidates.append(unescaped)
    parse_candidates.extend(escaped_candidates)

    seen: set[str] = set()
    ordered_candidates: List[str] = []
    for candidate in parse_candidates:
        if candidate not in seen:
            ordered_candidates.append(candidate)
            seen.add(candidate)

    last_error: Optional[Exception] = None
    for candidate in ordered_candidates:
        try:
            parsed = parse_latex(candidate, backend="lark")
        except (sympy.SympifyError, TypeError, Exception) as exc:
            last_error = exc
            try:
                parsed = parse_latex(candidate)
            except Exception as fallback_exc:
                last_error = fallback_exc
                continue
        if isinstance(parsed, lark.Tree):
            return list(parsed.children)
        return [parsed]

    try:
        raise last_error if last_error is not None else ValueError("no parse candidates")
    except Exception:
        warnings.warn(f"couldn't parse {expr}")
        return []


def symbolic_is_equiv(x1: str, x2: str) -> bool:
    try:
        parsed_x1s = parse_symbolic_latex(x1)
        parsed_x2s = parse_symbolic_latex(x2)
        if not parsed_x1s or not parsed_x2s:
            return False

        errors: List[str] = []
        for parsed_x1 in parsed_x1s:
            for parsed_x2 in parsed_x2s:
                try:
                    diff = parsed_x1 - parsed_x2
                except Exception as exc:
                    errors.append(f"couldn't subtract {x1} and {x2}: {exc}")
                    continue
                try:
                    simplified_diff = timed_call(sympy.simplify, args=(diff,), timeout=60)
                    if simplified_diff == 0:
                        return True
                    if hasattr(simplified_diff, "equals") and simplified_diff.equals(0):
                        return True
                except Exception as exc:
                    errors.append(f"couldn't compare simplified {x1} - {x2} with 0: {exc}")
                    continue
                try:
                    free_symbols = getattr(simplified_diff, "free_symbols", set())
                    if not free_symbols:
                        numeric_diff = timed_call(sympy.N, args=(sympy.Abs(simplified_diff),), timeout=60)
                        if float(numeric_diff) < 0.001:
                            return True
                except Exception as exc:
                    errors.append(f"Had some trouble simplifying when comparing {x1} and {x2}: {exc}")
        for error in errors:
            warnings.warn(error)
        return False
    except Exception as exc:  # pragma: no cover - defensive
        warnings.warn(f"Failed comparing {x1} and {x2}: {exc}")
        traceback.print_tb(exc.__traceback__)
        return False


def remove_nonnumeric_chars_at_ends(text: str) -> Tuple[str, int]:
    start_index = 0
    while start_index < len(text) and not text[start_index].isdigit():
        start_index += 1
    end_index = start_index
    while end_index < len(text) and text[end_index].isdigit():
        end_index += 1
    return text[start_index:end_index], len(text) - (end_index - start_index)


def extract_expression_completions_from_generation(generation: str, debug: bool = False) -> List[Any]:
    numbers: Optional[List[Any]] = None
    if "answer:" in generation.lower():
        lines = generation.lower().strip().split("\n")
        answer_line = None
        answer_index = None
        for i, line in enumerate(lines):
            if "answer:" in line:
                answer_line = line
                answer_index = i
        if answer_line is not None:
            answer_str = answer_line.split("answer:")[1].replace("answer:", "").replace("**", "").replace(".", "").strip()
            if answer_str == "" and answer_index is not None and answer_index < len(lines) - 1:
                answer_str = lines[answer_index + 1].replace("answer:", "").replace("**", "").replace(".", "").strip()
            numbers = []
            for item in answer_str.split(","):
                token = item.strip().split(" ")[-1].replace("$", "").replace("{", "").replace("}", "").replace("\\", "").replace("boxed", "").replace("<", "").replace(">", "")
                try:
                    numbers.append(int(token))
                except Exception:
                    if debug:
                        print("ERROR", token)
                    numbers.append("NO ANSWER")
            if len(numbers) == 0 or set(numbers) == {"NO ANSWER"}:
                numbers = None

    if numbers is None and "\\boxed" in generation:
        boxed = last_boxed_only_string(generation)
        string = remove_boxed(boxed) if boxed is not None else generation
        string = string.replace("\\text{", "").replace("}", "").replace("\\", "")
        numbers = []
        for item in string.strip().split(","):
            try:
                numbers.append(int(item.strip()))
            except Exception:
                numbers.append("NO ANSWER")
        if len(numbers) == 0 or set(numbers) == {"NO ANSWER"}:
            numbers = None

    if numbers is None:
        last_line = generation.strip().lower().split("\n")[-1]
        numbers = []
        for item in last_line.strip().split(","):
            token, _ = remove_nonnumeric_chars_at_ends(item)
            if len(token.strip()) == 0:
                continue
            try:
                numbers.append(int(token.strip()))
            except Exception:
                numbers.append("NO ANSWER")
        if len(numbers) == 0 or set(numbers) == {"NO ANSWER"}:
            numbers = None

    if numbers is None:
        split_string = "answer:"
        numbers = [part.strip() for part in generation.lower().split(split_string)[-1].split(",")]
        new_numbers: List[int] = []
        for i, token in enumerate(numbers):
            cleaned, num_removed = remove_nonnumeric_chars_at_ends(token)
            if cleaned and cleaned != "₂":
                new_numbers.append(int(cleaned))
            if i > 0 and num_removed > 0:
                break
        numbers = new_numbers

    return numbers or []


def mathcontest_extract_answer_value(question_text: str, letter: str) -> str:
    pattern = r'\\textbf{\(([A-E])\)\s?}(.*?)(?:\\qquad|\$)'
    matches = re.findall(pattern, question_text)
    answers = {match[0]: match[1].strip() for match in matches}
    answer = answers.get(letter, None)
    if not answer:
        return "FAILURE"
    answer = answer.strip().strip("$").strip("~")
    return answer


def mathcontest_score(ground_truth: str, answer_text: str, question_text: str) -> float:
    score = 0.0
    if not (isinstance(ground_truth, str) and len(ground_truth) == 1 and "A" <= ground_truth <= "E"):
        raise ValueError("ground_truth must be a single capital letter between A and E.")

    solution_matches = re.findall(r"<solution>(.*?)</solution>", answer_text)
    if solution_matches:
        solution_match = solution_matches[-1]
        if len(set(solution_match)) == 1 and next(iter(set(solution_match))).lower() == ground_truth.lower():
            score = 1.0

    if score == 0.0 and ground_truth * 4 in answer_text:
        score = 1.0

    if score == 0.0:
        normalized_answer = answer_text.replace("\\\\fbox{", "\\\\boxed{")
        last_boxed = last_boxed_only_string(normalized_answer)
        if last_boxed:
            boxed_res = remove_boxed(last_boxed).replace("\\text{", "").replace("}", "").replace("\\", "").lower()
            if boxed_res == ground_truth.lower():
                score = 1.0

    if score == 0.0:
        option_value = mathcontest_extract_answer_value(question_text, ground_truth)
        length_to_check = 20 + len(option_value)
        if option_value in answer_text[-length_to_check:]:
            score = 1.0

    if score == 0.0:
        last_line = answer_text.strip().split("\n")[-1]
        if last_line.strip().replace("*", "").lower() == ground_truth.lower():
            score = 1.0
        elif "(" in last_line and ")" in last_line:
            val = last_line.split("(")[1].split(")")[0]
            if val.lower() == ground_truth.lower():
                score = 1.0

    return score


def aime_score(ground_truth: str, answer_text: str) -> float:
    score = 0.0

    solution_matches = re.findall(r"<solution>(.*?)</solution>", answer_text)
    if solution_matches:
        solution_match = solution_matches[-1]
        if len(set(solution_match)) == 1 and next(iter(set(solution_match))).lower() == ground_truth.lower():
            score = 1.0

    if score == 0.0 and ground_truth in answer_text[-50:]:
        score = 1.0

    return score


def proof_rearrangement_score(ground_truth: str, answer_text: str) -> float:
    ground_truth_numbers = [int(n) for n in ground_truth.split(",") if str(n).strip()]
    completions = extract_expression_completions_from_generation(answer_text, debug=False)
    match = [
        (completions[i] == ground_truth_numbers[i]) if i < len(ground_truth_numbers) else 0
        for i in range(len(completions))
    ]
    return float(sum(match) / len(match)) if len(match) > 0 else 0.0


def amps_hard_score(ground_truth: str, answer_text: str) -> float:
    retval = 0.0
    parsed_answer = None

    if isinstance(ground_truth, list):
        ground_truth = ground_truth[-1]

    answer_text = answer_text.replace("+C", "")
    answer_text = answer_text.replace("+ C", "")
    answer_text = answer_text.replace("+ c", "")
    answer_text = answer_text.replace("+c", "")
    answer_text = answer_text.replace("\\\\fbox{", "\\\\boxed{")
    answer_text = answer_text.replace("\\dfrac", "\\frac")
    answer_text = answer_text.replace("\\tfrac", "\\frac")
    answer_text = answer_text.replace("\\left", "")
    answer_text = answer_text.replace("\\right", "")
    answer_text = answer_text.replace("\\bigl", "")
    answer_text = answer_text.replace("\\bigr", "")
    answer_text = answer_text.replace("\\Bigl", "")
    answer_text = answer_text.replace("\\Bigr", "")
    answer_text = answer_text.replace("\\,", "")
    answer_text = answer_text.replace("\\;", "")
    answer_text = answer_text.replace("\n", "")
    answer_text = answer_text.replace("\\cdot", "*")

    ground_truth = ground_truth.replace("\\left", "")
    ground_truth = ground_truth.replace("\\right", "")
    ground_truth = ground_truth.replace(" ^", "^")
    ground_truth = ground_truth.replace("\\ ", "*")

    last_boxed = last_boxed_only_string(answer_text)
    if last_boxed:
        parsed_answer = normalize_final_answer(remove_boxed(last_boxed))

    if parsed_answer is None:
        last_line = answer_text.split("\n")[-1]
        if last_line.count("$") >= 2:
            close_pos = last_line.rfind("$")
            if close_pos > 0 and last_line[close_pos - 1] == "$":
                close_pos -= 1
            open_pos = last_line.rfind("$", 0, close_pos)
            math_text = last_line[open_pos + 1 : close_pos]
            if "=" in math_text:
                math_text = math_text.split("=")[-1].strip()
            elif "\\quad \\text{or} \\quad" in math_text:
                math_text = math_text.split("\\quad \\text{or} \\quad")[-1].strip()
            parsed_answer = normalize_final_answer(math_text)

    if parsed_answer is not None:
        try:
            if symbolic_is_equiv(ground_truth, parsed_answer):
                retval = 1.0
        except TimeoutError:
            warnings.warn("Timeout when comparing ground truth and parsed answer")
        except Exception as exc:
            warnings.warn(f"Error when comparing ground truth and parsed answer: {exc}")
    else:
        trimmed_answer = answer_text[:-1] if answer_text.endswith(".") else answer_text
        if ground_truth == trimmed_answer[-len(ground_truth) :]:
            retval = 1.0

    return retval


def extract_solution_tag(answer_text: str) -> Optional[str]:
    solution_matches = re.findall(r"<solution>(.*?)</solution>", answer_text, re.IGNORECASE | re.DOTALL)
    if solution_matches:
        return solution_matches[-1].strip()
    return None


def normalize_integrals_answer(answer_text: str) -> str:
    answer_text = answer_text.strip()
    answer_text = answer_text.replace("\\\\", "\\")
    answer_text = answer_text.replace("\\dfrac", "\\frac")
    answer_text = answer_text.replace("\\tfrac", "\\frac")
    answer_text = answer_text.replace("\\left", "")
    answer_text = answer_text.replace("\\right", "")
    answer_text = answer_text.replace("\\bigl", "")
    answer_text = answer_text.replace("\\bigr", "")
    answer_text = answer_text.replace("\\Bigl", "")
    answer_text = answer_text.replace("\\Bigr", "")
    answer_text = answer_text.replace("\\cdot", "*")
    answer_text = answer_text.replace("\\,", "")
    answer_text = answer_text.replace("\\;", "")
    answer_text = answer_text.replace("\n", "")
    answer_text = answer_text.replace(" ", "")
    return answer_text


def parse_integrals_answer(answer_text: str) -> Optional[sympy.Expr]:
    answer_text = normalize_integrals_answer(answer_text)

    try:
        if "/" in answer_text and "\\" not in answer_text:
            parts = answer_text.split("/")
            if len(parts) == 2:
                return sympy.Rational(int(parts[0]), int(parts[1]))
    except (ValueError, TypeError):
        pass

    try:
        return sympy.Integer(int(answer_text))
    except (ValueError, TypeError):
        pass

    try:
        return sympy.Rational(answer_text).limit_denominator(10_000_000)
    except (ValueError, TypeError, sympy.SympifyError):
        pass

    frac_match = re.match(r"\\frac\{(\d+)\}\{(\d+)\}", answer_text)
    if frac_match:
        try:
            return sympy.Rational(int(frac_match.group(1)), int(frac_match.group(2)))
        except (ValueError, TypeError):
            pass

    try:
        return parse_latex(answer_text)
    except Exception:
        pass

    try:
        parsed = parse_latex(answer_text, backend="lark")
        if hasattr(parsed, "children"):
            parsed = parsed.children[0]
        return parsed
    except Exception:
        return None


def symbolic_expr_is_equiv(x1: sympy.Expr, x2: sympy.Expr) -> bool:
    try:
        diff = x1 - x2
        simplified = sympy.simplify(diff)
        if simplified == 0:
            return True
        if sympy.Abs(simplified).evalf() < 1e-5:
            return True
    except Exception as exc:
        warnings.warn(f"Error comparing expressions: {exc}")
    return False


def integrals_with_game_score(ground_truth: str, answer_text: str) -> float:
    score = 0.0
    parsed_model_answer = None

    gt_parsed = parse_integrals_answer(ground_truth)
    if gt_parsed is None:
        warnings.warn(f"Could not parse ground truth: {ground_truth}")
        return 0.0

    solution_text = extract_solution_tag(answer_text)
    if solution_text:
        parsed_model_answer = parse_integrals_answer(solution_text)
        if parsed_model_answer is not None and symbolic_expr_is_equiv(gt_parsed, parsed_model_answer):
            score = 1.0

    if score == 0.0:
        normalized_answer = answer_text.replace("\\\\fbox{", "\\\\boxed{")
        last_boxed = last_boxed_only_string(normalized_answer)
        if last_boxed:
            parsed_model_answer = parse_integrals_answer(remove_boxed(last_boxed))
            if parsed_model_answer is not None and symbolic_expr_is_equiv(gt_parsed, parsed_model_answer):
                score = 1.0

    if score == 0.0:
        last_part = answer_text[-200:] if len(answer_text) > 200 else answer_text
        if normalize_integrals_answer(ground_truth) in normalize_integrals_answer(last_part):
            score = 1.0

    return score


def scalar_answer_is_correct(predicted: str, ground_truth: str) -> bool:
    pred_norm = canonicalize_answer(predicted)
    gold_norm = canonicalize_answer(ground_truth)
    if not pred_norm or not gold_norm:
        return False

    if pred_norm == gold_norm:
        return True
    if pred_norm.replace(" ", "") == gold_norm.replace(" ", ""):
        return True

    pred_num = parse_numeric_answer(pred_norm)
    gold_num = parse_numeric_answer(gold_norm)
    if pred_num is not None and gold_num is not None:
        return math.isclose(pred_num, gold_num, rel_tol=1e-9, abs_tol=1e-9)

    return False


def livebench_math_answer_score(row: Dict[str, Any], predicted: Optional[str]) -> float:
    if predicted is None:
        return 0.0

    task = str(row.get("task", ""))
    subtask = str(row.get("subtask", ""))
    ground_truth = str(row.get("answer", ""))
    question_text = str(row.get("raw_question") or row.get("question") or "")

    if task == "math_comp":
        if len(ground_truth) == 1 and ground_truth in "ABCDE":
            return mathcontest_score(ground_truth, predicted, question_text)
        return aime_score(ground_truth, predicted)

    if task == "olympiad":
        return proof_rearrangement_score(ground_truth, predicted)

    if task == "AMPS_Hard":
        return amps_hard_score(ground_truth, predicted)

    if task == "integrals_with_game":
        return integrals_with_game_score(ground_truth, predicted)

    if scalar_answer_is_correct(predicted, ground_truth):
        return 1.0

    pred_parts = split_answer_list(canonicalize_answer(predicted))
    gold_parts = split_answer_list(canonicalize_answer(ground_truth))
    if pred_parts is not None and gold_parts is not None and len(pred_parts) == len(gold_parts):
        return 1.0 if all(scalar_answer_is_correct(p, g) for p, g in zip(pred_parts, gold_parts)) else 0.0

    return 0.0


def safe_livebench_math_answer_score(row: Dict[str, Any], predicted: Optional[str], timeout: int = SCORING_TIMEOUT_SECS) -> float:
    task = str(row.get("task", ""))
    if task not in TIMEOUT_SUBPROCESS_TASKS:
        return float(livebench_math_answer_score(row, predicted))
    return float(run_with_timeout(livebench_math_answer_score, args=(row, predicted), timeout=timeout))


def collect_livebench_math_task_scores(traces: Sequence[EvalItemTrace]) -> Dict[str, float]:
    scores_by_task: Dict[str, List[float]] = {}
    for trace in traces:
        task = trace.task or "unknown"
        scores_by_task.setdefault(task, []).append(float(trace.score))

    return {
        task: (sum(task_scores) / len(task_scores))
        for task, task_scores in sorted(scores_by_task.items())
        if task_scores
    }


def build_livebench_scoring_response(
    row: Dict[str, Any],
    parsed: Optional[TargetAnswer],
    raw_text: str,) -> str:
    if parsed is None:
        return str(raw_text or "").strip()

    reasoning = str(parsed.reasoning or "").strip()
    answer = str(parsed.answer or "").strip()
    task = str(row.get("task", "") or "")

    if task == "olympiad" and answer and "answer:" not in answer.lower():
        final_answer = f"Answer: {answer}"
    else:
        final_answer = answer

    parts = [part for part in [reasoning, final_answer] if part]
    if not parts:
        return str(raw_text or "").strip()
    return "\n\n".join(parts).strip()


def call_target_model(
    client: OpenAI,
    prompt: PromptProgram,
    question: str,
    answer_format_hint: Optional[str],
    model: str) -> Tuple[Optional[TargetAnswer], str, str]:
    system_text, user_text = prompt.render(question, answer_format_hint=answer_format_hint)
    request_input = [
        {"role": "system", "content": system_text},
        {"role": "user", "content": user_text},
    ]

    try:
        resp = client.responses.create(
            model=model,
            input=request_input,
            temperature=0.0,
            timeout=TARGET_REQUEST_TIMEOUT_SECS,
        )
        raw_text = str(getattr(resp, "output_text", "") or "").strip()
        parsed = parse_target_answer_from_text(raw_text)
        if parsed is None:
            return None, raw_text, "non_json"
        return parsed, raw_text, "ok"
    except Exception as exc:
        return None, f"[PARSE_ERROR] {exc}", "non_json"


def evaluate_single_item(
    client: OpenAI,
    prompt: PromptProgram,
    row: Dict[str, Any],
    idx: int,
    target_model: str) -> Dict[str, Any]:
    question = str(row["question"])
    gold = str(row["answer"])
    answer_format_hint = row.get("answer_format_hint")
    parsed, raw_text, err = call_target_model(
        client,
        prompt,
        question,
        answer_format_hint=str(answer_format_hint) if answer_format_hint else None,
        model=target_model,
    )

    if parsed is None:
        trace = EvalItemTrace(
            idx=idx,
            row_id=str(row.get("id", idx)),
            task=str(row.get("task", "")) or None,
            subtask=str(row.get("subtask", "")) or None,
            question=question,
            gold=gold,
            pred=None,
            score=0.0,
            full_credit=False,
            confidence=None,
            reasoning=None,
            error_type=err,
        )
    else:
        pred = parsed.answer
        scoring_response = build_livebench_scoring_response(row, parsed, raw_text)
        try:
            answer_score = safe_livebench_math_answer_score(row, scoring_response)
        except Exception:
            answer_score = 0.0
            err = "score_timeout"
        trace = EvalItemTrace(
            idx=idx,
            row_id=str(row.get("id", idx)),
            task=str(row.get("task", "")) or None,
            subtask=str(row.get("subtask", "")) or None,
            question=question,
            gold=gold,
            pred=pred,
            score=answer_score,
            full_credit=answer_score >= 1.0 - 1e-12,
            confidence=float(parsed.confidence),
            reasoning=parsed.reasoning,
            error_type=err,
        )

    if parsed is None and raw_text and not raw_text.startswith("[PARSE_ERROR]"):
        try:
            raw_score = safe_livebench_math_answer_score(row, raw_text)
            trace.score = raw_score
            trace.full_credit = raw_score >= 1.0 - 1e-12
        except Exception:
            trace.error_type = "score_timeout"

    return {
        "idx": idx,
        "row": row,
        "parsed": parsed,
        "raw_text": raw_text,
        "trace": trace,
        "err": err,
    }


def critique_one_trace_with_gpt5(
    client: OpenAI,
    trace: Dict[str, Any],
    model: str = "gpt-5",
) -> FailureCritique:
    critic_system = f"""You are a strict evaluation critic for math failures.
You are given one failed model attempt with:
- task metadata
- question
- gold answer
- predicted answer
- model confidence
- model reasoning

Your goal is to diagnose why the model failed.

Instructions:
1) Produce 1-3 failure_modes with:
   - label: 2-6 words, consistent across similar errors
   - definition: comprehensive explanation of the failure mode
   - why: brief, self-contained explanation for THIS example
   - basis: cite what in the trace shows this
2) Make labels concrete and clusterable.
3) If you cannot identify a clear failure mode, return an empty list.
4) Output only valid JSON matching the schema.
"""
    resp = client.responses.parse(
        model=model,
        input=[
            {"role": "system", "content": critic_system},
            {"role": "user", "content": json.dumps(trace, ensure_ascii=False)},
        ],
        text_format=FailureCritique,
    )
    return resp.output_parsed


def critique_single_trace(
    client: OpenAI,
    row: Dict[str, Any],
    trace: EvalItemTrace,
    optimizer_name: str,
) -> Dict[str, Any]:
    crit = critique_one_trace_with_gpt5(
        client,
        trace.model_dump(),
        model=optimizer_name,
    )
    return {
        "idx": trace.idx,
        "row_id": row.get("id"),
        "question": row.get("question", ""),
        "critique": crit,
        "impact": 1.0 - float(trace.score),
    }


def evaluate_prompt_tool(
    client: OpenAI,
    prompt: PromptProgram,
    items: List[Dict[str, Any]],
    logger: JsonlLogger,
    *,
    target_model: str,
    step: int = 0,
    mode: str = "train",
    k_topics: int = 10,
    clustering_sample_size: int = 100,
    optimizer_name: str = "gpt-5",
    seed: int = 0,
    max_workers: int = 20,
    enable_critiques: bool = True,) -> Dict[str, Any]:
    traces: List[EvalItemTrace] = []
    failure_labels: List[Dict[str, Any]] = []
    format_errors = 0
    confs: List[float] = []
    brier_terms: List[float] = []
    eval_results: List[Dict[str, Any]] = []
    incorrect_eval_results: List[Dict[str, Any]] = []
    max_workers = max(1, min(max_workers, len(items) if items else 1))

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                evaluate_single_item,
                client,
                prompt,
                row,
                idx,
                target_model,
            )
            for idx, row in enumerate(items)
        ]
        for future in tqdm(as_completed(futures), total=len(futures), desc=f"Evaluating {mode} items"):
            eval_results.append(future.result())

    eval_results.sort(key=lambda item: item["idx"])

    for result in eval_results:
        row = result["row"]
        parsed = result["parsed"]
        trace = result["trace"]
        traces.append(trace)
        logger.log(
            "target_trace",
            step,
            {
                "idx": trace.idx,
                "row_id": trace.row_id,
                "task": trace.task,
                "subtask": trace.subtask,
                "question": trace.question,
                "gold": trace.gold,
                "pred": parsed.answer if parsed else None,
                "confidence": float(parsed.confidence) if parsed else None,
                "reasoning": parsed.reasoning if parsed else None,
                "error": result["err"],
                "score": float(trace.score),
                "full_credit": trace.full_credit,
            },
        )

        if parsed is None:
            format_errors += 1
            continue

        conf = float(trace.confidence or 0.0)
        confs.append(conf)
        brier_terms.append((conf - float(trace.score)) ** 2)

        if mode == "train" and enable_critiques and trace.score < 1.0 - 1e-12:
            incorrect_eval_results.append(result)

    if mode == "train" and enable_critiques and incorrect_eval_results:
        critique_results: List[Dict[str, Any]] = []
        critique_workers = max(1, min(max_workers, len(incorrect_eval_results)))
        with ThreadPoolExecutor(max_workers=critique_workers) as executor:
            futures = [
                executor.submit(
                    critique_single_trace,
                    client,
                    result["row"],
                    result["trace"],
                    optimizer_name,
                )
                for result in incorrect_eval_results
            ]
            for future in tqdm(as_completed(futures), total=len(futures), desc="Critiquing train errors"):
                critique_results.append(future.result())

        critique_results.sort(key=lambda item: item["idx"])
        for critique_result in critique_results:
            crit = critique_result["critique"]
            logger.log(
                "item_critique",
                step,
                {
                    "idx": critique_result["idx"],
                    "row_id": critique_result["row_id"],
                    "question": critique_result["question"],
                    "critique": crit.model_dump(),
                },
            )
            start_id = len(failure_labels)
            for fm in crit.failure_modes:
                failure_labels.append(
                    {
                        "id": start_id,
                        "text": f"{fm.label}: {fm.definition}",
                        "example": fm.why,
                        "impact": critique_result["impact"],
                    }
                )
                start_id += 1

    n = len(traces)
    em = sum(float(trace.score) for trace in traces) / max(1, n)
    per_task_scores = collect_livebench_math_task_scores(traces)
    avg_conf = sum(confs) / max(1, len(confs))
    brier = sum(brier_terms) / max(1, len(brier_terms))
    fmt_rate = format_errors / max(1, n)

    report = EvalReport(
        iteration=step,
        prompt_program=prompt.model_dump(),
        metrics=EvalMetrics(
            n=n,
            task_score=em,
            avg_confidence=avg_conf,
            brier=brier,
            format_error_rate=fmt_rate,
            per_task_scores=per_task_scores,
        ),
        insights=None,
    )

    if mode in {"val", "test"}:
        return report.model_dump()

    if HAS_CLUSTER_FUSION and failure_labels:
        clustering_cfg = ClusterFusionConfig(
            k_topics=k_topics,
            partition=PartitionConfig(
                num_groups=max(2, 2 * k_topics),
                sample_size=clustering_sample_size,
                seed=seed,
                cosine_order=True,
            ),
            domain_guidance=(
                # "You will receive short failure-mode labels produced during iterative prompt tuning "
                # "for a math QA task. Each record describes a recurring failure pattern."
                "You will receive short failure-mode labels produced by an iterative prompt optimization method. "
                "Each record describes a recurring failure pattern in model behavior."
            ),
            feature_context="failure modes",
            text_field="text",
            topic_desc_mode="comprehensive",
        )
        topics = run_clusterfusion(failure_labels, clustering_cfg, get_topics=True)
        logger.log("failure_mode_clusters", step, {"topics": topics})
        selected_topics = [t for t in topics if t.get("prevalence", 0.0) >= 0.10] or topics[: min(len(topics), 3)]
        for topic in selected_topics:
            topic.pop("prevalence", None)
            topic.pop("topic_id", None)
        report.insights = TraceInsights(failure_modes=[FailureModeTopic(**topic) for topic in selected_topics])
    elif failure_labels:
        examples = [item["text"] for item in failure_labels[: min(5, len(failure_labels))]]
        report.insights = TraceInsights(
            failure_modes=[
                FailureModeTopic(
                    name="uncategorized failures",
                    definition="Representative failure labels collected without clustering.",
                    examples=examples,
                )
            ]
        )

    return report.model_dump()


def detect_parse_collapse(
    train_report: Dict[str, Any],
    previous_train_report: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    current_fmt = float(train_report["metrics"]["format_error_rate"])
    previous_fmt = (
        float(previous_train_report["metrics"]["format_error_rate"])
        if previous_train_report is not None
        else None
    )

    reasons: List[str] = []
    if current_fmt >= PARSE_COLLAPSE_FORMAT_ERROR_RATE_THRESHOLD:
        reasons.append(
            "current format_error_rate "
            f"{current_fmt:.3f} >= {PARSE_COLLAPSE_FORMAT_ERROR_RATE_THRESHOLD:.3f}"
        )
    if previous_fmt is not None:
        delta = current_fmt - previous_fmt
        if delta >= PARSE_COLLAPSE_FORMAT_ERROR_RATE_DELTA_THRESHOLD:
            reasons.append(
                "format_error_rate delta "
                f"{delta:.3f} >= {PARSE_COLLAPSE_FORMAT_ERROR_RATE_DELTA_THRESHOLD:.3f}"
            )

    if not reasons:
        return None

    return {
        "current_format_error_rate": current_fmt,
        "previous_format_error_rate": previous_fmt,
        "reason": "; ".join(reasons),
    }


def validate_prompt_patch(patch: PromptPatch) -> Optional[str]:
    for field_name in ("system", "instruction"):
        value = getattr(patch, field_name)
        if not value:
            continue
        normalized = value.lower()
        for marker in PATCH_META_MARKERS:
            if marker in normalized:
                return (
                    f"{field_name} contains optimizer/meta-edit language marker {marker!r}; "
                    "patch fields must be clean standalone target-facing prompt text."
                )
    return None


def apply_patch(prompt: PromptProgram, patch: PromptPatch) -> PromptProgram:
    data = prompt.model_dump()
    updates = patch.model_dump(exclude_none=True)
    updates.pop("rationale", None)
    data.update(updates)
    return PromptProgram(**data)


def prompt_complexity_chars(prompt_program: Optional[Dict[str, Any]]) -> int:
    if not prompt_program:
        return 0
    system_text = str(prompt_program.get("system", "") or "")
    instruction_text = str(prompt_program.get("instruction", "") or "")
    return len(system_text) + len(instruction_text)


def score_report(
    report: Dict[str, Any],
    w_task: float = 1.0,
    w_brier: float = 0.05,
    w_fmt: float = 0.10,
    metric_name: str = "task_score",
    prompt_complexity_weight: float = 0.0,
    prompt_complexity_unit: float = 1000.0,
) -> float:
    metrics = report["metrics"]
    prompt_complexity_penalty = 0.0
    if prompt_complexity_weight > 0.0:
        prompt_complexity_penalty = (
            prompt_complexity_weight
            * prompt_complexity_chars(report.get("prompt_program"))
            / prompt_complexity_unit
        )
    return (
        w_task * float(metrics[metric_name])
        - w_brier * float(metrics["brier"])
        - prompt_complexity_penalty
        # - w_fmt * float(metrics["format_error_rate"])
    )


def better_score(a: float, b: float, eps: float = 1e-12) -> bool:
    return a > b + eps


def build_current_summary(
    history_reports: List[Dict[str, Any]],
    best_report: Optional[Dict[str, Any]],) -> Dict[str, Any]:
    current_report = history_reports[-1]
    current_metrics = current_report["metrics"]
    prev_metrics = history_reports[-2]["metrics"] if len(history_reports) >= 2 else None
    best_metrics = best_report["metrics"] if best_report else None

    current_score = score_report(current_report)
    prev_score = score_report(history_reports[-2]) if len(history_reports) >= 2 else None
    best_score = score_report(best_report) if best_report else None

    non_improving_streak = 0
    for report in reversed(history_reports):
        if best_report is not None and report["iteration"] == best_report["iteration"]:
            break
        non_improving_streak += 1

    summary = {
        "iteration": current_report["iteration"],
        "metrics": current_metrics,
        "insights": current_report.get("insights"),
        "best_so_far": {
            "iteration": best_report["iteration"] if best_report else None,
            "metrics": best_metrics,
            "score": best_score,
        },
        "delta_vs_previous": None,
        "delta_vs_best": None,
        "non_improving_streak": non_improving_streak,
        "did_last_patch_improve_train": None,
    }

    if prev_metrics is not None and prev_score is not None:
        summary["delta_vs_previous"] = {
            "task_score": float(current_metrics["task_score"] - prev_metrics["task_score"]),
            "avg_confidence": float(current_metrics["avg_confidence"] - prev_metrics["avg_confidence"]),
            "brier": float(current_metrics["brier"] - prev_metrics["brier"]),
            "format_error_rate": float(current_metrics["format_error_rate"] - prev_metrics["format_error_rate"]),
            "score": float(current_score - prev_score),
        }
        summary["did_last_patch_improve_train"] = bool(current_score > prev_score + 1e-12)

    if best_metrics is not None and best_score is not None:
        summary["delta_vs_best"] = {
            "task_score": float(current_metrics["task_score"] - best_metrics["task_score"]),
            "avg_confidence": float(current_metrics["avg_confidence"] - best_metrics["avg_confidence"]),
            "brier": float(current_metrics["brier"] - best_metrics["brier"]),
            "format_error_rate": float(current_metrics["format_error_rate"] - best_metrics["format_error_rate"]),
            "score": float(current_score - best_score),
        }

    return summary



OPTIMIZER_INSTRUCTIONS = (
        "You are the Reflective Prompt Tuning (RPT) controller.\n\n"
        "Your goal is to iteratively improve a PromptProgram for a math QA task.\n\n"
        "At each iteration you must:\n"
        "  (1) Call `evaluate_prompt` exactly once on the CURRENT PromptProgram.\n"
        "  (2) Read the returned evaluation report with insights.\n"
        "  (3) Output either a PATCH or STOP.\n\n"
        "Optimization target:\n"
        "  - Primary: improve task_score on the training split.\n"
        "  - Secondary: improve calibration (lower Brier / reduce overconfidence) without hurting task_score.\n\n"
        "Decision guidance:\n"
        "  - When current_summary is provided, use it as the primary decision signal, especially current_summary.metrics and any deltas vs previous/best.\n"
        "  - Use history only to detect trajectory, regressions, and previously ineffective edits.\n"
        "Patch constraints:\n"
        "  - Edits should be targeted to the failure modes, and designed to address their underlying issues with concrete guidance.\n"
        "  - Do not reduce failure modes to a short generic instruction; provide actionable steps.\n"
        "  - Prefer revising, merging, deleting, or reorganizing existing instructions over adding new broad rules.\n"
        "  - Keep the output contract stable (JSON schema and required fields).\n"
        "  - Avoid redundant or conflicting rules; consolidate instructions when possible.\n"
        "Stop condition:\n"
        "  - Output STOP if training-set performance has plateaued or further edits are unlikely to help.\n\n"
        "Hard rule:\n"
        "  - Do NOT propose a PATCH or STOP decision before calling `evaluate_prompt` and receiving its result."
)

def run_rpt(
    client: OpenAI,
    train_items: List[Dict[str, Any]],
    val_items: List[Dict[str, Any]],
    test_items: List[Dict[str, Any]],
    seed_prompt: PromptProgram,
    logger: JsonlLogger,
    *,
    target_model: str,
    iters: int = 5,
    mode: str = "all_reports",
    test_every: int = 5,
    clustering_sample_size: int = 100,
    k_topics: int = 10,
    k_reports: int = 5,
    seed: int = 0,
    eval_workers: int = 20,
    optimizer_name: str = "gpt-5",
    prompt_complexity_weight: float = 0.0,
    prompt_complexity_unit: float = 1000.0,
) -> PromptProgram:
    optimizer_instructions = OPTIMIZER_INSTRUCTIONS
    tools = [
        {
            "type": "function",
            "name": "evaluate_prompt",
            "description": (
                "Evaluate the current PromptProgram on the LiveBench-math training split "
                "using the target model. Returns an evaluation report with metrics and failure insights."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt_program": {
                        "type": "object",
                        "description": (
                            "The full PromptProgram to evaluate on the target model. "
                            "Fields: system, instruction, enforce_json_only."
                        ),
                        "properties": {
                            "system": {"type": "string"},
                            "instruction": {"type": "string"},
                            "enforce_json_only": {"type": "boolean"},
                        },
                    }
                },
                "required": ["prompt_program"],
            },
        }
    ]

    prompt = seed_prompt
    train_history_reports: List[Dict[str, Any]] = []
    best_train_report: Optional[Dict[str, Any]] = None

    best_prompt = prompt
    best_val_report: Optional[Dict[str, Any]] = None
    best_val_score = float("-inf")

    for t in range(iters):
        step = t + 1

        if step % test_every == 0 or step == 1:
            test_report = evaluate_prompt_tool(
                client,
                prompt,
                test_items,
                logger=logger,
                target_model=target_model,
                step=step,
                mode="test",
                max_workers=eval_workers,
                enable_critiques=False,
            )
            logger.log("test_stats", step, test_report)

        logger.log("iter_prompt", step, prompt.model_dump())
        print(f"\n=== RPT Iteration {step}/{iters} (mode={mode}) ===")
        input_list: List[Dict[str, Any]] = [
            {
                "role": "user",
                "content": (
                    f"Iteration {step}/{iters}\n\n"
                    "Call `evaluate_prompt` on the CURRENT PromptProgram below.\n"
                    "Use this JSON as evaluate_prompt.prompt_program:\n\n"
                    f"{prompt.model_dump_json(indent=2)}"
                ),
            }
        ]

        response = client.responses.parse(
            model=optimizer_name,
            input=input_list,
            instructions=optimizer_instructions,
            tools=tools,
        )
        input_list += response.output

        got_tool_call = False
        parse_collapse: Optional[Dict[str, Any]] = None
        for item in response.output:
            if getattr(item, "type", None) != "function_call" or item.name != "evaluate_prompt":
                continue

            got_tool_call = True
            train_report = evaluate_prompt_tool(
                client,
                prompt,
                train_items,
                logger=logger,
                target_model=target_model,
                step=step,
                mode="train",
                k_topics=k_topics,
                clustering_sample_size=clustering_sample_size,
                optimizer_name=optimizer_name,
                seed=seed,
                max_workers=eval_workers,
            )
            logger.log("train_stats", step, train_report)
            train_history_reports.append(train_report)
            previous_train_report = train_history_reports[-2] if len(train_history_reports) >= 2 else None
            parse_collapse = detect_parse_collapse(train_report, previous_train_report)

            input_list.append(
                {
                    "type": "function_call_output",
                    "call_id": item.call_id,
                    "output": json.dumps(train_report, ensure_ascii=False),
                }
            )

            train_score = score_report(
                train_report,
                prompt_complexity_weight=prompt_complexity_weight,
                prompt_complexity_unit=prompt_complexity_unit,
            )
            if best_train_report is None or better_score(
                train_score,
                score_report(
                    best_train_report,
                    prompt_complexity_weight=prompt_complexity_weight,
                    prompt_complexity_unit=prompt_complexity_unit,
                ),
            ):
                best_train_report = train_report

            val_report = evaluate_prompt_tool(
                client,
                prompt,
                val_items,
                logger=logger,
                target_model=target_model,
                step=step,
                mode="val",
                max_workers=eval_workers,
                enable_critiques=False,
            )
            logger.log("val_stats", step, val_report)

            val_score = score_report(
                val_report,
                prompt_complexity_weight=prompt_complexity_weight,
                prompt_complexity_unit=prompt_complexity_unit,
            )
            if better_score(val_score, best_val_score):
                best_val_score = val_score
                best_val_report = val_report
                best_prompt = prompt
                logger.log(
                    "best_update",
                    step,
                    {
                        "selection_split": "val",
                        "score": best_val_score,
                        "val_metrics": val_report["metrics"],
                        "train_metrics": train_report["metrics"],
                        "prompt_program": prompt.model_dump(),
                    },
                )

        if not got_tool_call:
            raise RuntimeError("Optimizer model did not call evaluate_prompt. Fix the controller prompt or tool schema.")

        current_summary = build_current_summary(train_history_reports, best_train_report)
        if mode == "last_report":
            decision_payload = {
                "mode": "last_report",
                "current_prompt_program": prompt.model_dump(),
                "current_summary": current_summary,
            }
            mode_hint = "You are only given the current iteration summary."
        elif mode == "all_reports":
            decision_payload = {
                "mode": "all_reports",
                "current_prompt_program": prompt.model_dump(),
                "history": train_history_reports,
            }
            mode_hint = "You are given PAST report history, plus a separate current_summary for the current iteration.\nUse history for trajectory and current_summary for the decision now.\n"
        elif mode == "history_summary":
            decision_payload = {
                "mode": "history_summary",
                "current_prompt_program": prompt.model_dump(),
                "history": train_history_reports[:-1],
                "current_summary": current_summary,
            }
            mode_hint = "You are given past report history plus a current summary."
        elif mode == "last_k_reports":
            decision_payload = {
                "mode": "last_k_reports",
                "current_prompt_program": prompt.model_dump(),
                "history": train_history_reports[-k_reports:],
            }
            mode_hint = f"You are given the last {k_reports} evaluation reports."
        else:
            raise ValueError(f"Unknown mode: {mode}")

        retry_feedback: List[str] = []
        if parse_collapse is not None:
            logger.log("parse_collapse_detected", step, parse_collapse)
            retry_feedback.append(
                "Hard failure signal: the current prompt caused parse collapse on the training split "
                f"({parse_collapse['reason']}). Return a corrective PATCH that restores clean parseable "
                "target outputs, or STOP if no safe correction exists."
            )

        decision: Optional[LoopDecision] = None
        for attempt in range(1, PATCH_DECISION_MAX_RETRIES + 1):
            decision_input = list(input_list)
            decision_content = (
                f"{mode_hint}\n"
                "Using the most recent function_call_output evaluation report above, now decide whether to STOP or output a PATCH.\n"
                "Return a LoopDecision JSON with fields:\n"
                "  - action: 'patch' or 'stop'\n"
                "  - patch (only if action='patch')\n"
                "  - stop_reason (only if action='stop')\n\n"
            )
            if retry_feedback:
                decision_content += (
                    "Guardrail feedback from previous decision attempts:\n"
                    + "\n".join(f"- {msg}" for msg in retry_feedback)
                    + "\n\n"
                )
            decision_content += json.dumps(decision_payload, ensure_ascii=False, indent=2)
            decision_input.append({"role": "user", "content": decision_content})

            decision_resp = client.responses.parse(
                model=optimizer_name,
                input=decision_input,
                instructions=optimizer_instructions,
                text_format=LoopDecision,
            )
            decision = decision_resp.output_parsed
            if attempt > 1 or retry_feedback:
                logger.log(
                    "decision_attempt",
                    step,
                    {
                        "attempt": attempt,
                        "decision": decision.model_dump(),
                        "guardrail_feedback": retry_feedback,
                    },
                )

            if decision.action == "stop":
                break

            if decision.action != "patch" or decision.patch is None:
                retry_feedback.append(
                    "Rejected previous decision because action='patch' requires a non-null patch, "
                    "or action must be 'stop'."
                )
                continue

            rejection_reason = validate_prompt_patch(decision.patch)
            if rejection_reason is None:
                break

            logger.log(
                "patch_rejected_meta_text",
                step,
                {
                    "attempt": attempt,
                    "reason": rejection_reason,
                    "patch": decision.patch.model_dump(),
                },
            )
            retry_feedback.append(
                f"Rejected previous PATCH: {rejection_reason} "
                "Return a clean standalone PromptPatch. Put edit explanations only in rationale."
            )
        else:
            decision = LoopDecision(
                action="stop",
                stop_reason=(
                    "Forced stop after repeated invalid or guardrail-violating decision attempts "
                    f"(max retries={PATCH_DECISION_MAX_RETRIES})."
                ),
            )
            logger.log("decision_forced_stop", step, decision.model_dump())
        logger.log("decision", step, decision.model_dump())

        if decision.action == "stop":
            print(f"[STOP] {decision.stop_reason}")
            break

        if decision.action == "patch" and decision.patch is not None:
            prompt = apply_patch(prompt, decision.patch)
            continue

        raise RuntimeError("Invalid LoopDecision from optimizer model.")

    print("\n=== FINAL PROMPTPROGRAM ===")
    print(best_prompt.model_dump_json(indent=2))
    print("Best val score:", best_val_score, "Best val metrics:", best_val_report["metrics"] if best_val_report else None)

    final_test_report = evaluate_prompt_tool(
        client,
        best_prompt,
        test_items,
        logger=logger,
        target_model=target_model,
        step=iters,
        mode="test",
        max_workers=eval_workers,
        enable_critiques=False,
    )
    logger.log("final_test_stats", iters, final_test_report)
    return best_prompt


def run_seed_prompt_evaluation(
    client: OpenAI,
    train_items: List[Dict[str, Any]],
    val_items: List[Dict[str, Any]],
    test_items: List[Dict[str, Any]],
    seed_prompt: PromptProgram,
    logger: JsonlLogger,
    *,
    target_model: str,
    eval_workers: int) -> Dict[str, Dict[str, Any]]:
    train_report = evaluate_prompt_tool(
        client,
        seed_prompt,
        train_items,
        logger=logger,
        target_model=target_model,
        step=1,
        mode="train",
        max_workers=eval_workers,
    )
    logger.log("train_stats", 1, train_report)

    val_report = evaluate_prompt_tool(
        client,
        seed_prompt,
        val_items,
        logger=logger,
        target_model=target_model,
        step=1,
        mode="val",
        max_workers=eval_workers,
        enable_critiques=False,
    )
    logger.log("val_stats", 1, val_report)

    test_report = evaluate_prompt_tool(
        client,
        seed_prompt,
        test_items,
        logger=logger,
        target_model=target_model,
        step=1,
        mode="test",
        max_workers=eval_workers,
        enable_critiques=False,
    )
    logger.log("test_stats", 1, test_report)

    return {
        "train": train_report,
        "val": val_report,
        "test": test_report,
    }


def make_seed_prompt() -> PromptProgram:
    return PromptProgram(
        system=(
            "Solve the math problem step by step and give the final answer in exactly the format requested by the question."
        ),
        instruction=(
            "Your output should be a JSON with fields:\n"
            "- reasoning: your chain of thought / reasoning / thinking process, detailed analysis and calculations.\n"
            "- answer: final answer in exactly the format requested by the question.\n"
            "- confidence: a number in [0,1] representing your confidence in the final answer."
        ),
        enforce_json_only=True,
        # max_reasoning_sentences=8,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0, help="Seed for optimizer-side randomness.")
    ap.add_argument("--split_seed", type=int, default=DEFAULT_SPLIT_SEED, help="Seed used to shuffle and partition the LiveBench math pool.")
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--mode", type=str, default="all_reports", choices=["all_reports", "history_summary", "last_report", "last_k_reports"])
    ap.add_argument("--k_reports", type=int, default=5)
    ap.add_argument("--k_topics", type=int, default=10)
    ap.add_argument("--clustering_sample_size", type=int, default=100)
    ap.add_argument("--test_every", type=int, default=5)
    ap.add_argument("--eval_workers", type=int, default=20)
    ap.add_argument(
        "--optimizer_name",
        type=str,
        default="gpt-5",
        help="Model name used for optimizer/controller and critique calls.",
    )
    ap.add_argument("--target_model", type=str, default="gpt-4.1", help="Model being prompt-tuned/evaluated.")
    ap.add_argument(
        "--prompt_complexity_weight",
        type=float,
        default=0.0,
        help=(
            "Penalty weight subtracted from selection score as "
            "weight * prompt_chars / prompt_complexity_unit."
        ),
    )
    ap.add_argument(
        "--prompt_complexity_unit",
        type=float,
        default=1000.0,
        help="Normalization unit for prompt complexity penalty, in characters.",
    )
    ap.add_argument("--data_dir", type=str, default=DEFAULT_DATA_DIR)
    ap.add_argument("--prepare_only", action="store_true", help="Create/load cached train/val/test splits and exit.")
    ap.add_argument("--evaluate_only", action="store_true", help="Evaluate the seed prompt on the cached splits and exit.")
    args = ap.parse_args()

    random.seed(args.seed)

    train_items, val_items, test_items = load_or_create_livebench_math_splits(
        base_dir=args.data_dir,
        split_seed=args.split_seed,
    )
    ensure_disjoint_split_ids(
        {
            "train": train_items,
            "val": val_items,
            "test": test_items,
        }
    )

    print(
        "Loaded LiveBench math splits: "
        f"{len(train_items)} train, {len(val_items)} val, {len(test_items)} test "
        f"(split_seed={args.split_seed})."
    )

    if args.prepare_only:
        return

    logger = JsonlLogger(
        os.path.join(
            "logs",
            "livebench_math",
            "openai",
            args.optimizer_name,
            (
                f"log_{args.mode}_iters_{args.iters}_train_{len(train_items)}_val_{len(val_items)}"
                f"_test_{len(test_items)}_split_seed_{args.split_seed}_seed_{args.seed}"
                f"_k_topics_{args.k_topics}_cluster_desc_comprehensive"
                f"_optimizer_non_minimal"
                f"_pcw_{args.prompt_complexity_weight:g}_pcu_{args.prompt_complexity_unit:g}.jsonl"
            ),
        )
    )

    seed_prompt = make_seed_prompt()

    if args.evaluate_only:
        reports = run_seed_prompt_evaluation(
            client,
            train_items,
            val_items,
            test_items,
            seed_prompt,
            logger,
            target_model=args.target_model,
            eval_workers=args.eval_workers,
        )
        print(json.dumps(reports, indent=2))
        return

    best_prompt = run_rpt(
        client,
        train_items,
        val_items,
        test_items,
        seed_prompt,
        logger,
        target_model=args.target_model,
        iters=args.iters,
        mode=args.mode,
        test_every=args.test_every,
        k_topics=args.k_topics,
        clustering_sample_size=args.clustering_sample_size,
        k_reports=args.k_reports,
        seed=args.seed,
        eval_workers=args.eval_workers,
        optimizer_name=args.optimizer_name,
        prompt_complexity_weight=args.prompt_complexity_weight,
        prompt_complexity_unit=args.prompt_complexity_unit,
    )

    print("\n=== FINAL PROMPTPROGRAM ===")
    print(best_prompt.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
