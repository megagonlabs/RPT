from __future__ import annotations

from collections import defaultdict
import json
import os
from typing import Any, Dict, List, Optional, Tuple

from google import genai
from google.genai import types as genai_types

from .common import extract_json_object_text


def make_gemini_client(project: Optional[str], location: str) -> genai.Client:
    project_id = str(project or "").strip() or None
    if project_id:
        return genai.Client(
            vertexai=True,
            project=project_id,
            location=location,
            http_options=genai_types.HttpOptions(api_version="v1"),
        )
    return genai.Client(http_options=genai_types.HttpOptions(api_version="v1"))


def thinking_config(level: str) -> genai_types.ThinkingConfig:
    thinking_level = getattr(genai_types.ThinkingLevel, level.upper())
    return genai_types.ThinkingConfig(thinking_level=thinking_level)


def high_thinking_config() -> genai_types.ThinkingConfig:
    return thinking_config("HIGH")


def medium_thinking_config() -> genai_types.ThinkingConfig:
    return thinking_config("MEDIUM")


def parse_gemini_structured_response(response: Any, schema_model: Any) -> Any:
    parsed = getattr(response, "parsed", None)
    if parsed is not None:
        if isinstance(parsed, schema_model):
            return parsed
        return schema_model.model_validate(parsed)

    raw_text = str(getattr(response, "text", "") or "").strip()
    json_text = extract_json_object_text(raw_text)
    if not json_text:
        raise ValueError(f"Could not parse Gemini JSON response: {raw_text[:500]}")
    return schema_model.model_validate(json.loads(json_text))


def build_cleaned_log_path(log_path: str) -> str:
    root, ext = os.path.splitext(log_path)
    if ext:
        return f"{root}_cleaned{ext}"
    return f"{log_path}_cleaned"


def write_cleaned_log(
    log_path: str,
    cleaned_path: Optional[str] = None,
    *,
    required_events: set[str],
    final_test_event: str = "final_test_stats",
) -> str:
    if not os.path.exists(log_path):
        raise FileNotFoundError(f"Log not found: {log_path}")

    passthrough_records: List[Tuple[int, Dict[str, Any]]] = []
    pre_iter_records: Dict[int, List[Tuple[int, Dict[str, Any]]]] = defaultdict(list)
    attempts_by_step: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    current_attempt_idx: Dict[int, Optional[int]] = {}
    final_test_records: List[Tuple[int, Dict[str, Any]]] = []

    with open(log_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            text = line.strip()
            if not text:
                continue
            rec = json.loads(text)
            step = rec.get("step")
            event = rec.get("event")

            if event == final_test_event:
                final_test_records.append((idx, rec))
                continue

            if not isinstance(step, int) or step <= 0:
                passthrough_records.append((idx, rec))
                continue

            if event == "iter_prompt":
                attempt = {
                    "records": list(pre_iter_records.pop(step, [])),
                    "events": set(),
                }
                attempt["records"].append((idx, rec))
                attempt["events"].add(event)
                attempts_by_step[step].append(attempt)
                current_attempt_idx[step] = len(attempts_by_step[step]) - 1
                continue

            active_idx = current_attempt_idx.get(step)
            if active_idx is None:
                pre_iter_records[step].append((idx, rec))
                continue

            attempt = attempts_by_step[step][active_idx]
            attempt["records"].append((idx, rec))
            attempt["events"].add(event)
            if event == "decision":
                current_attempt_idx[step] = None

    selected_records: List[Tuple[int, Dict[str, Any]]] = list(passthrough_records)
    completed_steps = 0
    for step in range(1, max(attempts_by_step.keys(), default=0) + 1):
        attempts = attempts_by_step.get(step) or []
        chosen = None
        for attempt in reversed(attempts):
            if required_events.issubset(attempt["events"]):
                chosen = attempt
                break
        if chosen is None:
            break
        selected_records.extend(chosen["records"])
        completed_steps = step

    if final_test_records:
        selected_records.append(final_test_records[-1])

    selected_records.sort(key=lambda item: item[0])
    cleaned_path = cleaned_path or build_cleaned_log_path(log_path)
    cleaned_dir = os.path.dirname(cleaned_path)
    if cleaned_dir:
        os.makedirs(cleaned_dir, exist_ok=True)
    with open(cleaned_path, "w", encoding="utf-8") as out:
        for _, rec in selected_records:
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Wrote cleaned log with {completed_steps} completed iteration(s) to {cleaned_path}")
    return cleaned_path

