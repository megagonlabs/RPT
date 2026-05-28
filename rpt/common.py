from __future__ import annotations

from datetime import date, datetime
import json
import os
from pathlib import Path
import random
import re
from threading import Lock
import time
from typing import Any, Dict, Iterable, List, Optional


def ensure_parent_dir(path: str | Path) -> None:
    parent = Path(path).expanduser().parent
    if str(parent):
        parent.mkdir(parents=True, exist_ok=True)


def json_default(obj: Any) -> Any:
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()

    item = getattr(obj, "item", None)
    if callable(item):
        try:
            return item()
        except Exception:
            pass

    isoformat = getattr(obj, "isoformat", None)
    if callable(isoformat):
        try:
            return isoformat()
        except Exception:
            pass

    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")


class JsonlLogger:
    def __init__(self, path: str | Path):
        self.path = str(path)
        self._lock = Lock()
        log_dir = os.path.dirname(self.path)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

    def log(self, event: str, step: int, payload: Dict[str, Any]) -> None:
        rec = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "event": event,
            "step": step,
            "payload": payload,
        }
        with self._lock:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False, default=json_default) + "\n")


def load_jsonl(path: str | Path, sample_n: Optional[int] = None, seed: int = 0) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            text = line.strip()
            if text:
                items.append(json.loads(text))

    if sample_n is not None and 0 < sample_n < len(items):
        rnd = random.Random(seed)
        rnd.shuffle(items)
        items = items[:sample_n]
    return items


def write_jsonl(path: str | Path, items: Iterable[Dict[str, Any]]) -> None:
    ensure_parent_dir(path)
    with open(path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False, default=json_default) + "\n")


def extract_json_object_text(raw_text: str) -> Optional[str]:
    text = str(raw_text or "").strip()
    if not text:
        return None

    if text.startswith("```"):
        fence_match = re.match(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.DOTALL | re.IGNORECASE)
        if fence_match:
            text = fence_match.group(1).strip()

    try:
        json.loads(text)
        return text
    except Exception:
        pass

    decoder = json.JSONDecoder()
    for start_idx, ch in enumerate(text):
        if ch != "{":
            continue
        try:
            _, end_idx = decoder.raw_decode(text[start_idx:])
            return text[start_idx : start_idx + end_idx]
        except Exception:
            continue
    return None

