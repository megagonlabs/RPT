from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple


def add_jsonl_suffix(path: str | Path) -> Path:
    path = Path(path)
    if str(path).endswith(".jsonl"):
        return path
    return Path(f"{path}.jsonl")


def log_stem(path: str | Path) -> str:
    name = Path(path).name
    if name.endswith(".jsonl"):
        return name[:-len(".jsonl")]
    return name


def _contains_part(path: Path, part: str) -> bool:
    return part in path.parts


def _index_after_part(path: Path, part: str) -> Optional[str]:
    parts = path.parts
    try:
        idx = parts.index(part)
    except ValueError:
        return None
    if idx + 1 >= len(parts):
        return None
    return parts[idx + 1]


def infer_task_name_from_log_path(log_path: str | Path, logs_root: str | Path = "logs") -> Optional[str]:
    path = Path(log_path)
    task_name = _index_after_part(path, Path(logs_root).name)
    if task_name:
        return task_name
    if path.parent and path.parent != Path("."):
        return path.parent.name
    return None


def infer_model_name_from_log_path(log_path: str | Path, logs_root: str | Path = "logs") -> Optional[str]:
    path = Path(log_path)
    parts = path.parts
    logs_name = Path(logs_root).name
    if logs_name in parts:
        idx = parts.index(logs_name)
        rel_parts = parts[idx + 1 :]
        if len(rel_parts) >= 3:
            return rel_parts[-2]
        return None
    if path.parent and path.parent != Path("."):
        return path.parent.name
    return None


def resolve_log_path(
    log_ref: str | Path,
    task_name: Optional[str] = None,
    model_name: Optional[str] = None,
    logs_root: str | Path = "logs",
) -> Path:
    """Resolve a log reference that may be a full path, a logs/<task>/ stem, or a bare stem."""
    path = Path(log_ref)
    logs_root = Path(logs_root)
    with_suffix = add_jsonl_suffix(path)

    candidates: list[Path] = []

    def add(path_to_add: Path) -> None:
        if path_to_add not in candidates:
            candidates.append(path_to_add)

    if not path.is_absolute() and not _contains_part(path, logs_root.name):
        if task_name and len(path.parts) == 1:
            if model_name:
                add(logs_root / task_name / model_name / with_suffix.name)
            add(logs_root / task_name / with_suffix.name)
        elif len(path.parts) >= 2:
            add(logs_root / with_suffix)

    add(with_suffix)
    add(path)

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def analysis_dir_for_log(
    log_path: str | Path,
    root: str | Path = "clustering_results",
    task_name: Optional[str] = None,
    model_name: Optional[str] = None,
    logs_root: str | Path = "logs",
) -> Path:
    log_path = Path(log_path)
    task = task_name or infer_task_name_from_log_path(log_path, logs_root=logs_root)
    if not task:
        raise ValueError(
            "Could not infer task_name from the log path. Pass --task_name when using a bare log filename."
        )
    model = model_name or infer_model_name_from_log_path(log_path, logs_root=logs_root)
    if model:
        return Path(root) / task / model / log_stem(log_path)
    return Path(root) / task / log_stem(log_path)


def resolve_analysis_dir(
    ref: str | Path,
    task_name: Optional[str] = None,
    model_name: Optional[str] = None,
    root: str | Path = "clustering_results",
    logs_root: str | Path = "logs",
) -> Path:
    """Resolve a clustering-results directory from a log path, task/log stem, full dir, or bare stem."""
    path = Path(ref)
    root = Path(root)
    logs_root = Path(logs_root)

    if _contains_part(path, logs_root.name) or str(path).endswith(".jsonl"):
        return analysis_dir_for_log(
            resolve_log_path(path, task_name=task_name, model_name=model_name, logs_root=logs_root),
            root=root,
            task_name=task_name,
            model_name=model_name,
            logs_root=logs_root,
        )

    if path.is_absolute() or _contains_part(path, root.name):
        return path

    if len(path.parts) >= 2 and task_name is None:
        return root / path

    if task_name and model_name:
        return root / task_name / model_name / log_stem(path)

    if task_name:
        return root / task_name / log_stem(path)

    return root / log_stem(path)


def task_model_and_stem_from_analysis_dir(
    analysis_dir: str | Path,
    root: str | Path = "clustering_results",
) -> Tuple[Optional[str], Optional[str], str]:
    path = Path(analysis_dir)
    root_name = Path(root).name
    parts = path.parts
    if root_name in parts:
        idx = parts.index(root_name)
        rel_parts = parts[idx + 1 :]
        if len(rel_parts) >= 3:
            return rel_parts[0], rel_parts[-2], rel_parts[-1]
        if len(rel_parts) == 2:
            return rel_parts[0], None, rel_parts[-1]
        if len(rel_parts) == 1:
            return None, None, rel_parts[-1]
    if path.parent and path.parent != Path("."):
        return path.parent.name, None, path.name
    return None, None, path.name


def task_and_stem_from_analysis_dir(
    analysis_dir: str | Path,
    root: str | Path = "clustering_results",
) -> Tuple[Optional[str], str]:
    task, _, stem = task_model_and_stem_from_analysis_dir(analysis_dir, root=root)
    return task, stem


def log_path_for_analysis_dir(
    analysis_dir: str | Path,
    task_name: Optional[str] = None,
    model_name: Optional[str] = None,
    logs_root: str | Path = "logs",
    clustering_root: str | Path = "clustering_results",
) -> Path:
    inferred_task, inferred_model, stem = task_model_and_stem_from_analysis_dir(analysis_dir, root=clustering_root)
    task = task_name or inferred_task
    model = model_name or inferred_model
    if not task:
        raise ValueError(
            "Could not infer task_name from the clustering results directory. Pass --task_name."
        )
    logs_root = Path(logs_root)
    candidates = []
    if model:
        candidates.append(logs_root / task / model / f"{stem}.jsonl")
    candidates.append(logs_root / task / f"{stem}.jsonl")

    for candidate in candidates:
        if candidate.exists():
            return candidate

    task_root = logs_root / task
    if task_root.exists():
        matches = list(task_root.rglob(f"{stem}.jsonl"))
        if model:
            for match in matches:
                if model in match.parts:
                    return match
        if matches:
            return matches[0]

    return candidates[0]
