from __future__ import annotations

import os
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent


def _repo_path_from_env(env_name: str, *default_parts: str) -> Path:
    override = os.getenv(env_name)
    if override:
        return Path(override).expanduser()
    return REPO_ROOT.joinpath(*default_parts)


DATA_ROOT = _repo_path_from_env("RPT_DATA_ROOT", "data")


def _data_path_from_env(env_name: str, *default_parts: str) -> Path:
    override = os.getenv(env_name)
    if override:
        return Path(override).expanduser()
    return DATA_ROOT.joinpath(*default_parts)


HOTPOTQA_DATA_DIR = _data_path_from_env("RPT_HOTPOTQA_DATA_DIR", "hotpotqa")
LIVEBENCH_MATH_DATA_DIR = _data_path_from_env(
    "RPT_LIVEBENCH_MATH_DATA_DIR",
    "livebench_math",
)
XBRL_FORMULA_DATA_DIR = _data_path_from_env(
    "RPT_XBRL_FORMULA_DATA_DIR",
    "xbrl_formula",
)
