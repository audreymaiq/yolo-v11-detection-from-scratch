"""Training-config loading with CLI overrides."""

from pathlib import Path
from typing import Any

import yaml

from yolo11.cfg import DEFAULT_CFG_DICT


def load_config(path: str | Path, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return package defaults, updated by the YAML at ``path``, updated by non-``None`` ``overrides``."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, encoding="utf-8") as f:
        config = {**DEFAULT_CFG_DICT, **(yaml.safe_load(f) or {})}
    config.update({k: v for k, v in (overrides or {}).items() if v is not None})
    return config
