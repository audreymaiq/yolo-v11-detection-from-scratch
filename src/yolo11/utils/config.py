"""Training-config loading with CLI overrides."""

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    """Read a YAML config; keys in ``overrides`` whose value is not ``None`` replace the file's values."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    config.update({k: v for k, v in (overrides or {}).items() if v is not None})
    return config
