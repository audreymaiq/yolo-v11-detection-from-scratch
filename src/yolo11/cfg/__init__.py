"""Package default configuration: the schema of every setting the trainer, validator and dataset read."""

from pathlib import Path

import yaml

DEFAULT_CFG_PATH = Path(__file__).resolve().parent / "default.yaml"

with open(DEFAULT_CFG_PATH, encoding="utf-8") as f:
    DEFAULT_CFG_DICT: dict = yaml.safe_load(f)
DEFAULT_CFG_KEYS = frozenset(DEFAULT_CFG_DICT)
