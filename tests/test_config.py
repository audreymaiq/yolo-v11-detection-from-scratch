from pathlib import Path

import pytest
import yaml

from yolo11.cfg import DEFAULT_CFG_DICT, DEFAULT_CFG_KEYS
from yolo11.utils.config import load_config

CONFIGS = sorted((Path(__file__).resolve().parents[1] / "configs").glob("*.yaml"))


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.name)
def test_experiment_configs_only_use_known_keys(path):
    unknown = set(yaml.safe_load(path.read_text())) - DEFAULT_CFG_KEYS
    assert not unknown, f"{path.name} has keys missing from yolo11/cfg/default.yaml: {sorted(unknown)}"


def test_load_config_falls_back_to_defaults_and_applies_overrides(tmp_path):
    cfg_file = tmp_path / "exp.yaml"
    cfg_file.write_text("epochs: 3\n")
    cfg = load_config(cfg_file, {"batch": 4, "device": None})
    assert cfg["epochs"] == 3  # from the file
    assert cfg["batch"] == 4  # from overrides
    assert cfg["imgsz"] == DEFAULT_CFG_DICT["imgsz"]  # from package defaults
    assert cfg["device"] == DEFAULT_CFG_DICT["device"]  # None overrides are ignored
