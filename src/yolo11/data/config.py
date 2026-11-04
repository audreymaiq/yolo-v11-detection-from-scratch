"""Dataset YAML (``data.yaml``) loading."""

from pathlib import Path
from typing import Any

import yaml


def load_data_config(file: str | Path, channels: int | None = None, single_cls: bool = False) -> dict[str, Any]:
    """Load a dataset YAML and resolve its split paths to absolute paths.

    Split entries (``train``/``val``/``test``) are resolved against ``path`` when given, otherwise against the
    directory containing the YAML file.
    """
    file = Path(file)
    if not file.is_file():
        raise FileNotFoundError(f"Dataset config not found: {file}")
    with open(file, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if "val" not in data and "validation" in data:
        data["val"] = data.pop("validation")
    if "names" not in data and "nc" in data:
        data["names"] = [f"class_{i}" for i in range(data["nc"])]
    elif "names" in data and "nc" not in data:
        data["nc"] = len(data["names"])
    data["channels"] = channels or data.get("channels", 3)

    root = Path(data.get("path") or file.parent)
    if not root.is_absolute():
        root = (file.parent / root).resolve() if data.get("path") else root.resolve()
    data["path"] = root
    for k in ("train", "val", "test", "minival"):
        if k in data:
            if isinstance(data[k], str):
                data[k] = str((root / data[k]).resolve())
            else:
                data[k] = [str((root / x).resolve()) for x in data[k]]

    if single_cls:
        data["names"] = {0: "item"}
        data["nc"] = 1
    data["yaml_file"] = str(file)
    return data
