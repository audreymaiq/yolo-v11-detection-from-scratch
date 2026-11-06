"""Where a training run came from: the code version, the software environment and the data it was trained on.

The trainer writes this to ``<run>/provenance.json``, stores it in every checkpoint and logs it to the experiment
tracker; ``yolo11 export`` copies the essentials into the exported model's metadata.
"""

import hashlib
import os
import platform
import subprocess
from datetime import datetime
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np
import torch

from yolo11 import __version__

# Distributions whose versions change results; extras are recorded when installed.
PACKAGES = ("torch", "torchvision", "numpy", "opencv-python", "pillow", "scipy", "polars", "onnx", "onnxruntime")

# Set by images built without .git (see Dockerfile) to the commit they were built from.
GIT_COMMIT_ENV = "YOLO11_GIT_COMMIT"


def git_info() -> dict[str, Any] | None:
    """Commit, branch and whether tracked files have uncommitted changes, for the repo this package runs from.

    Returns ``None`` when the package is not running from a git checkout (e.g. installed from a wheel) and
    ``YOLO11_GIT_COMMIT`` is not set.
    """
    package = Path(__file__).resolve().parents[1]  # src/yolo11

    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=package, capture_output=True, text=True, timeout=10, check=True
        ).stdout.strip()

    try:
        git("ls-files", "--error-unmatch", "__init__.py")  # tracked here: we run from the checkout, not a copy
        return {
            "commit": git("rev-parse", "HEAD"),
            "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": bool(git("status", "--porcelain", "--untracked-files=no")),
        }
    except (OSError, subprocess.SubprocessError):
        commit = os.getenv(GIT_COMMIT_ENV)
        return {"commit": commit, "branch": None, "dirty": None} if commit else None


def environment_info(device: torch.device | str | None = None) -> dict[str, Any]:
    """Python, OS, package versions and the accelerator a run uses."""
    packages = {}
    for name in PACKAGES:
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            pass
    device = torch.device(device) if device is not None else None
    cuda = device is not None and device.type == "cuda"
    return {
        "yolo11": __version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": packages,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
        "device": str(device) if device is not None else None,
        "device_name": torch.cuda.get_device_name(device) if cuda else platform.processor() or None,
    }


def dataset_fingerprint(dataset) -> dict[str, Any]:
    """Identify the data a ``YOLODataset`` serves: its images, their sizes and their labels.

    The hash covers each image's path relative to the dataset root (so moving the dataset or switching OS keeps
    it), the image size and the labels after ``single_cls``/``classes``/``fraction`` are applied. Editing a label
    or adding/removing an image changes it; editing pixels without resizing the image does not.
    """
    root = Path(dataset.data["path"]) if dataset.data and dataset.data.get("path") else None
    h = hashlib.sha256()
    instances = 0

    def rel(f: str) -> str:
        try:
            return Path(os.path.relpath(f, root)).as_posix() if root else Path(f).as_posix()
        except ValueError:  # another drive on Windows
            return Path(f).as_posix()

    for name, lb in sorted((rel(lb["im_file"]), lb) for lb in dataset.labels):
        cls = np.asarray(lb["cls"], dtype=np.int64).reshape(-1)
        boxes = np.asarray(lb["bboxes"], dtype=np.float32).reshape(-1, 4)
        h.update(name.encode() + b"\0")
        h.update(np.asarray(lb["shape"], dtype=np.int64).tobytes())
        h.update(cls.tobytes() + boxes.tobytes())
        instances += len(cls)
    return {"images": len(dataset.labels), "instances": instances, "sha256": h.hexdigest()}


def collect_provenance(device=None, datasets: dict[str, Any] | None = None, data_yaml: str | None = None) -> dict:
    """Everything above for one run; ``datasets`` maps a split name to its ``YOLODataset``."""
    return {
        "date": datetime.now().astimezone().isoformat(),
        "git": git_info(),
        "environment": environment_info(device),
        "data": {
            "yaml": data_yaml,
            **{split: dataset_fingerprint(ds) for split, ds in (datasets or {}).items()},
        },
    }


def summary(provenance: dict | None) -> dict[str, Any]:
    """Flat key -> value essentials, for tracker params/tags and exported-model metadata."""
    if not provenance:
        return {}
    git = provenance.get("git") or {}
    env = provenance.get("environment") or {}
    out = {
        "git_commit": git.get("commit"),
        "git_dirty": git.get("dirty"),
        "torch": (env.get("packages") or {}).get("torch"),
        "device": env.get("device"),
    }
    for split, fp in (provenance.get("data") or {}).items():
        if isinstance(fp, dict) and "sha256" in fp:
            out[f"data_{split}_sha256"] = fp["sha256"]
            out[f"data_{split}_images"] = fp["images"]
    return {k: v for k, v in out.items() if v is not None}
