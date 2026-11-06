"""Dataset fingerprints and git detection behind provenance.json."""

import re
import subprocess
from pathlib import Path

import numpy as np

from yolo11.utils import provenance
from yolo11.utils.provenance import dataset_fingerprint, git_info, summary


class FakeDataset:
    def __init__(self, root, labels):
        self.data = {"path": root}
        self.labels = labels


def _labels(root):
    def lb(name, cls, boxes, shape=(96, 96)):
        return {"im_file": str(Path(root) / "images" / name), "shape": shape,
                "cls": np.array(cls, np.float32).reshape(-1, 1), "bboxes": np.array(boxes, np.float32).reshape(-1, 4)}

    return [lb("a.jpg", [0], [[0.5, 0.5, 0.2, 0.2]]), lb("b.jpg", [1, 0], [[0.1, 0.1, 0.1, 0.1], [0.4, 0.4, 0.2, 0.2]])]


def test_fingerprint_ignores_dataset_location_and_order(tmp_path):
    here, there = str(tmp_path / "data" / "coco"), str(tmp_path / "copy" / "coco")
    a = dataset_fingerprint(FakeDataset(here, _labels(here)))
    moved = dataset_fingerprint(FakeDataset(there, _labels(there)[::-1]))
    assert a == moved
    assert a["images"] == 2 and a["instances"] == 3 and re.fullmatch(r"[0-9a-f]{64}", a["sha256"])


def test_fingerprint_changes_with_labels_images_and_sizes():
    base = dataset_fingerprint(FakeDataset("/d", _labels("/d")))["sha256"]
    relabeled = _labels("/d")
    relabeled[0]["cls"][0, 0] = 1
    resized = _labels("/d")
    resized[1]["shape"] = (96, 128)
    for labels in (relabeled, resized, _labels("/d")[:1]):
        assert dataset_fingerprint(FakeDataset("/d", labels))["sha256"] != base


def test_git_info_in_this_checkout():
    info = git_info()
    assert re.fullmatch(r"[0-9a-f]{40}", info["commit"]) and isinstance(info["dirty"], bool)


def test_git_info_falls_back_to_env_without_git(monkeypatch):
    def no_git(*args, **kwargs):
        raise FileNotFoundError("git")

    monkeypatch.setattr(subprocess, "run", no_git)
    monkeypatch.delenv(provenance.GIT_COMMIT_ENV, raising=False)
    assert git_info() is None
    monkeypatch.setenv(provenance.GIT_COMMIT_ENV, "abc123")
    assert git_info() == {"commit": "abc123", "branch": None, "dirty": None}


def test_summary_flattens_essentials():
    prov = {
        "git": {"commit": "c", "dirty": False},
        "environment": {"packages": {"torch": "2.0"}, "device": "cpu"},
        "data": {"yaml": "d.yaml", "train": {"sha256": "t", "images": 3, "instances": 4}},
    }
    assert summary(prov) == {
        "git_commit": "c", "git_dirty": False, "torch": "2.0", "device": "cpu",
        "data_train_sha256": "t", "data_train_images": 3,
    }
    assert summary(None) == {}
