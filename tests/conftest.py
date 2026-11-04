from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

NAMES = ["square", "bar"]


def _make_split(root: Path, split: str, n: int, rng: np.random.Generator, size: int = 96) -> None:
    (root / "images" / split).mkdir(parents=True)
    (root / "labels" / split).mkdir(parents=True)
    for i in range(n):
        img = np.full((size, size, 3), 40, np.uint8)
        lines = []
        for _ in range(rng.integers(1, 3)):
            cls = int(rng.integers(0, 2))
            if cls == 0:
                w, h = int(rng.integers(20, 40)), int(rng.integers(20, 40))
            else:
                w, h = int(rng.integers(50, 80)), int(rng.integers(8, 16))
            x, y = int(rng.integers(0, size - w)), int(rng.integers(0, size - h))
            img[y : y + h, x : x + w] = (200, 60, 60) if cls == 0 else (60, 200, 60)
            lines.append(f"{cls} {(x + w / 2) / size:.6f} {(y + h / 2) / size:.6f} {w / size:.6f} {h / size:.6f}")
        cv2.imwrite(str(root / "images" / split / f"{i}.jpg"), img)
        (root / "labels" / split / f"{i}.txt").write_text("\n".join(lines) + "\n")


@pytest.fixture(scope="session")
def tiny_dataset(tmp_path_factory) -> Path:
    """A 2-class synthetic YOLO dataset; returns the path of its data.yaml."""
    root = tmp_path_factory.mktemp("tiny")
    rng = np.random.default_rng(0)
    _make_split(root, "train", 8, rng)
    _make_split(root, "val", 4, rng)
    data = {"path": str(root), "train": "images/train", "val": "images/val", "names": NAMES}
    (root / "data.yaml").write_text(yaml.safe_dump(data))
    return root / "data.yaml"
