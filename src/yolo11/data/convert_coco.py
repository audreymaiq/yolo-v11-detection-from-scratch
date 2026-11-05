"""Convert COCO detection boxes to YOLO labels without modifying existing data."""

from __future__ import annotations

import json
import math
import tempfile
from collections import defaultdict
from pathlib import Path, PurePosixPath
from typing import Any


def _label_name(file_name: str) -> Path:
    if not isinstance(file_name, str) or not file_name or "\\" in file_name:
        raise ValueError(f"Invalid COCO file_name: {file_name!r}")
    path = PurePosixPath(file_name)
    if path.is_absolute() or any(part in {".", ".."} or ":" in part for part in path.parts):
        raise ValueError(f"Unsafe COCO file_name: {file_name!r}")
    return Path(*path.with_suffix(".txt").parts)


def _load_coco(json_file: str | Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    with open(json_file, encoding="utf-8") as source:
        data = json.load(source)
    if not isinstance(data, dict) or any(not isinstance(data.get(key), list) for key in ("images", "categories", "annotations")):
        raise ValueError("COCO JSON must contain images, categories, and annotations lists")
    images, categories, annotations = (data[key] for key in ("images", "categories", "annotations"))
    if not images or not categories:
        raise ValueError("COCO JSON must contain at least one image and one category")
    return images, categories, annotations


def convert_coco_json(json_file: str | Path, output_dir: str | Path, expected_count: int | None = None) -> dict[str, int]:
    """Write a new label directory atomically; never overwrite an existing directory.

    COCO categories are sorted by ID. Crowd and invalid boxes are skipped;
    partially out-of-image boxes are clipped. Background images get empty files.
    """
    images, categories, annotations = _load_coco(json_file)
    category_ids = [category["id"] for category in categories]
    if len(set(category_ids)) != len(category_ids):
        raise ValueError("Duplicate COCO category ID")
    sorted_categories = sorted(categories, key=lambda category: category["id"])
    class_ids = {category["id"]: index for index, category in enumerate(sorted_categories)}

    image_info: dict[int, tuple[Path, int, int]] = {}
    label_names: set[Path] = set()
    for image in images:
        image_id = image["id"]
        if image_id in image_info:
            raise ValueError(f"Duplicate COCO image ID: {image_id}")
        label_name = _label_name(image["file_name"])
        if label_name in label_names:
            raise ValueError(f"Multiple images map to label file {label_name}")
        width, height = image["width"], image["height"]
        if not isinstance(width, int) or not isinstance(height, int) or width <= 0 or height <= 0:
            raise ValueError(f"Invalid image size for image ID {image_id}")
        image_info[image_id] = label_name, width, height
        label_names.add(label_name)
    if expected_count is not None and expected_count != len(images):
        raise ValueError(f"Expected {expected_count} images, COCO JSON contains {len(images)}")

    labels: dict[int, list[str]] = defaultdict(list)
    seen: dict[int, set[tuple[int, float, float, float, float]]] = defaultdict(set)
    stats = {"images": len(images), "annotations": len(annotations), "written": 0, "crowd": 0,
             "invalid": 0, "duplicates": 0, "clipped": 0}
    for annotation in annotations:
        image_id = annotation["image_id"]
        if image_id not in image_info:
            raise ValueError(f"Annotation refers to unknown image ID {image_id}")
        category_id = annotation["category_id"]
        if category_id not in class_ids:
            raise ValueError(f"Annotation refers to unknown category ID {category_id}")
        if annotation.get("iscrowd", False):
            stats["crowd"] += 1
            continue
        bbox = annotation.get("bbox")
        if not isinstance(bbox, list) or len(bbox) != 4 or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in bbox):
            stats["invalid"] += 1
            continue
        x, y, box_width, box_height = bbox
        if box_width <= 0 or box_height <= 0:
            stats["invalid"] += 1
            continue
        _, width, height = image_info[image_id]
        left = max(0.0, min(float(width), x))
        top = max(0.0, min(float(height), y))
        right = max(0.0, min(float(width), x + box_width))
        bottom = max(0.0, min(float(height), y + box_height))
        if right <= left or bottom <= top:
            stats["invalid"] += 1
            continue
        if (left, top, right, bottom) != (x, y, x + box_width, y + box_height):
            stats["clipped"] += 1
        cx = (left + right) / (2 * width)
        cy = (top + bottom) / (2 * height)
        bw = (right - left) / width
        bh = (bottom - top) / height
        key = (class_ids[category_id], cx, cy, bw, bh)
        if key in seen[image_id]:
            stats["duplicates"] += 1
            continue
        seen[image_id].add(key)
        labels[image_id].append(f"{key[0]} {cx:.12g} {cy:.12g} {bw:.12g} {bh:.12g}\n")
        stats["written"] += 1

    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}. Choose a new --output directory.")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output.name}-", dir=output.parent) as temporary:
        staged = Path(temporary) / "labels"
        staged.mkdir()
        for image_id, (name, _, _) in image_info.items():
            path = staged / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("".join(labels[image_id]), encoding="utf-8")
        class_metadata = [{"yolo_id": class_ids[c["id"]], "coco_id": c["id"], "name": c["name"]}
                          for c in sorted_categories]
        (staged / "classes.json").write_text(json.dumps(class_metadata, indent=2, ensure_ascii=False), encoding="utf-8")
        verify_output(staged, len(images), nc=len(categories))
        staged.rename(output)
    return stats


def verify_output(output_dir: str | Path, expected_count: int, nc: int = 80) -> None:
    """Raise if generated YOLO labels are incomplete or malformed."""
    files = list(Path(output_dir).rglob("*.txt"))
    if len(files) != expected_count:
        raise ValueError(f"Expected {expected_count} label files, found {len(files)}")
    for path in files:
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            parts = line.split()
            if len(parts) != 5:
                raise ValueError(f"{path}:{line_number}: expected five columns")
            try:
                class_id = int(parts[0])
                cx, cy, width, height = (float(value) for value in parts[1:])
            except ValueError as exc:
                raise ValueError(f"{path}:{line_number}: invalid numeric label") from exc
            if not 0 <= class_id < nc or not all(math.isfinite(v) for v in (cx, cy, width, height)):
                raise ValueError(f"{path}:{line_number}: invalid class or coordinate")
            if width <= 0 or height <= 0 or cx - width / 2 < -1e-7 or cy - height / 2 < -1e-7 or cx + width / 2 > 1 + 1e-7 or cy + height / 2 > 1 + 1e-7:
                raise ValueError(f"{path}:{line_number}: box lies outside the image")