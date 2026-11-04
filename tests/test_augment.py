import random

import numpy as np

from yolo11.data.augment import RandomPerspective
from yolo11.data.preprocessor import Instances


def _labels(size: int, box_xywh, mosaic_border=None) -> dict:
    labels = {
        "img": np.full((size, size, 3), 114, np.uint8),
        "cls": np.zeros((1, 1), np.float32),
        "instances": Instances(np.array([box_xywh], np.float32), bbox_format="xywh", normalized=True),
    }
    if mosaic_border is not None:
        labels["mosaic_border"] = mosaic_border
    return labels


def test_zoom_out_keeps_fully_visible_boxes(monkeypatch):
    # Force the smallest scale (s = 0.3) and no translation: the box shrinks but stays fully inside the image,
    # so it must survive box_candidates (area ratio is measured against the scaled source box).
    monkeypatch.setattr(random, "uniform", lambda a, b: a)
    rp = RandomPerspective(degrees=0.0, translate=0.0, scale=0.7, shear=0.0, perspective=0.0)
    out = rp(_labels(64, [0.5, 0.5, 0.5, 0.5]))
    assert len(out["instances"].bboxes) == 1
    assert len(out["cls"]) == 1


def test_translation_is_relative_to_output_size():
    # 2x mosaic canvas (128) cropped back to imgsz=64; translate=0.1 may shift by at most 0.1 * 64 px
    rp = RandomPerspective(degrees=0.0, translate=0.1, scale=0.0, shear=0.0, perspective=0.0)
    random.seed(0)
    for _ in range(200):
        out = rp(_labels(128, [0.5, 0.5, 0.25, 0.25], mosaic_border=(-32, -32)))
        assert out["img"].shape[:2] == (64, 64)
        cx, cy = out["instances"].bboxes[0, :2] * 64  # normalized xywh -> output pixels
        assert abs(cx - 32) <= 6.4 + 1e-3 and abs(cy - 32) <= 6.4 + 1e-3
