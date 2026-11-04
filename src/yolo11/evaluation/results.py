from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

from yolo11.utils.bbox_utils import xyxy2xywh, clip_boxes


class Boxes:

    def __init__(self, boxes: torch.Tensor | np.ndarray, orig_shape: tuple[int, int]) -> None:
        if boxes.ndim == 1:
            boxes = boxes[None, :]
        n = boxes.shape[-1]
        assert n in {6, 7}, f"expected 6 or 7 values but got {n}"
        self.data = boxes
        self.orig_shape = orig_shape
        self.is_track = n == 7

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx):
        return Boxes(self.data[idx], self.orig_shape)

    @property
    def xyxy(self) -> torch.Tensor | np.ndarray:
        return self.data[:, :4]

    @property
    def conf(self) -> torch.Tensor | np.ndarray:
        return self.data[:, -2]

    @property
    def cls(self) -> torch.Tensor | np.ndarray:
        return self.data[:, -1]

    @property
    def id(self) -> torch.Tensor | np.ndarray | None:
        return self.data[:, -3] if self.is_track else None

    @property
    @lru_cache(maxsize=2)
    def xywh(self) -> torch.Tensor | np.ndarray:
        return xyxy2xywh(self.xyxy)

    @property
    @lru_cache(maxsize=2)
    def xyxyn(self) -> torch.Tensor | np.ndarray:
        xyxy = self.xyxy.clone() if isinstance(self.xyxy, torch.Tensor) else np.copy(self.xyxy)
        xyxy[..., [0, 2]] /= self.orig_shape[1]
        xyxy[..., [1, 3]] /= self.orig_shape[0]
        return xyxy

    @property
    @lru_cache(maxsize=2)
    def xywhn(self) -> torch.Tensor | np.ndarray:
        xywh = xyxy2xywh(self.xyxy)
        xywh[..., [0, 2]] /= self.orig_shape[1]
        xywh[..., [1, 3]] /= self.orig_shape[0]
        return xywh


class Results:

    def __init__(self, orig_img, path, names, boxes=None) -> None:
        self.orig_img = orig_img
        self.orig_shape = orig_img.shape[:2]
        self.boxes = Boxes(boxes, self.orig_shape) if boxes is not None else None
        self.names = names
        self.path = path

    def __len__(self) -> int:
        return len(self.boxes) if self.boxes is not None else 0

    def save_txt(self, txt_file, save_conf=False):
        boxes = self.boxes
        texts = []
        
        if boxes:
            # Detect - detection only
            for j, d in enumerate(boxes):
                c = int(d.cls)
                conf = float(d.conf)
                id = int(d.id.item()) if d.is_track else None
                line = (c, *d.xywhn.view(-1))
                line += (conf,) * save_conf + (() if id is None else (id,))
                texts.append(("%g " * len(line)).rstrip() % line)

        if texts:
            Path(txt_file).parent.mkdir(parents=True, exist_ok=True)
            with open(txt_file, "a", encoding="utf-8") as f:
                f.writelines(text + "\n" for text in texts)

        return str(txt_file)