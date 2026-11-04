"""Inference on images with a trained YOLO11 checkpoint."""

import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from yolo11.data.augment import LetterBox
from yolo11.postprocessing.nms import non_max_suppression
from yolo11.utils.bbox_utils import scale_boxes
from yolo11.utils.torch_utils import check_imgsz, load_checkpoint, select_device

LOGGER = logging.getLogger(__name__)

IMG_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


class Predictor:
    """Load a checkpoint and run letterbox -> forward -> NMS -> rescale to original image coordinates.

    Class names come from the checkpoint itself, so no dataset/config file is needed.

    Example:
        >>> predictor = Predictor("runs/COCO2017/weights/best.pt", conf=0.25)
        >>> detections = predictor.predict("image.jpg")
        >>> detections[0]  # {'box': [x1, y1, x2, y2], 'confidence': 0.9, 'class_id': 0, 'label': 'person'}
    """

    def __init__(
        self,
        weights: str | Path,
        device: str | int | None = None,
        imgsz: int | None = None,
        conf: float = 0.25,
        iou: float = 0.7,
        max_det: int = 300,
        half: bool = False,
        classes: list[int] | None = None,
        agnostic: bool = False,
    ):
        self.device = select_device(device)
        self.model = load_checkpoint(weights, device=self.device, fuse=True)[0]
        self.half = half and self.device.type != "cpu"
        self.model = self.model.half() if self.half else self.model.float()
        self.model.eval()

        self.stride = max(int(self.model.stride.max()), 32)
        train_imgsz = getattr(getattr(self.model, "args", None), "get", lambda *_: None)("imgsz")
        self.imgsz = check_imgsz(imgsz or train_imgsz or 640, stride=self.stride, max_dim=1)
        names = getattr(self.model, "names", None) or {}
        self.names: dict[int, str] = dict(enumerate(names)) if isinstance(names, (list, tuple)) else dict(names)
        self.conf, self.iou, self.max_det = conf, iou, max_det
        self.classes, self.agnostic = classes, agnostic
        self.letterbox = LetterBox(new_shape=(self.imgsz, self.imgsz), auto=True, stride=self.stride)
        LOGGER.info(f"Loaded {weights} on {self.device} (imgsz={self.imgsz}, classes={len(self.names)})")

    @staticmethod
    def read_image(source: str | Path | np.ndarray) -> np.ndarray:
        """Return a BGR ``uint8`` HWC image from a path or an array."""
        if isinstance(source, np.ndarray):
            return source
        img = cv2.imread(str(source))
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {source}")
        return img

    def preprocess(self, im0: np.ndarray) -> torch.Tensor:
        im = self.letterbox(image=im0)[..., ::-1]  # BGR -> RGB
        im = torch.from_numpy(np.ascontiguousarray(im.transpose(2, 0, 1))).to(self.device)
        im = im.half() if self.half else im.float()
        return (im / 255.0).unsqueeze(0)

    @torch.no_grad()
    def predict(self, source: str | Path | np.ndarray) -> list[dict[str, Any]]:
        """Detect objects in one image. Boxes are ``xyxy`` pixels in the original image."""
        im0 = self.read_image(source)
        im = self.preprocess(im0)
        preds = self.model(im)
        det = non_max_suppression(
            preds,
            self.conf,
            self.iou,
            classes=self.classes,
            agnostic=self.agnostic,
            max_det=self.max_det,
        )[0]
        det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape[:2])
        return [
            {
                "box": [round(v, 2) for v in box],
                "confidence": round(conf, 5),
                "class_id": int(cls),
                "label": self.names.get(int(cls), str(int(cls))),
            }
            for *box, conf, cls in det.cpu().tolist()
        ]

    def predict_many(self, sources: list[str | Path | np.ndarray]) -> list[list[dict[str, Any]]]:
        return [self.predict(s) for s in sources]


def collect_images(source: str | Path) -> list[Path]:
    """Expand a file or directory into a sorted list of image files."""
    source = Path(source)
    if source.is_dir():
        return sorted(p for p in source.rglob("*") if p.suffix.lower() in IMG_SUFFIXES)
    if source.is_file():
        return [source]
    raise FileNotFoundError(f"Source not found: {source}")
