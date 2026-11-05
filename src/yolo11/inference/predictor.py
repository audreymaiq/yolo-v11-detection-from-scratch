"""Inference on images with a trained YOLO11 checkpoint or an exported model."""

import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from yolo11.data.augment import LetterBox
from yolo11.inference.backends import Backend, load_backend
from yolo11.postprocessing.nms import non_max_suppression
from yolo11.utils.bbox_utils import scale_boxes
from yolo11.utils.torch_utils import check_imgsz, select_device

LOGGER = logging.getLogger(__name__)

IMG_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


class Predictor:
    """Run letterbox -> forward -> NMS -> rescale to original image coordinates on a checkpoint or an export.

    ``weights`` may be a ``.pt`` checkpoint or a ``.onnx`` / ``.torchscript`` file from ``yolo11 export``. Class
    names and input size come from the model file itself, so no dataset/config file is needed.

    Example:
        >>> predictor = Predictor("runs/COCO2017/weights/best.onnx", conf=0.25)
        >>> detections = predictor.predict("image.jpg")
        >>> detections[0]  # {'box': [x1, y1, x2, y2], 'confidence': 0.9, 'class_id': 0, 'label': 'person'}
    """

    def __init__(
        self,
        weights: str | Path,
        device: str | int | None = None,
        imgsz: int | list[int] | None = None,
        conf: float = 0.25,
        iou: float = 0.7,
        max_det: int = 300,
        half: bool = False,
        classes: list[int] | None = None,
        agnostic: bool = False,
    ):
        self.backend: Backend = load_backend(weights, select_device(device), half)
        self.device = self.backend.device
        self.stride = self.backend.stride
        self.names = self.backend.names
        if self.backend.dynamic:
            self.imgsz = check_imgsz(imgsz or self.backend.imgsz or 640, stride=self.stride, min_dim=2)
        else:  # a fixed-shape export only runs at the size it was exported with
            self.imgsz = list(self.backend.imgsz)
            if imgsz and check_imgsz(imgsz, stride=self.stride, min_dim=2) != self.imgsz:
                LOGGER.warning(f"{weights} was exported for imgsz={self.imgsz}; ignoring imgsz={imgsz}")
        self.conf, self.iou, self.max_det = conf, iou, max_det
        self.classes, self.agnostic = classes, agnostic
        LOGGER.info(
            f"Loaded {weights} ({self.backend.format}) on {self.device} (imgsz={self.imgsz}, classes={len(self.names)})"
        )

    @staticmethod
    def read_image(source: str | Path | np.ndarray) -> np.ndarray:
        """Return a BGR ``uint8`` HWC image from a path or an array."""
        if isinstance(source, np.ndarray):
            return source
        img = cv2.imread(str(source))
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {source}")
        return img

    def preprocess(self, images: list[np.ndarray]) -> torch.Tensor:
        """Letterbox BGR images into one ``(B, C, H, W)`` batch scaled to ``[0, 1]``.

        As in the reference implementation, padding is reduced to a multiple of the stride only when all images have
        the same shape and the model accepts any input size; otherwise every image is padded to ``imgsz``.
        """
        same_shapes = len({im.shape for im in images}) == 1
        auto = same_shapes and self.backend.dynamic
        letterbox = LetterBox(new_shape=tuple(self.imgsz), auto=auto, stride=self.stride)
        batch = []
        for im0 in images:
            im = letterbox(image=im0)
            if self.backend.channels == 1:
                im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)[..., None] if im.ndim == 3 else im[..., None]
            else:
                im = im[..., ::-1]  # BGR -> RGB
            batch.append(im.transpose(2, 0, 1))
        x = torch.from_numpy(np.ascontiguousarray(np.stack(batch))).to(self.device)
        return (x.half() if self.backend.fp16 else x.float()) / 255.0

    @torch.no_grad()
    def predict(self, source: str | Path | np.ndarray) -> list[dict[str, Any]]:
        """Detect objects in one image. Boxes are ``xyxy`` pixels in the original image."""
        return self.predict_batch([source])[0]

    @torch.no_grad()
    def predict_batch(self, sources: list[str | Path | np.ndarray]) -> list[list[dict[str, Any]]]:
        """Detect objects in several images with one forward pass; returns one detection list per image."""
        images = [self.read_image(s) for s in sources]
        if not images:
            return []
        fixed = self.backend.batch
        if fixed is not None and len(images) != fixed:  # a fixed-batch export: run it in chunks of that size
            return [d for i in range(0, len(images), fixed) for d in self._predict_fixed(images[i : i + fixed], fixed)]
        x = self.preprocess(images)
        return self.postprocess(self.backend(x), x.shape[2:], images)

    def _predict_fixed(self, images: list[np.ndarray], batch: int) -> list[list[dict[str, Any]]]:
        x = self.preprocess(images)
        if len(images) < batch:  # pad the last chunk with blank images and drop their results
            x = torch.cat((x, x.new_zeros((batch - len(images), *x.shape[1:]))))
        return self.postprocess(self.backend(x), x.shape[2:], images)

    def postprocess(self, preds: torch.Tensor, input_hw, images: list[np.ndarray]) -> list[list[dict[str, Any]]]:
        dets = non_max_suppression(
            preds.float(),
            self.conf,
            self.iou,
            classes=self.classes,
            agnostic=self.agnostic,
            max_det=self.max_det,
        )
        results = []
        for det, im0 in zip(dets, images):
            det[:, :4] = scale_boxes(input_hw, det[:, :4], im0.shape[:2])
            results.append(
                [
                    {
                        "box": [round(v, 2) for v in box],
                        "confidence": round(conf, 5),
                        "class_id": int(cls),
                        "label": self.names.get(int(cls), str(int(cls))),
                    }
                    for *box, conf, cls in det.cpu().tolist()
                ]
            )
        return results

    def predict_many(self, sources: list[str | Path | np.ndarray], batch: int = 1) -> list[list[dict[str, Any]]]:
        """Detect objects in many images, ``batch`` images per forward pass."""
        return [d for i in range(0, len(sources), batch) for d in self.predict_batch(sources[i : i + batch])]


def collect_images(source: str | Path) -> list[Path]:
    """Expand a file or directory into a sorted list of image files."""
    source = Path(source)
    if source.is_dir():
        return sorted(p for p in source.rglob("*") if p.suffix.lower() in IMG_SUFFIXES)
    if source.is_file():
        return [source]
    raise FileNotFoundError(f"Source not found: {source}")
