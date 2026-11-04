import cv2
import numpy as np
import math
from numbers import Number
from pathlib import Path
import tqdm as TQDM
from itertools import repeat
import torch.nn.functional as F
from PIL import Image, ImageOps

from yolo11.utils.bbox_utils import xywh2xyxy, xyxy2xywh, xyxy2ltwh, ltwh2xyxy, ltwh2xywh, xywh2ltwh

_formats = ["xyxy", "xywh", "ltwh"]

class Preprocessor:
    def __init__(
        self,
        im_files: list[str],
        labels: list[dict],
        imgsz: int,
        batch_size: int,
        stride: int,
        pad: float,
        rect: bool,
        cache: bool | str | None,
        augment: bool,
        channels: int,
        prefix: str,
    ):
        self.im_files = im_files
        self.labels = labels
        self.imgsz = imgsz
        self.batch_size = batch_size
        self.stride = stride
        self.pad = pad
        self.rect = rect
        self.cache = cache.lower() if isinstance(cache, str) else "ram" if cache is True else None
        self.augment = augment
        self.channels = channels
        self.cv2_flag = cv2.IMREAD_GRAYSCALE if self.channels == 1 else cv2.IMREAD_COLOR
        self.prefix = prefix
        self.ni = len(self.labels)
        self.ims, self.im_hw0, self.im_hw = [None] * self.ni, [None] * self.ni, [None] * self.ni
        self.npy_files = [Path(f).with_suffix(".npy") for f in self.im_files]
        self.buffer = []
        self.max_buffer_length = min((self.ni, self.batch_size * 8, 1000)) if self.augment else 0
        if self.rect:
            self.set_rectangle()
        if self.cache:
            self.cache_images()

    def set_rectangle(self) -> None:
        """Set rectangle shapes."""
        bi = np.floor(np.arange(self.ni) / self.batch_size).astype(int)
        nb = bi[-1] + 1
        s = np.array([x["shape"] for x in self.labels])
        ar = s[:, 0] / s[:, 1]
        irect = ar.argsort()
        self.im_files = [self.im_files[i] for i in irect]
        self.labels = [self.labels[i] for i in irect]
        self.npy_files = [self.npy_files[i] for i in irect]
        ar = ar[irect]
        shapes = [[1, 1]] * nb
        for i in range(nb):
            ari = ar[bi == i]
            mini, maxi = ari.min(), ari.max()
            if maxi < 1:
                shapes[i] = [maxi, 1]
            elif mini > 1:
                shapes[i] = [1, 1 / mini]
        self.batch_shapes = np.ceil(np.array(shapes) * self.imgsz / self.stride + self.pad).astype(int) * self.stride
        self.batch = bi

    def cache_images(self) -> None:
        """Cache images."""
        b = 0
        gb = 1 << 30
        fcn = self.cache_images_to_disk if self.cache == "disk" else self.load_image
        storage = "Disk" if self.cache == "disk" else "RAM"
        results = []
        for i in range(self.ni):
            x = fcn(i)
            if self.cache == "disk":
                b += self.npy_files[i].stat().st_size if self.npy_files[i].exists() else 0
            else:
                self.ims[i], self.im_hw0[i], self.im_hw[i] = x
                b += self.ims[i].nbytes
            results.append(x)
        pbar = TQDM.tqdm(enumerate(results), total=self.ni)
        for _ in pbar:
            pbar.desc = f"Caching images ({b / gb:.1f}GB {storage})"
        pbar.close()

    def cache_images_to_disk(self, i: int) -> None:
        """Save to npy."""
        f = self.npy_files[i]
        if not f.exists():
            np.save(f.as_posix(), cv2.imread(self.im_files[i], self.cv2_flag))

    def load_image(self, i: int, rect_mode: bool = True) -> tuple[np.ndarray, tuple[int, int], tuple[int, int]]:
        """Load image"""
        im, f, fn = self.ims[i], self.im_files[i], self.npy_files[i]
        if im is None:
            if fn.exists():
                # Cache NPY (Luôn là BGR do cv2.imread tạo ra)
                try:
                    im = np.load(fn)
                    if self.channels == 3:
                        # Nếu cần 3 kênh:
                        if len(im.shape) == 2:  # Nếu cache là xám -> Giả lập RGB
                            im = cv2.cvtColor(im, cv2.COLOR_GRAY2RGB)
                        else: # Nếu cache là màu (BGR) -> Đổi sang RGB
                            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
                    elif self.channels == 1:
                         # Nếu cần 1 kênh:
                         if len(im.shape) == 3: # Nếu cache là màu -> Đổi sang Xám
                            im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
                except Exception as e:
                    print(f"WARNING: Corrupted cache {fn}, fallback to PIL. Error: {e}")
                    im = None 
            
            if im is None:
                # Ảnh gốc PIL (Luôn là RGB hoặc L)
                pil_im = Image.open(f)
                pil_im = ImageOps.exif_transpose(pil_im)
                
                # Ép kiểu theo config
                if self.channels == 1:
                    pil_im = pil_im.convert('L') # Bắt buộc ra Grayscale
                    im = np.array(pil_im)
                else: 
                    pil_im = pil_im.convert('RGB') # Bắt buộc ra RGB
                    im = np.array(pil_im)

            if im is None:
                raise FileNotFoundError(f"Image Not Found {f}")
            h0, w0 = im.shape[:2]
            if rect_mode and self.rect:
                bs = self.batch_shapes[self.batch[i]] if hasattr(self, 'batch') else (self.imgsz, self.imgsz)
                r = min(bs[0] / h0, bs[1] / w0)
                if r != 1:
                    w, h = min(math.ceil(w0 * r), bs[1]), min(math.ceil(h0 * r), bs[0])
                    im = cv2.resize(im, (w, h), interpolation=cv2.INTER_LINEAR)
            else:
                r = self.imgsz / max(h0, w0)
                if r != 1:
                    w, h = (min(math.ceil(w0 * r), self.imgsz), min(math.ceil(h0 * r), self.imgsz))
                    im = cv2.resize(im, (w, h), interpolation=cv2.INTER_LINEAR)
            
            # Đảm bảo tensor luôn là 3 chiều (H, W, C) cho PyTorch
            if im.ndim == 2:
                im = im[..., None]
            if self.augment:
                self.ims[i], self.im_hw0[i], self.im_hw[i] = im, (h0, w0), im.shape[:2]
                self.buffer.append(i)
                # Quản lý bộ nhớ FIFO
                if len(self.buffer) >= self.max_buffer_length:
                    j = self.buffer.pop(0)
                    if self.cache != "ram":
                        self.ims[j], self.im_hw0[j], self.im_hw[j] = None, None, None
            return im, (h0, w0), im.shape[:2]

        return self.ims[i], self.im_hw0[i], self.im_hw[i]

class Instances:
    
    def __init__(
        self,
        bboxes: np.ndarray,
        bbox_format: str = "xywh",
        normalized: bool = True,
    ) -> None:
        """Initialize with bboxes."""
        bboxes = np.array(bboxes) if not isinstance(bboxes, np.ndarray) else bboxes
        self._bboxes = Bboxes(bboxes=bboxes, format=bbox_format)
        self.normalized = normalized

    def convert_bbox(self, format: str) -> None:
        """Convert bounding box format."""
        self._bboxes.convert(format=format)

    @property
    def bbox_areas(self) -> np.ndarray:
        """Calculate the area of bounding boxes."""
        return self._bboxes.areas()

    def scale(self, scale_w: float, scale_h: float):
        """Scale coordinates by given factors."""
        if self.normalized:
            return  # Skip mul if normalized (proportions unchanged after resize)
        self._bboxes.mul((scale_w, scale_h, scale_w, scale_h))

    def denormalize(self, w: int, h: int) -> None:
        """Convert normalized to absolute."""
        if not self.normalized:
            return
        self._bboxes.mul((w, h, w, h))
        self.normalized = False

    def normalize(self, width: int, height: int) -> None:
        """Convert absolute to normalized."""
        if self.normalized:
            return
        self._bboxes.mul((1 / width, 1 / height, 1 / width, 1 / height))
        self.normalized = True

    def add_padding(self, padw: int, padh: int) -> None:
        """Add padding to coordinates."""
        if self.normalized:
            raise ValueError("add_padding on normalized requires img_w, img_h; denormalize first or pass sizes")
        self._bboxes.add((padw, padh, padw, padh))

    def flipud(self, h: int) -> None:
        """Flip vertically."""
        if self.normalized:
            h = 1
        self._bboxes.flipud(h)

    def fliplr(self, w: int) -> None:
        """Flip horizontally."""
        if self.normalized:
            w = 1
        self._bboxes.fliplr(w)

    def clip(self, w: int, h: int) -> None:
        """Clip to boundaries."""
        if self.normalized:
            w, h = 1, 1
        self._bboxes.clip(w, h)
        self.remove_zero_area_boxes()  # Sửa: thêm remove sau clip

    def remove_zero_area_boxes(self) -> np.ndarray:
        """Remove zero-area boxes."""
        good = self.bbox_areas > 0
        if not all(good):
            self._bboxes.bboxes = self._bboxes.bboxes[good]
        return good

    def update(self, bboxes: np.ndarray):
        """Update bboxes."""
        self._bboxes = Bboxes(bboxes=bboxes, format=self._bboxes.format)

    def __getitem__(self, index: int | np.ndarray | slice) -> 'Instances':
        """Retrieve instance(s)."""
        bboxes = self.bboxes[index]
        bbox_format = self._bboxes.format
        return Instances(
            bboxes=bboxes,
            bbox_format=bbox_format,
            normalized=self.normalized,
        )

    def __len__(self) -> int:
        return len(self.bboxes)

    @classmethod
    def concatenate(cls, instances_list: list['Instances'], axis=0) -> 'Instances':
        """Concatenate Instances."""
        assert isinstance(instances_list, (list, tuple))
        if not instances_list:
            return cls(np.empty(0))
        assert all(isinstance(instance, Instances) for instance in instances_list)
        if len(instances_list) == 1:
            return instances_list[0]
        bbox_format = instances_list[0]._bboxes.format
        normalized = instances_list[0].normalized
        cat_boxes = np.concatenate([ins.bboxes for ins in instances_list], axis=axis)
        return cls(cat_boxes, bbox_format=bbox_format, normalized=normalized)

    @property
    def bboxes(self) -> np.ndarray:
        return self._bboxes.bboxes

class Bboxes:
    
    def __init__(self, bboxes: np.ndarray, format: str = "xywh") -> None:
        assert format in _formats
        bboxes = bboxes[None, :] if bboxes.ndim == 1 else bboxes
        assert bboxes.ndim == 2 and bboxes.shape[1] == 4
        self.bboxes = bboxes
        self.format = format

    def convert(self, format: str) -> None:
        assert format in _formats
        if self.format == format:
            return
        if self.format == "xyxy":
            func = xyxy2xywh if format == "xywh" else xyxy2ltwh
        elif self.format == "xywh":
            func = xywh2xyxy if format == "xyxy" else xywh2ltwh
        else:
            func = ltwh2xyxy if format == "xyxy" else ltwh2xywh
        self.bboxes = func(self.bboxes)
        self.format = format

    def areas(self) -> np.ndarray:
        return (
            (self.bboxes[:, 2] - self.bboxes[:, 0]) * (self.bboxes[:, 3] - self.bboxes[:, 1])
            if self.format == "xyxy"
            else self.bboxes[:, 3] * self.bboxes[:, 2]
        )

    def mul(self, scale: int | tuple | list) -> None:
        """Mul with format handling."""
        if isinstance(scale, Number):
            scale = tuple(repeat(scale, 4))
        assert len(scale) == 4
        if self.format == "xywh" or self.format == "ltwh":
            self.bboxes[:, 0] *= scale[0]
            self.bboxes[:, 1] *= scale[1]
            self.bboxes[:, 2] *= scale[0]  # w *= scale_w
            self.bboxes[:, 3] *= scale[1]  # h *= scale_h
        else:  # xyxy
            self.bboxes *= np.array(scale)

    def add(self, offset: int | tuple | list) -> None:
        """Add with format handling (sửa: chỉ add vào coords, không vào w/h)."""
        if isinstance(offset, Number):
            offset = tuple(repeat(offset, 4))
        assert len(offset) == 4
        if self.format == "xywh":
            self.bboxes[:, 0] += offset[0]
            self.bboxes[:, 1] += offset[1]
            # Không add vào w/h
        elif self.format == "ltwh":
            self.bboxes[:, 0] += offset[0]
            self.bboxes[:, 1] += offset[1]
            # Không add vào w/h
        else:  # xyxy
            self.bboxes[:, 0] += offset[0]
            self.bboxes[:, 1] += offset[1]
            self.bboxes[:, 2] += offset[2]
            self.bboxes[:, 3] += offset[3]

    def flipud(self, h: int) -> None:
        """Flip vertically with format handling."""
        if self.format == "xyxy":
            y1 = self.bboxes[:, 1].copy()
            y2 = self.bboxes[:, 3].copy()
            self.bboxes[:, 1] = h - y2
            self.bboxes[:, 3] = h - y1
        elif self.format == "ltwh":
            self.bboxes[:, 1] = h - self.bboxes[:, 1] - self.bboxes[:, 3]
        else:  # xywh
            self.bboxes[:, 1] = h - self.bboxes[:, 1]

    def fliplr(self, w: int) -> None:
        """Flip horizontally with format handling."""
        if self.format == "xyxy":
            x1 = self.bboxes[:, 0].copy()
            x2 = self.bboxes[:, 2].copy()
            self.bboxes[:, 0] = w - x2
            self.bboxes[:, 2] = w - x1
        elif self.format == "ltwh":
            self.bboxes[:, 0] = w - self.bboxes[:, 0] - self.bboxes[:, 2]
        else:  # xywh
            self.bboxes[:, 0] = w - self.bboxes[:, 0]

    def clip(self, w: int, h: int) -> None:
        """Clip with format handling."""
        ori_format = self.format
        self.convert("xyxy")
        self.bboxes[:, [0, 2]] = self.bboxes[:, [0, 2]].clip(0, w)
        self.bboxes[:, [1, 3]] = self.bboxes[:, [1, 3]].clip(0, h)
        if ori_format != "xyxy":
            self.convert(ori_format)

    def __len__(self) -> int:
        return len(self.bboxes)

    @classmethod
    def concatenate(cls, boxes_list: list['Bboxes'], axis: int = 0) -> 'Bboxes':
        if not boxes_list:
            return cls(np.empty(0))
        if len(boxes_list) == 1:
            return boxes_list[0]
        return cls(np.concatenate([b.bboxes for b in boxes_list], axis=axis))

    def __getitem__(self, index: int | np.ndarray | slice) -> 'Bboxes':
        if isinstance(index, int):
            return Bboxes(self.bboxes[index].reshape(1, -1), format=self.format)
        b = self.bboxes[index]
        assert b.ndim == 2
        return Bboxes(b, format=self.format)

def update_labels_info(label: dict) -> dict:
    bboxes = label.pop("bboxes")
    bbox_format = label.pop("bbox_format")
    normalized = label.pop("normalized")
    label["instances"] = Instances(bboxes, bbox_format=bbox_format, normalized=normalized)
    return label

def scale_img(img, ratio=1.0, same_shape=False, gs=32, value=0.447):
    """
    Scale and pad an image tensor, optionally maintaining aspect ratio and padding to gs multiple.

    Args:
        img (torch.Tensor): Input image tensor.
        ratio (float, optional): Scaling ratio.
        same_shape (bool, optional): Whether to maintain the same shape.
        gs (int, optional): Grid size for padding.

    Returns:
        (torch.Tensor): Scaled and padded image tensor.
    """
    if ratio == 1.0:
        return img
    h, w = img.shape[2:]
    s = (int(h * ratio), int(w * ratio))  # new size
    img = F.interpolate(img, size=s, mode="bilinear", align_corners=False)  # resize
    if not same_shape:  # pad/crop img
        h, w = (math.ceil(x * ratio / gs) * gs for x in (h, w))
    return F.pad(img, [0, w - s[1], 0, h - s[0]], value=value)  # value = imagenet mean