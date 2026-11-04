import os
import gc
import glob
import torch
import hashlib
import logging
import numpy as np
import tqdm as TQDM
from typing import Any
from pathlib import Path
from copy import deepcopy
from itertools import repeat
from PIL import Image, ImageOps
from torch.utils.data import Dataset

from yolo11.data.augment import Compose, Format, LetterBox, v8_transforms
from yolo11.data.preprocessor import Preprocessor, update_labels_info, Instances
from yolo11.utils.bbox_utils import segments2boxes

LOGGER = logging.getLogger(__name__)

IMG_FORMATS = {"bmp", "dng", "jpeg", "jpg", "mpo", "png", "tif", "tiff", "webp", "pfm", "heic"} 

def build_yolo_dataset(
    cfg,
    img_path: str,
    batch: int,
    data: dict[str, Any],
    mode: str = "train",
    rect: bool = False,
    stride: int = 32,
) -> Dataset:
    return YOLODataset(
        img_path=img_path,
        imgsz=cfg.imgsz,
        batch_size=batch,
        augment=mode == "train",  # per-transform probabilities (mosaic, mixup, hsv_*, ...) control strength
        hyp=cfg,
        rect=getattr(cfg, "rect", False) or rect,
        cache=getattr(cfg, "cache", False) or None,
        single_cls=getattr(cfg, "single_cls", False),
        stride=stride,
        pad=0.0 if mode == "train" else 0.5,
        prefix=f"{mode}: ",
        data=data,
        fraction=getattr(cfg, "fraction", 1.0) if mode == "train" else 1.0,
        classes=getattr(cfg, "classes", None),
        pad_color=int(p if (p := getattr(cfg, 'pad_color', 114)) is not None else 114),
    )

class YOLODataset(Dataset):
    
    def __init__(
        self,
        img_path: str | list[str],
        imgsz: int = 640,
        cache: bool | str = False,
        augment: bool = True,
        hyp: dict[str, Any] = {},  # hyp dict
        prefix: str = "",
        rect: bool = False,
        batch_size: int = 16,
        stride: int = 32,
        pad: float = 0.5,
        single_cls: bool = False,
        classes: list[int] | None = None,
        fraction: float = 1.0,
        data: dict | None = None,
        pad_color: int = 114,
    ):
        super().__init__()
        self.pad_color = pad_color
        self.rect = rect
        self.img_path = img_path
        self.imgsz = imgsz
        self.augment = augment
        self.single_cls = single_cls
        self.prefix = prefix
        self.fraction = fraction
        self.data = data
        self.channels = data.get("channels", 3)
        
        self.im_files = self.get_img_files(img_path)
        
        # Get labels
        self.labels = self.get_labels()
        self.preprocessor = Preprocessor(
            self.im_files,
            self.labels,
            self.imgsz,
            batch_size,
            stride,
            pad,
            rect,
            cache,
            augment,
            self.channels,
            self.prefix,
        )
        self.im_files = self.preprocessor.im_files
        self.labels = self.preprocessor.labels
        self.ni = self.preprocessor.ni
        
        if self.rect:
            self.batch = self.preprocessor.batch
            self.batch_shapes = self.preprocessor.batch_shapes
        
        self.update_labels(include_class=classes)  # update for single_cls/classes
        
        # Cache images already done in preprocessor
        
        self.transforms = self.build_transforms(hyp=hyp)
    
    def get_img_files(self, img_path: str | list[str]) -> list[str]:
        """Read image files"""
        f = []
        for p in img_path if isinstance(img_path, list) else [img_path]:
            p = Path(p)
            if p.is_dir():
                f += glob.glob(str(p / "**" / "*.*"), recursive=True)
            elif p.is_file():
                with open(p, encoding="utf-8") as t:
                    t = t.read().strip().splitlines()
                    parent = str(p.parent) + os.sep
                    f += [x.replace("./", parent) if x.startswith("./") else x for x in t]
        im_files = sorted(x.replace("/", os.sep) for x in f if x.rpartition(".")[-1].lower() in IMG_FORMATS)  # IMG_FORMATS từ Ultralytics
        if self.fraction < 1:
            im_files = im_files[: round(len(im_files) * self.fraction)]
        return im_files
    
    def cache_labels(self, path: Path = Path("./labels.cache")) -> dict:
        """Cache dataset labels"""
        x = {"labels": []}
        nm, nf, ne, nc = 0, 0, 0, 0
        desc = f"Scanning {path.parent / path.stem}..."
        total = len(self.im_files)
        results = []
        for args in zip(self.im_files, self.label_files, repeat(self.prefix), repeat(len(self.data["names"])), repeat(self.single_cls)):
            results.append(verify_image_label(args))
        msgs = []
        pbar = TQDM.tqdm(results, desc=desc, total=total)
        for im_file, lb, shape, nm_f, nf_f, ne_f, nc_f, msg in pbar:
            nm += nm_f
            nf += nf_f
            ne += ne_f
            nc += nc_f
            if msg:
                msgs.append(msg)
            if im_file:
                x["labels"].append({
                    "im_file": im_file,
                    "shape": shape,
                    "cls": lb[:, 0:1],
                    "bboxes": lb[:, 1:],
                    "normalized": True,
                    "bbox_format": "xywh",
                })
            pbar.desc = f"{desc} {nf} images, {nm + ne} backgrounds, {nc} corrupt"
        pbar.close()
        if msgs:
            LOGGER.warning("\n".join(msgs))
        x["hash"] = get_hash(self.label_files + self.im_files)
        x["results"] = nf, nm, ne, nc, len(self.im_files)
        save_cache_file(path, x)
        return x

    def get_labels(self) -> list[dict]:
        """Get labels"""
        self.label_files = img2label_paths(self.im_files)
        cache_path = Path(self.label_files[0]).parent.with_suffix(".cache")
        try:
            cache = load_cache_file(cache_path)
            assert cache["hash"] == get_hash(self.label_files + self.im_files)  # dataset unchanged
            exists = True
        except Exception:
            cache = self.cache_labels(cache_path)
            exists = False
        nf, nm, ne, nc, n = cache.pop("results")
        if exists:
            d = f"Scanning {cache_path}... {nf} images, {nm + ne} backgrounds, {nc} corrupt"
            TQDM.tqdm(None, desc=self.prefix + d, total=n, initial=n)
        cache.pop("hash")  # loại pop
        labels = cache["labels"]
        self.im_files = [lb["im_file"] for lb in labels]
        lengths = ((len(lb["cls"]), len(lb["bboxes"])) for lb in labels)
        len_cls, len_boxes = (sum(x) for x in zip(*lengths))
        return labels
    
    def build_transforms(self, hyp: dict | None = None) -> Compose:
        """Build transforms"""
        if self.augment:
            hyp.mosaic = getattr(hyp, 'mosaic', 0.0) if self.augment and not self.rect else 0.0
            hyp.mixup = getattr(hyp, 'mixup', 0.0) if self.augment and not self.rect else 0.0
            hyp.cutmix = getattr(hyp, 'cutmix', 0.0) if self.augment and not self.rect else 0.0
            transforms = v8_transforms(self, self.imgsz, hyp, pad_value=self.pad_color)
        else:
            transforms = Compose([LetterBox(new_shape=(self.imgsz, self.imgsz), scaleup=False, padding_value=self.pad_color)])
        transforms.append(
            Format(
                bbox_format="xywh",
                normalize=True,
                batch_idx=True,
            )
        )
        return transforms
    
    @staticmethod
    def collate_fn(batch: list[dict]) -> dict:
        new_batch = {}
        if not batch:
            return new_batch
        keys = batch[0].keys()
        for k in keys:
            # Lấy tất cả values cho key 'k' từ mọi item trong batch
            values = [b[k] for b in batch]
            if k == 'img':
                # 'img' là tensor, dùng stack để tạo batch [B, C, H, W]
                new_batch[k] = torch.stack(values, 0)
            elif k in {'bboxes', 'cls', 'batch_idx'}:                
                # Xử lý đặc biệt cho batch_idx
                if k == 'batch_idx':
                    for i in range(len(values)):
                        values[i] += i # Áp dụng chỉ số batch (0, 1, 2...)
                new_batch[k] = torch.cat(values, 0)
            else:
                new_batch[k] = values 
                
        return new_batch
    
    def update_labels(self, include_class: list[int] | None) -> None:
        """Update labels"""
        include_class_array = np.array(include_class).reshape(1, -1) if include_class else None
        for i in range(len(self.labels)):
            if include_class is not None:
                cls = self.labels[i]["cls"]
                bboxes = self.labels[i]["bboxes"]
                j = (cls == include_class_array).any(1)
                self.labels[i]["cls"] = cls[j]
                self.labels[i]["bboxes"] = bboxes[j]
            if self.single_cls:
                self.labels[i]["cls"][:, 0] = 0
    
    def __getitem__(self, index: int) -> dict[str, Any]:
        data = self.transforms(self.get_image_and_label(index))
        return data
    
    def get_image_and_label(self, index: int) -> dict[str, Any]:
        label = deepcopy(self.labels[index])
        label.pop("shape", None)
        label["img"], label["ori_shape"], label["resized_shape"] = self.preprocessor.load_image(index, rect_mode=self.rect)
        label["ratio_pad"] = (
            (
                label["resized_shape"][0] / label["ori_shape"][0],
                label["resized_shape"][1] / label["ori_shape"][1],
            ),
            (0.0, 0.0),  # (padw, padh) - Initialized to 0, updated by LetterBox
        )
        if self.rect:
            label["rect_shape"] = self.batch_shapes[self.batch[index]]
        label["instances"] = Instances(label["bboxes"], bbox_format="xywh", normalized=True)
        return update_labels_info(label)
    
    def close_mosaic(self, hyp: dict[str, Any] | Any) -> None:
        """ Close mosaic augmentation."""
        LOGGER.info("Closing dataloader mosaic")
        is_dict = isinstance(hyp, dict)
        def set_hyp_attr(obj, name, val):
            if is_dict:
                obj[name] = val
            else:
                setattr(obj, name, val)
        set_hyp_attr(hyp, "mosaic", 0.0)
        set_hyp_attr(hyp, "mixup", 0.0)
        set_hyp_attr(hyp, "cutmix", 0.0)
        self.transforms = self.build_transforms(hyp=hyp)
        LOGGER.info("Mosaic, MixUp, and CutMix augmentations have been disabled.")

    def __len__(self) -> int:
        return len(self.labels)

def get_hash(paths: list[str]) -> str:
    """Hash file paths together with each file's size and modification time."""
    h = hashlib.sha256()
    for p in paths:
        h.update(os.fsencode(p) + b"\0")
        try:
            st = os.stat(p)
            h.update(f"{st.st_size}:{st.st_mtime_ns}".encode())
        except OSError:
            pass
        h.update(b"\0")
    return h.hexdigest()


def load_cache_file(path: Path) -> dict:
    """Load a ``*.cache`` dict saved by ``save_cache_file``."""
    gc.disable()  # faster unpickling of many small objects
    try:
        return np.load(str(path), allow_pickle=True).item()
    finally:
        gc.enable()  # also when loading fails, e.g. on the first run when no cache exists yet


def save_cache_file(path: Path, x: dict) -> None:
    """Save ``x`` to exactly ``path``; ``np.save`` given a filename would append ``.npy``."""
    try:
        with open(path, "wb") as f:
            np.save(f, x, allow_pickle=True)
    except OSError as e:
        Path(path).unlink(missing_ok=True)  # do not leave a partially written cache
        LOGGER.warning(f"Cache not saved to {path}: {e}")


def img2label_paths(img_paths: list[str]) -> list[str]:
    """Convert image paths to label paths by replacing 'images' with 'labels' and extension with '.txt'."""
    sa, sb = f"{os.sep}images{os.sep}", f"{os.sep}labels{os.sep}"  # /images/, /labels/ substrings
    return [sb.join(x.rsplit(sa, 1)).rsplit(".", 1)[0] + ".txt" for x in img_paths]

def verify_image_label(args: tuple) -> list:
    """Verify one image-label pair."""
    im_file, lb_file, prefix, num_cls, single_cls = args
    nm, nf, ne, nc, msg = 0, 0, 0, 0, ""
    try:
        im = Image.open(im_file)
        im.verify()
        shape = exif_size(im)
        shape = (shape[1], shape[0])
        assert (shape[0] > 9) & (shape[1] > 9), f"image size {shape} <10 pixels"
        assert im.format.lower() in IMG_FORMATS, f"invalid image format {im.format}"
        if im.format.lower() in {"jpg", "jpeg"}:
            with open(im_file, "rb") as f:
                f.seek(-2, 2)
                if f.read() != b"\xff\xd9":
                    ImageOps.exif_transpose(Image.open(im_file)).save(im_file, "JPEG", subsampling=0, quality=100)
                    msg = f"{prefix}{im_file}: corrupt JPEG restored and saved"

        if os.path.isfile(lb_file):
            nf = 1
            with open(lb_file, encoding="utf-8") as f:
                lb = [x.split() for x in f.read().strip().splitlines() if len(x)]
                if any(len(x) > 6 for x in lb):  # polygon (segmentation) labels: use their bounding boxes
                    classes = np.array([x[0] for x in lb], dtype=np.float32)
                    segments = [np.array(x[1:], dtype=np.float32).reshape(-1, 2) for x in lb]
                    lb = np.concatenate((classes.reshape(-1, 1), segments2boxes(segments)), 1)
                lb = np.array(lb, dtype=np.float32)
            if nl := len(lb):
                assert lb.shape[1] == 5, f"labels require 5 columns, {lb.shape[1]} columns detected"
                points = lb[:, 1:]
                assert points.max() <= 1.01, f"non-normalized or out of bounds coordinates {points[points > 1.01]}"
                assert lb.min() >= -0.01, f"negative class labels or coordinate {lb[lb < -0.01]}"

                max_cls = 0 if single_cls else lb[:, 0].max()
                assert max_cls < num_cls, (
                    f"Label class {int(max_cls)} exceeds dataset class count {num_cls}. "
                    f"Possible class labels are 0-{num_cls - 1}"
                )
                _, i = np.unique(lb, axis=0, return_index=True)
                if len(i) < nl:
                    lb = lb[i]
                    msg = f"{prefix}{im_file}: {nl - len(i)} duplicate labels removed"
            else:
                ne = 1
                lb = np.zeros((0, 5), dtype=np.float32)
        else:
            nm = 1
            lb = np.zeros((0, 5), dtype=np.float32)
        lb = lb[:, :5]
        return im_file, lb, shape, nm, nf, ne, nc, msg
    except Exception as e:
        nc = 1
        msg = f"{prefix}{im_file}: ignoring corrupt image/label: {e}"
        return [None, None, None, nm, nf, ne, nc, msg]
    
def exif_size(img: Image.Image) -> tuple[int, int]:
    """Return exif-corrected PIL size."""
    s = img.size  # (width, height)
    if img.format == "JPEG":  # only support JPEG images
        try:
            if exif := img.getexif():
                rotation = exif.get(274, None)  # the EXIF key for the orientation tag is 274
                if rotation in {6, 8}:  # rotation 270 or 90
                    s = s[1], s[0]
        except Exception:
            pass
    return s