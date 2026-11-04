from types import SimpleNamespace
from pathlib import Path
from typing import Any
import tqdm as TQDM
import numpy as np
import logging
import torch
import torchvision
import time
import json
import re
import os

from yolo11.data.yolo_dataset import build_yolo_dataset
from yolo11.data.dataloader import build_dataloader
from yolo11.utils.torch_utils import check_imgsz, load_checkpoint
from yolo11.data.config import load_data_config
from yolo11.registry.metric_registry import MetricRegistry
from yolo11.utils.bbox_utils import xywh2xyxy, box_iou
from yolo11.postprocessing.nms import non_max_suppression

LOGGER = logging.getLogger(__name__)

class DetectionValidator:
    def __init__(self, dataloader=None, save_dir=None, args=None, _callbacks=None):
        if args is None:
            args = {}
        self.args = SimpleNamespace(**args)
        self.dataloader = dataloader
        self.stride = None
        self.data = None
        self.device = None
        self.batch_i = None
        self.training = True
        self.names = None
        self.seen = None
        self.stats = None
        self.nc = None
        self.iouv = None
        self.jdict = None
        self.speed = {"preprocess": 0.0, "inference": 0.0, "loss": 0.0, "postprocess": 0.0}
        
        # Save directory: Explicit is better than implicit
        # Priority: passed save_dir (training) > args.save_dir/name (standalone) > error
        if save_dir:
            # Training mode: Trainer passes its save_dir
            self.save_dir = Path(save_dir) if not isinstance(save_dir, Path) else save_dir
        elif hasattr(self.args, 'save_dir') and self.args.save_dir:
            # Standalone validation mode
            base_dir = Path(self.args.save_dir)
            name = getattr(self.args, 'name', 'val') or 'val'
            self.save_dir = base_dir / name
        else:
            raise ValueError(
                "Config MUST specify 'save_dir' for validation.\n"
                "Example: save_dir: 'runs/COCO2017'"
            )
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Config defaults matching BaseValidator
        if not hasattr(self.args, 'conf') or self.args.conf is None:
            self.args.conf = 0.001  # detect default
        if not hasattr(self.args, 'imgsz'):
            self.args.imgsz = 640
        self.args.imgsz = check_imgsz(self.args.imgsz, max_dim=1)
        
        # Plots and callbacks (matching BaseValidator)
        self.plots = {}
        self.callbacks = _callbacks or {}  # Simplified callback system
        
        # DetectionValidator specific
        self.args.task = "detect"
        self.iouv = torch.linspace(0.5, 0.95, 10)
        self.niou = self.iouv.numel()
        self.metrics = MetricRegistry.get_metric('detect')
        self.confusion_matrix = None  # Will be initialized in init_metrics

    @torch.no_grad()
    def __call__(self, trainer=None, model=None):
        self.training = trainer is not None
        augment = getattr(self.args, 'augment', False) and (not self.training)
        if self.training:
            self.device = trainer.device
            self.data = trainer.data
            self.args.half = self.device.type != "cpu" and trainer.amp
            model = trainer.ema.ema or trainer.model
            model = model.half() if getattr(self.args, 'half', False) else model.float()
            self.loss = torch.zeros_like(trainer.loss_items, device=trainer.device)
            model.eval()
        else:
            device_str = str(getattr(self.args, 'device', 'cuda:0' if torch.cuda.is_available() else 'cpu'))  # Fallback
            if device_str.lower() == 'cpu':
                self.device = torch.device('cpu')
            elif torch.backends.mps.is_available() and 'mps' in device_str.lower():
                self.device = torch.device('mps')
            else:
                # Extract numeric ID (handles 'cuda:0', '0', 'cuda', '')
                device_id = re.sub(r'\D', '', device_str) or '0'  # Non-digits → '', default '0'
                self.device = torch.device(f'cuda:{device_id}')
            if isinstance(model, (str, Path)):
                if not str(model).endswith(".pt"):
                    raise NotImplementedError("Only .pt models are supported.")
                model = load_checkpoint(model, device=self.device, fuse=True)[0]
                model = model.half() if getattr(self.args, 'half', False) else model.float()
                model.eval()
            else:
                model = model.half() if getattr(self.args, 'half', False) else model.float()
                model.eval()
            self.args.half = 'float16' in str(next(model.parameters()).dtype)
            self.stride = max(int(model.stride.max()), 32)
            imgsz = check_imgsz(self.args.imgsz, stride=self.stride)
            self.pt = True
            self.dynamic = getattr(model, 'dynamic', False)
            self.end2end = False
            if str(getattr(self.args, 'data', '')).rsplit(".", 1)[-1] in {"yaml", "yml"}:
                self.data = load_data_config(self.args.data)
            else:
                raise FileNotFoundError(f"Dataset '{getattr(self.args, 'data', None)}' for task={self.args.task} not found")
            if self.device.type in {"cpu", "mps"}:
                self.args.workers = 0
            if not (self.pt or (self.dynamic and not hasattr(model, 'imx'))):
                self.args.rect = False
            self.dataloader = self.dataloader or self.get_dataloader(self.data.get(getattr(self.args, 'split', 'val')), getattr(self.args, 'batch', 1), mode="val")
            model.eval()
            # Warmup
            warmup_imsz = (1 if self.pt else getattr(self.args, 'batch', 1), self.data.get('channels', 3), imgsz, imgsz)
            im = torch.empty(*warmup_imsz, dtype=torch.half if self.args.half else torch.float, device=self.device)
            model(im)  # warmup
        preprocess_time = 0.0
        inference_time = 0.0
        loss_time = 0.0
        postprocess_time = 0.0
        bar = TQDM.tqdm(self.dataloader, desc="Validation", total=len(self.dataloader))
        self.init_metrics(model.module if hasattr(model, 'module') else model)
        for batch_i, batch in enumerate(bar):
            self.batch_i = batch_i
            # Preprocess
            if self.device.type == 'cuda':
                torch.cuda.synchronize(self.device)
            start = time.perf_counter()
            batch = self.preprocess(batch)
            if self.device.type == 'cuda':
                torch.cuda.synchronize(self.device)
            preprocess_time += time.perf_counter() - start
            # Inference
            if self.device.type == 'cuda':
                torch.cuda.synchronize(self.device)
            start = time.perf_counter()
            preds = model(batch["img"], augment=augment)
            if self.device.type == 'cuda':
                torch.cuda.synchronize(self.device)
            inference_time += time.perf_counter() - start
            # Loss
            if self.device.type == 'cuda':
                torch.cuda.synchronize(self.device)
            start = time.perf_counter()
            if self.training:
                self.loss += model.loss(batch, preds)[1]
            if self.device.type == 'cuda':
                torch.cuda.synchronize(self.device)
            loss_time += time.perf_counter() - start
            # Postprocess
            if self.device.type == 'cuda':
                torch.cuda.synchronize(self.device)
            start = time.perf_counter()
            preds = self.postprocess(preds)
            if self.device.type == 'cuda':
                torch.cuda.synchronize(self.device)
            postprocess_time += time.perf_counter() - start
            self.update_metrics(preds, batch)
        stats = self.get_stats()
        num_images = len(self.dataloader.dataset)
        self.speed['preprocess'] = preprocess_time / num_images * 1000
        self.speed['inference'] = inference_time / num_images * 1000
        self.speed['loss'] = loss_time / num_images * 1000
        self.speed['postprocess'] = postprocess_time / num_images * 1000
        self.finalize_metrics()
        self.print_results()
        if self.training:
            model.float()
            results = {**stats, **trainer.label_loss_items(self.loss.cpu() / len(self.dataloader), prefix="val")}
            return {k: round(float(v), 5) for k, v in results.items()}  # return results as 5 decimal place floats
        else:
            LOGGER.info(
                "Speed: {:.1f}ms preprocess, {:.1f}ms inference, {:.1f}ms loss, {:.1f}ms postprocess per image".format(
                    *tuple(self.speed.values())
                )
            )
            if getattr(self.args, 'save_json', False) and self.jdict:
                pred_file = self.save_dir / "predictions.json"
                with open(pred_file, "w", encoding="utf-8") as f:
                    json.dump(self.jdict, f)
                LOGGER.info(f"Saved {len(self.jdict)} predictions to {pred_file}")
            LOGGER.info(f"Results saved to {self.save_dir}")
            return stats

    def match_predictions(
        self, pred_classes: torch.Tensor, true_classes: torch.Tensor, iou: torch.Tensor, use_scipy: bool = False
    ) -> torch.Tensor:
        """Match predictions to ground truth objects using IoU."""
        # Dx10 matrix, where D - detections, 10 - IoU thresholds
        correct = np.zeros((pred_classes.shape[0], self.iouv.shape[0])).astype(bool)
        # LxD matrix where L - labels (rows), D - detections (columns)
        correct_class = true_classes[:, None] == pred_classes
        iou = iou * correct_class  # zero out the wrong classes
        iou = iou.cpu().numpy()
        for i, threshold in enumerate(self.iouv.cpu().tolist()):
            matches = np.nonzero(iou >= threshold)  # IoU > threshold and classes match
            matches = np.array(matches).T
            if matches.shape[0]:
                if matches.shape[0] > 1:
                    matches = matches[iou[matches[:, 0], matches[:, 1]].argsort()[::-1]]
                    matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                    matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
                correct[matches[:, 1].astype(int), i] = True
        return torch.tensor(correct, dtype=torch.bool, device=pred_classes.device)

    def get_dataloader(self, dataset_path, batch_size, mode="val"):
        """Construct and return dataloader."""
        dataset = self.build_dataset(dataset_path, mode="val", batch=batch_size)
        shuffle = mode == "train"
        if getattr(dataset, "rect", False) and shuffle:
            LOGGER.warning("'rect=True' is incompatible with DataLoader shuffle, setting shuffle=False")
            shuffle = False
        return build_dataloader(
            dataset,
            batch_size,
            getattr(self.args, 'workers', 8),
            shuffle=shuffle,
            drop_last=getattr(self.args, 'compile', False),
            pin_memory=self.training,
        )

    def build_dataset(self, img_path: str, mode: str = "val", batch: int | None = None) -> torch.utils.data.Dataset:
        """Build YOLO Dataset."""
        return build_yolo_dataset(self.args, img_path, batch, self.data, mode=mode, stride=self.stride)

    def preprocess(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Preprocess batch of images for YOLO validation."""
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(self.device, non_blocking=self.device.type == "cuda")
        batch["img"] = (batch["img"].half() if getattr(self.args, 'half', False) else batch["img"].float()) / 255
        return batch

    def postprocess(self, preds: torch.Tensor) -> list[dict[str, torch.Tensor]]:
        """Apply Non-maximum suppression to prediction outputs."""
        outputs = non_max_suppression(
            preds,
            getattr(self.args, 'conf', 0.001),
            getattr(self.args, 'iou', 0.45),
            nc=0 if self.args.task == "detect" else self.nc,  # ← FIXED: Conditional nc
            multi_label=True,
            agnostic=getattr(self.args, 'single_cls', False) or getattr(self.args, 'agnostic_nms', False),
            max_det=getattr(self.args, 'max_det', 300),
            end2end=getattr(self, 'end2end', False),
            rotated=self.args.task == "obb" if hasattr(self.args, 'task') else False,  # ← FIXED: Conditional rotated
        )
        return [{"bboxes": x[:, :4], "conf": x[:, 4], "cls": x[:, 5], "extra": x[:, 6:]} for x in outputs]

    def init_metrics(self, model: torch.nn.Module) -> None:
        """Initialize evaluation metrics for YOLO detection validation."""
        val = self.data.get(getattr(self.args, 'split', 'val'), "")  # validation path
        self.is_coco = (
            isinstance(val, str)
            and "coco" in val
            and (val.endswith(f"{os.sep}val2017.txt") or val.endswith(f"{os.sep}test-dev2017.txt"))
        )  # is COCO - strict path check
        self.is_lvis = isinstance(val, str) and "lvis" in val and not self.is_coco  # is LVIS
        
        # For COCO: use coco80_to_coco91_class, otherwise sequential mapping
        if self.is_coco:
            # COCO 80 to 91 class mapping (hardcoded for compatibility)
            self.class_map = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90]
        else:
            # Sequential mapping for non-COCO datasets
            self.class_map = list(range(1, len(model.names) + 1))
        self.names = model.names
        self.nc = len(model.names)
        self.end2end = getattr(model, "end2end", False)
        self.seen = 0
        self.jdict = []
        self.metrics.names = model.names

    def update_metrics(self, preds: list[dict[str, torch.Tensor]], batch: dict[str, Any]) -> None:
        """Update metrics with new predictions and ground truth."""
        for si, pred in enumerate(preds):
            self.seen += 1
            pbatch = self._prepare_batch(si, batch)
            predn = self._prepare_pred(pred)
            cls = pbatch["cls"].cpu().numpy()
            no_pred = predn["cls"].shape[0] == 0
            self.metrics.update_stats(
                {
                    **self._process_batch(predn, pbatch),
                    "target_cls": cls,
                    "target_img": np.unique(cls),
                    "conf": np.zeros(0) if no_pred else predn["conf"].cpu().numpy(),
                    "pred_cls": np.zeros(0) if no_pred else predn["cls"].cpu().numpy(),
                }
            )
            
            if no_pred:
                continue
            
            # Save predictions
            if getattr(self.args, 'save_json', False) or getattr(self.args, 'save_txt', False):
                predn_scaled = self.scale_preds(predn, pbatch)
            if getattr(self.args, 'save_json', False):
                self.pred_to_json(predn_scaled, pbatch)
            if getattr(self.args, 'save_txt', False):
                self.save_one_txt(
                    predn_scaled,
                    getattr(self.args, 'save_conf', False),
                    pbatch["ori_shape"],
                    self.save_dir / "labels" / f"{Path(pbatch['im_file']).stem}.txt",
                )

    def finalize_metrics(self) -> None:
        """Set final values for metrics speed and confusion matrix."""
        self.metrics.speed = self.speed
        self.metrics.save_dir = self.save_dir

    def get_stats(self) -> dict[str, Any]:
        """Calculate and return metrics statistics."""
        self.metrics.process()
        self.metrics.clear_stats()
        return self.metrics.results_dict

    def print_results(self) -> None:
        """Print training/validation set metrics per class."""
        pf = "%22s" + "%11i" * 2 + "%11.3g" * len(self.metrics.keys)  # print format
        LOGGER.info(pf % ("all", self.seen, self.metrics.nt_per_class.sum(), *self.metrics.mean_results()))
        if self.metrics.nt_per_class.sum() == 0:
            LOGGER.warning(f"no labels found in {self.args.task} set, can not compute metrics without labels")
        # Print results per class
        if getattr(self.args, 'verbose', False) and not self.training and self.nc > 1 and len(self.metrics.stats):
            for i, c in enumerate(self.metrics.ap_class_index):
                LOGGER.info(
                    pf
                    % (
                        self.names[c],
                        self.metrics.nt_per_image[c],
                        self.metrics.nt_per_class[c],
                        *self.metrics.class_result(i),
                    )
                )

    def _prepare_batch(self, si: int, batch: dict[str, Any]) -> dict[str, Any]:
        """Prepare a batch of images and annotations for validation."""
        idx = batch["batch_idx"] == si
        cls = batch["cls"][idx].squeeze(-1)
        bbox = batch["bboxes"][idx]
        ori_shape = batch["ori_shape"][si]
        imgsz = batch["img"].shape[2:]
        ratio_pad = batch["ratio_pad"][si]
        if cls.shape[0]:
            bbox = xywh2xyxy(bbox) * torch.tensor(imgsz, device=self.device)[[1, 0, 1, 0]]  # target boxes
        return {
            "cls": cls,
            "bboxes": bbox,
            "ori_shape": ori_shape,
            "imgsz": imgsz,
            "ratio_pad": ratio_pad,
            "im_file": batch["im_file"][si],
        }

    def _prepare_pred(self, pred: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Prepare predictions for evaluation against ground truth."""
        if getattr(self.args, 'single_cls', False):
            pred["cls"] *= 0
        return pred

    def scale_preds(self, predn: dict[str, torch.Tensor], pbatch: dict[str, Any]) -> dict[str, torch.Tensor]:
        from yolo11.utils.bbox_utils import scale_boxes
        
        return {
            **predn,
            "bboxes": scale_boxes(
                pbatch["imgsz"],
                predn["bboxes"].clone(),
                pbatch["ori_shape"],
                ratio_pad=pbatch["ratio_pad"],
            ),
        }

    def save_one_txt(self, predn: dict[str, torch.Tensor], save_conf: bool, shape: tuple[int, int], file: Path) -> None:
        from yolo11.evaluation.results import Results

        Results(
            np.zeros((shape[0], shape[1]), dtype=np.uint8),
            path=None,
            names=self.names,
            boxes=torch.cat([predn["bboxes"], predn["conf"].unsqueeze(-1), predn["cls"].unsqueeze(-1)], dim=1),
        ).save_txt(file, save_conf=save_conf)

    def pred_to_json(self, predn: dict[str, torch.Tensor], pbatch: dict[str, Any]) -> None:
        from yolo11.utils.bbox_utils import xyxy2xywh
        
        path = Path(pbatch["im_file"])
        stem = path.stem
        image_id = int(stem) if stem.isnumeric() else stem
        box = xyxy2xywh(predn["bboxes"])  # xywh
        box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
        for b, s, c in zip(box.tolist(), predn["conf"].tolist(), predn["cls"].tolist()):
            self.jdict.append(
                {
                    "image_id": image_id,
                    "file_name": path.name,
                    "category_id": self.class_map[int(c)],
                    "bbox": [round(x, 3) for x in b],
                    "score": round(s, 5),
                }
            )

    def _process_batch(self, preds: dict[str, torch.Tensor], batch: dict[str, Any]) -> dict[str, np.ndarray]:
        """Return correct prediction matrix."""
        if batch["cls"].shape[0] == 0 or preds["cls"].shape[0] == 0:
            return {"tp": np.zeros((preds["cls"].shape[0], self.niou), dtype=bool)}
        iou = box_iou(batch["bboxes"], preds["bboxes"])
        return {"tp": self.match_predictions(preds["cls"], batch["cls"], iou).cpu().numpy()}