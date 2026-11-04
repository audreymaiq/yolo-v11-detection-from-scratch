import argparse
import json
import logging
from pathlib import Path

from yolo11.utils.config import load_config
from yolo11.utils.logger import get_logger


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--weights", required=True, help="Checkpoint (.pt)")
    parser.add_argument("--data", required=True, help="Dataset YAML")
    parser.add_argument("--config", help="Optional training config YAML providing defaults (imgsz, conf, iou, ...)")
    parser.add_argument("--split", default="val", help="Dataset split: val | test (default: val)")
    parser.add_argument("--imgsz", type=int, help="Inference size (default: from config, else 640)")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--conf", type=float, help="Confidence threshold (default 0.001)")
    parser.add_argument("--iou", type=float, help="NMS IoU threshold (default 0.7)")
    parser.add_argument("--max-det", type=int, help="Max detections per image (default 300)")
    parser.add_argument("--device", help="cpu, 0, cuda:0 (default: cuda if available)")
    parser.add_argument("--half", action="store_true", help="FP16 inference (CUDA only)")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--save-json", action="store_true", help="Also write COCO-style predictions.json")
    parser.add_argument("--save-dir", default="runs/val", help="Output directory (default: runs/val)")
    parser.add_argument("--output", help="Write the metrics to this JSON file")


def run(args: argparse.Namespace) -> dict:
    import torch

    from yolo11.evaluation.validator import DetectionValidator
    from yolo11.inference.predictor import resolve_device

    get_logger("yolo11")
    base = load_config(args.config) if args.config else {}
    options = {"imgsz": args.imgsz, "conf": args.conf, "iou": args.iou, "max_det": args.max_det}
    cfg = {**base, **{k: v for k, v in options.items() if v is not None}}
    cfg.update(
        data=args.data, split=args.split, batch=args.batch, half=args.half, save_json=args.save_json,
        workers=args.workers, rect=False, device=str(resolve_device(args.device)), save_dir=args.save_dir,
        name=None, plots=False, verbose=True,
    )
    cfg.setdefault("imgsz", 640)
    cfg.setdefault("conf", 0.001)
    cfg.setdefault("iou", 0.7)
    cfg.setdefault("max_det", 300)

    validator = DetectionValidator(save_dir=Path(args.save_dir), args=cfg)
    with torch.no_grad():
        stats = validator(model=args.weights)
    stats = {k: float(v) for k, v in stats.items()}
    logging.getLogger("yolo11").info("Metrics: " + json.dumps(stats))
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return stats
