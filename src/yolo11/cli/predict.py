import argparse
import json
import time
from pathlib import Path

from yolo11.utils.logger import get_logger


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--weights", required=True, help="Checkpoint (.pt)")
    parser.add_argument("--source", required=True, help="Image file or directory of images")
    parser.add_argument("--imgsz", type=int, help="Inference size (default: the size used in training)")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.7, help="NMS IoU threshold")
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--classes", type=int, nargs="+", help="Keep only these class ids")
    parser.add_argument("--agnostic-nms", action="store_true")
    parser.add_argument("--device", help="cpu, 0, cuda:0 (default: cuda if available)")
    parser.add_argument("--half", action="store_true", help="FP16 inference (CUDA only)")
    parser.add_argument("--output-dir", default="runs/predict", help="Where to save results")
    parser.add_argument("--save-json", action="store_true", help="Write detections to predictions.json")
    parser.add_argument("--no-save", action="store_true", help="Do not save annotated images")


def run(args: argparse.Namespace) -> dict[str, list]:
    import cv2

    from yolo11.inference.predictor import Predictor, collect_images
    from yolo11.utils.visualize_utils import draw_detections

    logger = get_logger("yolo11")
    images = collect_images(args.source)
    if not images:
        logger.warning(f"No images found in {args.source}")
        return {}

    predictor = Predictor(
        args.weights, device=args.device, imgsz=args.imgsz, conf=args.conf, iou=args.iou,
        max_det=args.max_det, half=args.half, classes=args.classes, agnostic=args.agnostic_nms,
    )
    out_dir = Path(args.output_dir) / time.strftime("exp_%Y%m%d_%H%M%S")
    results: dict[str, list] = {}
    for path in images:
        img = predictor.read_image(path)
        t0 = time.perf_counter()
        detections = predictor.predict(img)
        logger.info(f"{path.name}: {len(detections)} objects ({(time.perf_counter() - t0) * 1000:.1f} ms)")
        results[str(path)] = detections
        if not args.no_save:
            out_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out_dir / path.name), draw_detections(img, detections))
    if args.save_json:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "predictions.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    if not args.no_save or args.save_json:
        logger.info(f"Results saved to {out_dir.resolve()}")
    return results
