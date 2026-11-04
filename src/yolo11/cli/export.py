import argparse

from yolo11.utils.logger import get_logger


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--weights", required=True, help="Checkpoint (.pt)")
    parser.add_argument("--format", choices=("onnx", "torchscript"), default="onnx")
    parser.add_argument("--output", help="Output file or directory (default: next to the weights)")
    parser.add_argument("--imgsz", type=int, nargs="+", help="Input size: H or H W (default: training size)")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--half", action="store_true", help="FP16 (CUDA only)")
    parser.add_argument("--dynamic", action="store_true", help="ONNX: dynamic batch/height/width")
    parser.add_argument("--simplify", action="store_true", help="ONNX: run onnxsim (pip install onnxsim)")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset")


def run(args: argparse.Namespace):
    from yolo11.export.exporter import export_model

    get_logger("yolo11")
    imgsz = None
    if args.imgsz:
        imgsz = args.imgsz[0] if len(args.imgsz) == 1 else list(args.imgsz[:2])
    return export_model(
        args.weights, fmt=args.format, output=args.output, imgsz=imgsz, batch=args.batch, device=args.device,
        half=args.half, dynamic=args.dynamic, simplify=args.simplify, opset=args.opset,
    )
