"""Command line entry point: ``yolo11 <train|val|predict|export|convert-coco> [options]``."""

import argparse
import sys

from yolo11 import __version__
from yolo11.cli import convert_coco, evaluate, export, predict, train

COMMANDS = {
    "train": (train, "Train a detection model"),
    "val": (evaluate, "Evaluate a checkpoint on a dataset split (mAP)"),
    "predict": (predict, "Run inference on images"),
    "export": (export, "Export a checkpoint to ONNX / TorchScript"),
    "convert-coco": (convert_coco, "Convert COCO annotations to YOLO labels"),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="yolo11", description="YOLO11 object detection")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)
    for name, (module, help_text) in COMMANDS.items():
        module.add_arguments(sub.add_parser(name, help=help_text, description=help_text))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    COMMANDS[args.command][0].run(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
