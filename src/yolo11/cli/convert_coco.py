import argparse

from yolo11.data.convert_coco import convert_coco_json, verify_output


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--json", required=True, help="COCO instances_*.json")
    parser.add_argument("--output", required=True, help="Directory for the YOLO label .txt files")
    parser.add_argument("--verify-count", type=int, default=0, help="Verify the output has this many label files")


def run(args: argparse.Namespace) -> None:
    convert_coco_json(args.json, args.output)
    if args.verify_count > 0:
        verify_output(args.output, args.verify_count)
