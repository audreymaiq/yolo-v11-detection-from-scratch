import argparse

from yolo11.data.convert_coco import convert_coco_json


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--json", required=True, help="COCO instances_*.json")
    parser.add_argument("--output", required=True, help="Directory for the YOLO label .txt files")
    parser.add_argument("--verify-count", type=int, default=0, help="Verify the output has this many label files")


def run(args: argparse.Namespace) -> None:
    stats = convert_coco_json(args.json, args.output, expected_count=args.verify_count or None)
    print(f"Converted {stats['images']} images and {stats['written']} boxes to {args.output}")
    print(f"Skipped: {stats['crowd']} crowd, {stats['invalid']} invalid, {stats['duplicates']} duplicates; clipped {stats['clipped']} boxes")