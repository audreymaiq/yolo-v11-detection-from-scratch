# YOLO11 Detection

YOLO11 object detection implemented in PyTorch (train from scratch or from a checkpoint, evaluate with COCO mAP,
predict, export to ONNX / TorchScript).

## Install

```bash
pip install -e .            # core
pip install -e ".[export]"  # + ONNX export
pip install -e ".[dev]"     # + pytest
```

## Usage

All commands are available as `yolo11 <command>` or `python -m yolo11 <command>`.

```bash
# Train (edit configs/training.yaml first: data, model size, epochs, ...)
yolo11 train --config configs/training.yaml
yolo11 train --config configs/training.yaml --resume runs/COCO2017/weights/last.pt

# Evaluate (mAP50 / mAP50-95)
yolo11 val --weights runs/COCO2017/weights/best.pt --data datasets/COCO2017/data.yaml

# Predict on an image or a folder
yolo11 predict --weights runs/COCO2017/weights/best.pt --source path/to/images --conf 0.25

# Export, then predict with the exported model
yolo11 export --weights runs/COCO2017/weights/best.pt --format onnx --simplify
yolo11 predict --weights runs/COCO2017/weights/best.onnx --source path/to/images --batch 8

# COCO json -> YOLO labels
yolo11 convert-coco --json annotations/instances_val2017.json --output labels/val2017
```

Dataset layout follows the YOLO convention: `images/<split>/x.jpg` with labels in `labels/<split>/x.txt`
(`class cx cy w h`, normalized), described by a `data.yaml` (`path`, `train`, `val`, `names`).

## Layout

```
configs/            training configs (training.yaml, debug.yaml)
src/yolo11/
  cli/              command line interface (train, val, predict, export, convert-coco)
  models/           architecture (blocks, Detect head, DetectionModel) + yolo11.yaml (n/s/m/l/x)
  data/             dataset, dataloader, augmentations, COCO converter
  training/         trainer, losses, TaskAlignedAssigner, optimizers
  evaluation/       validator, mAP metrics, results containers
  inference/        Predictor and backends for .pt, .onnx and .torchscript models
  export/           ONNX / TorchScript exporter
  postprocessing/   NMS
  registry/         model / loss / optimizer / metric registries
  experiments/      optional MLflow / W&B tracking
  utils/            bbox ops, torch helpers, plotting, logging, config
tests/              pytest suite
runs/               training outputs (git-ignored)
```

## Notes

- Checkpoints saved before the package was renamed from `sgod_model_builder` to `yolo11` still load
  (see `yolo11/utils/legacy.py`).
- `yolo11 predict` and `yolo11.inference.Predictor` run `.pt`, `.onnx` and `.torchscript` models with the same
  preprocessing and NMS. ONNX needs `pip install -e ".[export]"` (onnxruntime); CUDA is used when onnxruntime has it.
- Exported models output raw `(B, 4+nc, N)` predictions (xywh pixels + class scores); NMS is not in the graph.
  Class names, stride, input size, batch and export options are embedded in the file as JSON values (ONNX
  `metadata_props`, TorchScript extra file `config.txt`). Exports without `--dynamic` only accept the exported batch
  size and input size; `Predictor` pads images and batches to fit. Exports made before metadata was embedded are
  still read from their `<model>.json`.

## License

The architecture, loss and training logic follow [Ultralytics](https://github.com/ultralytics/ultralytics)
(AGPL-3.0); redistribution of this code is subject to that license.
