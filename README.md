# YOLO11 Detection

YOLO11 object detection implemented in PyTorch (train from scratch or from a checkpoint, evaluate with COCO mAP,
predict, export to ONNX / TorchScript).

## Install

The exact versions every environment uses (development, CI, Docker) are pinned in `uv.lock`:

```bash
uv sync --extra export                     # .venv with the locked versions, ONNX export and dev tools (pytest, ruff)
uv sync --extra export --extra tracking    # + MLflow / W&B
```

Without uv, pip resolves the ranges in `pyproject.toml` instead (versions may differ from the lock):

```bash
pip install -e ".[export]"   # core + ONNX export
pip install --group dev      # pytest, ruff (pip >= 25.1)
```

After changing dependencies in `pyproject.toml`, run `uv lock` and commit `uv.lock`; CI fails while they disagree.

### Docker

The image installs the locked versions; Linux torch wheels bundle CUDA, so the host needs only the NVIDIA driver and
the NVIDIA Container Toolkit:

```bash
docker build --build-arg GIT_COMMIT=$(git rev-parse HEAD) -t yolo11 .
docker run --gpus all --shm-size=8g -v "$PWD/datasets:/app/datasets" -v "$PWD/runs:/app/runs"     yolo11 train --config configs/training.yaml
```

`GIT_COMMIT` is what the run's provenance records as its commit, since the image has no `.git`.

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

## Runs: provenance, tracking, resume

Each run directory holds `args.yaml` (the full config), `results.csv` (per-epoch losses, metrics and learning rates)
and `provenance.json`: the git commit and whether the checkout had uncommitted changes, Python/OS/package versions and
device, and a SHA-256 fingerprint of the train and val data (image paths relative to the dataset root, image sizes and
labels). The same provenance is stored in every checkpoint, and `yolo11 export` copies its essentials, with the
checkpoint's SHA-256, into the exported model's `source` metadata.

Set `mlflow: True` or `wandb: True` in the config to log the config, provenance, per-epoch metrics and, at the end, the
run directory (`MLFLOW_TRACKING_URI`, `EXPERIMENT_NAME`, `PROJECT_NAME` select the server/project).

`--resume runs/<name>/weights/last.pt` continues an interrupted run exactly: checkpoints written during training keep
the FP32 weights, optimizer, EMA, gradients accumulated since the last optimizer step, early-stopping state and the
random number generators, and the shuffle order depends only on `seed` and the epoch. With `workers: 0` and
`deterministic: True` a resumed run ends bit-identical to one that never stopped (`tests/test_resume.py`, on CPU; on
GPU, CUDA kernels without a deterministic variant can still make runs differ slightly); with worker
processes the augmentations after a resume are reproducible but not the ones the uninterrupted run would have drawn.
Resuming on data whose fingerprint changed logs a warning. Training state is stripped from `last.pt`/`best.pt` (and
weights halved) when training finishes.

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
  utils/            bbox ops, torch helpers, plotting, logging, config, provenance
tests/              pytest suite
runs/               training outputs (git-ignored)
.github/workflows/  CI: lock check + ruff, tests (Linux py3.10/3.13, Windows py3.11), Docker build
```

## Notes

- Checkpoints saved before the package was renamed from `sgod_model_builder` to `yolo11` still load
  (see `yolo11/utils/legacy.py`).
- `yolo11 predict` and `yolo11.inference.Predictor` run `.pt`, `.onnx` and `.torchscript` models with the same
  preprocessing and NMS. ONNX needs the `export` extra (onnxruntime); CUDA is used when onnxruntime has it.
- Exported models output raw `(B, 4+nc, N)` predictions (xywh pixels + class scores); NMS is not in the graph.
  Class names, stride, input size, batch and export options are embedded in the file as JSON values (ONNX
  `metadata_props`, TorchScript extra file `config.txt`). Exports without `--dynamic` only accept the exported batch
  size and input size; `Predictor` pads images and batches to fit. Exports made before metadata was embedded are
  still read from their `<model>.json`.

## License

The architecture, loss and training logic follow [Ultralytics](https://github.com/ultralytics/ultralytics)
(AGPL-3.0); redistribution of this code is subject to that license.
