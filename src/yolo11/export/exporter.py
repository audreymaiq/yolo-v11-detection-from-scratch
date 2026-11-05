"""Export a trained YOLO11 checkpoint to deployment formats (ONNX, TorchScript)."""

import inspect
import json
import logging
from datetime import datetime
from pathlib import Path

import torch

from yolo11 import __version__
from yolo11.inference.backends import TORCHSCRIPT_METADATA
from yolo11.models.head import Detect
from yolo11.utils.torch_utils import check_imgsz, load_checkpoint, select_device

LOGGER = logging.getLogger(__name__)

FORMATS = ("onnx", "torchscript")


def export_model(
    weights: str | Path,
    fmt: str = "onnx",
    output: str | Path | None = None,
    imgsz: int | list[int] | None = None,
    batch: int = 1,
    device: str | int | None = "cpu",
    half: bool = False,
    dynamic: bool = False,
    simplify: bool = False,
    opset: int = 17,
) -> Path:
    """Export ``weights`` and return the path of the written file.

    The exported graph takes a ``(B, C, H, W)`` RGB (or grayscale) tensor scaled to ``[0, 1]`` and returns
    ``(B, 4 + nc, N)`` with ``xywh`` boxes in input pixels followed by per-class sigmoid scores. NMS is *not* part of
    the graph. Class names, stride, input size and export options are embedded in the file (ONNX
    ``metadata_props``, TorchScript extra file ``config.txt``); ``yolo11.inference.Predictor`` reads them.
    """
    if fmt not in FORMATS:
        raise ValueError(f"Unsupported export format '{fmt}'. Choose from {FORMATS}.")
    device = select_device(device)
    half = half and device.type != "cpu"

    model = load_checkpoint(weights, device=device, fuse=True)[0]
    model = (model.half() if half else model.float()).eval()
    for p in model.parameters():
        p.requires_grad = False
    detect = model.model[-1]
    if not isinstance(detect, Detect):
        raise TypeError(f"Expected a Detect head, got {type(detect).__name__}")
    detect.dynamic = dynamic  # rebuild anchors from the input shape instead of tracing them as constants
    detect.export = True
    detect.format = fmt

    stride = max(int(model.stride.max()), 32)
    train_imgsz = (getattr(model, "args", None) or {}).get("imgsz")
    imgsz = check_imgsz(imgsz or train_imgsz or 640, stride=stride, max_dim=2)
    h, w = (imgsz, imgsz) if isinstance(imgsz, int) else imgsz
    channels = int(getattr(model, "yaml", {}).get("channels", 3))
    im = torch.zeros(batch, channels, h, w, device=device, dtype=torch.half if half else torch.float)
    with torch.no_grad():
        for _ in range(2):  # dry runs build anchors and catch shape errors before export
            y = model(im)
    LOGGER.info(f"Output shape: {tuple(y.shape)}")

    weights = Path(weights)
    ext = {"onnx": ".onnx", "torchscript": ".torchscript"}[fmt]
    output = Path(output) if output else weights.with_suffix(ext)
    if output.is_dir() or output.suffix == "":
        output = output / weights.with_suffix(ext).name
    output.parent.mkdir(parents=True, exist_ok=True)

    names = getattr(model, "names", {})
    metadata = {
        "description": f"YOLO11 detection model exported from {weights.name}",
        "version": __version__,
        "date": datetime.now().astimezone().isoformat(),
        "license": "AGPL-3.0",
        "task": "detect",
        "stride": stride,
        "batch": batch,
        "imgsz": [h, w],
        "channels": channels,
        "names": dict(enumerate(names)) if isinstance(names, (list, tuple)) else dict(names),
        "dynamic": dynamic,
        "half": half,
    }
    if fmt == "onnx":
        _export_onnx(model, im, output, opset, dynamic, simplify, metadata)
    else:
        ts = torch.jit.trace(model, im, strict=False)
        ts.save(str(output), _extra_files={TORCHSCRIPT_METADATA: json.dumps(metadata, ensure_ascii=False)})
    LOGGER.info(f"Exported {weights} -> {output} ({output.stat().st_size / 1e6:.1f} MB)")
    return output


def _export_onnx(model, im, output: Path, opset: int, dynamic: bool, simplify: bool, metadata: dict) -> None:
    import onnx

    dynamic_axes = None
    if dynamic:
        dynamic_axes = {"images": {0: "batch", 2: "height", 3: "width"}, "output0": {0: "batch", 2: "anchors"}}
    kwargs = {"dynamo": False} if "dynamo" in inspect.signature(torch.onnx.export).parameters else {}  # torch>=2.5
    torch.onnx.export(
        model,
        im,
        str(output),
        opset_version=opset,
        input_names=["images"],
        output_names=["output0"],
        dynamic_axes=dynamic_axes,
        **kwargs,
    )
    model_onnx = onnx.load(str(output))
    if simplify:
        try:
            import onnxsim

            simplified, ok = onnxsim.simplify(model_onnx)
            if ok:
                model_onnx = simplified
            else:
                LOGGER.warning("onnxsim could not validate the simplified model; keeping the original.")
        except ImportError:
            LOGGER.warning("`onnxsim` is not installed; skipping simplification (pip install onnxsim).")
    for key, value in metadata.items():  # JSON values, readable from any language through the ONNX model metadata
        prop = model_onnx.metadata_props.add()
        prop.key, prop.value = key, json.dumps(value, ensure_ascii=False)
    onnx.save(model_onnx, str(output))
