"""Export a trained YOLO11 checkpoint to deployment formats (ONNX, TorchScript)."""

import inspect
import json
import logging
from pathlib import Path

import torch

from yolo11.inference.predictor import resolve_device
from yolo11.models.head import Detect
from yolo11.utils.torch_utils import check_imgsz, load_checkpoint

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

    The exported graph takes a ``(B, 3, H, W)`` RGB tensor scaled to ``[0, 1]`` and returns ``(B, 4 + nc, N)`` with
    ``xywh`` boxes in input pixels followed by per-class sigmoid scores. NMS is *not* part of the graph.
    """
    if fmt not in FORMATS:
        raise ValueError(f"Unsupported export format '{fmt}'. Choose from {FORMATS}.")
    device = resolve_device(device)
    half = half and device.type != "cpu"

    model = load_checkpoint(weights, device=device, fuse=True)[0]
    model = (model.half() if half else model.float()).eval()
    for p in model.parameters():
        p.requires_grad = False
    detect = model.model[-1]
    if not isinstance(detect, Detect):
        raise TypeError(f"Expected a Detect head, got {type(detect).__name__}")
    detect.export = True
    detect.format = fmt

    stride = max(int(model.stride.max()), 32)
    train_imgsz = (getattr(model, "args", None) or {}).get("imgsz")
    imgsz = check_imgsz(imgsz or train_imgsz or 640, stride=stride, max_dim=2)
    h, w = (imgsz, imgsz) if isinstance(imgsz, int) else imgsz
    im = torch.zeros(batch, 3, h, w, device=device, dtype=torch.half if half else torch.float)
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

    if fmt == "onnx":
        _export_onnx(model, im, output, opset, dynamic, simplify)
    else:
        torch.jit.trace(model, im, strict=False).save(str(output))

    names = getattr(model, "names", {})
    meta = {"names": dict(enumerate(names)) if isinstance(names, (list, tuple)) else names, "imgsz": [h, w], "stride": stride}
    output.with_suffix(".json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    LOGGER.info(f"Exported {weights} -> {output} ({output.stat().st_size / 1e6:.1f} MB)")
    return output


def _export_onnx(model, im, output: Path, opset: int, dynamic: bool, simplify: bool) -> None:
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
    if simplify:
        try:
            import onnx
            import onnxsim

            model_onnx, ok = onnxsim.simplify(onnx.load(str(output)))
            if ok:
                onnx.save(model_onnx, str(output))
            else:
                LOGGER.warning("onnxsim could not validate the simplified model; keeping the original.")
        except ImportError:
            LOGGER.warning("`onnxsim` is not installed; skipping simplification (pip install onnxsim).")
