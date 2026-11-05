"""Run a checkpoint or an exported model behind one interface.

Every backend takes a ``(B, C, H, W)`` tensor scaled to ``[0, 1]`` and returns raw predictions ``(B, 4 + nc, N)``
(``xywh`` boxes in input pixels followed by class scores), so preprocessing and NMS are shared by all formats.
Exported models describe themselves through the metadata that ``yolo11 export`` embeds in the file.
"""

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch

from yolo11.utils.torch_utils import load_checkpoint

LOGGER = logging.getLogger(__name__)

TORCHSCRIPT_METADATA = "config.txt"  # torch.jit extra file holding the export metadata, as in the reference

_ORT_DTYPES = {"tensor(float16)": np.float16, "tensor(float)": np.float32}


def read_sidecar_metadata(path: Path) -> dict[str, Any]:
    """Metadata of exports made before it was embedded, which wrote it to ``<model>.json``."""
    sidecar = path.with_suffix(".json")
    return json.loads(sidecar.read_text(encoding="utf-8")) if sidecar.is_file() else {}


class Backend:
    """Common attributes of every backend.

    Attributes:
        format (str): ``pt``, ``torchscript`` or ``onnx``.
        names (dict[int, str]): Class names.
        stride (int): Largest model stride.
        imgsz (list[int] | None): ``[h, w]`` the model was exported or trained at.
        batch (int | None): Fixed batch size, or ``None`` when any batch size is accepted.
        dynamic (bool): Whether the input height and width may differ from ``imgsz``.
        channels (int): Number of input channels.
        fp16 (bool): Whether the model expects half-precision input.
    """

    format = ""

    def __init__(self, device: torch.device):
        self.device = device
        self.names: dict[int, str] = {}
        self.stride = 32
        self.imgsz: list[int] | None = None
        self.batch: int | None = None
        self.dynamic = False
        self.channels = 3
        self.fp16 = False

    def apply_metadata(self, metadata: dict[str, Any], path: Path) -> None:
        if not metadata:
            raise ValueError(
                f"{path} has no yolo11 metadata (class names, stride, input size). Export it with `yolo11 export`."
            )
        self.names = {int(k): v for k, v in metadata["names"].items()}
        self.stride = int(metadata["stride"])
        imgsz = metadata["imgsz"]
        self.imgsz = [int(imgsz), int(imgsz)] if isinstance(imgsz, (int, str)) else [int(x) for x in imgsz]
        self.channels = int(metadata.get("channels", 3))

    def forward(self, im: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def __call__(self, im: torch.Tensor) -> torch.Tensor:
        return self.forward(im)


class PyTorchBackend(Backend):
    """A ``.pt`` checkpoint; accepts any batch and any input size that is a multiple of the stride."""

    format = "pt"

    def __init__(self, weights: Path, device: torch.device, half: bool = False):
        super().__init__(device)
        self.model = load_checkpoint(weights, device=device, fuse=True)[0]
        self.fp16 = half and device.type != "cpu"
        self.model = (self.model.half() if self.fp16 else self.model.float()).eval()
        names = getattr(self.model, "names", None) or {}
        names = dict(enumerate(names)) if isinstance(names, (list, tuple)) else names
        self.names = {int(k): v for k, v in names.items()}
        self.stride = max(int(self.model.stride.max()), 32)
        args = getattr(self.model, "args", None)
        train_imgsz = args.get("imgsz") if isinstance(args, dict) else getattr(args, "imgsz", None)
        if train_imgsz:
            self.imgsz = [train_imgsz, train_imgsz] if isinstance(train_imgsz, int) else list(train_imgsz)
        self.channels = int(getattr(self.model, "yaml", {}).get("channels", 3))
        self.dynamic = True

    def forward(self, im: torch.Tensor) -> torch.Tensor:
        y = self.model(im)
        return y[0] if isinstance(y, (list, tuple)) else y


class TorchScriptBackend(Backend):
    """A ``.torchscript`` file written by ``yolo11 export``; traced for one input shape."""

    format = "torchscript"

    def __init__(self, weights: Path, device: torch.device, half: bool = False):
        super().__init__(device)
        extra_files = {TORCHSCRIPT_METADATA: ""}
        self.model = torch.jit.load(str(weights), map_location=device, _extra_files=extra_files).eval()
        metadata = json.loads(extra_files[TORCHSCRIPT_METADATA]) if extra_files[TORCHSCRIPT_METADATA] else {}
        self.apply_metadata(metadata or read_sidecar_metadata(weights), weights)
        self.fp16 = bool(metadata.get("half", False)) or (half and device.type != "cpu")
        self.model = self.model.half() if self.fp16 else self.model.float()
        self.batch = int(metadata.get("batch", 1))  # a traced graph only supports the traced batch and size

    def forward(self, im: torch.Tensor) -> torch.Tensor:
        y = self.model(im)
        return y[0] if isinstance(y, (list, tuple)) else y


class ONNXBackend(Backend):
    """An ``.onnx`` model run with ONNX Runtime; CUDA is used when requested and available."""

    format = "onnx"

    def __init__(self, weights: Path, device: torch.device):
        super().__init__(device)
        try:
            import onnxruntime
        except ImportError as e:
            raise ImportError("ONNX inference requires onnxruntime: pip install -e \".[export]\"") from e

        providers = ["CPUExecutionProvider"]
        if device.type == "cuda":
            if "CUDAExecutionProvider" in onnxruntime.get_available_providers():
                providers.insert(0, ("CUDAExecutionProvider", {"device_id": device.index or 0}))
            else:
                LOGGER.warning("CUDA requested but onnxruntime has no CUDAExecutionProvider; running ONNX on CPU.")
                self.device = torch.device("cpu")
        self.session = onnxruntime.InferenceSession(str(weights), providers=providers)

        metadata = {k: json.loads(v) for k, v in self.session.get_modelmeta().custom_metadata_map.items()}
        self.apply_metadata(metadata or read_sidecar_metadata(weights), weights)
        model_input = self.session.get_inputs()[0]
        if model_input.type not in _ORT_DTYPES:
            raise TypeError(f"Unsupported ONNX input type {model_input.type}")
        self.input_name = model_input.name
        self.input_dtype = _ORT_DTYPES[model_input.type]
        self.fp16 = self.input_dtype == np.float16
        batch, _, height, width = model_input.shape
        self.batch = batch if isinstance(batch, int) else None
        self.dynamic = not (isinstance(height, int) and isinstance(width, int))
        if not self.dynamic:
            self.imgsz = [height, width]

    def forward(self, im: torch.Tensor) -> torch.Tensor:
        x = im.cpu().numpy().astype(self.input_dtype, copy=False)
        y = self.session.run(None, {self.input_name: x})[0]
        return torch.from_numpy(y).to(self.device).float()


def load_backend(weights: str | Path, device: torch.device, half: bool = False) -> Backend:
    """Pick the backend from the file suffix: ``.pt``, ``.torchscript`` or ``.onnx``."""
    weights = Path(weights)
    if not weights.is_file():
        raise FileNotFoundError(f"Model not found: {weights}")
    suffix = weights.suffix.lower()
    if suffix == ".pt":
        return PyTorchBackend(weights, device, half)
    if suffix == ".torchscript":
        return TorchScriptBackend(weights, device, half)
    if suffix == ".onnx":
        if half:
            LOGGER.info("--half is ignored for ONNX: precision is fixed at export (yolo11 export --half).")
        return ONNXBackend(weights, device)
    raise ValueError(f"Unsupported model format '{suffix}'. Use a .pt checkpoint or a .onnx/.torchscript export.")
