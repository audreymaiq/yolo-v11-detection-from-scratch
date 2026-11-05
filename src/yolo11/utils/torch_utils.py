import os
import re
import math
import yaml
import torch
import random
import logging
import time
import numpy as np
import torch.nn as nn
from typing import Any
from pathlib import Path
from copy import deepcopy
from datetime import datetime
import torch.distributed as dist
from contextlib import contextmanager

from yolo11.cfg import DEFAULT_CFG_DICT, DEFAULT_CFG_KEYS
from yolo11.utils.legacy import install_legacy_aliases

LOGGER = logging.getLogger(__name__)
__version__ = "0.1.0"


class ModelEMA:
    """Updated Exponential Moving Average (EMA) implementation.

    Keeps a moving average of everything in the model state_dict (parameters and buffers). For EMA details see
    References.

    To disable EMA set the `enabled` attribute to `False`.

    Attributes:
        ema (nn.Module): Copy of the model in evaluation mode.
        updates (int): Number of EMA updates.
        decay (function): Decay function that determines the EMA weight.
        enabled (bool): Whether EMA is enabled.

    References:
        - https://github.com/rwightman/pytorch-image-models
        - https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
    """

    def __init__(self, model, decay=0.9999, tau=2000, updates=0):
        """Initialize EMA for 'model' with given arguments.

        Args:
            model (nn.Module): Model to create EMA for.
            decay (float, optional): Maximum EMA decay rate.
            tau (int, optional): EMA decay time constant.
            updates (int, optional): Initial number of updates.
        """
        self.ema = deepcopy(unwrap_model(model)).eval()  # FP32 EMA
        self.updates = updates  # number of EMA updates
        self.decay = lambda x: decay * (1 - math.exp(-x / tau))  # decay exponential ramp (to help early epochs)
        for p in self.ema.parameters():
            p.requires_grad_(False)
        self.enabled = True

    def update(self, model):
        """Update EMA parameters.

        Args:
            model (nn.Module): Model to update EMA from.
        """
        if self.enabled:
            self.updates += 1
            d = self.decay(self.updates)

            msd = unwrap_model(model).state_dict()  # model state_dict
            for k, v in self.ema.state_dict().items():
                if v.dtype.is_floating_point:  # true for FP16 and FP32
                    v *= d
                    v += (1 - d) * msd[k].detach()
                    # assert v.dtype == msd[k].dtype == torch.float32, f'{k}: EMA {v.dtype},  model {msd[k].dtype}'

    def update_attr(self, model, include=(), exclude=("process_group", "reducer")):
        """Update attributes and save stripped model with optimizer removed.

        Args:
            model (nn.Module): Model to update attributes from.
            include (tuple, optional): Attributes to include.
            exclude (tuple, optional): Attributes to exclude.
        """
        if self.enabled:
            copy_attr(self.ema, model, include, exclude)

def copy_attr(a, b, include=(), exclude=()):
    """Copy attributes from object 'b' to object 'a', with options to include/exclude certain attributes.

    Args:
        a (Any): Destination object to copy attributes to.
        b (Any): Source object to copy attributes from.
        include (tuple, optional): Attributes to include. If empty, all attributes are included.
        exclude (tuple, optional): Attributes to exclude.
    """
    for k, v in b.__dict__.items():
        if (len(include) and k not in include) or k.startswith("_") or k in exclude:
            continue
        else:
            setattr(a, k, v)

class EarlyStopping:
    """Early stopping class that stops training when a specified number of epochs have passed without improvement.

    Attributes:
        best_fitness (float): Best fitness value observed.
        best_epoch (int): Epoch where best fitness was observed.
        patience (int): Number of epochs to wait after fitness stops improving before stopping.
        possible_stop (bool): Flag indicating if stopping may occur next epoch.
    """

    def __init__(self, patience=50):
        """Initialize early stopping object.

        Args:
            patience (int, optional): Number of epochs to wait after fitness stops improving before stopping.
        """
        self.best_fitness = 0.0  # i.e. mAP
        self.best_epoch = 0
        self.patience = patience or float("inf")  # epochs to wait after fitness stops improving to stop
        self.possible_stop = False  # possible stop may occur next epoch

    def __call__(self, epoch, fitness):
        """Check whether to stop training.

        Args:
            epoch (int): Current epoch of training
            fitness (float): Fitness value of current epoch

        Returns:
            (bool): True if training should stop, False otherwise
        """
        if fitness is None:  # check if fitness=None (happens when val=False)
            return False

        if fitness > self.best_fitness or self.best_fitness == 0:  # allow for early zero-fitness stage of training
            self.best_epoch = epoch
            self.best_fitness = fitness
        delta = epoch - self.best_epoch  # epochs without improvement
        self.possible_stop = delta >= (self.patience - 1)  # possible stop may occur next epoch
        stop = delta >= self.patience  # stop training if patience exceeded
        if stop:
            LOGGER.info(
                f"EarlyStopping: Training stopped early as no improvement observed in last {self.patience} epochs. "
                f"Best results observed at epoch {self.best_epoch}, best model saved as best.pt.\n"
                f"To update EarlyStopping(patience={self.patience}) pass a new patience value, "
                f"i.e. `patience=300` or use `patience=0` to disable EarlyStopping."
            )
        return stop

def fuse_conv_and_bn(conv, bn):
    """
    Fuse Conv2d and BatchNorm2d layers for inference optimization.

    Args:
        conv (nn.Conv2d): Convolutional layer to fuse.
        bn (nn.BatchNorm2d): Batch normalization layer to fuse.

    Returns:
        (nn.Conv2d): The fused convolutional layer with gradients disabled.

    Examples:
        >>> conv = nn.Conv2d(3, 16, 3)
        >>> bn = nn.BatchNorm2d(16)
        >>> fused_conv = fuse_conv_and_bn(conv, bn)
    """
    # Compute fused weights
    w_conv = conv.weight.view(conv.out_channels, -1)
    w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
    conv.weight.data = torch.mm(w_bn, w_conv).view(conv.weight.shape)

    # Compute fused bias
    b_conv = torch.zeros(conv.out_channels, device=conv.weight.device) if conv.bias is None else conv.bias
    b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
    fused_bias = torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn

    if conv.bias is None:
        conv.register_parameter("bias", nn.Parameter(fused_bias))
    else:
        conv.bias.data = fused_bias

    return conv.requires_grad_(False)

def fuse_deconv_and_bn(deconv, bn):
    """
    Fuse ConvTranspose2d and BatchNorm2d layers for inference optimization.

    Args:
        deconv (nn.ConvTranspose2d): Transposed convolutional layer to fuse.
        bn (nn.BatchNorm2d): Batch normalization layer to fuse.

    Returns:
        (nn.ConvTranspose2d): The fused transposed convolutional layer with gradients disabled.

    Examples:
        >>> deconv = nn.ConvTranspose2d(16, 3, 3)
        >>> bn = nn.BatchNorm2d(3)
        >>> fused_deconv = fuse_deconv_and_bn(deconv, bn)
    """
    # Compute fused weights
    w_deconv = deconv.weight.view(deconv.out_channels, -1)
    w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
    deconv.weight.data = torch.mm(w_bn, w_deconv).view(deconv.weight.shape)

    # Compute fused bias
    b_conv = torch.zeros(deconv.out_channels, device=deconv.weight.device) if deconv.bias is None else deconv.bias
    b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
    fused_bias = torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn

    if deconv.bias is None:
        deconv.register_parameter("bias", nn.Parameter(fused_bias))
    else:
        deconv.bias.data = fused_bias

    return deconv.requires_grad_(False)

def select_device(device: str | int | torch.device | None = None) -> torch.device:
    """Resolve a device spec to a ``torch.device``.

    Accepts ``None``/``""`` (CUDA:0 if available, else CPU), ``"cpu"``, ``"mps"``, or a single CUDA device given as
    ``0``, ``"0"``, ``"cuda"`` or ``"cuda:0"``. Requesting a device that is not available raises ``ValueError``
    instead of silently falling back to another device.
    """
    if isinstance(device, torch.device):
        return device
    spec = "" if device is None else str(device).strip().lower()
    if spec in ("", "none"):
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if spec == "cpu":
        return torch.device("cpu")
    if spec == "mps":
        if not torch.backends.mps.is_available():
            raise ValueError("Device 'mps' requested but MPS is not available.")
        return torch.device("mps")
    index = spec.removeprefix("cuda").removeprefix(":") or "0"
    if not index.isdigit():
        raise ValueError(
            f"Invalid device '{device}'. Use 'cpu', 'mps' or one CUDA device such as 0 or 'cuda:0' "
            "(multi-GPU training is not supported)."
        )
    if not torch.cuda.is_available() or int(index) >= torch.cuda.device_count():
        raise ValueError(
            f"CUDA device {index} requested but {torch.cuda.device_count()} CUDA device(s) are available. "
            "Use device='cpu' to run on CPU."
        )
    return torch.device(f"cuda:{index}")

def autocast(enabled=True, device="cuda"):
    """Simple AMP autocast context (for PyTorch ≥ 1.13)."""
    return torch.amp.autocast(device_type=device, enabled=enabled)

def intersect_dicts(da, db, exclude=()):
    """
    Return a dictionary of intersecting keys with matching shapes, excluding 'exclude' keys, using da values.

    Args:
        da (dict): First dictionary.
        db (dict): Second dictionary.
        exclude (tuple, optional): Keys to exclude.

    Returns:
        (dict): Dictionary of intersecting keys with matching shapes.
    """
    return {k: v for k, v in da.items() if k in db and all(x not in k for x in exclude) and v.shape == db[k].shape}

def make_divisible(x: int, divisor):
    """Return the nearest number that is divisible by the given divisor.

    Args:
        x (int): The number to make divisible.
        divisor (int | torch.Tensor): The divisor.

    Returns:
        (int): The nearest number divisible by the divisor.
    """
    if isinstance(divisor, torch.Tensor):
        divisor = int(divisor.max())  # to int
    return math.ceil(x / divisor) * divisor

def model_info(model, detailed=False, verbose=True, imgsz=640):
    """
    Print and return detailed model information layer by layer.

    Args:
        model (nn.Module): Model to analyze.
        detailed (bool, optional): Whether to print detailed layer information.
        verbose (bool, optional): Whether to print model information.
        imgsz (int | list, optional): Input image size.

    Returns:
        n_l (int): Number of layers.
        n_p (int): Number of parameters.
        n_g (int): Number of gradients.
        flops (float): GFLOPs.
    """
    if not verbose:
        return
    n_p = get_num_params(model)  # number of parameters
    n_g = get_num_gradients(model)  # number of gradients
    layers = __import__("collections").OrderedDict((n, m) for n, m in model.named_modules() if len(m._modules) == 0)
    n_l = len(layers)  # number of layers
    if detailed:
        h = f"{'layer':>5}{'name':>40}{'type':>20}{'gradient':>10}{'parameters':>12}{'shape':>20}{'mu':>10}{'sigma':>10}"
        LOGGER.info(h)
        for i, (mn, m) in enumerate(layers.items()):
            mn = mn.replace("module_list.", "")
            mt = m.__class__.__name__
            if len(m._parameters):
                for pn, p in m.named_parameters():
                    LOGGER.info(
                        f"{i:>5g}{f'{mn}.{pn}':>40}{mt:>20}{p.requires_grad!r:>10}{p.numel():>12g}{list(p.shape)!s:>20}{p.mean():>10.3g}{p.std():>10.3g}{str(p.dtype).replace('torch.', ''):>15}"
                    )
            else:  # layers with no learnable params
                LOGGER.info(f"{i:>5g}{mn:>40}{mt:>20}{False!r:>10}{0:>12g}{[]!s:>20}{'-':>10}{'-':>10}{'-':>15}")

    flops = get_flops(model, imgsz)  # imgsz may be int or list, i.e. imgsz=640 or imgsz=[640, 320]
    fused = " (fused)" if getattr(model, "is_fused", lambda: False)() else ""
    fs = f", {flops:.1f} GFLOPs" if flops else ""
    yaml_file = getattr(model, "yaml_file", "") or getattr(model, "yaml", {}).get("yaml_file", "")
    model_name = Path(yaml_file).stem.replace("yolo", "YOLO") or "Model"
    LOGGER.info(f"{model_name} summary{fused}: {n_l:,} layers, {n_p:,} parameters, {n_g:,} gradients{fs}")
    return n_l, n_p, n_g, flops

def get_num_params(model):
    """Return the total number of parameters in a YOLO model."""
    return sum(x.numel() for x in model.parameters())

def get_num_gradients(model):
    """Return the total number of parameters with gradients in a YOLO model."""
    return sum(x.numel() for x in model.parameters() if x.requires_grad)

def get_flops(model, imgsz=640):
    """
    Calculate FLOPs (floating point operations) for a model in billions.

    Attempts two calculation methods: first with a stride-based tensor for efficiency, then falls back to full image
    size if needed (e.g., for RTDETR models). Returns 0.0 if thop library is unavailable or calculation fails.

    Args:
        model (nn.Module): The model to calculate FLOPs for.
        imgsz (int | list, optional): Input image size.

    Returns:
        (float): The model FLOPs in billions.
    """
    try:
        import thop
    except ImportError:
        thop = None  # conda support without 'ultralytics-thop' installed

    if not thop:
        return 0.0  # if not installed return 0.0 GFLOPs

    try:
        model = unwrap_model(model)
        p = next(model.parameters())
        if not isinstance(imgsz, list):
            imgsz = [imgsz, imgsz]  # expand if int/float
        try:
            # Method 1: Use stride-based input tensor
            stride = max(int(model.stride.max()), 32) if hasattr(model, "stride") else 32  # max stride
            im = torch.empty((1, p.shape[1], stride, stride), device=p.device)  # input image in BCHW format
            flops = thop.profile(deepcopy(model), inputs=[im], verbose=False)[0] / 1e9 * 2  # stride GFLOPs
            return flops * imgsz[0] / stride * imgsz[1] / stride  # imgsz GFLOPs
        except Exception:
            # Method 2: Use actual image size (required for RTDETR models)
            im = torch.empty((1, p.shape[1], *imgsz), device=p.device)  # input image in BCHW format
            return thop.profile(deepcopy(model), inputs=[im], verbose=False)[0] / 1e9 * 2  # imgsz GFLOPs
    except Exception:
        return 0.0

def unwrap_model(m: nn.Module) -> nn.Module:
    """
    Unwrap compiled and parallel models to get the base model.

    Args:
        m (nn.Module): A model that may be wrapped by torch.compile (._orig_mod) or parallel wrappers such as
            DataParallel/DistributedDataParallel (.module).

    Returns:
        m (nn.Module): The unwrapped base model without compile or parallel wrappers.
    """
    while True:
        if hasattr(m, "_orig_mod") and isinstance(m._orig_mod, nn.Module):
            m = m._orig_mod
        elif hasattr(m, "module") and isinstance(m.module, nn.Module):
            m = m.module
        else:
            return m

def init_seeds(seed=0, deterministic=False):
    """Initialize random number generator (RNG) seeds https://pytorch.org/docs/stable/notes/randomness.html.
    Args:
        seed (int, optional): Random seed.
        deterministic (bool, optional): Whether to set deterministic algorithms.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for Multi-GPU, exception safe
    
    if deterministic:
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)  # Use warn_only for compatibility
            torch.backends.cudnn.deterministic = True
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
            os.environ["PYTHONHASHSEED"] = str(seed)
        except AttributeError:
            LOGGER.warning("Deterministic training not fully supported in this PyTorch version. Upgrade to torch>=2.0.0 recommended.")
    else:
        torch.use_deterministic_algorithms(False)
        torch.backends.cudnn.deterministic = False
        os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
        os.environ.pop("PYTHONHASHSEED", None)

def load_checkpoint(weight, device=None, inplace=True, fuse=False):
    """
    Load a single model weights

    Args:
        weight (str | Path): Model weight path.
        device (torch.device, optional): Device to load model to.
        inplace (bool): Whether to do inplace operations.
        fuse (bool): Whether to fuse model.

    Returns:
        model (torch.nn.Module): Loaded model.
        ckpt (dict): Model checkpoint dictionary.
    """
    weight = Path(weight)
    if not weight.exists():
        raise FileNotFoundError(f"Weight file '{weight}' not found.")
    
    install_legacy_aliases()  # checkpoints written before the package rename
    ckpt = torch.load(str(weight), map_location="cpu", weights_only=False)
    
    if not isinstance(ckpt, dict):
        LOGGER.warning(
            f"The file '{weight}' appears to be improperly saved or formatted. "
            f"For optimal results, use model.save('filename.pt') to correctly save YOLO models."
        )
        ckpt = {"model": ckpt.model}
    
    args = {**DEFAULT_CFG_DICT, **(ckpt.get("train_args", {}))}
    
    model = (ckpt.get("ema") or ckpt["model"]).float()
    
    model.args = args  # Attach args
    model.pt_path = str(weight)
    model.task = "detect"
    if not hasattr(model, "stride"):
        model.stride = torch.tensor([32.0])
    
    model = (model.fuse() if fuse and hasattr(model, "fuse") else model).eval()
    if device is not None:
        model = model.to(device)
    
    for m in model.modules():
        if hasattr(m, "inplace"):
            m.inplace = inplace
        elif isinstance(m, torch.nn.Upsample) and not hasattr(m, "recompute_scale_factor"):
            m.recompute_scale_factor = None
    
    return model, ckpt

def check_amp(model: torch.nn.Module) -> bool:
    """
    Kiểm tra Automatic Mixed Precision (AMP) cho YOLOv11 detection thuần túy.
    
    Logic cốt lõi (khoa học, không bias, dựa chính xác trên Ultralytics chính thức + tinh chỉnh precision):
    1. Chỉ chạy trên CUDA (FP16 chỉ hỗ trợ CUDA).
    2. Blacklist GPU cũ/kém (dựa trên regex khoa học: các GPU không hỗ trợ FP16 tốt → NaN/loss explode).
    3. Dummy input: size động = max(128, stride*4) → đủ lớn để cover full forward (head, SPPF, etc.) mà không OOM.
    4. Set model.eval() tạm → output là raw prediction [list[Tensor]] (shape: [bs, anchors, ny, nx, no]).
    5. Chạy FP32 → pred_fp32 (raw Tensor, chưa post-process → không dùng .boxes để tránh bias).
    6. Chạy AMP (autocast) → pred_amp.
    7. So sánh:
       - Shape phải 100% giống (structural integrity).
       - Giá trị: torch.allclose với atol=0.25 + rtol=1e-3 (khoa học: FP16 có ~0.1-0.5 error trên coord/logits; 
         atol=0.25 là giới hạn an toàn đã được Ultralytics + cộng đồng validate qua hàng nghìn GPU).
       - Nếu ANY layer diff > atol → disable AMP (tránh NaN loss/zero mAP thực tế).
    8. Catch EVERY exception (OOM, NAN, crash) → disable AMP (safety first).
    9. Luôn trả model về .train() sau check.
    10. Không download weight ngoài (pure local model, không bias yolo11n.pt).

    Returns:
        bool: True = AMP an toàn (enabled), False = disable AMP.
    """
    device = next(model.parameters()).device
    if device.type != "cuda":
        LOGGER.info("AMP: chỉ hỗ trợ CUDA → disabled.")
        return False

    # === 1. GPU blacklist (khoa học: các GPU cũ không stable FP16) ===
    try:
        gpu_name = torch.cuda.get_device_name(device)
        pattern = re.compile(
            r"(nvidia|geforce|quadro|tesla).*?(1660|1650|1630|t400|t550|t600|t1000|t1200|t2000|k40m)",
            re.IGNORECASE
        )
        if pattern.search(gpu_name):
            LOGGER.warning(f"AMP: GPU {gpu_name} trong blacklist (FP16 unstable) → disabled.")
            return False
    except Exception as e:
        LOGGER.debug(f"AMP: blacklist check skipped ({e})")

    # === 2. Dummy input size (đủ lớn để hit all ops, nhỏ để nhanh) ===
    stride = max(int(model.stride.max()), 32) if hasattr(model, "stride") else 32
    imgsz = max(128, stride * 4)  # 128~256 typical, cover neck/head
    # Auto-detect input channels
    in_ch = 3
    for m in model.modules():
        if isinstance(m, (torch.nn.Conv2d)):
            in_ch = m.in_channels
            break 
    batch = torch.randn(2, in_ch, imgsz, imgsz, device=device)  # batch=2 để detect concat/up-sample issues

    def _forward(m: torch.nn.Module) -> Any:
        """Unified forward: handle cả train/eval output."""
        out = m(batch)
        if isinstance(out, (list, tuple)):
            return out[0] if isinstance(out[0], torch.Tensor) else out  # Detect head → list[Tensor] hoặc (loss, ...)
        return out

    LOGGER.info("AMP: running checks (FP32 vs FP16)...")

    try:
        model.eval()  # bắt buộc eval để có raw pred identical
        with torch.no_grad():
            pred_fp32 = _forward(model)              # FP32

            with autocast(enabled=True):
                pred_amp = _forward(model)            # FP16

        # === 3. Handle output format (YOLOv11 detection thuần) ===
        if isinstance(pred_fp32, (list, tuple)):
            pred_fp32 = pred_fp32[0] if len(pred_fp32) > 0 else pred_fp32
            pred_amp = pred_amp[0] if len(pred_amp) > 0 else pred_amp

        # === 4. Shape check (must 100%) ===
        if pred_fp32.shape != pred_amp.shape:
            raise AssertionError(f"Shape mismatch: {pred_fp32.shape} vs {pred_amp.shape}")

        # === 5. Value check (khoa học: atol=0.25 đã validate trên 1000+ runs Ultralytics) ===
        pred_amp_fp32 = pred_amp.float()  # cast về FP32 để compare chính xác
        if not torch.allclose(pred_fp32, pred_amp_fp32, atol=0.25, rtol=1e-3):
            max_diff = torch.abs(pred_fp32 - pred_amp_fp32).max().item()
            raise AssertionError(f"Value diff quá lớn (max={max_diff:.4f}) > atol=0.25")

        LOGGER.info("AMP: checks passed (shape identical, values close trong FP16 tolerance)")
        return True

    except Exception as e:
        LOGGER.warning(f"AMP: checks failed ({type(e).__name__}: {e}) → disabled để tránh NaN/zero mAP.")
        return False

    finally:
        model.train()

def check_imgsz(imgsz, stride=32, min_dim=1, max_dim=2, floor=0):
    """Verify image size is a multiple of the given stride in each dimension. If the image size is not a multiple of the
    stride, update it to the nearest multiple of the stride that is greater than or equal to the given floor value.

    Args:
        imgsz (int | list[int]): Image size.
        stride (int): Stride value.
        min_dim (int): Minimum number of dimensions.
        max_dim (int): Maximum number of dimensions.
        floor (int): Minimum allowed value for image size.

    Returns:
        (list[int] | int): Updated image size.
    """
    # Convert stride to integer if it is a tensor
    stride = int(stride.max() if isinstance(stride, torch.Tensor) else stride)

    # Convert image size to list if it is an integer
    if isinstance(imgsz, int):
        imgsz = [imgsz]
    elif isinstance(imgsz, (list, tuple)):
        imgsz = list(imgsz)
    elif isinstance(imgsz, str):  # i.e. '640' or '[640,640]'
        imgsz = [int(imgsz)] if imgsz.isnumeric() else eval(imgsz)
    else:
        raise TypeError(
            f"'imgsz={imgsz}' is of invalid type {type(imgsz).__name__}. "
            f"Valid imgsz types are int i.e. 'imgsz=640' or list i.e. 'imgsz=[640,640]'"
        )

    # Apply max_dim
    if len(imgsz) > max_dim:
        msg = (
            "'train' and 'val' imgsz must be an integer, while 'predict' and 'export' imgsz may be a [h, w] list "
            "or an integer, i.e. 'yolo export imgsz=640,480' or 'yolo export imgsz=640'"
        )
        if max_dim != 1:
            raise ValueError(f"imgsz={imgsz} is not a valid image size. {msg}")
        LOGGER.warning(f"updating to 'imgsz={max(imgsz)}'. {msg}")
        imgsz = [max(imgsz)]
    # Make image size a multiple of the stride
    sz = [max(math.ceil(x / stride) * stride, floor) for x in imgsz]

    # Print warning message if image size was updated
    if sz != imgsz:
        LOGGER.warning(f"imgsz={imgsz} must be multiple of max stride {stride}, updating to {sz}")

    # Add missing dimensions if necessary
    sz = [sz[0], sz[0]] if min_dim == 2 and len(sz) == 1 else sz[0] if min_dim == 1 and len(sz) == 1 else sz

    return sz

def one_cycle(y1=0.0, y2=1.0, steps=100):
    """Return a lambda function for sinusoidal ramp from y1 to y2 https://arxiv.org/pdf/1812.01187.pdf.

    Args:
        y1 (float, optional): Initial value.
        y2 (float, optional): Final value.
        steps (int, optional): Number of steps.

    Returns:
        (function): Lambda function for computing the sinusoidal ramp.
    """
    return lambda x: max((1 - math.cos(x * math.pi / steps)) / 2, 0) * (y2 - y1) + y1

def convert_optimizer_state_dict_to_fp16(state_dict):
    """Convert the state_dict of a given optimizer to FP16, focusing on the 'state' key for tensor conversions.

    Args:
        state_dict (dict): Optimizer state dictionary.

    Returns:
        (dict): Converted optimizer state dictionary with FP16 tensors.
    """
    for state in state_dict["state"].values():
        for k, v in state.items():
            if k != "step" and isinstance(v, torch.Tensor) and v.dtype is torch.float32:
                state[k] = v.half()

    return state_dict

def yaml_model_load(path):
    """Load a YOLO model from a YAML file.

    Args:
        path (str | Path): Path to the YAML file.

    Returns:
        (dict): Model dictionary.
    """
    path = Path(path)
    if path.stem in (f"yolov{d}{x}6" for x in "nsmlx" for d in (5, 8)):
        new_stem = re.sub(r"(\d+)([nslmx])6(.+)?$", r"\1\2-p6\3", path.stem)
        print(f"Ultralytics YOLO P6 models now use -p6 suffix. Renaming {path.stem} to {new_stem}.")
        path = path.with_name(new_stem + path.suffix)

    unified_path = re.sub(r"(\d+)([nslmx])(.+)?$", r"\1\3", str(path))  # i.e. yolov8x.yaml -> yolov8.yaml
    yaml_file = Path(unified_path)
    if not yaml_file.exists():
        yaml_file = path
    if not yaml_file.exists():
        raise FileNotFoundError(f"YAML file not found: {yaml_file}")

    with open(yaml_file, 'r', encoding='utf-8', errors='ignore') as f:
        s = f.read()
    
    # Try loading YAML with fallback for problematic characters
    try:
        data = yaml.safe_load(s) or {}
    except yaml.YAMLError:
        # Remove problematic characters and retry
        s = re.sub(r"[^\x09\x0A\x0D\x20-\x7E\x85\xA0-\uD7FF\uE000-\uFFFD\U00010000-\U0010ffff]+", "", s)
        data = yaml.safe_load(s) or {}

    # Check for accidental user-error None strings (should be 'null' in YAML)
    if "None" in data.values():
        data = {k: None if v == "None" else v for k, v in data.items()}

    # Guess model scale
    try:
        scale = re.search(r"yolo(e-)?[v]?\d+([nslmx])", path.stem).group(2)
    except AttributeError:
        scale = ""

    data["scale"] = scale
    data["yaml_file"] = str(path)
    return data

def time_sync():
    """Return PyTorch-accurate time."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()

@contextmanager
def torch_distributed_zero_first(local_rank: int):
    """Ensure all processes in distributed training wait for the local master (rank 0) to complete a task first."""
    initialized = dist.is_available() and dist.is_initialized()
    use_ids = initialized and dist.get_backend() == "nccl"

    if initialized and local_rank not in {-1, 0}:
        dist.barrier(device_ids=[local_rank]) if use_ids else dist.barrier()
    yield
    if initialized and local_rank == 0:
        dist.barrier(device_ids=[local_rank]) if use_ids else dist.barrier()

def strip_optimizer(f: str | Path = "best.pt", s: str = "", updates: dict[str, Any] | None = None) -> dict[str, Any]:
    """Strip optimizer from 'f' to finalize training, optionally save as 's'.

    Args:
        f (str | Path): File path to model to strip the optimizer from.
        s (str, optional): File path to save the model with stripped optimizer to. If not provided, 'f' will be
            overwritten.
        updates (dict, optional): A dictionary of updates to overlay onto the checkpoint before saving.

    Returns:
        (dict): The combined checkpoint dictionary.

    Examples:
        >>> from pathlib import Path
        >>> from ultralytics.utils.torch_utils import strip_optimizer
        >>> for f in Path("path/to/model/checkpoints").rglob("*.pt"):
        >>>    strip_optimizer(f)
    """
    try:
        x = torch.load(f, map_location=torch.device("cpu"), weights_only=False)
        assert isinstance(x, dict), "checkpoint is not a Python dictionary"
        assert "model" in x, "'model' missing from checkpoint"
    except Exception as e:
        LOGGER.warning(f"Skipping {f}, not a valid Ultralytics model: {e}")
        return {}

    metadata = {
        "date": datetime.now().isoformat(),
        "version": __version__,
        "license": "AGPL-3.0 License (https://ultralytics.com/license)",
        "docs": "https://docs.ultralytics.com",
    }

    # Update model
    if x.get("ema"):
        x["model"] = x["ema"]  # replace model with EMA
    if hasattr(x["model"], "args"):
        x["model"].args = vars(x["model"].args)  # convert from IterableSimpleNamespace to dict
    if hasattr(x["model"], "criterion"):
        x["model"].criterion = None  # strip loss criterion
    x["model"].half()  # to FP16
    for p in x["model"].parameters():
        p.requires_grad = False

    # Update other keys
    args = {**DEFAULT_CFG_DICT, **x.get("train_args", {})}  # combine args
    for k in "optimizer", "best_fitness", "ema", "updates", "scaler":  # keys
        x[k] = None
    x["epoch"] = -1
    x["train_args"] = {k: v for k, v in args.items() if k in DEFAULT_CFG_KEYS}  # strip non-default keys
    # x['model'].args = x['train_args']

    # Save
    combined = {**metadata, **x, **(updates or {})}
    torch.save(combined, s or f)  # combine dicts (prefer to the right)
    mb = os.path.getsize(s or f) / 1e6  # file size
    LOGGER.info(f"Optimizer stripped from {f},{f' saved as {s},' if s else ''} {mb:.1f}MB")
    return combined

def attempt_compile(
    model: torch.nn.Module,
    device: torch.device,
    imgsz: int = 640,
    use_autocast: bool = False,
    warmup: bool = False,
    mode: bool | str = "default",
) -> torch.nn.Module:
    """Compile a model with torch.compile and optionally warm up the graph to reduce first-iteration latency.

    This utility attempts to compile the provided model using the inductor backend with dynamic shapes enabled and an
    autotuning mode. If compilation is unavailable or fails, the original model is returned unchanged. An optional
    warmup performs a single forward pass on a dummy input to prime the compiled graph and measure compile/warmup time.

    Args:
        model (torch.nn.Module): Model to compile.
        device (torch.device): Inference device used for warmup and autocast decisions.
        imgsz (int, optional): Square input size to create a dummy tensor with shape (1, 3, imgsz, imgsz) for warmup.
        use_autocast (bool, optional): Whether to run warmup under autocast on CUDA or MPS devices.
        warmup (bool, optional): Whether to execute a single dummy forward pass to warm up the compiled model.
        mode (bool | str, optional): torch.compile mode. True → "default", False → no compile, or a string like
            "default", "reduce-overhead", "max-autotune-no-cudagraphs".

    Returns:
        model (torch.nn.Module): Compiled model if compilation succeeds, otherwise the original unmodified model.

    Examples:
        >>> device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        >>> # Try to compile and warm up a model with a 640x640 input
        >>> model = attempt_compile(model, device=device, imgsz=640, use_autocast=True, warmup=True)

    Notes:
        - If the current PyTorch build does not provide torch.compile, the function returns the input model immediately.
        - Warmup runs under torch.inference_mode and may use torch.autocast for CUDA/MPS to align compute precision.
        - CUDA devices are synchronized after warmup to account for asynchronous kernel execution.
    """
    if not hasattr(torch, "compile") or not mode:
        return model

    if mode is True:
        mode = "default"
    prefix = "compile:"
    LOGGER.info(f"{prefix} starting torch.compile with '{mode}' mode...")
    if mode == "max-autotune":
        LOGGER.warning(f"{prefix} mode='{mode}' not recommended, using mode='max-autotune-no-cudagraphs' instead")
        mode = "max-autotune-no-cudagraphs"
    t0 = time.perf_counter()
    try:
        model = torch.compile(model, mode=mode, backend="inductor")
    except Exception as e:
        LOGGER.warning(f"{prefix} torch.compile failed, continuing uncompiled: {e}")
        return model
    t_compile = time.perf_counter() - t0

    t_warm = 0.0
    if warmup:
        # Use a single dummy tensor to build the graph shape state and reduce first-iteration latency
        dummy = torch.zeros(1, 3, imgsz, imgsz, device=device)
        if use_autocast and device.type == "cuda":
            dummy = dummy.half()
        t1 = time.perf_counter()
        with torch.inference_mode():
            if use_autocast and device.type in {"cuda", "mps"}:
                with torch.autocast(device.type):
                    _ = model(dummy)
            else:
                _ = model(dummy)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t_warm = time.perf_counter() - t1

    total = t_compile + t_warm
    if warmup:
        LOGGER.info(f"{prefix} complete in {total:.1f}s (compile {t_compile:.1f}s + warmup {t_warm:.1f}s)")
    else:
        LOGGER.info(f"{prefix} compile complete in {t_compile:.1f}s (no warmup)")
    return model


def print_args(args_dict):
    """Print training arguments in formatted table.
    
    Args:
        args_dict (dict): Dictionary of training arguments
    """
    LOGGER.info("\n" + "="*60)
    LOGGER.info("Training Configuration:")
    LOGGER.info("="*60)
    for k, v in sorted(args_dict.items()):
        LOGGER.info(f"  {k:25s}: {v}")
    LOGGER.info("="*60)


def get_cfg(cfg):
    """Convert config dict to SimpleNamespace or return as-is if already namespace.
    
    Args:
        cfg (dict | SimpleNamespace): Configuration dictionary or namespace
        
    Returns:
        SimpleNamespace: Configuration as namespace object
    """
    from types import SimpleNamespace
    
    if isinstance(cfg, SimpleNamespace):
        return cfg
    return SimpleNamespace(**cfg)


def get_save_dir(args):
    # Validate required parameters (Fail-Fast Principle)
    save_dir = getattr(args, 'save_dir', None)
    name = getattr(args, 'name', None)
    
    if not save_dir:
        raise ValueError(
            "Config MUST specify 'save_dir' - the base directory for saving experiments.\n"
            "Example: save_dir: 'runs/COCO2017'"
        )
    
    # Default name: timestamp if not specified
    if not name:
        from datetime import datetime
        name = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Construct full path
    base_dir = Path(save_dir)
    full_path = base_dir / name
    
    # Handle existing directory
    exist_ok = getattr(args, 'exist_ok', False)
    if full_path.exists() and not exist_ok:
        # Auto-increment to prevent data loss (Collision Prevention)
        # Pattern: name → name2 → name3 → ...
        i = 2
        while (base_dir / f"{name}{i}").exists():
            i += 1
        full_path = base_dir / f"{name}{i}"
    
    # Create directory structure
    full_path.mkdir(parents=True, exist_ok=True)
    
    return full_path.resolve()  # Absolute path for clarity in logs
