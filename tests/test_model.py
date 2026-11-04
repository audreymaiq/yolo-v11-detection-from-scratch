import torch

from yolo11.registry import ModelRegistry
from yolo11.utils.legacy import install_legacy_aliases


def test_forward_shapes():
    model = ModelRegistry.get_model("yolov11n", ch=3, nc=2, verbose=False).eval()
    with torch.no_grad():
        y, feats = model(torch.zeros(1, 3, 64, 64))
    assert y.shape == (1, 4 + 2, (8 * 8) + (4 * 4) + (2 * 2))
    assert len(feats) == 3
    assert model.stride.tolist() == [8.0, 16.0, 32.0]


def test_scales_build():
    for name in ("yolov11n", "yolov11s"):
        model = ModelRegistry.get_model(name, ch=3, nc=3, verbose=False)
        assert sum(p.numel() for p in model.parameters()) > 1e6


def test_legacy_aliases_resolve_to_current_classes():
    install_legacy_aliases()
    from sgod_model_builder.models.conv import Conv as LegacyConv

    from yolo11.models.conv import Conv

    assert LegacyConv is Conv
