import pytest
import torch

from yolo11.utils.torch_utils import select_device


@pytest.mark.parametrize("spec", [None, "", "cpu", "CPU", torch.device("cpu")])
def test_cpu_and_auto(spec, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert select_device(spec) == torch.device("cpu")


@pytest.mark.parametrize("spec", [0, "0", "cuda", "cuda:0"])
def test_cuda_specs_resolve_to_cuda0(spec, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    assert select_device(spec) == torch.device("cuda:0")


@pytest.mark.parametrize("spec", ["0", "cuda:1"])
def test_unavailable_cuda_raises(spec, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1 if spec == "cuda:1" else 0)
    with pytest.raises(ValueError, match="CUDA device"):
        select_device(spec)


@pytest.mark.parametrize("spec", ["0,1", "[0, 1]", "gpu"])
def test_invalid_specs_raise(spec):
    with pytest.raises(ValueError, match="Invalid device"):
        select_device(spec)
