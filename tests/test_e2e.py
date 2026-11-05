"""Train 1 epoch on a tiny synthetic dataset, then evaluate / predict / export the resulting checkpoint."""

from pathlib import Path

import pytest
import yaml

from yolo11.cli.main import main
from yolo11.models.model import DetectionModel

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def run_dir(tiny_dataset, tmp_path_factory):
    out = tmp_path_factory.mktemp("runs")
    cfg = yaml.safe_load((ROOT / "configs" / "training.yaml").read_text())
    cfg.update(
        data=str(tiny_dataset), save_dir=str(out), name="e2e", epochs=1, batch=2, imgsz=64, workers=0,
        device="cpu", amp=False, cache=False, plots=False, close_mosaic=0, save_period=0, model="yolov11n",
        warmup_epochs=0, resume=False, compile=False,
    )
    cfg_path = out / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))

    def no_tta(self, x):
        raise AssertionError("test-time augmentation must not run during training or its final evaluation")

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(DetectionModel, "_predict_augment", no_tta)
        assert main(["train", "--config", str(cfg_path)]) == 0
    return out / "e2e"


def test_train_writes_checkpoints(run_dir):
    assert (run_dir / "weights" / "last.pt").is_file()
    assert (run_dir / "results.csv").is_file()


def test_final_checkpoints_keep_training_args(run_dir):
    from yolo11.inference.predictor import Predictor
    from yolo11.utils.torch_utils import load_checkpoint

    for name in ("best.pt", "last.pt"):  # both are stripped after training
        model = load_checkpoint(run_dir / "weights" / name)[0]
        assert model.args["imgsz"] == 64
    assert Predictor(run_dir / "weights" / "best.pt", device="cpu").imgsz == 64  # no --imgsz needed


def test_val(run_dir, tiny_dataset, tmp_path):
    out = tmp_path / "metrics.json"
    main([
        "val", "--weights", str(run_dir / "weights" / "last.pt"), "--data", str(tiny_dataset), "--imgsz", "64",
        "--batch", "2", "--device", "cpu", "--workers", "0", "--save-dir", str(tmp_path / "val"), "--output", str(out),
    ])
    assert out.is_file() and "metrics/mAP50-95(B)" in out.read_text()


def test_val_uses_rectangular_batches_like_training(run_dir, tiny_dataset, tmp_path, monkeypatch):
    import yolo11.evaluation.validator as validator_module

    rects = []
    build = validator_module.build_yolo_dataset

    def spy(cfg, *args, **kwargs):
        dataset = build(cfg, *args, **kwargs)
        rects.append(dataset.rect)
        return dataset

    monkeypatch.setattr(validator_module, "build_yolo_dataset", spy)
    main([
        "val", "--weights", str(run_dir / "weights" / "best.pt"), "--data", str(tiny_dataset), "--imgsz", "64",
        "--batch", "2", "--device", "cpu", "--workers", "0", "--save-dir", str(tmp_path),
    ])
    assert rects == [True]


def test_predict(run_dir, tiny_dataset, tmp_path):
    src = tiny_dataset.parent / "images" / "val"
    main([
        "predict", "--weights", str(run_dir / "weights" / "last.pt"), "--source", str(src), "--imgsz", "64",
        "--conf", "0.001", "--device", "cpu", "--output-dir", str(tmp_path), "--save-json",
    ])
    assert list(tmp_path.glob("exp_*/predictions.json"))
    assert list(tmp_path.glob("exp_*/0.jpg"))


def test_export_onnx_matches_torch(run_dir, tmp_path):
    onnx = pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")
    import numpy as np
    import torch

    from yolo11.utils.torch_utils import load_checkpoint

    out = tmp_path / "m.onnx"
    main(["export", "--weights", str(run_dir / "weights" / "last.pt"), "--format", "onnx", "--imgsz", "64",
          "--output", str(out)])
    onnx.checker.check_model(onnx.load(str(out)))
    x = np.random.rand(1, 3, 64, 64).astype("float32")
    got = ort.InferenceSession(str(out)).run(None, {"images": x})[0]
    model = load_checkpoint(run_dir / "weights" / "last.pt", fuse=True)[0].eval()
    model.model[-1].export = True
    with torch.no_grad():
        want = model(torch.from_numpy(x)).numpy()
    assert np.allclose(got, want, atol=1e-3)


def test_dynamic_onnx_runs_other_shapes(run_dir, tmp_path):
    ort = pytest.importorskip("onnxruntime")
    import numpy as np
    import torch

    from yolo11.utils.torch_utils import load_checkpoint

    out = tmp_path / "dynamic.onnx"
    main(["export", "--weights", str(run_dir / "weights" / "last.pt"), "--format", "onnx", "--imgsz", "64",
          "--dynamic", "--output", str(out)])
    session = ort.InferenceSession(str(out))
    model = load_checkpoint(run_dir / "weights" / "last.pt", fuse=True)[0].eval()
    model.model[-1].export = True
    for shape in [(1, 3, 64, 64), (2, 3, 96, 128)]:  # the export size and a different batch and size
        x = np.random.rand(*shape).astype("float32")
        got = session.run(None, {"images": x})[0]
        with torch.no_grad():
            want = model(torch.from_numpy(x)).numpy()
        assert got.shape == want.shape
        assert np.allclose(got, want, atol=1e-3)


def test_export_torchscript(run_dir, tmp_path):
    out = tmp_path / "m.torchscript"
    main(["export", "--weights", str(run_dir / "weights" / "last.pt"), "--format", "torchscript", "--imgsz", "64",
          "--output", str(out)])
    assert out.is_file()
