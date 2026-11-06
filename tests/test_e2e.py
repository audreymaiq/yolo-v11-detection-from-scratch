"""Train 1 epoch on a tiny synthetic dataset, then evaluate / predict / export the resulting checkpoint."""

import json
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
    assert Predictor(run_dir / "weights" / "best.pt", device="cpu").imgsz == [64, 64]  # no --imgsz needed


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


@pytest.fixture(scope="module")
def exports(run_dir, tmp_path_factory):
    """The last checkpoint exported as static ONNX, dynamic ONNX, static ONNX with batch 2, and TorchScript."""
    pytest.importorskip("onnxruntime")
    out = tmp_path_factory.mktemp("exports")
    weights = str(run_dir / "weights" / "last.pt")
    variants = {
        "onnx": ["--format", "onnx"],
        "onnx_dynamic": ["--format", "onnx", "--dynamic"],
        "onnx_batch2": ["--format", "onnx", "--batch", "2"],
        "torchscript": ["--format", "torchscript"],
    }
    files = {}
    for name, flags in variants.items():
        files[name] = out / f"{name}.{flags[1]}"
        main(["export", "--weights", weights, "--imgsz", "64", "--output", str(files[name]), *flags])
    return files


def test_exports_embed_metadata(exports):
    import onnx
    import torch

    assert not list(exports["onnx"].parent.glob("*.json"))  # no sidecar file
    props = {p.key: json.loads(p.value) for p in onnx.load(str(exports["onnx_dynamic"])).metadata_props}
    assert props["names"] == {"0": "square", "1": "bar"}
    assert props["imgsz"] == [64, 64] and props["stride"] == 32 and props["dynamic"] is True
    extra = {"config.txt": ""}
    torch.jit.load(str(exports["torchscript"]), _extra_files=extra)
    assert json.loads(extra["config.txt"])["imgsz"] == [64, 64]


def test_exports_record_their_source(run_dir, exports):
    import hashlib

    import onnx

    weights = run_dir / "weights" / "last.pt"
    provenance = json.loads((run_dir / "provenance.json").read_text())
    props = {p.key: json.loads(p.value) for p in onnx.load(str(exports["onnx"])).metadata_props}
    source = props["source"]
    assert source["weights"] == "last.pt"
    assert source["sha256"] == hashlib.sha256(weights.read_bytes()).hexdigest()
    assert source["data_train_sha256"] == provenance["data"]["train"]["sha256"]
    assert source.get("git_commit") == (provenance["git"] or {}).get("commit")


def test_run_records_provenance(run_dir):
    import torch

    provenance = json.loads((run_dir / "provenance.json").read_text())
    assert provenance["environment"]["packages"]["torch"] and provenance["environment"]["device"] == "cpu"
    assert provenance["data"]["train"]["images"] == 8 and provenance["data"]["val"]["images"] == 4
    ckpt = torch.load(run_dir / "weights" / "best.pt", map_location="cpu", weights_only=False)
    assert ckpt["provenance"] == provenance


def test_finetune_records_initial_weights(run_dir, tmp_path):
    cfg = yaml.safe_load((run_dir.parent / "cfg.yaml").read_text())
    cfg.update(model=str(run_dir / "weights" / "best.pt"), save_dir=str(tmp_path), name="ft")
    (tmp_path / "ft.yaml").write_text(yaml.safe_dump(cfg))
    assert main(["train", "--config", str(tmp_path / "ft.yaml")]) == 0
    init = json.loads((tmp_path / "ft" / "provenance.json").read_text())["init_weights"]
    parent = json.loads((run_dir / "provenance.json").read_text())
    assert init["path"] == cfg["model"] and init["data_train_sha256"] == parent["data"]["train"]["sha256"]


@pytest.mark.parametrize("fmt", ["onnx", "onnx_dynamic", "onnx_batch2", "torchscript"])
def test_exported_models_match_checkpoint(run_dir, tiny_dataset, exports, fmt):
    import numpy as np

    from yolo11.inference.predictor import Predictor

    images = [str(p) for p in sorted((tiny_dataset.parent / "images" / "val").glob("*.jpg"))]
    reference = Predictor(run_dir / "weights" / "last.pt", device="cpu", conf=0.001)
    exported = Predictor(exports[fmt], device="cpu", conf=0.001)
    assert exported.names == reference.names == {0: "square", 1: "bar"}
    assert exported.imgsz == [64, 64]

    x = reference.preprocess([reference.read_image(images[0])])
    want = reference.backend(x).numpy()
    if exported.backend.batch == 2:
        x = x.repeat(2, 1, 1, 1)
    np.testing.assert_allclose(exported.backend(x)[:1].numpy(), want, atol=1e-3)
    detections = exported.predict_many(images, batch=3)  # three images: a full and a padded chunk for batch 2
    assert len(detections) == len(images)
    assert all({"box", "confidence", "class_id", "label"} <= det.keys() for dets in detections for det in dets)


def test_predict_batch_matches_single_images(run_dir, tiny_dataset):
    from yolo11.inference.predictor import Predictor

    images = [str(p) for p in sorted((tiny_dataset.parent / "images" / "val").glob("*.jpg"))]
    predictor = Predictor(run_dir / "weights" / "last.pt", device="cpu", conf=0.001)
    assert predictor.predict_many(images, batch=len(images)) == [predictor.predict(p) for p in images]


def test_exports_without_metadata_are_rejected(exports, tmp_path):
    import onnx

    from yolo11.inference.predictor import Predictor

    model = onnx.load(str(exports["onnx"]))
    del model.metadata_props[:]
    bare = tmp_path / "bare.onnx"
    onnx.save(model, str(bare))
    with pytest.raises(ValueError, match="no yolo11 metadata"):
        Predictor(bare, device="cpu")
    # exports made before metadata was embedded kept it in a sidecar <model>.json
    bare.with_suffix(".json").write_text(json.dumps({"names": {"0": "a", "1": "b"}, "imgsz": [64, 64], "stride": 32}))
    assert Predictor(bare, device="cpu").names == {0: "a", 1: "b"}


def test_predict_cli_with_onnx(exports, tiny_dataset, tmp_path):
    src = tiny_dataset.parent / "images" / "val"
    results = main([
        "predict", "--weights", str(exports["onnx_dynamic"]), "--source", str(src), "--conf", "0.001",
        "--device", "cpu", "--batch", "2", "--output-dir", str(tmp_path), "--save-json",
    ])
    assert results == 0
    saved = json.loads(next(tmp_path.glob("exp_*/predictions.json")).read_text())
    assert len(saved) == len(list(src.glob("*.jpg")))