from yolo11.data.config import load_data_config


def test_load_data_config_resolves_paths(tiny_dataset):
    data = load_data_config(tiny_dataset)
    assert data["nc"] == 2
    assert data["train"].endswith("train") and data["val"].endswith("val")
    assert data["channels"] == 3


def test_label_cache_is_written_and_reused(tiny_dataset, monkeypatch):
    import gc

    from yolo11.data.yolo_dataset import YOLODataset

    data = load_data_config(tiny_dataset)
    labels_dir = tiny_dataset.parent / "labels"
    for f in labels_dir.glob("*.cache*"):
        f.unlink()

    YOLODataset(data["val"], imgsz=64, augment=False, data=data, batch_size=2)
    assert (labels_dir / "val.cache").is_file()
    assert not list(labels_dir.glob("*.npy"))
    assert gc.isenabled()

    def rescan(*args, **kwargs):
        raise AssertionError("labels were rescanned although a valid cache exists")

    monkeypatch.setattr(YOLODataset, "cache_labels", rescan)
    YOLODataset(data["val"], imgsz=64, augment=False, data=data, batch_size=2)


def test_partial_mosaic_samples_share_keys_and_collate(tiny_dataset):
    import random
    from types import SimpleNamespace

    from yolo11.data.yolo_dataset import YOLODataset

    data = load_data_config(tiny_dataset)
    hyp = SimpleNamespace(
        mosaic=0.5, mixup=0.0, cutmix=0.0, scale=0.5, degrees=0.0, translate=0.1, shear=0.0, perspective=0.0,
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, grayscale=0.0, flipud=0.0, fliplr=0.5,
    )
    ds = YOLODataset(data["train"], imgsz=64, augment=True, hyp=hyp, data=data, batch_size=4)
    random.seed(0)
    samples = [ds[i % len(ds)] for i in range(32)]  # p=0.5 over 32 draws mixes both paths
    assert len({tuple(sorted(s)) for s in samples}) == 1
    assert all(s["img"].shape[1:] == (64, 64) for s in samples)
    for b in range(0, len(samples), 4):
        batch = YOLODataset.collate_fn(samples[b : b + 4])
        assert batch["img"].shape == (4, 3, 64, 64)


def test_training_augmentation_follows_mode_not_tta_flag(tiny_dataset):
    from types import SimpleNamespace

    from yolo11.data.yolo_dataset import build_yolo_dataset

    data = load_data_config(tiny_dataset)
    cfg = SimpleNamespace(imgsz=64, augment=False, mosaic=0.0, mixup=0.0, cutmix=0.0)  # augment = TTA flag
    assert build_yolo_dataset(cfg, data["train"], 2, data, mode="train").augment
    assert not build_yolo_dataset(cfg, data["val"], 2, data, mode="val").augment


def test_dataset_item(tiny_dataset):
    from yolo11.data.yolo_dataset import YOLODataset

    data = load_data_config(tiny_dataset)
    ds = YOLODataset(data["val"], imgsz=64, augment=False, data=data, batch_size=2)
    item = ds[0]
    assert item["img"].shape == (3, 64, 64)
    assert item["cls"].shape[1] == 1 and item["bboxes"].shape[1] == 4
    assert len(ds) == 4
