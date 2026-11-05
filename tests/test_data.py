import pytest

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


def test_polygon_labels_become_boxes(tmp_path):
    import cv2
    import numpy as np

    from yolo11.data.yolo_dataset import verify_image_label

    img, lbl = tmp_path / "a.jpg", tmp_path / "a.txt"
    cv2.imwrite(str(img), np.zeros((32, 32, 3), np.uint8))
    lbl.write_text("1 0.1 0.2 0.5 0.2 0.5 0.6 0.1 0.6\n0 0.4 0.4 0.6 0.4 0.6 0.6 0.4 0.6\n")  # two polygons
    im_file, lb, *_, msg = verify_image_label((str(img), str(lbl), "", 2, False))
    assert im_file == str(img), msg  # image kept, not reported as corrupt
    np.testing.assert_allclose(lb, [[1, 0.3, 0.4, 0.4, 0.4], [0, 0.5, 0.5, 0.2, 0.2]], atol=1e-6)  # file order


def test_dataset_item(tiny_dataset):
    from yolo11.data.yolo_dataset import YOLODataset

    data = load_data_config(tiny_dataset)
    ds = YOLODataset(data["val"], imgsz=64, augment=False, data=data, batch_size=2)
    item = ds[0]
    assert item["img"].shape == (3, 64, 64)
    assert item["cls"].shape[1] == 1 and item["bboxes"].shape[1] == 4
    assert len(ds) == 4


def test_truncated_jpeg_is_restored_without_touching_hard_links(tmp_path):
    import os
    from PIL import Image
    from yolo11.data.yolo_dataset import verify_image_label

    image = tmp_path / "bad.jpg"
    label = tmp_path / "bad.txt"
    Image.new("RGB", (32, 32)).save(image)
    contents = image.read_bytes() + b"\x00\x00"  # trailing bytes after the end marker, image data intact
    image.write_bytes(contents)
    linked = tmp_path / "linked.jpg"
    os.link(image, linked)
    label.write_text("", encoding="utf-8")

    result = verify_image_label((str(image), str(label), "", 1, False))
    assert result[0] == str(image)
    assert result[6] == 0  # corrupt count: the image is kept
    assert "corrupt JPEG restored and saved" in result[7]
    assert image.read_bytes().endswith(b"\xff\xd9")
    assert linked.read_bytes() == contents  # the hard-linked original is left alone
    assert sorted(p.name for p in tmp_path.iterdir()) == ["bad.jpg", "bad.txt", "linked.jpg"]  # no temp file left


@pytest.mark.parametrize("fmt", ["JPEG", "PNG", "WEBP"])
@pytest.mark.parametrize("orientation", range(1, 9))
def test_exif_size_matches_decoded_image(tmp_path, fmt, orientation):
    import cv2
    from PIL import Image
    from yolo11.data.yolo_dataset import exif_size

    path = tmp_path / f"image.{fmt.lower()}"
    image = Image.new("RGB", (40, 20))
    exif = image.getexif()
    exif[274] = orientation
    image.save(path, fmt, exif=exif.tobytes())
    with Image.open(path) as im:
        im.verify()
        assert exif_size(im) == cv2.imread(str(path)).shape[1::-1]