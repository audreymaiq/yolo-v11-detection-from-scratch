from yolo11.data.config import load_data_config


def test_load_data_config_resolves_paths(tiny_dataset):
    data = load_data_config(tiny_dataset)
    assert data["nc"] == 2
    assert data["train"].endswith("train") and data["val"].endswith("val")
    assert data["channels"] == 3


def test_dataset_item(tiny_dataset):
    from yolo11.data.yolo_dataset import YOLODataset

    data = load_data_config(tiny_dataset)
    ds = YOLODataset(data["val"], imgsz=64, augment=False, data=data, batch_size=2)
    item = ds[0]
    assert item["img"].shape == (3, 64, 64)
    assert item["cls"].shape[1] == 1 and item["bboxes"].shape[1] == 4
    assert len(ds) == 4
