"""The converter must preserve source data and produce valid detection labels."""

import json

import pytest

from yolo11.data.convert_coco import convert_coco_json, verify_output


def _sample():
    return {
        "images": [
            {"id": 1, "file_name": "nested/a.jpg", "width": 100, "height": 50},
            {"id": 2, "file_name": "b.jpg", "width": 100, "height": 50},
        ],
        "categories": [{"id": 7, "name": "seven"}, {"id": 3, "name": "three"}],
        "annotations": [
            {"image_id": 1, "category_id": 7, "bbox": [10, 5, 20, 10]},
            {"image_id": 1, "category_id": 7, "bbox": [10, 5, 20, 10]},
            {"image_id": 1, "category_id": 3, "bbox": [-5, 0, 15, 10]},
            {"image_id": 1, "category_id": 3, "bbox": [50, 25, 0.1, 0.1]},
            {"image_id": 1, "category_id": 3, "bbox": [20, 20, 5, 5], "iscrowd": 1},
        ],
    }


def test_converter_preserves_small_boxes_and_background_images(tmp_path):
    source = tmp_path / "instances.json"
    source.write_text(json.dumps(_sample()), encoding="utf-8")
    output = tmp_path / "labels"

    stats = convert_coco_json(source, output, expected_count=2)

    assert stats == {"images": 2, "annotations": 5, "written": 3, "crowd": 1,
                     "invalid": 0, "duplicates": 1, "clipped": 1}
    assert source.is_file()
    assert (output / "b.txt").read_text() == ""
    labels = (output / "nested" / "a.txt").read_text().splitlines()
    assert len(labels) == 3
    assert labels[0].split()[0] == "1"  # sorted COCO category IDs: 3 -> 0, 7 -> 1
    assert labels[1].split() == ["0", "0.05", "0.1", "0.1", "0.2"]
    assert float(labels[2].split()[3]) > 0  # a box smaller than 1 px² is retained
    classes = json.loads((output / "classes.json").read_text())
    assert [entry["coco_id"] for entry in classes] == [3, 7]
    verify_output(output, 2, nc=2)


def test_converter_never_replaces_existing_output(tmp_path):
    source = tmp_path / "instances.json"
    source.write_text(json.dumps(_sample()), encoding="utf-8")
    output = tmp_path / "labels"
    output.mkdir()
    marker = output / "keep.txt"
    marker.write_text("important", encoding="utf-8")
    with pytest.raises(FileExistsError):
        convert_coco_json(source, output)
    assert marker.read_text(encoding="utf-8") == "important"


@pytest.mark.parametrize("file_name", ["../escape.jpg", "/root/escape.jpg", "C:/escape.jpg", "a\\b.jpg"])
def test_converter_rejects_unsafe_image_names_before_writing(tmp_path, file_name):
    data = _sample()
    data["images"][0]["file_name"] = file_name
    source = tmp_path / "instances.json"
    source.write_text(json.dumps(data), encoding="utf-8")
    output = tmp_path / "labels"
    with pytest.raises(ValueError):
        convert_coco_json(source, output)
    assert not output.exists()


def test_verify_output_fails_for_bad_labels(tmp_path):
    (tmp_path / "bad.txt").write_text("0 0.9 0.5 0.5 0.2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="outside the image"):
        verify_output(tmp_path, 1, nc=1)