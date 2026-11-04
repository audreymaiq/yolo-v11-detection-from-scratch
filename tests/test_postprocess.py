import torch

from yolo11.postprocessing.nms import non_max_suppression
from yolo11.utils.bbox_utils import scale_boxes, xywh2xyxy


def test_nms_suppresses_overlapping_boxes():
    nc = 2
    # (B, 4+nc, N): two near-identical boxes of class 0, one box of class 1
    pred = torch.zeros(1, 4 + nc, 3)
    pred[0, :4, 0] = torch.tensor([50.0, 50, 20, 20])
    pred[0, :4, 1] = torch.tensor([51.0, 50, 20, 20])
    pred[0, :4, 2] = torch.tensor([20.0, 20, 10, 10])
    pred[0, 4, :2] = torch.tensor([0.9, 0.8])
    pred[0, 5, 2] = 0.7
    out = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.5)[0]
    assert out.shape == (2, 6)
    assert sorted(out[:, 5].tolist()) == [0.0, 1.0]


def test_scale_boxes_undoes_letterbox():
    # 100x200 (HxW) image letterboxed into 64x64: gain 0.32, vertical pad 16
    boxes = torch.tensor([[0.0, 16.0, 64.0, 48.0]])
    out = scale_boxes((64, 64), boxes.clone(), (100, 200))
    assert torch.allclose(out, torch.tensor([[0.0, 0.0, 200.0, 100.0]]), atol=3)


def test_xywh2xyxy():
    assert xywh2xyxy(torch.tensor([[10.0, 10, 4, 6]])).tolist() == [[8.0, 7.0, 12.0, 13.0]]
