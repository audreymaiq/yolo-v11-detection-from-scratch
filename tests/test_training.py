"""TaskAlignedAssigner, the detection loss and ModelEMA on hand-made inputs with known answers."""

import math
from types import SimpleNamespace

import pytest
import torch

from yolo11.registry import ModelRegistry
from yolo11.training.assigners import TaskAlignedAssigner
from yolo11.training.losses import v8DetectionLoss
from yolo11.utils.torch_utils import ModelEMA

# ---------------------------------------------------------------- assigner
# 8x8 grid of anchors with stride 8 over a 64x64 image: centers at 4, 12, ..., 60 on each axis.
CENTERS = torch.stack(torch.meshgrid(torch.arange(4.0, 64, 8), torch.arange(4.0, 64, 8), indexing="xy"), -1).view(-1, 2)
NA = len(CENTERS)


def _assign(gt_boxes, gt_cls, pd_boxes, scores=None, topk=13, mask=None):
    gt_boxes = torch.tensor(gt_boxes, dtype=torch.float).view(1, -1, 4)
    gt_labels = torch.tensor(gt_cls, dtype=torch.float).view(1, -1, 1)
    mask = torch.ones(1, gt_boxes.shape[1], 1) if mask is None else torch.tensor(mask, dtype=torch.float).view(1, -1, 1)
    pd_boxes = torch.tensor(pd_boxes, dtype=torch.float).expand(1, NA, 4).clone()
    scores = torch.full((1, NA, 2), 0.9) if scores is None else scores
    return TaskAlignedAssigner(topk=topk, num_classes=2, alpha=0.5, beta=6.0)(
        scores, pd_boxes, CENTERS, gt_labels, gt_boxes, mask
    )


def _inside(box):
    x1, y1, x2, y2 = box
    return (CENTERS[:, 0] > x1) & (CENTERS[:, 0] < x2) & (CENTERS[:, 1] > y1) & (CENTERS[:, 1] < y2)


def test_assigner_assigns_only_anchors_inside_the_box():
    box = [0, 0, 24, 24]  # anchor centers 4, 12, 20 on each axis fall inside: 9 anchors
    labels, boxes, scores, fg, _ = _assign([box], [1], box)
    fg = fg[0]
    assert torch.equal(fg, _inside(box)) and fg.sum() == 9
    assert (labels[0][fg] == 1).all()
    assert torch.equal(boxes[0][fg], torch.tensor(box, dtype=torch.float).expand(9, 4))
    assert (scores[0][~fg] == 0).all() and (scores[0][fg][:, 0] == 0).all()
    assert torch.allclose(scores[0][fg][:, 1], torch.ones(9))  # predictions match the box: IoU 1, full target


def test_assigner_keeps_topk_by_alignment():
    box = [0, 0, 24, 24]
    scores = torch.full((1, NA, 2), 0.1)
    best = _inside(box).nonzero().flatten()[[0, 4, 8]]  # 3 of the 9 candidates predict the class confidently
    scores[0, best, 0] = 0.9
    *_, fg, _ = _assign([box], [0], box, scores=scores, topk=3)
    assert torch.equal(fg[0].nonzero().flatten(), best)


def test_assigner_without_boxes_assigns_nothing():
    labels, boxes, scores, fg, _ = TaskAlignedAssigner(num_classes=2)(
        torch.rand(1, NA, 2), torch.rand(1, NA, 4), CENTERS, torch.zeros(1, 0, 1), torch.zeros(1, 0, 4),
        torch.zeros(1, 0, 1),
    )
    assert not fg.any() and not scores.any() and (labels == 2).all()


def test_assigner_ignores_padded_boxes():
    box = [0, 0, 24, 24]
    single = _assign([box], [1], box)
    padded = _assign([box, [0, 0, 0, 0]], [1, 0], box, mask=[1, 0])
    for a, b in zip(single, padded):
        assert torch.equal(a, b)


def test_assigner_gives_shared_anchors_to_the_best_overlapping_box():
    a, b = [0, 0, 32, 32], [16, 16, 48, 48]  # anchors at 20 and 28 on both axes lie in both boxes
    labels, boxes, _, fg, gt_idx = _assign([a, b], [0, 1], b)  # every anchor predicts box b
    shared = _inside(a) & _inside(b)
    assert shared.sum() == 4 and fg[0][shared].all()
    assert (gt_idx[0][shared] == 1).all() and (labels[0][shared] == 1).all()
    only_a = _inside(a) & ~_inside(b)
    assert (labels[0][only_a & fg[0]] == 0).all()


# ---------------------------------------------------------------- loss
@pytest.fixture
def model():
    torch.manual_seed(0)
    m = ModelRegistry.get_model("yolov11n", ch=3, nc=2, verbose=False).train()
    m.args = SimpleNamespace(box=7.5, cls=0.5, dfl=1.5)
    return m


def _batch():
    torch.manual_seed(1)
    return {
        "img": torch.rand(2, 3, 64, 64),
        "batch_idx": torch.tensor([0.0, 0.0, 1.0]),
        "cls": torch.tensor([[0.0], [1.0], [1.0]]),
        "bboxes": torch.tensor([[0.3, 0.3, 0.4, 0.4], [0.7, 0.6, 0.3, 0.5], [0.5, 0.5, 0.6, 0.6]]),  # xywh, 0-1
    }


def test_loss_preprocess_groups_targets_per_image(model):
    crit = v8DetectionLoss(model)
    b = _batch()
    targets = torch.cat((b["batch_idx"].view(-1, 1), b["cls"], b["bboxes"]), 1)
    out = crit.preprocess(targets, 2, scale_tensor=torch.tensor([64.0, 64, 64, 64]))
    assert out.shape == (2, 2, 5)  # 2 images, at most 2 boxes each, (cls, x1, y1, x2, y2)
    assert torch.allclose(out[0, 0], torch.tensor([0, 6.4, 6.4, 32.0, 32.0]))  # xywh 0.3,0.3,0.4,0.4 * 64
    assert torch.allclose(out[1, 0], torch.tensor([1, 12.8, 12.8, 51.2, 51.2]))
    assert not out[1, 1].any()  # padding row for the image with one box


def test_loss_is_finite_and_scaled_by_batch(model):
    b = _batch()
    total, items = v8DetectionLoss(model)(model(b["img"]), b)
    assert torch.isfinite(total).all() and (items > 0).all()
    assert torch.allclose(total, items * 2)  # summed over the batch of 2


def test_loss_without_targets_is_classification_only(model):
    b = _batch()
    b.update(batch_idx=torch.zeros(0), cls=torch.zeros(0, 1), bboxes=torch.zeros(0, 4))
    _, items = v8DetectionLoss(model)(model(b["img"]), b)
    assert items[0] == 0 and items[2] == 0 and items[1] > 0  # box, cls, dfl


@pytest.mark.parametrize("as_dict", [False, True])
def test_loss_gains_come_from_args(model, as_dict):
    b = _batch()
    preds = [p.detach() for p in model(b["img"])]
    _, base = v8DetectionLoss(model)(preds, b)
    gains = {"box": 15.0, "cls": 0.25, "dfl": 3.0}
    model.args = gains if as_dict else SimpleNamespace(**gains)  # checkpoints load args as a dict
    _, scaled = v8DetectionLoss(model)(preds, b)
    assert torch.allclose(scaled, base * torch.tensor([2.0, 0.5, 2.0]))


def test_loss_backpropagates_and_decreases(model):
    b = _batch()
    crit = v8DetectionLoss(model)
    opt = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    first = None
    for _ in range(30):
        total, items = crit(model(b["img"]), b)
        opt.zero_grad()
        total.sum().backward()
        if first is None:
            first = items.sum().item()
            grads = {n: p.grad for n, p in model.named_parameters()}
            has_grad = lambda prefix: all(  # noqa: E731
                g is not None and g.abs().sum() > 0 for n, g in grads.items() if n.startswith(prefix)
            )
            assert has_grad("model.0.") and has_grad("model.23.cv3.")  # backbone; class branch sees every anchor
            # the box branch learns only on levels that got positive anchors (at 64 px, the 2x2 P5 level gets none)
            assert any(has_grad(f"model.23.cv2.{i}.") for i in range(3))
        opt.step()
    assert items.sum().item() < 0.7 * first  # it learns this one batch


# ---------------------------------------------------------------- EMA
def test_ema_update_follows_its_decay():
    torch.manual_seed(0)
    net = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.BatchNorm1d(4))
    ema = ModelEMA(net, decay=0.9, tau=10)
    before = {k: v.clone() for k, v in ema.ema.state_dict().items()}
    with torch.no_grad():
        for p in net.parameters():
            p.add_(1.0)
    net(torch.randn(8, 4))  # moves the BatchNorm running statistics too
    ema.update(net)

    d = 0.9 * (1 - math.exp(-1 / 10))  # first update: decay ramps up from 0
    assert ema.updates == 1
    for k, v in ema.ema.state_dict().items():
        if v.dtype.is_floating_point:
            assert torch.allclose(v, d * before[k] + (1 - d) * net.state_dict()[k]), k
        else:  # num_batches_tracked is not averaged
            assert torch.equal(v, before[k]), k


def test_ema_decay_ramps_to_its_maximum():
    ema = ModelEMA(torch.nn.Linear(2, 2), decay=0.9999, tau=2000)
    assert ema.decay(1) < 0.001 and ema.decay(2000) == pytest.approx(0.9999 * (1 - math.exp(-1)))
    assert ema.decay(100_000) == pytest.approx(0.9999, abs=1e-12)


def test_ema_copy_is_frozen_and_independent():
    net = torch.nn.Linear(2, 2)
    ema = ModelEMA(net)
    assert not ema.ema.training and not any(p.requires_grad for p in ema.ema.parameters())
    with torch.no_grad():
        net.weight.add_(1.0)
    assert not torch.equal(ema.ema.weight, net.weight)  # a copy, not a view


def test_ema_update_attr_copies_only_included():
    net, other = torch.nn.Linear(2, 2), torch.nn.Linear(2, 2)
    ema = ModelEMA(net)
    other.names, other.extra = {0: "a"}, 1
    ema.update_attr(other, include=["names"])
    assert ema.ema.names == {0: "a"} and not hasattr(ema.ema, "extra")
