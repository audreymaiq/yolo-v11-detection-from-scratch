import numpy as np
import pytest

from yolo11.evaluation.metrics import ap_per_class, compute_ap


def test_partial_recall_does_not_invent_precision_between_last_recall_and_one():
    # One true positive among two targets gives a maximum recall of 0.5.
    # The old curve used [0, 0.5, 1] and incorrectly gave AP about 0.75.
    ap, _, _ = compute_ap(np.array([0.5]), np.array([1.0]))
    assert ap == pytest.approx(0.495, abs=1e-6)


def test_ap_per_class_with_one_missed_target():
    tp = np.ones((1, 10), dtype=bool)
    result = ap_per_class(tp, np.array([0.9]), np.array([0]), np.array([0, 0]))
    assert result[5].shape == (1, 10)
    np.testing.assert_allclose(result[5], 0.495, atol=1e-6)