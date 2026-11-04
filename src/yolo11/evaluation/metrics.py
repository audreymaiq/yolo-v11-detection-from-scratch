import numpy as np
from typing import Any

# NumPy 2.0 renamed `trapz` to `trapezoid`
_trapezoid = getattr(np, "trapezoid", None) or np.trapz

class DetMetrics:
    """Utility class for computing detection metrics such as precision, recall, and mean average precision (mAP).

    Attributes:
        names (dict[int, str]): A dictionary of class names.
        speed (dict[str, float]): A dictionary for storing execution times of different parts of the detection process.
        task (str): The task type, set to 'detect'.
        stats (dict[str, list]): A dictionary containing lists for true positives, confidence scores, predicted classes,
            target classes, and target images.
        nt_per_class: Number of targets per class.
        nt_per_image: Number of targets per image.
        p (list): Precision for each class. Shape: (nc,).
        r (list): Recall for each class. Shape: (nc,).
        f1 (list): F1 score for each class. Shape: (nc,).
        all_ap (list): AP scores for all classes and all IoU thresholds. Shape: (nc, 10).
        ap_class_index (list): Index of class for each AP score. Shape: (nc,).
        nc (int): Number of classes.

    Methods:
        update_stats: Update statistics by appending new values to existing stat collections.
        process: Process predicted results for object detection and update metrics.
        clear_stats: Clear the stored statistics.
        keys: Return a list of keys for accessing specific metrics.
        mean_results: Mean of results, returns mp, mr, map50, map.
        class_result: Class-aware result, returns p[i], r[i], ap50[i], ap[i].
        maps: mAP of each class.
        fitness: Model fitness as a weighted combination of metrics.
        ap_class_index: Return the average precision index per class (property).
        results_dict: Return dictionary of computed performance metrics and statistics.
        summary: Generate a summarized representation of per-class detection metrics as a list of dictionaries.
    """

    def __init__(self, names: dict[int, str] = {}) -> None:
        """Initialize a DetMetrics instance with class names.

        Args:
            names (dict[int, str], optional): Dictionary of class names.
        """
        self.names = names
        self.speed = {"preprocess": 0.0, "inference": 0.0, "loss": 0.0, "postprocess": 0.0}
        self.task = "detect"
        self.stats = dict(tp=[], conf=[], pred_cls=[], target_cls=[], target_img=[])
        self.nt_per_class = None
        self.nt_per_image = None
        self.p = []  # (nc, )
        self.r = []  # (nc, )
        self.f1 = []  # (nc, )
        self.all_ap = []  # (nc, 10)
        self.ap_class_index = []  # (nc, )
        self.nc = 0

    def update_stats(self, stat: dict[str, Any]) -> None:
        """Update statistics by appending new values to existing stat collections.

        Args:
            stat (dict[str, any]): Dictionary containing new statistical values to append. Keys should match existing
                keys in self.stats.
        """
        for k in self.stats.keys():
            self.stats[k].append(stat[k])

    def process(self) -> dict[str, np.ndarray]:
        """Process predicted results for object detection and update metrics.

        Returns:
            (dict[str, np.ndarray]): Dictionary containing concatenated statistics arrays.
        """
        stats = {k: np.concatenate(v, 0) for k, v in self.stats.items()}  # to numpy
        if not stats:
            return stats
        results = ap_per_class(
            stats["tp"],
            stats["conf"],
            stats["pred_cls"],
            stats["target_cls"],
            names=self.names,
        )[2:]
        self.nc = len(self.names)
        self.p, self.r, self.f1, self.all_ap, self.ap_class_index = results
        self.nt_per_class = np.bincount(stats["target_cls"].astype(int), minlength=len(self.names))
        self.nt_per_image = np.bincount(stats["target_img"].astype(int), minlength=len(self.names))
        return stats

    def clear_stats(self):
        """Clear the stored statistics."""
        for v in self.stats.values():
            v.clear()

    @property
    def keys(self) -> list[str]:
        """Return a list of keys for accessing specific metrics."""
        return ["metrics/precision(B)", "metrics/recall(B)", "metrics/mAP50(B)", "metrics/mAP50-95(B)"]

    def mean_results(self) -> list[float]:
        """Return mean of results, mp, mr, map50, map."""
        return [self.mp, self.mr, self.map50, self.map]

    def class_result(self, i: int) -> tuple[float, float, float, float]:
        """Return class-aware result, p[i], r[i], ap50[i], ap[i]."""
        return self.p[i], self.r[i], self.ap50[i], self.ap[i]

    @property
    def maps(self) -> np.ndarray:
        """Return mAP of each class."""
        maps = np.zeros(self.nc) + self.map
        for i, c in enumerate(self.ap_class_index):
            maps[c] = self.ap[i]
        return maps

    @property
    def fitness(self) -> float:
        """Return model fitness as a weighted combination of metrics."""
        w = [0.0, 0.0, 0.0, 1.0]  # weights for [P, R, mAP@0.5, mAP@0.5:0.95]
        return (np.nan_to_num(np.array(self.mean_results())) * w).sum()

    @property
    def results_dict(self) -> dict[str, float]:
        """Return dictionary of computed performance metrics and statistics."""
        keys = [*self.keys, "fitness"]
        values = ((float(x) if hasattr(x, "item") else x) for x in ([*self.mean_results(), self.fitness]))
        return dict(zip(keys, values))

    def summary(self, normalize: bool = True, decimals: int = 5) -> list[dict[str, Any]]:
        """Generate a summarized representation of per-class detection metrics as a list of dictionaries.

        Args:
            normalize (bool): For Detect metrics, everything is normalized by default [0-1].
            decimals (int): Number of decimal places to round the metrics values to.

        Returns:
            (list[dict[str, Any]]): A list of dictionaries, each representing one class with corresponding metric
                values.
        """
        per_class = {
            "Box-P": self.p,
            "Box-R": self.r,
            "Box-F1": self.f1,
        }
        return [
            {
                "Class": self.names[self.ap_class_index[i]],
                "Images": self.nt_per_image[self.ap_class_index[i]],
                "Instances": self.nt_per_class[self.ap_class_index[i]],
                **{k: round(v[i], decimals) for k, v in per_class.items()},
                "mAP50": round(self.class_result(i)[2], decimals),
                "mAP50-95": round(self.class_result(i)[3], decimals),
            }
            for i in range(len(per_class["Box-P"]))
        ]

    @property
    def ap50(self) -> np.ndarray | list:
        """Return the Average Precision (AP) at an IoU threshold of 0.5 for all classes."""
        return self.all_ap[:, 0] if len(self.all_ap) else []

    @property
    def ap(self) -> np.ndarray | list:
        """Return the Average Precision (AP) at an IoU threshold of 0.5-0.95 for all classes."""
        return self.all_ap.mean(1) if len(self.all_ap) else []

    @property
    def mp(self) -> float:
        """Return the Mean Precision of all classes."""
        return np.mean(self.p) if len(self.p) else 0.0

    @property
    def mr(self) -> float:
        """Return the Mean Recall of all classes."""
        return np.mean(self.r) if len(self.r) else 0.0

    @property
    def map50(self) -> float:
        """Return the mean Average Precision (mAP) at an IoU threshold of 0.5."""
        return self.all_ap[:, 0].mean() if len(self.all_ap) else 0.0

    @property
    def map75(self) -> float:
        """Return the mean Average Precision (mAP) at an IoU threshold of 0.75."""
        return self.all_ap[:, 5].mean() if len(self.all_ap) else 0.0

    @property
    def map(self) -> float:
        """Return the mean Average Precision (mAP) over IoU thresholds of 0.5 - 0.95 in steps of 0.05."""
        return self.all_ap.mean() if len(self.all_ap) else 0.0

def ap_per_class(
    tp: np.ndarray,
    conf: np.ndarray,
    pred_cls: np.ndarray,
    target_cls: np.ndarray,
    names: dict[int, str] = {},
    eps: float = 1e-16,
) -> tuple:
    """Compute the average precision per class for object detection evaluation.

    Args:
        tp (np.ndarray): Binary array indicating whether the detection is correct (True) or not (False).
        conf (np.ndarray): Array of confidence scores of the detections.
        pred_cls (np.ndarray): Array of predicted classes of the detections.
        target_cls (np.ndarray): Array of true classes of the detections.
        names (dict[int, str], optional): Dictionary of class names (unused in simplified version).
        eps (float, optional): A small value to avoid division by zero.

    Returns:
        tp (np.ndarray): True positive counts at threshold given by max F1 metric for each class.
        fp (np.ndarray): False positive counts at threshold given by max F1 metric for each class.
        p (np.ndarray): Precision values at threshold given by max F1 metric for each class.
        r (np.ndarray): Recall values at threshold given by max F1 metric for each class.
        f1 (np.ndarray): F1-score values at threshold given by max F1 metric for each class.
        ap (np.ndarray): Average precision for each class at different IoU thresholds.
        unique_classes (np.ndarray): An array of unique classes that have data.
    """
    # Sort by objectness
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # Find unique classes
    unique_classes, nt = np.unique(target_cls, return_counts=True)
    nc = unique_classes.shape[0]  # number of classes

    # Average precision
    ap = np.zeros((nc, tp.shape[1]))

    for ci, c in enumerate(unique_classes):
        i = pred_cls == c
        n_l = nt[ci]  # number of labels
        n_p = i.sum()  # number of predictions
        if n_p == 0 or n_l == 0:
            continue

        # Accumulate FPs and TPs
        fpc = (1 - tp[i]).cumsum(0)
        tpc = tp[i].cumsum(0)

        # Recall
        recall = tpc / (n_l + eps)  # recall curve

        # Precision
        precision = tpc / (tpc + fpc)  # precision curve

        # AP from recall-precision curve
        for j in range(tp.shape[1]):
            ap[ci, j], _, _ = compute_ap(recall[:, j], precision[:, j])

    # Compute F1 (harmonic mean of precision and recall)
    # Use smoothed curves for max F1 index
    x = np.linspace(0, 1, 1000)
    p_curve = np.zeros((nc, 1000))
    r_curve = np.zeros((nc, 1000))
    for ci, c in enumerate(unique_classes):
        i = pred_cls == c
        if i.sum() == 0 or nt[ci] == 0:
            continue
        recall = tp[i].cumsum(0) / (nt[ci] + eps)
        precision = tp[i].cumsum(0) / (tp[i].cumsum(0) + (1 - tp[i]).cumsum(0) + eps)
        r_curve[ci] = np.interp(-x, -conf[i], recall[:, 0], left=0)
        p_curve[ci] = np.interp(-x, -conf[i], precision[:, 0], left=1)

    f1_curve = 2 * p_curve * r_curve / (p_curve + r_curve + eps)
    i = smooth(f1_curve.mean(0), 0.1).argmax()  # max F1 index
    p, r, f1 = p_curve[:, i], r_curve[:, i], f1_curve[:, i]  # max-F1 precision, recall, F1 values
    tp = (r * nt).round()  # true positives
    fp = (tp / (p + eps) - tp).round()  # false positives
    return tp, fp, p, r, f1, ap, unique_classes.astype(int)

def compute_ap(recall: np.ndarray, precision: np.ndarray) -> tuple:
    """Compute the average precision (AP) given the recall and precision curves.

    Args:
        recall (np.ndarray): The recall curve.
        precision (np.ndarray): The precision curve.

    Returns:
        (float): Average precision.
        (np.ndarray): Precision envelope curve.
        (np.ndarray): Recall curve with sentinel values appended.
    """
    # Append sentinel values to beginning and end
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))

    # Compute the precision envelope
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))

    # Integrate area under curve
    method = 1  # methods: 1 - continuous, 2 - discrete (MS COCO)
    if method == 1:
        x = np.linspace(0, 1, 101)  # 101-point interp (COCO)
        ap = _trapezoid(np.interp(x, mrec, mpre), x)  # integrate
    else:  # 2
        i = np.where(mrec[1:] != mrec[:-1])[0]  # points where x-axis (recall) changes
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])  # area under curve

    return ap, mpre, mrec

def smooth(y: np.ndarray, f: float = 0.05) -> np.ndarray:
    """Box filter of fraction f."""
    nf = round(len(y) * f * 2) // 2 + 1  # number of filter elements (must be odd)
    p = np.ones(nf // 2)  # ones padding
    yp = np.concatenate((p * y[0], y, p * y[-1]), 0)  # y padded
    return np.convolve(yp, np.ones(nf) / nf, mode="valid")  # y-smoothed