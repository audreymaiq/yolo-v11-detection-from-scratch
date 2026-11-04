import logging
import warnings
from pathlib import Path
from typing import Callable
import numpy as np
import math
import cv2
import torch
from PIL import Image, ImageDraw

from yolo11.utils.bbox_utils import xywh2xyxy

LOGGER = logging.getLogger(__name__)

class Colors:
    def __init__(self):
        self.palette = [
            (4, 42, 255), (11, 219, 235), (243, 243, 243), (0, 223, 183), (17, 31, 104),
            (255, 111, 221), (255, 68, 79), (204, 237, 0), (0, 243, 68), (189, 0, 255),
            (0, 180, 255), (221, 0, 186), (0, 255, 255), (38, 192, 0), (1, 255, 179),
            (125, 36, 255), (123, 0, 104), (255, 27, 108), (252, 109, 47), (162, 255, 11)
        ]

    def __call__(self, i, bgr=False):
        c = self.palette[int(i) % len(self.palette)]
        return (c[2], c[1], c[0]) if bgr else c

colors = Colors()

class Annotator:
    def __init__(self, im, line_width=None):
        self.im = im.copy() if not im.flags.writeable else im
        if self.im.shape[2] == 1:  # grayscale
            self.im = cv2.cvtColor(self.im, cv2.COLOR_GRAY2BGR)
        elif self.im.shape[2] > 3:  # multispectral
            self.im = np.ascontiguousarray(self.im[..., :3])
        
        # Line width đơn giản
        self.lw = line_width or max(round(sum(self.im.shape[:2]) / 2 * 0.003), 2)
        self.tf = max(self.lw - 1, 1)  # font thickness
        self.sf = self.lw / 3  # font scale

    def box_label(self, box, label="", color=(128, 128, 128), txt_color=(255, 255, 255)):
        if isinstance(box, torch.Tensor):
            box = box.tolist()

        p1 = (int(box[0]), int(box[1]))
        p2 = (int(box[2]), int(box[3]))
        
        cv2.rectangle(self.im, p1, p2, color, thickness=self.lw, lineType=cv2.LINE_AA)
        
        if label:
            w, h = cv2.getTextSize(label, 0, fontScale=self.sf, thickness=self.tf)[0]
            h += 3  # pad
            outside = p1[1] >= h
            if p1[0] > self.im.shape[1] - w:
                p1 = (self.im.shape[1] - w, p1[1])
            p2_text = (p1[0] + w, p1[1] - h if outside else p1[1] + h)
            cv2.rectangle(self.im, p1, p2_text, color, -1, cv2.LINE_AA)  # filled
            cv2.putText(self.im, label, (p1[0], p1[1] - 2 if outside else p1[1] + h - 1),
                        0, self.sf, txt_color, thickness=self.tf, lineType=cv2.LINE_AA)

    def result(self):
        return self.im

def plot_labels(boxes, cls, names=(), save_dir=Path(""), on_plot=None):
    """Plot training labels including class histograms and box statistics.
    Args:
        boxes (np.ndarray): Bounding box coordinates in format [x, y, width, height].
        cls (np.ndarray): Class indices.
        names (dict, optional): Dictionary mapping class indices to class names.
        save_dir (Path, optional): Directory to save the plot.
        on_plot (Callable, optional): Function to call after plot is saved.
    """
    import matplotlib.pyplot as plt
    import polars as pl
    from matplotlib.colors import LinearSegmentedColormap
    # Filter matplotlib>=3.7.2 warning
    warnings.filterwarnings("ignore", category=UserWarning, message="The figure layout has changed to tight")
    warnings.filterwarnings("ignore", category=FutureWarning)
    # Plot dataset labels
    LOGGER.info(f"Plotting labels to {save_dir / 'labels.jpg'}... ")
    boxes = boxes[:1000000]  # limit to 1M boxes
    # Handle empty boxes
    if boxes.size == 0 or boxes.shape[0] == 0:
        schema = {"x": pl.Float64, "y": pl.Float64, "width": pl.Float64, "height": pl.Float64}
        x = pl.DataFrame(schema=schema)  # empty df with schema
    else:
        x = pl.DataFrame(boxes, schema=["x", "y", "width", "height"])
    # Matplotlib labels
    subplot_3_4_color = LinearSegmentedColormap.from_list("white_blue", ["white", "blue"])
    ax = plt.subplots(2, 2, figsize=(8, 8), tight_layout=True)[1].ravel()
   
    cls = np.asarray(cls).ravel()  # Đảm bảo np 1D array (handle scalar/single/multi)
    if cls.size == 0:
        nc = len(names) if names else 0  # fallback to len(names)
        ax[0].text(0.5, 0.5, 'No labels', horizontalalignment='center', verticalalignment='center')
        ax[0].set_xlim(0, nc if nc > 0 else 1)
        ax[0].set_ylim(0, 1)
        y = (None, None, [plt.Rectangle((0,0),1,1) for _ in range(nc)])  # dummy patches for loop
    else:
        nc = int(cls.max() + 1)
        y = ax[0].hist(cls, bins=np.linspace(0, nc, nc + 1) - 0.5, rwidth=0.8)
    
    for i in range(nc):
        if cls.size > 0 and i < len(y[2].patches):
            y[2].patches[i].set_color([c / 255 for c in colors(i, bgr=False)])  # Matplotlib needs RGB
    ax[0].set_ylabel("instances")
    if 0 < len(names) < 30:
        ax[0].set_xticks(range(len(names)))
        ax[0].set_xticklabels(list(names.values()), rotation=90, fontsize=10)
        if cls.size > 0:
            ax[0].bar_label(y[2])
    else:
        ax[0].set_xlabel("classes")
    boxes_plot = np.column_stack([0.5 - boxes[:, 2:4] / 2, 0.5 + boxes[:, 2:4] / 2]) * 1000 if boxes.size > 0 else np.zeros((0,4))
    img = Image.fromarray(np.ones((1000, 1000, 3), dtype=np.uint8) * 255)
    for cls_val, box in zip(cls[:500], boxes_plot[:500]):
        ImageDraw.Draw(img).rectangle(box.tolist(), width=1, outline=colors(cls_val, bgr=False))  # PIL needs RGB
    ax[1].imshow(img)
    ax[1].axis("off")
    ax[2].hist2d(x["x"], x["y"], bins=50, cmap=subplot_3_4_color)
    ax[2].set_xlabel("x")
    ax[2].set_ylabel("y")
    ax[3].hist2d(x["width"], x["height"], bins=50, cmap=subplot_3_4_color)
    ax[3].set_xlabel("width")
    ax[3].set_ylabel("height")
    for a in {0, 1, 2, 3}:
        for s in {"top", "right", "left", "bottom"}:
            ax[a].spines[s].set_visible(False)
    fname = save_dir / "labels.jpg"
    plt.savefig(fname, dpi=200)
    plt.close()
    if on_plot:
        on_plot(fname)

def plot_images(
    labels,
    images=np.zeros((0, 3, 640, 640), dtype=np.float32),
    paths=None,
    fname="images.jpg",
    names=None,
    max_size=1920,
    max_subplots=16,
    save=True,
    conf_thres=0.25,
):
    # Chuẩn hóa labels sang np
    for k in ["cls", "bboxes", "conf", "batch_idx"]:
        if k in labels and isinstance(labels[k], torch.Tensor):
            labels[k] = labels[k].cpu().numpy()
        if k == "cls" and labels[k].ndim == 2:
            labels[k] = labels[k].squeeze(1)

    cls = labels.get("cls", np.zeros(0, dtype=np.int64))
    batch_idx = labels.get("batch_idx", np.zeros(len(cls), dtype=np.int64))
    bboxes = labels.get("bboxes", np.zeros((0, 4), dtype=np.float32))
    confs = labels.get("conf", None)
    images = labels.get("img", images)  # default images

    if isinstance(images, torch.Tensor):
        images = images.cpu().float().numpy()

    # Giả định 3-ch, bỏ handle 2/4+ ch để đơn giản
    if images.max() <= 1:
        images *= 255

    bs, _, h, w = images.shape
    bs = min(bs, max_subplots)
    ns = math.ceil(bs ** 0.5)  # square grid

    mosaic = np.full((int(ns * h), int(ns * w), 3), 255, dtype=np.uint8)
    for i in range(bs):
        x, y = int(w * (i // ns)), int(h * (i % ns))
        mosaic[y:y+h, x:x+w] = images[i].transpose(1, 2, 0)

    # Resize nếu cần
    scale = max_size / ns / max(h, w)
    if scale < 1:
        h = math.ceil(scale * h)
        w = math.ceil(scale * w)
        mosaic = cv2.resize(mosaic, (int(ns * w), int(ns * h)))

    # Annotate (dùng cv2 Annotator)
    fs = max(int((h + w) * ns * 0.01), 18)
    annotator = Annotator(mosaic, line_width=round(fs / 10))

    for i in range(bs):
        x, y = int(w * (i // ns)), int(h * (i % ns))
        cv2.rectangle(mosaic, (x, y), (x + w, y + h), (255, 255, 255), thickness=2)  # borders (dùng cv2 thay annotator.rectangle)

        if paths:  # Giữ nếu cần, hoặc loại
            cv2.putText(mosaic, paths[i][:40], (x + 5, y + 5 + fs), 0, fs / 3, (220, 220, 220), thickness=2, lineType=cv2.LINE_AA)

        if len(cls) > 0:
            idx = batch_idx == i
            classes = cls[idx].astype(int)
            conf = confs[idx] if confs is not None else None
            boxes = bboxes[idx]
            if len(boxes):
                # Scale boxes
                if boxes[:, :4].max() <= 1.1:  # normalized
                    boxes[..., [0, 2]] *= w
                    boxes[..., [1, 3]] *= h
                elif scale < 1:
                    boxes[..., :4] *= scale
                boxes[..., 0] += x
                boxes[..., 1] += y
                boxes = xywh2xyxy(boxes)  # Chuyển xywh sang xyxy nếu cần
                for j, box in enumerate(boxes.astype(np.int64).tolist()):
                    c = classes[j]
                    color = colors(c, bgr=False)  # rgb for annotator on rgb image
                    c_name = names.get(c, str(c)) if names else str(c)
                    if conf is None or conf[j] > conf_thres:
                        label = c_name if conf is None else f"{c_name} {conf[j]:.1f}"
                        annotator.box_label(box, label, color=color)

    result_im = annotator.result()
    if save:
        result_im_bgr = cv2.cvtColor(result_im, cv2.COLOR_RGB2BGR)
        cv2.imwrite(fname, result_im_bgr)
    else:
        return result_im

def plot_results(file: str = "path/to/results.csv", dir: str = "", on_plot: Callable | None = None):
    """Plot training results from a results CSV file. The function supports various types of data including
    segmentation, pose estimation, and classification. Plots are saved as 'results.png' in the directory where the
    CSV is located.

    Args:
        file (str, optional): Path to the CSV file containing the training results.
        dir (str, optional): Directory where the CSV file is located if 'file' is not provided.
        on_plot (callable, optional): Callback function to be executed after plotting. Takes filename as an argument.

    Examples:
        >>> from ultralytics.utils.plotting import plot_results
        >>> plot_results("path/to/results.csv", segment=True)
    """
    import matplotlib.pyplot as plt  # scope for faster 'import ultralytics'
    import polars as pl
    from scipy.ndimage import gaussian_filter1d

    save_dir = Path(file).parent if file else Path(dir)
    files = list(save_dir.glob("results*.csv"))
    assert len(files), f"No results.csv files found in {save_dir.resolve()}, nothing to plot."

    loss_keys, metric_keys = [], []
    for i, f in enumerate(files):
        try:
            data = pl.read_csv(f, infer_schema_length=None)
            if i == 0:
                for c in data.columns:
                    if "loss" in c:
                        loss_keys.append(c)
                    elif "metric" in c:
                        metric_keys.append(c)
                loss_mid, metric_mid = len(loss_keys) // 2, len(metric_keys) // 2
                columns = (
                    loss_keys[:loss_mid] + metric_keys[:metric_mid] + loss_keys[loss_mid:] + metric_keys[metric_mid:]
                )
                fig, ax = plt.subplots(2, len(columns) // 2, figsize=(len(columns) + 2, 6), tight_layout=True)
                ax = ax.ravel()
            x = data.select(data.columns[0]).to_numpy().flatten()
            for i, j in enumerate(columns):
                y = data.select(j).to_numpy().flatten().astype("float")
                ax[i].plot(x, y, marker=".", label=f.stem, linewidth=2, markersize=8)  # actual results
                ax[i].plot(x, gaussian_filter1d(y, sigma=3), ":", label="smooth", linewidth=2)  # smoothing line
                ax[i].set_title(j, fontsize=12)
        except Exception as e:
            LOGGER.error(f"Plotting error for {f}: {e}")
    ax[1].legend()
    fname = save_dir / "results.png"
    fig.savefig(fname, dpi=200)
    plt.close()
    if on_plot:
        on_plot(fname)

def feature_visualization(x, module_type: str, stage: int, n: int = 32, save_dir: Path = Path("runs/detect/exp")):
    """Visualize feature maps of a given model module during inference.

    Args:
        x (torch.Tensor): Features to be visualized.
        module_type (str): Module type.
        stage (int): Module stage within the model.
        n (int, optional): Maximum number of feature maps to plot.
        save_dir (Path, optional): Directory to save results.
    """
    import matplotlib.pyplot as plt  # scope for faster 'import ultralytics'

    if "Detect" in module_type:
        return
    
    if isinstance(x, torch.Tensor):
        _, channels, height, width = x.shape  # batch, channels, height, width
        if height > 1 and width > 1:
            f = save_dir / f"stage{stage}_{module_type.rsplit('.', 1)[-1]}_features.png"  # filename

            blocks = torch.chunk(x[0].cpu(), channels, dim=0)  # select batch index 0, block by channels
            n = min(n, channels)  # number of plots
            _, ax = plt.subplots(math.ceil(n / 8), 8, tight_layout=True)  # 8 rows x n/8 cols
            ax = ax.ravel()
            plt.subplots_adjust(wspace=0.05, hspace=0.05)
            for i in range(n):
                ax[i].imshow(blocks[i].squeeze())  # cmap='gray'
                ax[i].axis("off")

            LOGGER.info(f"Saving {f}... ({n}/{channels})")
            plt.savefig(f, dpi=300, bbox_inches="tight")
            plt.close()
            np.save(str(f.with_suffix(".npy")), x[0].cpu().numpy())  # npy save

def draw_detections(image: np.ndarray, detections: list[dict]) -> np.ndarray:
    """Draw ``Predictor`` detections on a BGR image and return the annotated copy."""
    annotator = Annotator(np.ascontiguousarray(image).copy())
    for det in detections:
        label = f"{det['label']} {det['confidence']:.2f}"
        annotator.box_label(det["box"], label, color=colors(det["class_id"], bgr=True))
    return annotator.result()
