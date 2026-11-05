import random
import math
import cv2
import torch
import numpy as np

from yolo11.utils.bbox_utils import bbox_ioa
from yolo11.data.preprocessor import Instances

def v8_transforms(dataset, imgsz, hyp, stretch=False, pad_value=114):
    """Build augmentation pipeline for YOLO training (Safe Version)"""
    # helper to handle both dict and object (Namespace)
    is_dict = isinstance(hyp, dict)
    def get_hyp(k, default=0.0):
        if is_dict:
            return hyp.get(k, default)
        return getattr(hyp, k, default)

    # 1. Mosaic Check
    mosaic_prob = get_hyp("mosaic")
    use_mosaic = mosaic_prob > 0.0
    mosaic = Mosaic(dataset, imgsz=imgsz, p=mosaic_prob, pad_value=pad_value) if use_mosaic else None

    # 2. Affine Transforms
    scale_raw = get_hyp("scale", 0.5)
    aug_scale = 0.5 if isinstance(scale_raw, str) else float(scale_raw)
    affine = RandomPerspective(
        degrees=get_hyp("degrees"),
        translate=get_hyp("translate"),
        scale=aug_scale,
        shear=get_hyp("shear"),
        perspective=get_hyp("perspective"),
        pre_transform=None if stretch else LetterBox(new_shape=(imgsz, imgsz), padding_value=pad_value),
        pad_val=pad_value,
    )
    pre_transform = Compose([mosaic, affine]) if mosaic else affine
    transforms_list = [pre_transform]
    
    # 3. MixUp
    mixup_prob = get_hyp("mixup")
    if mixup_prob > 0.0:
        transforms_list.append(MixUp(dataset, pre_transform=pre_transform, p=mixup_prob))
        
    # 4. CutMix
    cutmix_prob = get_hyp("cutmix")
    if cutmix_prob > 0.0:
        transforms_list.append(CutMix(dataset, pre_transform=pre_transform, p=cutmix_prob))
        
    # 5. Color/Flip Transforms
    transforms_list.extend([
        RandomHSV(hgain=get_hyp("hsv_h"), sgain=get_hyp("hsv_s"), vgain=get_hyp("hsv_v")),
        RandomGrayscale(p=get_hyp("grayscale")),
        RandomFlip(direction="vertical", p=get_hyp("flipud")),
        RandomFlip(direction="horizontal", p=get_hyp("fliplr")),
    ])
    
    return Compose(transforms_list)

class BaseTransform:
    """Base class for image transformations in a simplified Ultralytics-style library.
    Focused only on detection tasks (image and instances). Removed semantic support.
    Methods:
        apply_image: Apply image transformations to labels.
        apply_instances: Apply transformations to object instances in labels.
        __call__: Apply label transformations to image and instances.
    Examples:
        >>> transform = BaseTransform()
        >>> labels = {"img": np.array(...), "instances": [...]}
        >>> transformed_labels = transform(labels)
    """
    def __init__(self) -> None:
        """Initialize the BaseTransform object for detection tasks."""
        pass

    def apply_image(self, labels):
        """Apply image transformations to labels.
        Override in subclasses for specific logic. Base returns unchanged.
        Args:
            labels (dict): Contains 'img' key with image data.
        Returns:
            (dict): Updated labels.
        Examples:
            >>> transform = BaseTransform()
            >>> labels = {"img": np.random.rand(640, 640, 3)}
            >>> transform.apply_image(labels)
        """
        pass

    def apply_instances(self, labels):
        """Apply transformations to object instances in labels.
        For detection: transforms bounding boxes and classes.
        Override in subclasses. Base returns unchanged.
        Args:
            labels (dict): Contains 'instances' key with boxes/cls.
        Returns:
            (dict): Updated labels.
        Examples:
            >>> transform = BaseTransform()
            >>> labels = {"instances": {"bboxes": torch.rand(5, 4), "cls": torch.randint(0, 80, (5,))}}
            >>> transform.apply_instances(labels)
        """
        pass

    def __call__(self, labels):
        """Apply transformations to image and instances.
        Only calls detection-relevant methods.
        Args:
            labels (dict): Contains 'img' and optionally 'instances'.
        Returns:
            (dict): Transformed labels.
        Examples:
            >>> transform = BaseTransform()
            >>> labels = {"img": np.random.rand(640, 640, 3), "instances": {}}
            >>> transformed_labels = transform(labels)
        """
        self.apply_image(labels)
        self.apply_instances(labels)
        return labels

class Compose:
    """A class for composing multiple image transformations.
    No changes needed; it's simple and task-agnostic.
    Attributes:
        transforms (list[Callable]): List of transformation functions.
    Methods:
        __call__: Apply transforms sequentially.
        append: Add a transform.
        insert: Insert a transform at index.
        __getitem__: Get transform(s).
        __setitem__: Set transform(s).
        tolist: Convert to list.
    Examples:
        >>> transforms = [RandomFlip(), RandomPerspective(30)]
        >>> compose = Compose(transforms)
        >>> transformed_data = compose(data)
    """
    def __init__(self, transforms):
        """Initialize with list of transforms.
        Args:
            transforms (list[Callable]): Transforms to apply.
        """
        self.transforms = transforms if isinstance(transforms, list) else [transforms]

    def __call__(self, data):
        """Apply transforms to data.
        Args:
            data (dict): Input data (labels).
        Returns:
            (dict): Transformed data.
        """
        for t in self.transforms:
            data = t(data)
        return data

    def append(self, transform):
        """Append a transform.
        Args:
            transform (BaseTransform): Transform to add.
        """
        self.transforms.append(transform)

    def insert(self, index, transform):
        """Insert a transform at index.
        Args:
            index (int): Position to insert.
            transform (BaseTransform): Transform to insert.
        """
        self.transforms.insert(index, transform)

    def __getitem__(self, index: list | int) -> 'Compose':
        """Get transform(s) by index.
        Args:
            index (int | list[int]): Index or list of indices.
        Returns:
            (Compose): New Compose with selected transforms.
        """
        if isinstance(index, int):
            return self.transforms[index]
        elif isinstance(index, list):
            return Compose([self.transforms[i] for i in index])
        else:
            raise AssertionError(f"Index should be int or list, got {type(index)}")

    def __setitem__(self, index: list | int, value: list | BaseTransform) -> None:
        """Set transform(s) by index.
        Args:
            index (int | list[int]): Index or list.
            value (BaseTransform | list[BaseTransform]): Value(s) to set.
        """
        if isinstance(index, int):
            index, value = [index], [value]
        if not isinstance(value, list):
            value = [value]
        for i, v in zip(index, value):
            if i >= len(self.transforms):
                raise AssertionError(f"Index {i} out of range {len(self.transforms)}.")
            self.transforms[i] = v

    def tolist(self):
        """Convert transforms to list.
        Returns:
            (list): List of transforms.
        """
        return self.transforms

    def __repr__(self):
        """String representation.
        Returns:
            (str): Compose string.
        """
        return f"{self.__class__.__name__}([{', '.join([str(t) for t in self.transforms])}])"

class BaseMixTransform:
    """Base class for mix transformations like Mosaic, MixUp, CutMix.
    Simplified for detection only. Handles probability-based mixing of images and labels (bboxes + cls).
    Attributes:
        dataset: The dataset object to get additional images/labels.
        pre_transform (Callable | None): Optional transform before mixing.
        p (float): Probability of applying the mix.
    Methods:
        __call__: Apply mix transform if probability met.
        _mix_transform: To be implemented by subclasses.
        get_indexes: Get random indexes for mixing.
    Examples:
        >>> transform = BaseMixTransform(dataset, pre_transform=Compose([...]), p=0.5)
        >>> mixed_labels = transform(labels)
    """
    def __init__(self, dataset, pre_transform=None, p=0.0) -> None:
        """Initialize for detection mix transforms.
        Args:
            dataset: Dataset to fetch extra images.
            pre_transform: Transform to apply before mixing.
            p: Probability [0.0, 1.0].
        """
        self.dataset = dataset
        self.pre_transform = pre_transform
        self.p = p

    def __call__(self, labels):
        """Apply pre_transform and mix transform to labels.
        Args:
            labels (dict): {'img': np.ndarray, 'bboxes': torch.Tensor(N,4), 'cls': torch.Tensor(N), ...}
        Returns:
            (dict): Mixed labels.
        """
        if random.uniform(0, 1) > self.p:
            return labels
        indexes = self.get_indexes()
        if isinstance(indexes, int):
            indexes = [indexes]
        mix_labels = [self.dataset.get_image_and_label(i) for i in indexes]
        if self.pre_transform is not None:
            for i, data in enumerate(mix_labels):
                mix_labels[i] = self.pre_transform(data)
        labels['mix_labels'] = mix_labels
        labels = self._mix_transform(labels)
        labels.pop('mix_labels', None)
        return labels

    def _mix_transform(self, labels):
        """Implement specific mix logic in subclasses.
        Args:
            labels (dict): With 'mix_labels' for extra data.
        Returns:
            (dict): Mixed labels.
        """
        raise NotImplementedError

    def get_indexes(self):
        """Get random index(es) for mixing.
        Returns:
            (int or list[int]): Random index(es).
        """
        return random.randint(0, len(self.dataset) - 1)

class Mosaic(BaseMixTransform):
    """Mosaic augmentation for detection (simplified to 2x2 grid).
    Combines 4 images into one mosaic. Focused on detection (bboxes + cls).
    Attributes:
        imgsz (int): Target image size.
        border (tuple): Border for padding.
    Methods:
        get_indexes: Get 3 random indexes (for 4 total images).
        _mix_transform: Apply mosaic4.
        _mosaic4: Create 2x2 mosaic.
        _update_labels: Add padding to bboxes.
        _cat_labels: Concat labels and clip.
    Examples:
        >>> mosaic = Mosaic(dataset, imgsz=640, p=1.0)
        >>> augmented_labels = mosaic(labels)
    """
    def __init__(self, dataset, imgsz=640, p=1.0, pad_value=114):
        """Initialize Mosaic for 2x2 grid.
        Args:
            dataset: Dataset.
            imgsz: Target size.
            p: Probability.
            pad_value: Padding value.
        """
        super().__init__(dataset=dataset, p=p)
        self.imgsz = imgsz
        self.pad_value = pad_value
        self.border = (-imgsz // 2, -imgsz // 2)  # width, height

    def get_indexes(self):
        """Get 3 random indexes for mosaic4.
        Returns:
            (list[int]): 3 indexes.
        """
        return [random.randint(0, len(self.dataset) - 1) for _ in range(3)]

    def _mix_transform(self, labels):
        """Apply mosaic4.
        Args:
            labels (dict): With 'mix_labels'.
        Returns:
            (dict): Mosaic labels.
        """
        if len(labels.get('mix_labels', [])) == 0:
            return labels
        return self._mosaic4(labels)

    def _mosaic4(self, labels):
        mosaic_labels = []
        s = self.imgsz
        yc, xc = (int(random.uniform(-x, 2 * s + x)) for x in self.border)
        for i in range(4):
            if i == 0:  # top left
                labels_patch = labels
            else:
                labels_patch = labels['mix_labels'][i - 1]
            img = labels_patch['img']
            
            # CHUẨN HÓA TENSOR (Zero-copy View)
            # np.atleast_3d trả về view O(1), rất nhanh.
            img = np.atleast_3d(img)
            h, w, c = img.shape
            
            # Khởi tạo Canvas (Chỉ làm 1 lần ở ảnh đầu tiên)
            if i == 0:  
                # TỐI ƯU HÓA TỐC ĐỘ KHỞI TẠO
                # np.zeros nhanh hơn np.full khoảng 2-3 lần (do dùng calloc/memset)
                # Nếu pad_value là 0, dùng zeros. Nếu khác, dùng full.
                if self.pad_value == 0:
                    img4 = np.zeros((s * 2, s * 2, c), dtype=np.uint8)
                else:
                    img4 = np.full((s * 2, s * 2, c), self.pad_value, dtype=np.uint8)
                
                # Tọa độ cắt ghép (Giữ nguyên logic chuẩn)
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h
            elif i == 1:  # top right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # bottom left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
            elif i == 3:  # bottom right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)
            # img4.shape[2] là chuẩn (theo ảnh đầu tiên), img.shape[2] là ảnh hiện tại
            canvas_c = img4.shape[2]
            if c != canvas_c:
                if canvas_c == 1 and c == 3:
                    # Case Nguy Hiểm: Canvas Xám, Ảnh Màu -> Phải chuyển Ảnh sang Xám
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                    img = np.atleast_3d(img) # Đảm bảo lại rank-3
                elif canvas_c == 3 and c == 1:
                    # Case An Toàn: Canvas Màu, Ảnh Xám -> Broadcasting tự lo (1->3)
                    pass 
            img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]
            padw = x1a - x1b
            padh = y1a - y1b
            self._update_labels(labels_patch, padw, padh, (x1b, y1b, x2b, y2b))
            mosaic_labels.append(labels_patch)
        final_labels = self._cat_labels(mosaic_labels)
        final_labels['img'] = img4
        return final_labels

    @staticmethod
    def _update_labels(labels, padw, padh, clip_rect=None):
        """Add padding to bboxes"""
        instances = labels['instances']
        instances.denormalize(labels['img'].shape[1], labels['img'].shape[0])
        instances.convert_bbox("xyxy")
        
        if clip_rect is not None:
            # Clip to the source patch region [x1, y1, x2, y2]
            # Note: clip_rect is (x1b, y1b, x2b, y2b)
            bboxes = instances.bboxes
            bboxes[:, 0] = np.clip(bboxes[:, 0], clip_rect[0], clip_rect[2])
            bboxes[:, 1] = np.clip(bboxes[:, 1], clip_rect[1], clip_rect[3])
            bboxes[:, 2] = np.clip(bboxes[:, 2], clip_rect[0], clip_rect[2])
            bboxes[:, 3] = np.clip(bboxes[:, 3], clip_rect[1], clip_rect[3])
            instances.update(bboxes)

        instances.add_padding(padw, padh)
        instances.convert_bbox("xywh")

    def _cat_labels(self, mosaic_labels):
        """Concat labels, clip to mosaic size, remove zero-area boxes.
        Args:
            mosaic_labels (list[dict]): List of 4 labels.
        Returns:
            (dict): Combined labels.
        """
        if not mosaic_labels:
            return {}
        cls = []
        bboxes = []
        imgsz = self.imgsz * 2
        for lab in mosaic_labels:
            instances = lab['instances']
            instances.denormalize(lab['img'].shape[1], lab['img'].shape[0])
            instances.convert_bbox("xyxy")
            cls.append(torch.tensor(lab['cls']))
            bboxes.append(instances.bboxes)
        boxes = torch.clamp(torch.from_numpy(np.concatenate(bboxes, 0)), 0, imgsz)
        good = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) > 0
        final_labels = {
            'im_file': mosaic_labels[0].get('im_file', None),
            'ori_shape': mosaic_labels[0].get('ori_shape', None),
            'resized_shape': (imgsz, imgsz),
            'cls': torch.cat(cls, 0)[good],
            'mosaic_border': self.border,  # consumed by RandomPerspective
        }
        instances = Instances(boxes[good].numpy(), bbox_format="xyxy", normalized=False)
        instances.convert_bbox("xywh")
        instances.normalize(imgsz, imgsz)
        final_labels['instances'] = instances
        return final_labels

class MixUp(BaseMixTransform):
    """MixUp augmentation for detection.
    Blends two images and concatenates bboxes/cls.
    Methods:
        _mix_transform: Apply blend.
    Examples:
        >>> mixup = MixUp(dataset, p=0.5)
        >>> augmented_labels = mixup(labels)
    """
    def __init__(self, dataset, pre_transform=None, p=0.0) -> None:
        """Initialize MixUp.
        Args:
            dataset: Dataset.
            pre_transform: Pre-mix transform.
            p: Probability.
        """
        super().__init__(dataset=dataset, pre_transform=pre_transform, p=p)

    def _mix_transform(self, labels):
        """Apply MixUp.
        Args:
            labels (dict): With 'mix_labels' (list of 1 dict).
        Returns:
            (dict): Mixed labels.
        """
        # Lấy ảnh 2
        if len(labels.get('mix_labels', [])) == 0:
            return labels
        labels2 = labels["mix_labels"][0]
        # Chuẩn hóa đầu vào
        # Ép tất cả về dạng 3D (H, W, C) để xử lý toán học an toàn
        img1 = np.atleast_3d(labels["img"])
        img2 = np.atleast_3d(labels2["img"])
        r = np.random.beta(32.0, 32.0)
        # Xử lý tương thích kênh màu
        # Nếu img1 là 3 kênh, img2 là 1 kênh -> không cộng được trực tiếp
        if img1.shape[2] != img2.shape[2]:
             # Đơn giản hóa: Ép về cùng số kênh lớn nhất (thường là 3)
             max_c = max(img1.shape[2], img2.shape[2])
             if img1.shape[2] < max_c:
                 img1 = np.repeat(img1, max_c, axis=2)
             if img2.shape[2] < max_c:
                 img2 = np.repeat(img2, max_c, axis=2)
        # Tính toán MixUp
        mixed_img = (img1 * r + img2 * (1 - r)).astype(np.uint8)
        labels["img"] = mixed_img
        instances1 = labels['instances']
        instances2 = labels2['instances']
        instances1.denormalize(labels["img"].shape[1], labels["img"].shape[0])
        instances2.denormalize(labels2["img"].shape[1], labels2["img"].shape[0])
        instances1.convert_bbox("xyxy")
        instances2.convert_bbox("xyxy")
        cat_instances = Instances.concatenate([instances1, instances2])
        cat_instances.convert_bbox("xywh")
        cat_instances.normalize(labels["img"].shape[1], labels["img"].shape[0])
        labels['instances'] = cat_instances
        labels["cls"] = torch.cat([torch.as_tensor(labels["cls"]), torch.as_tensor(labels2["cls"])], 0)
        labels['instances'].remove_zero_area_boxes()
        return labels

class CutMix(BaseMixTransform):
    """CutMix augmentation for detection.
    Cuts a region from second image and pastes, adjusts bboxes.
    Attributes:
        beta (float): For sampling ratio.
        num_areas (int): Num tries for cut area.
    Methods:
        _mix_transform: Apply cut.
        _rand_bbox: Random bbox for cut.
    Examples:
        >>> cutmix = CutMix(dataset, p=0.5)
        >>> augmented_labels = cutmix(labels)
    """
    def __init__(self, dataset, pre_transform=None, p=0.0, beta=1.0) -> None:
        """Initialize CutMix.
        Args:
            dataset: Dataset.
            pre_transform: Pre-mix transform.
            p: Probability.
            beta: Beta param.
            num_areas: Num areas to try.
        """
        super().__init__(dataset=dataset, pre_transform=pre_transform, p=p)
        self.beta = beta

    def _rand_bbox(self, width, height):
        """Generate random bbox for cut.
        Returns:
            (tuple[int]): x1, y1, x2, y2
        """
        lam = np.random.beta(self.beta, self.beta)
        cut_ratio = np.sqrt(1.0 - lam)
        cut_w = int(width * cut_ratio)
        cut_h = int(height * cut_ratio)
        cx = np.random.randint(width)
        cy = np.random.randint(height)
        x1 = np.clip(cx - cut_w // 2, 0, width)
        y1 = np.clip(cy - cut_h // 2, 0, height)
        x2 = np.clip(cx + cut_w // 2, 0, width)
        y2 = np.clip(cy + cut_h // 2, 0, height)
        return x1, y1, x2, y2

    def _mix_transform(self, labels):
        """Apply CutMix.
        Args:
            labels (dict): With 'mix_labels' (list of 1 dict).
        Returns:
            (dict): Mixed labels.
        """        
        # Chuẩn hóa đầu vào
        img1 = np.atleast_3d(labels["img"])
        h, w = img1.shape[:2]
        
        labels2 = labels.pop("mix_labels")[0]
        img2 = np.atleast_3d(labels2["img"])

        area = self._rand_bbox(w, h)
        x1, y1, x2, y2 = map(int, area)
        
        # Lấy vùng patch từ ảnh 2
        patch = img2[y1:y2, x1:x2]
        # Nếu dán Patch Xám vào Nền Màu -> Nhân bản Patch lên
        if img1.shape[2] == 3 and patch.shape[2] == 1:
             patch = np.repeat(patch, 3, axis=2)
        # Nếu dán Patch Màu vào Nền Xám -> Chuyển Patch sang Xám
        elif img1.shape[2] == 1 and patch.shape[2] == 3:
             patch = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)
             patch = np.atleast_3d(patch) # Giữ rank-3

        # Gán
        img1[y1:y2, x1:x2] = patch
        labels["img"] = img1
        instances1 = labels['instances']
        instances1.denormalize(w, h)
        instances1.convert_bbox("xyxy")
        ioa1 = bbox_ioa(np.array([area]), instances1.bboxes).squeeze(0)
        if ioa1.sum() > 0:
            return labels
        instances2 = labels2['instances']
        instances2.denormalize(w, h)
        instances2.convert_bbox("xyxy")
        ioa2 = bbox_ioa(np.array([area]), instances2.bboxes).squeeze(0)
        indexes2 = np.nonzero(ioa2 >= 0.3)[0]  # Threshold 0.3 từ official variant cho partial paste
        if len(indexes2) == 0:
            return labels
        bboxes2 = instances2.bboxes[indexes2]
        cls2 = labels2["cls"][indexes2]
        x1, y1, x2, y2 = map(int, area)
        labels["img"][y1:y2, x1:x2] = labels2["img"][y1:y2, x1:x2]
        bboxes2 += np.array([-x1, -y1, -x1, -y1])
        np.clip(bboxes2, [0, 0, 0, 0], [x2 - x1, y2 - y1, x2 - x1, y2 - y1], out=bboxes2)
        bboxes2 += np.array([x1, y1, x1, y1])
        cat_bboxes = np.concatenate([instances1.bboxes, bboxes2], 0)
        cat_cls = torch.cat([torch.tensor(labels["cls"]), torch.tensor(cls2)], 0)
        cat_instances = Instances(cat_bboxes, bbox_format="xyxy", normalized=False)
        cat_instances.convert_bbox("xywh")
        cat_instances.normalize(w, h)
        cat_instances.remove_zero_area_boxes()
        labels['instances'] = cat_instances
        labels["cls"] = cat_cls
        return labels

class RandomPerspective(BaseTransform):
    """Random perspective/affine transforms for detection.
    Applies rotate, translate, scale, shear, perspective to img and bboxes.
    Attributes:
        degrees, translate, scale, shear, perspective, border.
    Methods:
        __call__: Apply transform.
    Examples:
        >>> transform = RandomPerspective()
        >>> result = transform(labels)
    """
    def __init__(
        self,
        degrees=0.0,
        translate=0.1,
        scale=0.5,
        shear=0.0,
        perspective=0.0,
        border=(0, 0),
        pre_transform=None,
        pad_val=114,
    ):
        """Initialize.
        Args:
            degrees: Rotation range.
            translate: Translate fraction.
            scale: Scale range.
            shear: Shear degrees.
            perspective: Perspective factor.
            border: Mosaic border.
            pre_transform: Pre-transform.
            pad_val: Padding value (int).
        """
        super().__init__()
        self.degrees = float(degrees)
        self.translate = float(translate)
        self.scale = float(scale)
        self.shear = float(shear)
        self.perspective = float(perspective)
        self.border = border
        self.pre_transform = pre_transform
        self.pad_val = pad_val
        self.pre_transform_shape = None

    def __call__(self, labels):
        """Warp image and boxes; mosaic and non-mosaic samples leave with the same keys."""
        if self.pre_transform and "mosaic_border" not in labels:
            labels = self.pre_transform(labels)
        labels.pop("ratio_pad", None)  # only meaningful for un-augmented (val) images
        border = labels.pop("mosaic_border", self.border)
        self.apply_image(labels, border)
        self.apply_instances(labels)
        labels.pop("M", None)
        labels.pop("scale", None)
        labels["resized_shape"] = labels["img"].shape[:2]
        return labels

    def apply_image(self, labels, border=(0, 0)):
        """Apply affine to img; ``border`` is negative for a 2x mosaic canvas to crop it back to imgsz."""
        img = labels["img"]
        height, width = img.shape[:2]

        self.pre_transform_shape = (height, width)

        border = tuple(border)
        self.size = (width + border[1] * 2, height + border[0] * 2)  # output (w, h)
        # Center determination
        C = np.eye(3, dtype=np.float32)
        C[0, 2] = -width / 2
        C[1, 2] = -height / 2
        # Perspective, Rotation, Shear
        P = np.eye(3, dtype=np.float32)
        P[2, 0] = random.uniform(-self.perspective, self.perspective)
        P[2, 1] = random.uniform(-self.perspective, self.perspective)

        R = np.eye(3, dtype=np.float32)
        a = random.uniform(-self.degrees, self.degrees)
        s = random.uniform(1 - self.scale, 1 + self.scale)
        R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

        S = np.eye(3, dtype=np.float32)
        S[0, 1] = math.tan(random.uniform(-self.shear, self.shear) * math.pi / 180)
        S[1, 0] = math.tan(random.uniform(-self.shear, self.shear) * math.pi / 180)

        # Translation: place the image center around the output center, +/- translate * output size
        T = np.eye(3, dtype=np.float32)
        T[0, 2] = random.uniform(0.5 - self.translate, 0.5 + self.translate) * self.size[0]
        T[1, 2] = random.uniform(0.5 - self.translate, 0.5 + self.translate) * self.size[1]

        M = T @ S @ R @ P @ C

        if (border != (0, 0)) or not np.allclose(M, np.eye(3)):
            # Dynamic border value based on channels
            c = img.shape[2] if len(img.shape) == 3 else 1
            pad_curr = (self.pad_val,) * c
            
            if self.perspective:
                img = cv2.warpPerspective(img, M, dsize=self.size, borderValue=pad_curr)
            else:
                img = cv2.warpAffine(img, M[:2], dsize=self.size, borderValue=pad_curr)
        labels["img"] = img
        labels["M"] = M
        labels["scale"] = s
        return labels

    def apply_instances(self, labels):
        """Apply to bboxes using correct denormalization logic."""
        if "instances" not in labels or len(labels["instances"].bboxes) == 0:
            return labels
        M = labels.pop("M")
        s = labels.pop("scale", 1.0)
        instances = labels["instances"]
        was_normalized = instances.normalized
        if self.pre_transform_shape is not None:
            h, w = self.pre_transform_shape
        else:
            h, w = labels["img"].shape[:2] 
        instances.denormalize(w, h) 
        instances.convert_bbox("xyxy")
        bboxes = instances.bboxes
        n = len(bboxes)
        # Transform points
        if n:
            xy = np.ones((n * 4, 3), dtype=bboxes.dtype)
            xy[:, :2] = bboxes[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(n * 4, 2)
            xy = xy @ M.T
            xy = (xy[:, :2] / xy[:, 2:3] if self.perspective else xy[:, :2]).reshape(n, 8)
            
            x = xy[:, [0, 2, 4, 6]]
            y = xy[:, [1, 3, 5, 7]]
            new_bboxes = np.stack((x.min(1), y.min(1), x.max(1), y.max(1))).T

            # Clip boxes to the NEW image size (self.size)
            target_w, target_h = self.size
            new_bboxes[:, [0, 2]] = new_bboxes[:, [0, 2]].clip(0, target_w)
            new_bboxes[:, [1, 3]] = new_bboxes[:, [1, 3]].clip(0, target_h)

            # Filter candidates against the source boxes at the sampled scale, so zooming out alone
            # does not count as losing area; only clipping/distortion does
            i = RandomPerspective.box_candidates(bboxes.T * s, new_bboxes.T)
            
            instances.update(new_bboxes[i])
            labels["cls"] = labels["cls"][i]
            
            instances.convert_bbox("xywh")
            if was_normalized:
                instances.normalize(target_w, target_h)
            instances.remove_zero_area_boxes()
            
        return labels

    @staticmethod
    def box_candidates(box1, box2, wh_thr=2, ar_thr=100, area_thr=0.1, eps=1e-16):
        """Filter transformed bboxes."""
        w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
        w2, h2 = box2[2] - box2[0], box2[3] - box2[1]
        ar = np.maximum(w2 / (h2 + eps), h2 / (w2 + eps))
        return (w2 > wh_thr) & (h2 > wh_thr) & (w2 * h2 / (w1 * h1 + eps) > area_thr) & (ar < ar_thr)

class RandomHSV(BaseTransform):
    """Random HSV adjust on image.
    Adjusts hue, saturation, value within gains.
    Methods:
        apply_image: Apply to img.
    Examples:
        >>> hsv = RandomHSV(0.015, 0.7, 0.4)
        >>> labels = hsv(labels)
    """
    def __init__(self, hgain=0.5, sgain=0.5, vgain=0.5) -> None:
        """Initialize gains."""
        self.hgain = float(hgain)
        self.sgain = float(sgain)
        self.vgain = float(vgain)

    def apply_image(self, labels):
        """Apply random HSV."""
        img = labels["img"]
        # Skip HSV for Grayscale (2D or 1-channel 3D)
        if len(img.shape) == 2 or (len(img.shape) == 3 and img.shape[2] == 1):
            return labels
        if img.shape[-1] != 3:
            return labels
        if self.hgain or self.sgain or self.vgain:
            r = np.random.uniform(-1, 1, 3) * [self.hgain, self.sgain, self.vgain]
            x = np.arange(0, 256, dtype=r.dtype)
            lut_hue = ((x + r[0] * 180) % 180).astype(img.dtype)
            lut_sat = np.clip(x * (r[1] + 1), 0, 255).astype(img.dtype)
            lut_val = np.clip(x * (r[2] + 1), 0, 255).astype(img.dtype)
            lut_sat[0] = 0
            hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_RGB2HSV))
            im_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
            cv2.cvtColor(im_hsv, cv2.COLOR_HSV2RGB, dst=img)
        return labels

class RandomGrayscale(BaseTransform):
    """Randomly convert image to grayscale with probability p.
    Maintains 3-channel format (R=G=B).
    """
    def __init__(self, p=0.1):
        super().__init__()
        self.p = float(p)

    def apply_image(self, labels):
        if random.random() < self.p:
            img = labels['img']
            # Convert to gray then back to BGR/RGB to keep 3 channels
            # Input img is usually RGB or BGR depending on cv2 reading, but RandomHSV treats it as RGB (skips if not 3 channels).
            # Here want R=G=B.
            if img is not None and len(img.shape) == 3 and img.shape[2] == 3:
                gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                labels['img'] = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        return labels

class RandomFlip(BaseTransform):
    """Random flip image horizontal/vertical with p.
    Flips img and adjusts bboxes.
    Attributes:
        p (float): Probability.
        direction (str): 'horizontal' or 'vertical'.
    Methods:
        apply_image: Flip img.
        apply_instances: Flip bboxes.
    Examples:
        >>> flip = RandomFlip(0.5, "horizontal")
        >>> labels = flip(labels)
    """
    def __init__(self, p=0.5, direction="horizontal") -> None:
        """Initialize."""
        super().__init__()
        assert direction in {"horizontal", "vertical"}
        assert 0 <= p <= 1.0
        self.p = float(p)
        self.direction = direction
        self.axis = 1 if direction == "horizontal" else 0
        self.flipped = False

    def apply_image(self, labels):
        """Flip img if random < p."""
        self.flipped = random.random() < self.p
        if self.flipped:
            labels["img"] = np.flip(labels["img"], axis=self.axis)
        return labels

    def apply_instances(self, labels):
        """Flip bboxes if flipped."""
        if "instances" not in labels or not self.flipped:
            return labels
            
        h, w = labels["img"].shape[:2]
        instances = labels["instances"]
        instances.denormalize(w, h) 
        # Chuyển sang xyxy để lật các góc
        instances.convert_bbox("xyxy") 
        if self.axis == 1:  # horizontal
            instances.fliplr(w)
        else:  # vertical
            instances.flipud(h)
            
        instances.convert_bbox("xywh")
        instances.normalize(w, h)
        instances.remove_zero_area_boxes()
        return labels

    def __call__(self, labels):
        self.apply_image(labels)
        self.apply_instances(labels)
        return labels

class LetterBox(BaseTransform):
    """Resize and pad image for detection.
    Resizes while preserving aspect ratio, pads to target shape, updates bboxes.
    Attributes:
        new_shape (tuple): Target (h, w).
        auto (bool): Min rectangle.
        scale_fill (bool): Stretch without pad.
        scaleup (bool): Allow upscale.
        stride (int): For padding round.
        center (bool): Center or top-left.
        padding_value (int): Pad value.
        interpolation (int): Resize interp.
    Methods:
        __call__: Apply to labels or image.
    Examples:
        >>> letterbox = LetterBox((640, 640))
        >>> result = letterbox(labels)
    """
    def __init__(
        self,
        new_shape=(640, 640),
        auto=False,
        scale_fill=False,
        scaleup=True,
        center=True,
        stride=32,
        padding_value=114,
        interpolation=cv2.INTER_LINEAR,
    ):
        """Initialize."""
        super().__init__()
        self.new_shape = new_shape
        self.auto = auto
        self.scale_fill = scale_fill
        self.scaleup = scaleup
        self.center = center
        self.stride = stride
        self.padding_value = padding_value
        self.interpolation = interpolation

    def __call__(self, labels=None, image=None):
        if labels is None:
            labels = {}
        img = labels.get('img') if image is None else image
        shape = img.shape[:2]  # ori h, w
        new_shape = labels.pop('rect_shape', self.new_shape)
        if isinstance(new_shape, (int, str)):
            new_shape = (int(new_shape), int(new_shape))
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not self.scaleup:
            r = min(r, 1.0)
        ratio = r, r
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))  # new_w, new_h (after resize, before pad)
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
        if self.auto:
            dw, dh = dw % self.stride, dh % self.stride  # mod to stride
        elif self.scale_fill:
            dw, dh = 0.0, 0.0
            new_unpad = (new_shape[1], new_shape[0])
            ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]
        if self.center:
            dw /= 2
            dh /= 2
        if shape[::-1] != new_unpad:  # resize if needed
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        # Tự động tạo tuple value dựa trên số kênh của ảnh (1 hoặc 3)
        # Nếu ảnh 1 kênh -> value=(0,), nếu 3 kênh -> value=(0,0,0)
        c = img.shape[2] if len(img.shape) == 3 else 1
        pad_color_tuple = (self.padding_value,) * c
        img = cv2.copyMakeBorder(
            img, top, bottom, left, right, 
            cv2.BORDER_CONSTANT, 
            value=pad_color_tuple # Truyền tuple động
        )
        if len(labels):
            labels = self._update_labels(labels, ratio, left, top, shape[1], shape[0], new_shape[1], new_shape[0])
        if image is None:
            labels['img'] = img
            return labels
        return img

    @staticmethod
    def _update_labels(labels, ratio, padw, padh, ori_w, ori_h, new_w, new_h):
        """Update bboxes with scale and pad."""
        if 'instances' in labels:
            instances = labels['instances']
            was_normalized = instances.normalized
            instances.denormalize(ori_w, ori_h)  # denorm with ori w, h (note: denormalize expects w, h order)
            instances.scale(*ratio)  # scale abs coords with resize ratio
            instances.add_padding(padw, padh)  # add abs pad to coords (shift)
            instances.clip(new_w, new_h)  # clip to new w, h (abs)
            if was_normalized:
                instances.normalize(new_w, new_h)  # norm with new w, h
            instances.remove_zero_area_boxes()
            if isinstance(ratio, float):
                ratio = (ratio, ratio)
            # Accumulate ratio with existing ratio from Preprocessor
            if 'ratio_pad' in labels:
                 # labels['ratio_pad'] = ((h_ratio_prev, w_ratio_prev), (padw_prev, padh_prev))
                 # Current ratio is (w_ratio_new, h_ratio_new)
                 # New Total Ratio = Prev Ratio * New Ratio
                 prev_ratio = labels['ratio_pad'][0] # (h, w)
                 # ratio is (w, h) so we match indices: ratio[1] is h, ratio[0] is w
                 new_h_ratio = prev_ratio[0] * ratio[1]
                 new_w_ratio = prev_ratio[1] * ratio[0]
                 
                 # Pad Accumulation:
                 # padw_total = padw_prev * w_ratio_new + padw_new
                 # But Preprocessor pad is usually 0. Let's strictly accumulate.
                 prev_pad = labels['ratio_pad'][1]
                 new_padw = prev_pad[0] * ratio[0] + padw
                 new_padh = prev_pad[1] * ratio[1] + padh
                 
                 labels['ratio_pad'] = (
                     (new_h_ratio, new_w_ratio), 
                     (new_padw, new_padh)
                 )
            else:
                 labels['ratio_pad'] = (
                     (ratio[1], ratio[0]), 
                     (padw, padh)
                 )
        return labels

class Format:
    """Format annotations for detection collate.
    Standardizes img to tensor, bboxes to specified format, normalizes if set.
    Attributes:
        bbox_format (str): 'xywh' or 'xyxy'.
        normalize (bool): Normalize bboxes [0,1].
        batch_idx (bool): Keep batch idx.
    Methods:
        __call__: Format labels.
        _format_img: Img to tensor.
    Examples:
        >>> formatter = Format(bbox_format="xyxy", normalize=True)
        >>> formatted = formatter(labels)
    """
    def __init__(
        self,
        bbox_format="xyxy",
        normalize=True,
        batch_idx=True,
    ):
        """Initialize for detection."""
        self.bbox_format = bbox_format
        self.normalize = normalize
        self.batch_idx = batch_idx

    def __call__(self, labels):
        """Format for collate_fn.
        Args:
            labels (dict): {'img': np, 'cls': np/torch (N,), 'bboxes': np/torch (N,4) xyxy abs, ...}
        Returns:
            (dict): {'img': torch, 'cls': torch, 'bboxes': torch, 'batch_idx': torch if set}
        """
        img = labels['img']
        h, w = img.shape[:2]
        labels['img'] = self._format_img(img)
        if 'instances' in labels and 'cls' in labels:
            cls = labels['cls']
            instances = labels['instances']
            bboxes = instances.bboxes
            nl = len(bboxes) # Số lượng nhãn

            # Áp dụng formatting
            if self.bbox_format == 'xywh':
                instances.convert_bbox('xywh')
                bboxes = instances.bboxes
            if self.normalize:
                instances.normalize(w, h)
                bboxes = instances.bboxes

            labels['cls'] = torch.as_tensor(cls, dtype=torch.float32)
            labels['bboxes'] = torch.from_numpy(bboxes)

        else:
            nl = 0
            labels['cls'] = torch.empty((0, 1), dtype=torch.float32)
            labels['bboxes'] = torch.empty((0, 4), dtype=torch.float32)
        if self.batch_idx:
            labels['batch_idx'] = torch.zeros(nl)
        labels.pop('instances', None)         
        return labels

    def _format_img(self, img):
        """Img np to torch tensor CHW."""
        if len(img.shape) < 3:
            img = np.expand_dims(img, -1)
        img = img.transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        return torch.from_numpy(img)