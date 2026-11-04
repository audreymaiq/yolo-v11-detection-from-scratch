import torch.nn as nn
from typing import Dict, Type, Any
from yolo11.training.losses import v8DetectionLoss, VarifocalLoss, FocalLoss, DFLoss, BboxLoss

class LossRegistry:
    _losses: Dict[str, Type[nn.Module]] = {}
    
    @classmethod
    def register(cls, name: str, loss_class: Type[nn.Module]):
        """
        Đăng ký một lớp loss function.
        Cú pháp: LossRegistry.register(name, loss_class)
        """
        if name in cls._losses:
            raise ValueError(f"Loss '{name}' is already registered.")
        if not issubclass(loss_class, nn.Module):
            raise ValueError(f"Loss class '{loss_class.__name__}' must subclass nn.Module for consistency.")
        cls._losses[name] = loss_class
    
    @classmethod
    def get_loss(cls, name: str, **kwargs: Any) -> nn.Module:
        """
        Lấy một instance của loss function bằng tên đã đăng ký.
        Nó sẽ tìm class tương ứng và khởi tạo nó với các tham số **kwargs.
        """
        if name not in cls._losses:
            available = ", ".join(cls.list_losses())
            raise ValueError(f"Loss '{name}' not registered. Available losses are: [{available}]")
        
        loss_class = cls._losses[name]
        return loss_class(**kwargs)
    
    @classmethod
    def list_losses(cls) -> list[str]:
        """Liệt kê tất cả các tên loss đã được đăng ký."""
        return list(cls._losses.keys())

# --- Đăng ký các loss function từ losses.py ---
LossRegistry.register('yolov11_loss', v8DetectionLoss)  # Custom detection loss
LossRegistry.register('varifocal_loss', VarifocalLoss)  # Varifocal for imbalance
LossRegistry.register('focal_loss', FocalLoss)  # Focal for hard negatives
LossRegistry.register('df_loss', DFLoss)  # Distribution Focal Loss
LossRegistry.register('bbox_loss', BboxLoss)  # Bbox-specific loss

# --- Đăng ký các loss chuẩn từ torch.nn ---
LossRegistry.register('cross_entropy', nn.CrossEntropyLoss)
LossRegistry.register('mse', nn.MSELoss)
LossRegistry.register('l1', nn.L1Loss)
LossRegistry.register('bce', nn.BCELoss)
LossRegistry.register('bce_with_logits', nn.BCEWithLogitsLoss)