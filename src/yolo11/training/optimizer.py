import torch.optim as optim
import torch.nn as nn
from typing import List

def build_param_groups(model: nn.Module) -> List[List[nn.Parameter]]:
    """
    Xây dựng các nhóm tham số cho optimizer:
    - g[0]: weights cần decay
    - g[1]: norms và logit_scale không decay
    - g[2]: biases không decay
   
    Args:
        model (nn.Module): Model để phân nhóm tham số.
   
    Returns:
        List[List[nn.Parameter]]: Danh sách 3 nhóm tham số.
    """
    g = [], [], [] # optimizer parameter groups
    bn = tuple(v for k, v in nn.__dict__.items() if "Norm" in k) # normalization layers, i.e. BatchNorm2d()
    for module_name, module in model.named_modules():
        for param_name, param in module.named_parameters(recurse=False):
            fullname = f"{module_name}.{param_name}" if module_name else param_name
            if "bias" in fullname:
                g[2].append(param) # biases (no decay)
            elif isinstance(module, bn) or "logit_scale" in fullname:
                g[1].append(param) # no decay
            else:
                g[0].append(param) # apply decay
    return g

def adam_optimizer(model: nn.Module, lr: float, momentum: float, decay: float, **kwargs) -> optim.Optimizer:
    """
    Tạo Adam optimizer với các nhóm tham số.
   
    Args:
        model (nn.Module): Model.
        lr (float): Learning rate.
        momentum (float): Momentum (beta1).
        decay (float): Weight decay.
        **kwargs: Các tham số khác (không sử dụng).
   
    Returns:
        optim.Optimizer: Instance của optimizer.
    """
    g = build_param_groups(model)
    optimizer = optim.Adam(g[2], lr=lr, betas=(momentum, 0.999), weight_decay=0.0)
    optimizer.add_param_group({"params": g[0], "weight_decay": decay})
    optimizer.add_param_group({"params": g[1], "weight_decay": 0.0})
    return optimizer

def adamax_optimizer(model: nn.Module, lr: float, momentum: float, decay: float, **kwargs) -> optim.Optimizer:
    """
    Tạo Adamax optimizer với các nhóm tham số.
   
    Args:
        model (nn.Module): Model.
        lr (float): Learning rate.
        momentum (float): Momentum (beta1).
        decay (float): Weight decay.
        **kwargs: Các tham số khác (không sử dụng).
   
    Returns:
        optim.Optimizer: Instance của optimizer.
    """
    g = build_param_groups(model)
    optimizer = optim.Adamax(g[2], lr=lr, betas=(momentum, 0.999), weight_decay=0.0)
    optimizer.add_param_group({"params": g[0], "weight_decay": decay})
    optimizer.add_param_group({"params": g[1], "weight_decay": 0.0})
    return optimizer

def adamw_optimizer(model: nn.Module, lr: float, momentum: float, decay: float, **kwargs) -> optim.Optimizer:
    """
    Tạo AdamW optimizer với các nhóm tham số.
   
    Args:
        model (nn.Module): Model.
        lr (float): Learning rate.
        momentum (float): Momentum (beta1).
        decay (float): Weight decay.
        **kwargs: Các tham số khác (không sử dụng).
   
    Returns:
        optim.Optimizer: Instance của optimizer.
    """
    g = build_param_groups(model)
    optimizer = optim.AdamW(g[2], lr=lr, betas=(momentum, 0.999), weight_decay=0.0)
    optimizer.add_param_group({"params": g[0], "weight_decay": decay})
    optimizer.add_param_group({"params": g[1], "weight_decay": 0.0})
    return optimizer

def nadam_optimizer(model: nn.Module, lr: float, momentum: float, decay: float, **kwargs) -> optim.Optimizer:
    """
    Tạo NAdam optimizer với các nhóm tham số.
   
    Args:
        model (nn.Module): Model.
        lr (float): Learning rate.
        momentum (float): Momentum (beta1).
        decay (float): Weight decay.
        **kwargs: Các tham số khác (không sử dụng).
   
    Returns:
        optim.Optimizer: Instance của optimizer.
    """
    g = build_param_groups(model)
    optimizer = optim.NAdam(g[2], lr=lr, betas=(momentum, 0.999), weight_decay=0.0)
    optimizer.add_param_group({"params": g[0], "weight_decay": decay})
    optimizer.add_param_group({"params": g[1], "weight_decay": 0.0})
    return optimizer

def radam_optimizer(model: nn.Module, lr: float, momentum: float, decay: float, **kwargs) -> optim.Optimizer:
    """
    Tạo RAdam optimizer với các nhóm tham số.
   
    Args:
        model (nn.Module): Model.
        lr (float): Learning rate.
        momentum (float): Momentum (beta1).
        decay (float): Weight decay.
        **kwargs: Các tham số khác (không sử dụng).
   
    Returns:
        optim.Optimizer: Instance của optimizer.
    """
    g = build_param_groups(model)
    optimizer = optim.RAdam(g[2], lr=lr, betas=(momentum, 0.999), weight_decay=0.0)
    optimizer.add_param_group({"params": g[0], "weight_decay": decay})
    optimizer.add_param_group({"params": g[1], "weight_decay": 0.0})
    return optimizer

def rmsprop_optimizer(model: nn.Module, lr: float, momentum: float, decay: float, **kwargs) -> optim.Optimizer:
    """
    Tạo RMSProp optimizer với các nhóm tham số.
   
    Args:
        model (nn.Module): Model.
        lr (float): Learning rate.
        momentum (float): Momentum.
        decay (float): Weight decay.
        **kwargs: Các tham số khác (không sử dụng).
   
    Returns:
        optim.Optimizer: Instance của optimizer.
    """
    g = build_param_groups(model)
    optimizer = optim.RMSprop(g[2], lr=lr, momentum=momentum)
    optimizer.add_param_group({"params": g[0], "weight_decay": decay})
    optimizer.add_param_group({"params": g[1], "weight_decay": 0.0})
    return optimizer

def sgd_optimizer(model: nn.Module, lr: float, momentum: float, decay: float, **kwargs) -> optim.Optimizer:
    """
    Tạo SGD optimizer với các nhóm tham số.
   
    Args:
        model (nn.Module): Model.
        lr (float): Learning rate.
        momentum (float): Momentum.
        decay (float): Weight decay.
        **kwargs: Các tham số khác (không sử dụng).
   
    Returns:
        optim.Optimizer: Instance của optimizer.
    """
    g = build_param_groups(model)
    optimizer = optim.SGD(g[2], lr=lr, momentum=momentum, nesterov=True)
    optimizer.add_param_group({"params": g[0], "weight_decay": decay})
    optimizer.add_param_group({"params": g[1], "weight_decay": 0.0})
    return optimizer