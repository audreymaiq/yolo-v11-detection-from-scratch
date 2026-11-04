from typing import Dict, Callable, Any
import torch.optim as optim
import torch.nn as nn
from yolo11.training.optimizer import adam_optimizer, adamax_optimizer, adamw_optimizer, nadam_optimizer, radam_optimizer, rmsprop_optimizer, sgd_optimizer

import logging
LOGGER = logging.getLogger(__name__)

class OptimizerRegistry:
    _optimizers: Dict[str, Callable] = {}

    @classmethod
    def register(cls, name: str, optimizer_builder: Callable):
        """
        Register an optimizer builder function.
        
        Args:
            name (str): The name to register the optimizer under (e.g., 'sgd', 'adamw').
            optimizer_builder (Callable): A function that takes model, lr, momentum, decay, **kwargs, returning an optimizer instance.
        """
        if name in cls._optimizers:
            raise ValueError(f"Optimizer '{name}' is already registered.")
        cls._optimizers[name] = optimizer_builder

    @classmethod
    def get_optimizer(cls, name: str, model: nn.Module, lr: float, momentum: float, decay: float, iterations: int, **kwargs: Any) -> optim.Optimizer:
        """
        Get an optimizer instance by its registered name.
        
        Args:
            name (str): The name of the optimizer (e.g., 'sgd' or 'auto').
            model (nn.Module): The model to optimize.
            lr (float): Learning rate.
            momentum (float): Momentum factor.
            decay (float): Weight decay.
            iterations (int): Total iterations for auto selection.
            **kwargs: Additional kwargs (e.g., nc for auto).
        
        Returns:
            An instance of torch.optim.Optimizer.
        """
        original_name = name
        name = name.lower()
        
        if name == "auto":
            # Logic "auto" từ code gốc
            nc = kwargs.get("nc", 10)  # Lấy nc từ kwargs (truyền từ trainer)
            lr_fit = round(0.002 * 5 / (4 + nc), 6)
            if iterations > 10000:
                name = "sgd"
                selected_lr = 0.01  # Default cho SGD lớn
                selected_momentum = 0.9
            else:
                name = "adamw"
                selected_lr = lr_fit
                selected_momentum = 0.9
            LOGGER.info(f"Optimizer 'auto' selected: {name.upper()} with lr={selected_lr}, momentum={selected_momentum}")
            lr = selected_lr
            momentum = selected_momentum
            # Có thể cập nhật args nếu cần, ví dụ: kwargs["warmup_bias_lr"] = 0.0
        
        if name not in cls._optimizers:
            available = ", ".join(cls.list_optimizers())
            raise ValueError(f"Optimizer '{original_name}' not registered. Available optimizers are: [{available}]")
        
        optimizer_builder = cls._optimizers[name]
        optimizer = optimizer_builder(model=model, lr=lr, momentum=momentum, decay=decay, **kwargs)
        
        # Extract actual parameter counts from optimizer.param_groups (matching ultralytics)
        # param_groups[0] = g[2] (bias), param_groups[1] = g[0] (decay weights), param_groups[2] = g[1] (norm weights)
        g2_count = len(optimizer.param_groups[0]['params'])  # bias
        g0_count = len(optimizer.param_groups[1]['params'])  # decay weights
        g1_count = len(optimizer.param_groups[2]['params'])  # norm weights
        
        LOGGER.info(
            f"optimizer: {type(optimizer).__name__}(lr={lr}, momentum={momentum}) with parameter groups "
            f"{g1_count} weight(decay=0.0), {g0_count} weight(decay={decay}), {g2_count} bias(decay=0.0)"
        )
        return optimizer

    @classmethod
    def list_optimizers(cls) -> list[str]:
        """List all registered optimizer names."""
        return list(cls._optimizers.keys())

# --- Register các optimizer  ---
OptimizerRegistry.register('adam', adam_optimizer)
OptimizerRegistry.register('adamax', adamax_optimizer)
OptimizerRegistry.register('adamw', adamw_optimizer)
OptimizerRegistry.register('nadam', nadam_optimizer)
OptimizerRegistry.register('radam', radam_optimizer)
OptimizerRegistry.register('rmsprop', rmsprop_optimizer)
OptimizerRegistry.register('sgd', sgd_optimizer)