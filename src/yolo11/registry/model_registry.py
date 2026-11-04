from pathlib import Path
from typing import Dict, Any
import torch.nn as nn
from yolo11.models.model import DetectionModel

YAML_PATH = Path(__file__).resolve().parents[1] / "models" / "yolo11.yaml"

class ModelRegistry:
    _models: Dict[str, Any] = {}
    
    @classmethod
    def register(cls, name: str, model_builder_func: Any):
        """
        Register a model factory function.
        
        Args:
            name (str): The name to register the model under.
            model_builder_func (function): The function that creates the model instance.
        """
        if name in cls._models:
            raise ValueError(f"Model '{name}' is already registered.")
        cls._models[name] = model_builder_func
    
    @classmethod
    def get_model(cls, name: str, **kwargs: Any) -> nn.Module:
        """
        Get a model instance by its registered name.
        
        Args:
            name (str): The name of the model (e.g., 'yolov11n').
            **kwargs: Arguments to pass to the model factory function (e.g., nc=80, ch=3, verbose=True).
            
        Returns:
            An instance of the nn.Module.
        """
        if name not in cls._models:
            available = ", ".join(cls.list_models())
            raise ValueError(f"Model '{name}' not registered. Available models are: [{available}]")
        
        model_builder = cls._models[name]
        return model_builder(**kwargs)
    
    @classmethod
    def list_models(cls) -> list:
        """List all registered model names."""
        return list(cls._models.keys())

# --- Register YOLOv11 model variants ---
ModelRegistry.register('yolov11n', lambda **kwargs: DetectionModel(cfg=str(YAML_PATH), scale='n', **kwargs))
ModelRegistry.register('yolov11s', lambda **kwargs: DetectionModel(cfg=str(YAML_PATH), scale='s', **kwargs))
ModelRegistry.register('yolov11m', lambda **kwargs: DetectionModel(cfg=str(YAML_PATH), scale='m', **kwargs))
ModelRegistry.register('yolov11l', lambda **kwargs: DetectionModel(cfg=str(YAML_PATH), scale='l', **kwargs))
ModelRegistry.register('yolov11x', lambda **kwargs: DetectionModel(cfg=str(YAML_PATH), scale='x', **kwargs))