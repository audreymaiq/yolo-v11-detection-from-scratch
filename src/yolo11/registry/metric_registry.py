from typing import Dict, Any

from yolo11.evaluation.metrics import DetMetrics

class MetricRegistry:
    _metrics: Dict[str, Any] = {}
    
    @classmethod
    def register(cls, name: str, metric_builder_func: Any):
        """
        Register a metric factory function.
        
        Args:
            name (str): The name to register the metric under.
            metric_builder_func (function): The function that creates the metric instance.
        """
        if name in cls._metrics:
            raise ValueError(f"Metric '{name}' is already registered.")
        cls._metrics[name] = metric_builder_func
    
    @classmethod
    def get_metric(cls, name: str, **kwargs: Any) -> Any:
        """
        Get a metric instance by its registered name.
        
        Args:
            name (str): The name of the metric (e.g., 'detect').
            **kwargs: Arguments to pass to the metric factory function (e.g., names={}).
            
        Returns:
            An instance of the metric class.
        """
        if name not in cls._metrics:
            available = ", ".join(cls.list_metrics())
            raise ValueError(f"Metric '{name}' not registered. Available metrics are: [{available}]")
        
        metric_builder = cls._metrics[name]
        return metric_builder(**kwargs)
    
    @classmethod
    def list_metrics(cls) -> list:
        """List all registered metric names."""
        return list(cls._metrics.keys())

# --- Register YOLOv11 metric for detection ---
MetricRegistry.register('detect', lambda **kwargs: DetMetrics(**kwargs))