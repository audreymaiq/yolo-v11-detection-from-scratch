"""Optional experiment tracking (MLflow / Weights & Biases). Both are lazy imports, so neither is required."""

import logging
import os
from typing import Any

LOGGER = logging.getLogger(__name__)

# Flat training-config keys worth logging as run parameters.
LOGGED_PARAMS = ("model", "data", "epochs", "batch", "imgsz", "optimizer", "lr0", "lrf", "weight_decay", "mosaic", "seed")


class ExperimentTracker:
    def __init__(self, config: dict[str, Any], use_mlflow: bool = False, use_wandb: bool = False):
        self.config = config
        self.use_mlflow = use_mlflow
        self.use_wandb = use_wandb
        self._mlflow = self._wandb = None
        if use_mlflow:
            self._init_mlflow()
        if use_wandb:
            self._init_wandb()

    def _init_mlflow(self) -> None:
        import mlflow

        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
        mlflow.set_experiment(os.getenv("EXPERIMENT_NAME", "yolo11"))
        mlflow.start_run()
        mlflow.log_params({k: self.config[k] for k in LOGGED_PARAMS if self.config.get(k) is not None})
        self._mlflow = mlflow

    def _init_wandb(self) -> None:
        import wandb

        wandb.init(project=os.getenv("PROJECT_NAME", "yolo11"), config=self.config, name=self.config.get("name"))
        self._wandb = wandb

    def log_params(self, params: dict[str, Any]) -> None:
        """Record run-level values such as provenance (MLflow tags, W&B config)."""
        if self._mlflow:
            self._mlflow.set_tags({k: str(v) for k, v in params.items()})
        if self._wandb:
            self._wandb.config.update(params, allow_val_change=True)

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        if self._mlflow:
            for key, value in metrics.items():
                self._mlflow.log_metric(key.replace("(", "_").replace(")", ""), value, step=step)
        if self._wandb:
            self._wandb.log(metrics, step=step)

    def log_artifacts(self, path: str) -> None:
        if self._mlflow:
            self._mlflow.log_artifacts(path)
        if self._wandb:
            self._wandb.save(f"{path}/*")

    def finish(self) -> None:
        if self._mlflow:
            self._mlflow.end_run()
        if self._wandb:
            self._wandb.finish()
