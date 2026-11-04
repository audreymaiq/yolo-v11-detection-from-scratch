"""Backward compatibility for checkpoints saved before the package was renamed.

Checkpoints store the whole ``nn.Module`` via pickle, so they reference classes by their import path
(``sgod_model_builder.models.conv.Conv``, ...). ``install_legacy_aliases`` registers those old module
names in ``sys.modules`` and points them at their current location, so old ``.pt`` files keep loading.
"""

import importlib
import sys
import types

LEGACY_ROOT = "sgod_model_builder"

# old sub-module -> current module
_RENAMED = {
    "models.activation": "yolo11.models.activation",
    "models.blocks": "yolo11.models.blocks",
    "models.conv": "yolo11.models.conv",
    "models.head": "yolo11.models.head",
    "models.model": "yolo11.models.model",
    "shared.bbox_utils": "yolo11.utils.bbox_utils",
    "shared.torch_utils": "yolo11.utils.torch_utils",
    "shared.logger": "yolo11.utils.logger",
    "shared.seed": "yolo11.utils.seed",
    "shared.visualize_utils": "yolo11.utils.visualize_utils",
    "training.assigners": "yolo11.training.assigners",
    "training.losses": "yolo11.training.losses",
    "training.optimizer": "yolo11.training.optimizer",
    "training.detection_trainer": "yolo11.training.trainer",
    "datasets.yolo_dataset": "yolo11.data.yolo_dataset",
    "datasets.dataloader": "yolo11.data.dataloader",
    "preprocessing.augment": "yolo11.data.augment",
    "preprocessing.detection_preprocessor": "yolo11.data.preprocessor",
    "postprocessing.postprocess": "yolo11.postprocessing.nms",
    "val.detection_validator": "yolo11.evaluation.validator",
    "evaluation.metrics": "yolo11.evaluation.metrics",
    "evaluation.results": "yolo11.evaluation.results",
    "registry.loss_registry": "yolo11.registry.loss_registry",
    "registry.metric_registry": "yolo11.registry.metric_registry",
    "registry.model_registry": "yolo11.registry.model_registry",
    "registry.optimizer_registry": "yolo11.registry.optimizer_registry",
}

_installed = False


def install_legacy_aliases() -> None:
    """Make ``sgod_model_builder.*`` imports resolve to the ``yolo11`` package (idempotent)."""
    global _installed
    if _installed:
        return
    packages = {LEGACY_ROOT}
    for old in _RENAMED:
        parts = old.split(".")
        packages.update(f"{LEGACY_ROOT}.{'.'.join(parts[:i])}" for i in range(1, len(parts)))
    for name in sorted(packages):  # placeholders so ``__import__('a.b.c')`` can resolve its parents
        sys.modules.setdefault(name, types.ModuleType(name))
    for old, new in _RENAMED.items():
        sys.modules[f"{LEGACY_ROOT}.{old}"] = importlib.import_module(new)
    _installed = True
