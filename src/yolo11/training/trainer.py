from types import SimpleNamespace
from copy import copy, deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any
import tqdm as tqdm
import numpy as np
import warnings
import logging
import random
import torch
import json
import math
import time
import yaml

from yolo11.data.yolo_dataset import build_yolo_dataset
from yolo11.data.dataloader import build_dataloader
from yolo11.data.config import load_data_config
from yolo11.registry.model_registry import ModelRegistry
from yolo11.registry.optimizer_registry import OptimizerRegistry
from yolo11.evaluation.validator import DetectionValidator
from yolo11.models.model import DetectionModel
from yolo11.utils.visualize_utils import plot_images, plot_labels, plot_results
from yolo11.utils.torch_utils import ModelEMA, EarlyStopping, init_seeds, unwrap_model, autocast, check_amp, check_imgsz, load_checkpoint, attempt_compile, one_cycle, strip_optimizer, torch_distributed_zero_first, print_args, get_cfg, get_save_dir, select_device, get_rng_state, set_rng_state
from yolo11.utils.provenance import collect_provenance, summary
from yolo11.training.losses import v8DetectionLoss
from yolo11 import __version__

LOGGER = logging.getLogger(__name__) 

class DetectionTrainer:

    def __init__(self, cfg, logger=None, tracker=None, overrides=None):
        if not isinstance(cfg, dict):
            raise TypeError("DetectionTrainer cfg must be a dictionary, not a path.")
        config_dict = cfg
        if overrides:
            config_dict = {**config_dict, **overrides}
        self.args = SimpleNamespace(**config_dict)
        self.tracker = tracker  # experiments.tracker.ExperimentTracker or None

        self.check_resume(config_dict)
        
        # Device setup
        self.device = select_device(getattr(self.args, 'device', None))
        self.args.device = str(self.device)
        
        # Initialize attributes
        self.validator = None
        self.metrics = None
        self.plots = {}
        init_seeds(getattr(self.args, 'seed', 42) + 1, deterministic=getattr(self.args, 'deterministic', False))
        
        # Dirs
        if self.resume:
            # <run>/weights/last.pt -> <run>; derived from the checkpoint path so moved runs keep working
            self.save_dir = Path(self.args.resume).resolve().parent.parent
        else:
            self.save_dir = get_save_dir(self.args)
        self.args.name = self.save_dir.name
        self.wdir = self.save_dir / "weights"
        self.wdir.mkdir(parents=True, exist_ok=True)
        self.args.save_dir = str(self.save_dir)
        
        # Save args
        with open(str(self.save_dir / "args.yaml"), "w", encoding="utf-8") as f:
            yaml.safe_dump(vars(self.args), f, sort_keys=False)
        
        self.last, self.best = self.wdir / "last.pt", self.wdir / "best.pt"
        self.save_period = getattr(self.args, 'save_period', -1)  # -1 = save only last/best
        self.batch_size = getattr(self.args, 'batch', getattr(self.args, 'batch_size', 16))
        self.epochs = getattr(self.args, 'epochs', 100)
        self.start_epoch = 0
        
        if self.device.type in {"cpu", "mps"}:
            self.args.workers = 0
        elif not hasattr(self.args, 'workers'):
            self.args.workers = 8  # Default workers for CUDA
        
        # Dataset first (model depends on data attributes)
        self.data = self.get_dataset()
        
        # Model - store path/name only, will be loaded in _setup_train()
        self.model = self.args.model
        
        # Training attributes
        self.ema = None
        self.lf = None
        self.scheduler = None
        self.best_fitness = None
        self.fitness = None
        self.loss = None
        self.tloss = None
        self.csv = self.save_dir / "results.csv"
        if self.csv.exists() and not self.resume:
            self.csv.unlink()
        self.plot_idx = [0, 1, 2]
        self.nan_recovery_attempts = 0
        self.args.task = "detect"
        
        # Print training info
        LOGGER.info(f"Training run: {self.args.name}")
        LOGGER.info(f"Save directory: {self.save_dir}")
        LOGGER.info(f"Device: {self.device}")
        
        # Print args for debugging
        print_args(vars(self.args))

    def train(self):
        self._do_train()
    
    def check_resume(self, overrides):
        """Check if resume checkpoint exists and update arguments accordingly."""
        resume = getattr(self.args, 'resume', False)  # Default to False if not set
        if resume:
            try:
                checkpoint_path = Path(resume)
                
                if not checkpoint_path.exists():
                    raise FileNotFoundError(
                        f"Resume checkpoint not found: {checkpoint_path}\n"
                        "Please provide a valid checkpoint path, e.g., 'runs/detect/train/weights/last.pt'"
                    )
                
                # Load checkpoint args
                ckpt_args = load_checkpoint(checkpoint_path)[0].args
                
                # Check that resume data YAML exists, otherwise use current data
                if not isinstance(ckpt_args["data"], dict) and not Path(ckpt_args["data"]).exists():
                    ckpt_args["data"] = self.args.data

                resume = True
                self.args = get_cfg(ckpt_args)
                self.args.model = self.args.resume = str(checkpoint_path.resolve())  # reinstate model
                for k in (
                    "imgsz",
                    "batch",
                    "device",
                    "close_mosaic",
                    "augmentations",
                    "save_period",
                    "workers",
                    "cache",
                    "patience",
                    "time",
                    "freeze",
                    "val",
                    "plots",
                ):  # allow arg updates to reduce memory or update device on resume
                    if k in overrides:
                        setattr(self.args, k, overrides[k])

                # Handle augmentations parameter for resume: check if user provided custom augmentations
                if ckpt_args.get("augmentations") is not None:
                    # Augmentations were saved in checkpoint as reprs but can't be restored automatically
                    LOGGER.warning(
                        "Custom Albumentations transforms were used in the original training run but are not "
                        "being restored. To preserve custom augmentations when resuming, you need to pass the "
                        "'augmentations' parameter again to get expected results. Example: \n"
                        f"model.train(resume=True, augmentations={ckpt_args['augmentations']})"
                    )

            except FileNotFoundError:
                raise
            except Exception as e:
                raise RuntimeError(f"Failed to resume from '{resume}': {e}") from e
        self.resume = resume

    def _do_train(self):
        self._setup_train()

        nb = len(self.train_loader)
        nw = max(round(self.args.warmup_epochs * nb), 100) if self.args.warmup_epochs > 0 else -1
        self.epoch_time = None
        self.epoch_time_start = time.time()
        self.train_time_start = time.time()
        
        LOGGER.info(
            f"Image sizes {self.args.imgsz} train, {self.args.imgsz} val\n"
            f"Using {self.train_loader.num_workers} dataloader workers\n"
            f"Logging results to {self.save_dir}\n"
            f"Starting training for {self.epochs} epochs..."
        )
        
        if getattr(self.args, "close_mosaic", 10) != 0:
            base_idx = (self.epochs - getattr(self.args, "close_mosaic", 10)) * nb
            self.plot_idx.extend([base_idx, base_idx + 1, base_idx + 2])
        
        epoch = self.start_epoch
        self.optimizer.zero_grad()
        if self.resume_grads:  # gradients accumulated since the last optimizer step of the resumed run
            for name, p in unwrap_model(self.model).named_parameters():
                if name in self.resume_grads:
                    p.grad = self.resume_grads[name].to(p.device)
            self.resume_grads = None
        
        while True:
            self.epoch = epoch
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # suppress lr_scheduler warning
                self.scheduler.step()
            
            self._model_train()
            
            # Update dataloader mosaic
            if epoch == (self.epochs - getattr(self.args, "close_mosaic", 10)):
                self._close_dataloader_mosaic()
                if hasattr(self.train_loader, "reset"):
                    self.train_loader.reset(epoch)  # same order and worker seeds as a run resumed here
            
            LOGGER.info(self.progress_string())
            pbar = tqdm.tqdm(enumerate(self.train_loader), total=nb, desc=f"Epoch {epoch + 1}/{self.epochs}")
            self.tloss = None
            
            for i, batch in pbar:
                ni = i + nb * epoch
                
                # Warmup
                if ni <= nw:
                    xi = [0, nw]
                    self.accumulate = max(1, int(np.interp(ni, xi, [1, getattr(self.args, "nbs", 64) / self.batch_size]).round()))
                    for j, x in enumerate(self.optimizer.param_groups):
                        x["lr"] = np.interp(
                            ni, xi, [getattr(self.args, "warmup_bias_lr", 0.0) if j == 0 else 0.0, x["initial_lr"] * self.lf(epoch)]
                        )
                        if "momentum" in x:
                            x["momentum"] = np.interp(ni, xi, [getattr(self.args, "warmup_momentum", 0.8), getattr(self.args, "momentum", 0.937)])
                
                # Forward
                with autocast(self.amp):
                    batch = self.preprocess_batch(batch)
                    if getattr(self.args, "compile", False):
                        # Decouple inference and loss calculations for improved compile performance
                        preds = self.model(batch["img"])
                        loss, self.loss_items = unwrap_model(self.model).loss(batch, preds)
                    else:
                        loss, self.loss_items = self.model(batch)
                    self.loss = loss.sum() if isinstance(loss, torch.Tensor) and loss.ndim > 0 else loss
                    self.tloss = (
                        self.loss_items if self.tloss is None else (self.tloss * i + self.loss_items) / (i + 1)
                    )
                
                # Backward
                self.scaler.scale(self.loss).backward()
                
                # Optimize
                if ni - self.last_opt_step >= self.accumulate:
                    self.optimizer_step()
                    self.last_opt_step = ni
                
                # Log
                loss_len = self.tloss.shape[0] if len(self.tloss.shape) else 1
                mem = self._get_memory()
                pbar.set_description(
                    ("%11s" * 2 + "%11.4g" * (2 + loss_len))
                    % (
                        f"{epoch + 1}/{self.epochs}",
                        f"{mem:.3g}G",
                        *(self.tloss if loss_len > 1 else torch.unsqueeze(self.tloss, 0)),
                        batch["cls"].shape[0],
                        batch["img"].shape[-1],
                    )
                )
                
                if self.args.plots and ni in self.plot_idx:
                    self.plot_training_samples(batch, ni)
            
            self.lr = {f"lr/pg{ir}": x["lr"] for ir, x in enumerate(self.optimizer.param_groups)}
            
            # EMA attribute update
            if self.ema:
                self.ema.update_attr(self.model, include=["yaml", "nc", "args", "names", "stride", "class_weights"])
            
            # Validation
            final_epoch = epoch + 1 >= self.epochs
            if self.args.val or final_epoch or self.stopper.possible_stop or self.stop:
                self._clear_memory(threshold=0.5)
                self.metrics, self.fitness = self.validate()
            
            if self._handle_nan_recovery(epoch):
                continue
            
            self.nan_recovery_attempts = 0  # Reset counter after successful epoch
            
            # Save metrics
            epoch_metrics = {**self.label_loss_items(self.tloss), **self.metrics, **self.lr}
            self.save_metrics(metrics=epoch_metrics)
            if self.tracker:
                self.tracker.log_metrics({k: float(v) for k, v in epoch_metrics.items()}, step=epoch + 1)
            self.stop |= self.stopper(epoch + 1, self.fitness) or final_epoch
            
            # Save model
            if self.args.save or final_epoch:
                self.save_model()
            
            # Epoch time tracking
            t = time.time()
            self.epoch_time = t - self.epoch_time_start
            self.epoch_time_start = t
            
            self._clear_memory(0.5)
            
            if self.stop:
                break
            epoch += 1
        
        seconds = time.time() - self.train_time_start
        LOGGER.info(f"\n{epoch - self.start_epoch + 1} epochs completed in {seconds / 3600:.3f} hours.")
        self.final_eval()
        if self.args.plots:
            self.plot_metrics()
        if self.tracker:
            if self.metrics:
                self.tracker.log_metrics({f"final/{k}": float(v) for k, v in self.metrics.items()}, step=epoch + 1)
            self.tracker.log_artifacts(str(self.save_dir))
        self._clear_memory()

    def _setup_train(self):
        ckpt = self.setup_model()
        self.model = self.model.to(self.device)
        self.set_model_attributes()
        self.model = attempt_compile(self.model, device=self.device, mode=getattr(self.args, 'compile', False))

        freeze_list = getattr(self.args, 'freeze', None) or []
        if isinstance(freeze_list, int):
            freeze_list = range(freeze_list)
        freeze_list = list(freeze_list) if not isinstance(freeze_list, list) else freeze_list
        always_freeze_names = [".dfl"]
        freeze_layer_names = [f"model.{x}." for x in freeze_list] + always_freeze_names
        self.freeze_layer_names = freeze_layer_names  # Store for _model_train()
        for k, v in self.model.named_parameters():
            if any(x in k for x in freeze_layer_names):
                LOGGER.info(f"Freezing layer '{k}'")
                v.requires_grad = False
            elif not v.requires_grad and v.dtype.is_floating_point:  # only floating point Tensor can require gradients
                LOGGER.warning(
                    f"setting 'requires_grad=True' for frozen layer '{k}'. "
                    "See ultralytics.engine.trainer for customization of frozen layers."
                )
                v.requires_grad = True

        # Check AMP (Automatic Mixed Precision)
        self.amp = torch.tensor(getattr(self.args, 'amp', self.device.type != 'cpu')).to(self.device)  # Auto-enable on CUDA
        if self.amp:
            # Verify hardware support (check_amp validates GPU compatibility)
            self.amp = torch.tensor(check_amp(self.model), device=self.device)
        self.amp = bool(self.amp)  # convert to boolean

        # Create GradScaler with PyTorch version compatibility
        try:
            # PyTorch 2.4+ API
            self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp)
        except TypeError:
            # PyTorch < 2.4 fallback
            self.scaler = torch.cuda.amp.GradScaler(enabled=self.amp)

        gs = max(int(self.model.stride.max() if hasattr(self.model, "stride") else 32), 32)
        self.args.imgsz = check_imgsz(getattr(self.args, 'imgsz', 640), stride=gs, floor=gs, max_dim=1)
        self.stride = gs

        batch_size = self.batch_size
        self.train_loader = self.get_dataloader(self.data["train"], batch_size=batch_size, mode="train")
        self.test_loader = self.get_dataloader(self.data.get("val") or self.data.get("test"), batch_size * 2, mode="val")
        self.validator = self.get_validator()
        self.ema = ModelEMA(self.model)
        self.last_opt_step = -1  # batch index (ni) of the last optimizer step
        self.resume_grads = None

        if "loss" in self.model.__dict__:
            LOGGER.info("Fix: Removed invalid 'loss' attribute from model.")
            del self.model.__dict__["loss"]
        loss_object = v8DetectionLoss(self.model)
        self.model.criterion = loss_object
        self.criterion = loss_object

        if self.args.plots:
            self.plot_training_labels()
        metric_keys = self.validator.metrics.keys + self.label_loss_items(prefix="val")
        self.metrics = dict(zip(metric_keys, [0] * len(metric_keys)))

        self.accumulate = max(round(getattr(self.args, "nbs", 64) / self.batch_size), 1)
        
        weight_decay = getattr(self.args, "weight_decay", 0.0005) * self.batch_size * self.accumulate / getattr(self.args, "nbs", 64)
        iterations = math.ceil(len(self.train_loader.dataset) / max(self.batch_size, getattr(self.args, "nbs", 64))) * self.epochs
        self.optimizer = self.build_optimizer(model=self.model, name=getattr(self.args, "optimizer", "auto"), lr=getattr(self.args, "lr0", 0.01), momentum=getattr(self.args, "momentum", 0.937), decay=weight_decay, iterations=iterations)
        self._setup_scheduler()
        self.stopper, self.stop = EarlyStopping(patience=getattr(self.args, "patience", 100)), False
        self.setup_provenance(ckpt)
        self.resume_training(ckpt)
        self.scheduler.last_epoch = self.start_epoch - 1

    def _setup_scheduler(self):
        """Initialize training learning rate scheduler."""  
        if getattr(self.args, "cos_lr", False):
            self.lf = one_cycle(1, getattr(self.args, "lrf", 0.01), self.epochs)  # cosine 1->hyp['lrf']
        else:
            self.lf = lambda x: max(1 - x / self.epochs, 0) * (1.0 - getattr(self.args, "lrf", 0.01)) + getattr(self.args, "lrf", 0.01)  # linear
        
        # Use torch.optim.lr_scheduler.LambdaLR instead of registry
        from torch import optim
        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=self.lf)

    def setup_model(self):
        """Load, create, or download model for any task."""
        if isinstance(self.model, torch.nn.Module):  # if model is loaded beforehand. No setup needed
            return
        
        cfg, weights = self.model, None
        ckpt = None
        if str(self.model).endswith(".pt"):
            weights, ckpt = load_checkpoint(self.model)  # Load to CPU by default
            cfg = weights.yaml if hasattr(weights, 'yaml') else None
        elif isinstance(getattr(self.args, 'pretrained', None), (str, Path)):
            weights, _ = load_checkpoint(self.args.pretrained)  # Load to CPU by default
        
        self.model = self.get_model(cfg=cfg, weights=weights, verbose=True)
        return ckpt

    def set_model_attributes(self):
        self.model.nc = self.data["nc"]
        self.model.names = self.data["names"]
        self.model.args = self.args
        self.loss_names = getattr(self.model, 'loss_names', ["box_loss", "cls_loss", "dfl_loss"])

    def get_dataset(self):
        try:
            channels = getattr(self.args, "ch", None) or getattr(self.args, "channels", None)
            return load_data_config(self.args.data, channels=channels, single_cls=getattr(self.args, "single_cls", False))
        except Exception as e:
            raise RuntimeError(f"Dataset '{self.args.data}' error: {e}") from e

    def build_dataset(self, img_path: str, mode: str = "train", batch: int | None = None):
        gs = max(int(unwrap_model(self.model).stride.max() if self.model else 0), 32)
        return build_yolo_dataset(self.args, img_path, batch, self.data, mode=mode, rect=mode == "val", stride=gs)

    def get_dataloader(self, dataset_path: str, batch_size: int = 16, mode: str = "train"):
        assert mode in {"train", "val"}
        with torch_distributed_zero_first(0):  # init dataset *.cache only once (safe for single-GPU)
            dataset = self.build_dataset(dataset_path, mode, batch_size)
        shuffle = mode == "train"
        if getattr(dataset, "rect", False) and shuffle:
            LOGGER.warning("'rect=True' is incompatible with DataLoader shuffle, setting shuffle=False")
            shuffle = False
        return build_dataloader(
            dataset,
            batch=batch_size,
            workers=self.args.workers if mode == "train" else self.args.workers * 2,
            shuffle=shuffle,
            drop_last=getattr(self.args, "compile", False) and mode == "train",
            pin_memory=True,
            seed=getattr(self.args, "seed", 0),
        )

    def preprocess_batch(self, batch: dict) -> dict:
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(self.device, non_blocking=self.device.type == "cuda")
        batch["img"] = batch["img"].float() / 255

        if getattr(self.args, "multi_scale", False):
            imgs = batch["img"]
            sz = (
                random.randrange(int(self.args.imgsz * 0.5), int(self.args.imgsz * 1.5 + self.stride))
                // self.stride
                * self.stride
            )
            sf = sz / max(imgs.shape[2:])
            if sf != 1:
                ns = [math.ceil(x * sf / self.stride) * self.stride for x in imgs.shape[2:]]
                imgs = torch.nn.functional.interpolate(imgs, size=ns, mode="bilinear", align_corners=False)
            batch["img"] = imgs
        
        return batch

    def _close_dataloader_mosaic(self):
        if hasattr(self.train_loader.dataset, "mosaic"):
            self.train_loader.dataset.mosaic = False
        if hasattr(self.train_loader.dataset, "close_mosaic"):
            LOGGER.info("Closing dataloader mosaic")
            self.train_loader.dataset.close_mosaic(hyp=copy(self.args))
    
    def _model_train(self):
        """Set model in training mode and freeze BN stats for frozen layers."""
        self.model.train()
        # Freeze BN stats for frozen layers
        freeze_layer_names = getattr(self, 'freeze_layer_names', [])
        for n, m in self.model.named_modules():
            if any(f in n for f in freeze_layer_names) and isinstance(m, torch.nn.BatchNorm2d):
                m.eval()
    
    def _get_memory(self, fraction=False):
        """Get GPU memory utilization in GB or as a fraction of total memory."""
        memory, total = 0, 0
        if self.device.type == "mps":
            memory = torch.mps.driver_allocated_memory()
            if fraction:
                return __import__("psutil").virtual_memory().percent / 100
        elif self.device.type != "cpu":
            memory = torch.cuda.memory_reserved()
            if fraction:
                total = torch.cuda.get_device_properties(self.device).total_memory
        return ((memory / total) if total > 0 else 0) if fraction else (memory / 2**30)
    
    def _clear_memory(self, threshold: float | None = None):
        """Clear GPU memory by calling garbage collector and emptying cache."""
        if threshold:
            assert 0 <= threshold <= 1, "Threshold must be between 0 and 1."
            if self._get_memory(fraction=True) <= threshold:
                return
        import gc
        gc.collect()
        if self.device.type == "mps":
            torch.mps.empty_cache()
        elif self.device.type == "cpu":
            return
        else:
            torch.cuda.empty_cache()
    
    def _load_checkpoint_state(self, ckpt):
        """Load model, optimizer, scaler, EMA, best_fitness and early-stopping state from checkpoint."""
        state = ckpt.get("train_state") or {}
        if state.get("model") is not None:  # the trained weights; checkpoints without them resume from EMA
            unwrap_model(self.model).load_state_dict(state["model"])
        if ckpt.get("optimizer") is not None:
            self.optimizer.load_state_dict(ckpt["optimizer"])
        if ckpt.get("scaler") is not None:
            self.scaler.load_state_dict(ckpt["scaler"])
        if self.ema and ckpt.get("ema"):
            self.ema = ModelEMA(self.model)  # validation with EMA creates inference tensors that can't be updated
            self.ema.ema.load_state_dict(ckpt["ema"].float().state_dict())
            self.ema.updates = ckpt["updates"]
        self.best_fitness = ckpt.get("best_fitness", 0.0)
        if state.get("stopper"):
            self.stopper.best_fitness = state["stopper"]["best_fitness"]
            self.stopper.best_epoch = state["stopper"]["best_epoch"]
        self.last_opt_step = state.get("last_opt_step", -1)
        self.resume_grads = state.get("grads")

    def _handle_nan_recovery(self, epoch):
        """Detect and recover from NaN/Inf loss and fitness collapse by loading last checkpoint."""
        loss_nan = self.loss is not None and not self.loss.isfinite()
        fitness_nan = self.fitness is not None and not np.isfinite(self.fitness)
        fitness_collapse = self.best_fitness and self.best_fitness > 0 and self.fitness == 0
        corrupted = loss_nan and (fitness_nan or fitness_collapse)
        reason = "Loss NaN/Inf" if loss_nan else "Fitness NaN/Inf" if fitness_nan else "Fitness collapse"
        if not corrupted:
            return False
        if epoch == self.start_epoch or not self.last.exists():
            LOGGER.warning(f"{reason} detected but can not recover from last.pt...")
            return False  # Cannot recover on first epoch, let training continue
        self.nan_recovery_attempts += 1
        if self.nan_recovery_attempts > 3:
            raise RuntimeError(f"Training failed: NaN persisted for {self.nan_recovery_attempts} epochs")
        LOGGER.warning(f"{reason} detected (attempt {self.nan_recovery_attempts}/3), recovering from last.pt...")
        self._model_train()  # set model to train mode before loading checkpoint to avoid inference tensor errors
        _, ckpt = load_checkpoint(self.last, device=self.device)
        ema_state = ckpt["ema"].float().state_dict()
        if not all(torch.isfinite(v).all() for v in ema_state.values() if isinstance(v, torch.Tensor)):
            raise RuntimeError(f"Checkpoint {self.last} is corrupted with NaN/Inf weights")
        self._load_checkpoint_state(ckpt)  # Load optimizer/scaler/EMA/best_fitness
        unwrap_model(self.model).load_state_dict(ema_state)  # restart from the EMA weights, not the diverged ones
        self.optimizer.zero_grad()  # drop the non-finite gradients of the failed epoch
        self.resume_grads = None
        del ckpt, ema_state
        self.scheduler.last_epoch = epoch - 1
        return True

    def get_model(self, cfg: str | dict | None = None, weights: str | None = None, verbose: bool = True):
        # Determine channels from data
        ch = self.data.get("channels", 3)
        nc = self.data["nc"]
        
        # Handle different cfg types
        if isinstance(cfg, dict):
            # From checkpoint resume - cfg is weights.yaml dict
            model = DetectionModel(cfg=cfg, ch=ch, nc=nc, verbose=verbose)
        elif cfg and isinstance(cfg, str) and Path(cfg).suffix in {'.yaml', '.yml'}:
            # cfg is YAML file path
            model = DetectionModel(cfg=cfg, ch=ch, nc=nc, verbose=verbose)
        else:
            # Load from registry or default
            model_name = cfg or 'yolov11n'
            # Normalize: yolo11n → yolov11n for registry
            if model_name.startswith('yolo') and not model_name.startswith('yolov'):
                model_name = model_name.replace('yolo', 'yolov', 1)
            model = ModelRegistry.get_model(model_name, ch=ch, nc=nc, verbose=verbose)
        
        # Load weights if provided (weights is the actual model, not path)
        if weights:
            model.load(weights)

        return model

    def build_optimizer(self, model, name="auto", lr=0.001, momentum=0.9, decay=1e-5, iterations=1e5):
        """Build optimizer matching ultralytics logic exactly.
        
        When name='auto', selects optimizer based on iterations and sets warmup_bias_lr for Adam-based optimizers.
        """
        # Handle auto mode: set warmup_bias_lr
        if name.lower() == "auto":
            # Set unconditionally for auto mode (both SGD and AdamW cases)
            self.args.warmup_bias_lr = 0.0  # no higher than 0.01 for Adam
        
        return OptimizerRegistry.get_optimizer(
            name=name,
            model=model,
            lr=lr,
            momentum=momentum,
            decay=decay,
            iterations=iterations,
            nc=self.data.get("nc", 10)  # Pass nc for auto mode
        )

    def optimizer_step(self):
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()
        if self.ema:
            self.ema.update(self.model)

    def get_validator(self):
        return DetectionValidator(
            self.test_loader,
            save_dir=self.save_dir,
            args=vars(self.args).copy()
        )

    def validate(self):
        metrics = self.validator(self)
        if metrics is None:
            return None, None
        fitness = metrics.pop("fitness", -self.loss.detach().cpu().numpy())  # use loss as fitness measure if not found
        if not self.best_fitness or self.best_fitness < fitness:
            self.best_fitness = fitness
        return metrics, fitness

    def label_loss_items(self, loss_items: list[float] | None = None, prefix: str = "train"):
        keys = [f"{prefix}/{x}" for x in self.loss_names]
        if loss_items is not None:
            loss_items = [round(float(x), 5) for x in loss_items]
            return dict(zip(keys, loss_items))
        else:
            return keys

    def save_metrics(self, metrics):
        keys, vals = list(metrics.keys()), list(metrics.values())
        n = len(metrics) + 2
        t = time.time() - self.train_time_start
        self.csv.parent.mkdir(parents=True, exist_ok=True)
        s = "" if self.csv.exists() else (("%s," * n % tuple(["epoch", "time", *keys])).rstrip(",") + "\n")
        with open(self.csv, "a", encoding="utf-8") as f:
            f.write(s + ("%.6g," * n % tuple([self.epoch + 1, t, *vals])).rstrip(",") + "\n")
    
    def read_results_csv(self):
        try:
            import polars as pl
            return pl.read_csv(self.csv, infer_schema_length=None).to_dict(as_series=False)
        except Exception:
            return {}

    def final_eval(self):
        model = self.best if self.best.exists() else None
        with torch_distributed_zero_first(0):  # strip only on primary process
            ckpt = strip_optimizer(self.last) if self.last.exists() else {}
            if model:
                # update best.pt train_metrics from last.pt
                strip_optimizer(self.best, updates={"train_results": ckpt.get("train_results")})
        if model:
            LOGGER.info(f"\nValidating {model}...")
            self.validator.args.plots = self.args.plots
            self.validator.args.compile = False  # disable final val compile as too slow
            self.metrics = self.validator(model=model)
            self.metrics.pop("fitness", None)

    def progress_string(self):
        return ("\n" + "%11s" * (4 + len(self.loss_names))) % (
            "Epoch",
            "GPU_mem",
            *self.loss_names,
            "Instances",
            "Size",
        )

    def plot_training_samples(self, batch: dict[str, Any], ni: int) -> None:
        plot_images(
            labels=batch,
            paths=batch["im_file"],
            fname=self.save_dir / f"train_batch{ni}.jpg",
            names=self.data["names"],
        )

    def plot_training_labels(self):
        boxes = np.concatenate([lb["bboxes"] for lb in self.train_loader.dataset.labels], 0)
        cls = np.concatenate([lb["cls"] for lb in self.train_loader.dataset.labels], 0)
        plot_labels(boxes, cls.squeeze(), names=self.data["names"], save_dir=self.save_dir)

    def plot_metrics(self):
        plot_results(file=self.csv)  # save results.png

    def setup_provenance(self, ckpt):
        """Record the code, environment and data of this run in provenance.json, checkpoints and the tracker.

        A resumed run keeps the provenance of the run it continues and appends this session under ``resumes``,
        warning when the data differs from what the run started on.
        """
        current = collect_provenance(
            self.device,
            {"train": self.train_loader.dataset, "val": self.test_loader.dataset},
            data_yaml=str(self.args.data),
        )
        if ckpt is not None and not self.resume:  # fine-tuning: record which checkpoint (and run) we started from
            current["init_weights"] = {"path": str(self.args.model), **summary(ckpt.get("provenance"))}
        original = (ckpt or {}).get("provenance") if self.resume else None
        if original:
            for split in ("train", "val"):
                before = (original.get("data") or {}).get(split) or {}
                now = current["data"][split]
                if before.get("sha256") and before["sha256"] != now["sha256"]:
                    LOGGER.warning(
                        f"The {split} data differs from the data this run started on ({now['images']} images now, "
                        f"{before['images']} before). Resuming on it anyway."
                    )
            self.provenance = {**original, "resumes": [*original.get("resumes", []), current]}
        else:
            self.provenance = current
        (self.save_dir / "provenance.json").write_text(json.dumps(self.provenance, indent=2), encoding="utf-8")
        if (current["git"] or {}).get("dirty"):
            LOGGER.warning("Training from a checkout with uncommitted changes: its commit alone does not reproduce it.")
        LOGGER.info(f"Provenance: {summary(current)}")
        if self.tracker:
            self.tracker.log_params(summary(current))

    def resume_training(self, ckpt):
        """Resume YOLO training from given epoch and best fitness."""
        if ckpt is None or not self.resume:
            return
        start_epoch = ckpt.get("epoch", -1) + 1
        assert start_epoch > 0, (
            f"{self.args.model} training to {self.epochs} epochs is finished, nothing to resume.\n"
            f"To train further, start a new run from it: set `model: {self.args.model}` in the config "
            f"and run 'yolo11 train --config <config>' without --resume."
        )
        LOGGER.info(f"Resuming training {self.args.model} from epoch {start_epoch + 1} to {self.epochs} total epochs")
        if self.epochs < start_epoch:
            LOGGER.info(
                f"{self.model} has been trained for {ckpt['epoch']} epochs. Fine-tuning for {self.epochs} more epochs."
            )
            self.epochs += ckpt["epoch"]  # finetune additional epochs
        self._load_checkpoint_state(ckpt)
        self.start_epoch = start_epoch
        close_mosaic = getattr(self.args, 'close_mosaic', 10)  # Safe access with default
        if close_mosaic and start_epoch > (self.epochs - close_mosaic):  # at == the training loop closes it
            self._close_dataloader_mosaic()
        self.train_loader.reset(start_epoch)  # continue the shuffle order (and worker seeds) at this epoch
        state = ckpt.get("train_state") or {}
        if state.get("rng"):
            set_rng_state(state["rng"])  # last: nothing after this may draw random numbers before the first batch

    def save_model(self):
        """Save model training checkpoints with additional metadata."""
        import io
        
        # Serialize ckpt to a byte buffer once (more efficient). Training state stays in FP32 so that resuming
        # continues exactly where this epoch ended; final_eval strips it and halves the weights.
        buffer = io.BytesIO()
        torch.save(
            {
                "epoch": self.epoch,
                "best_fitness": self.best_fitness,
                "model": None,  # final checkpoints and inference use EMA
                "ema": deepcopy(unwrap_model(self.ema.ema)),
                "updates": self.ema.updates,
                "optimizer": deepcopy(self.optimizer.state_dict()),
                "scaler": self.scaler.state_dict(),
                "train_state": {
                    "model": deepcopy(unwrap_model(self.model).state_dict()),
                    "rng": get_rng_state(),
                    "last_opt_step": self.last_opt_step,
                    "grads": {  # pending accumulation when the epoch did not end on an optimizer step
                        n: p.grad.detach().clone()
                        for n, p in unwrap_model(self.model).named_parameters()
                        if p.grad is not None
                    },
                    "stopper": {"best_fitness": self.stopper.best_fitness, "best_epoch": self.stopper.best_epoch},
                },
                "train_args": vars(self.args),
                "train_metrics": {**self.metrics, **{"fitness": self.fitness}},
                "train_results": self.read_results_csv(),
                "provenance": self.provenance,
                "date": datetime.now().isoformat(),
                "version": __version__,
            },
            buffer,
        )
        serialized_ckpt = buffer.getvalue()
        
        # Save checkpoints
        self.wdir.mkdir(parents=True, exist_ok=True)
        self.last.write_bytes(serialized_ckpt)
        if self.best_fitness == self.fitness:
            self.best.write_bytes(serialized_ckpt)
        if (self.save_period > 0) and (self.epoch % self.save_period == 0):
            (self.wdir / f"epoch{self.epoch}.pt").write_bytes(serialized_ckpt)