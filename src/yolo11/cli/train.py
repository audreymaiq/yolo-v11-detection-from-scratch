import argparse
import logging

from yolo11.utils.config import load_config
from yolo11.utils.logger import get_logger
from yolo11.utils.seed import set_seed


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", required=True, help="Training config YAML (see configs/training.yaml)")
    parser.add_argument("--data", help="Override `data` (dataset YAML)")
    parser.add_argument("--name", help="Override run name (results go to <save_dir>/<name>)")
    parser.add_argument("--epochs", type=int, help="Override epochs")
    parser.add_argument("--batch", type=int, help="Override batch size")
    parser.add_argument("--device", help="Override device, e.g. cpu, 0, cuda:0")
    parser.add_argument("--resume", help="Resume from a checkpoint (e.g. runs/exp/weights/last.pt)")
    parser.add_argument("--seed", type=int, help="Override random seed")
    parser.add_argument("--debug", action="store_true", help="Verbose logging, experiment trackers disabled")


def run(args: argparse.Namespace) -> None:
    from yolo11.experiments.tracker import ExperimentTracker
    from yolo11.training.trainer import DetectionTrainer

    overrides = {k: getattr(args, k) for k in ("data", "name", "epochs", "batch", "device", "resume", "seed")}
    config = load_config(args.config, overrides)

    logger = get_logger("yolo11", log_file="logs/train.log", log_level="DEBUG" if args.debug else "INFO")
    logger.info(f"Config: {args.config}")
    set_seed(config.get("seed", 42))

    experiments = config.get("experiments", {})
    use_mlflow = experiments.get("use_mlflow", False) and not args.debug
    use_wandb = experiments.get("use_wandb", False) and not args.debug
    tracker = ExperimentTracker(config, use_mlflow=use_mlflow, use_wandb=use_wandb)
    try:
        DetectionTrainer(config, logger=logger, tracker=tracker).train()
        logger.info("Training completed.")
    except Exception:
        logging.getLogger("yolo11").exception("Training failed")
        raise
    finally:
        tracker.finish()
