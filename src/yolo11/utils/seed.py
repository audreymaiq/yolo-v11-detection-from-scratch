from yolo11.utils.torch_utils import init_seeds


def set_seed(seed: int = 42, deterministic: bool = True):
    """Seed Python, NumPy and torch (see ``torch_utils.init_seeds``, which the trainer uses with ``config['seed']``)."""
    init_seeds(seed, deterministic=deterministic)
