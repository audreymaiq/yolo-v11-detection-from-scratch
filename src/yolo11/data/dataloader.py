import os
import torch
import random
import numpy as np
from typing import Any
from collections.abc import Iterator
from torch.utils.data import dataloader

class _RepeatSampler:
    """Sampler that repeats forever for infinite iteration.
    
    This sampler wraps another sampler and yields its contents indefinitely,
    allowing for infinite iteration over a dataset without recreating the sampler.
    
    Mathematical property: For sampler S with n elements, _RepeatSampler(S) 
    yields S[0], S[1], ..., S[n-1], S[0], S[1], ... infinitely.
    """

    def __init__(self, sampler: Any):
        """Initialize with a sampler to repeat indefinitely."""
        self.sampler = sampler

    def __iter__(self) -> Iterator:
        """Iterate over the sampler indefinitely."""
        while True:
            yield from iter(self.sampler)


class InfiniteDataLoader(dataloader.DataLoader):
    """DataLoader that reuses workers for infinite iteration"""

    def __init__(self, *args: Any, **kwargs: Any):
        """Initialize InfiniteDataLoader with same args as DataLoader."""
        # PyTorch < 2.0 doesn't support prefetch_factor
        TORCH_2_0 = int(torch.__version__.split(".")[0]) >= 2
        if not TORCH_2_0:
            kwargs.pop("prefetch_factor", None)
        super().__init__(*args, **kwargs)
        # Wrap batch_sampler to repeat indefinitely
        # object.__setattr__ bypasses potential property restrictions
        object.__setattr__(self, "batch_sampler", _RepeatSampler(self.batch_sampler))
        self.iterator = super().__iter__()

    def __len__(self) -> int:
        """Return length of the underlying sampler (one epoch's worth)."""
        return len(self.batch_sampler.sampler)

    def __iter__(self) -> Iterator:
        """Yield batches for one epoch from the infinite iterator."""
        for _ in range(len(self)):
            yield next(self.iterator)

    def __del__(self):
        """Ensure workers are properly terminated when DataLoader is deleted."""
        try:
            if not hasattr(self, 'iterator') or not hasattr(self.iterator, "_workers"):
                return
            for w in self.iterator._workers:
                if w.is_alive():
                    w.terminate()
            self.iterator._shutdown_workers()
        except Exception:
            pass

    def reset(self):
        """Reset iterator for dataset modifications during training."""
        self.iterator = self._get_iterator()


def seed_worker(worker_id: int) -> None:
    """Set dataloader worker seed for reproducibility"""
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_dataloader(
    dataset,
    batch: int,
    workers: int,
    shuffle: bool = True,
    drop_last: bool = False,
    pin_memory: bool = True,
) -> InfiniteDataLoader:
    """Build InfiniteDataLoader for training or validation.
    
    Args:
        dataset: Dataset to load from.
        batch: Batch size.
        workers: Maximum number of worker processes.
        shuffle: Whether to shuffle data.
        drop_last: Whether to drop incomplete final batch.
        pin_memory: Whether to use pinned (page-locked) memory.
    
    Returns:
        InfiniteDataLoader configured for efficient training.
    """
    # Ensure batch doesn't exceed dataset size
    batch = min(batch, len(dataset))
    
    # Worker calculation: CPU cores per GPU, capped by user request
    nd = torch.cuda.device_count()  # Number of CUDA devices
    nw = min(os.cpu_count() // max(nd, 1), workers)
    
    sampler = None
    
    # Generator for reproducible shuffling
    generator = torch.Generator()
    generator.manual_seed(6148914691236517205)  # Ultralytics magic seed
    
    # Prefetch factor: Number of batches to load ahead per worker
    # Rationale: Overlaps data loading (I/O) with model forward/backward (GPU)
    # 4 is higher than default 2, trading memory for throughput
    prefetch_factor = 4 if nw > 0 else None
    
    # Pin memory: Only beneficial when GPU is present
    # Pinned memory enables async CPU→GPU transfer with non_blocking=True
    use_pin_memory = nd > 0 and pin_memory
    
    # Drop last: Only drop if there's actually an incomplete batch
    # Avoid dropping full batches unnecessarily
    should_drop_last = drop_last and (len(dataset) % batch != 0)
    
    return InfiniteDataLoader(
        dataset=dataset,
        batch_size=batch,
        shuffle=shuffle and sampler is None,
        num_workers=nw,
        sampler=sampler,
        prefetch_factor=prefetch_factor,
        pin_memory=use_pin_memory,
        collate_fn=getattr(dataset, "collate_fn", None),
        worker_init_fn=seed_worker,
        generator=generator,
        drop_last=should_drop_last,
    )