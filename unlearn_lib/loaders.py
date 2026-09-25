"""
unlearn_lib.loaders
Build a DataLoader from an explicit index tensor.

The existing `load_split_loader` helpers take a *filename* and so can only serve
the precomputed single-class splits. Multi-class and subsampled forget sets are
computed at runtime (see unlearn_lib.splits), so they need an indices-based
entry point. DataLoader settings mirror
`Baseline CIFAR Training/common.py:184` exactly, so loaders built here and there
behave identically.
"""
from __future__ import annotations

from typing import Optional, Sequence, Union

import torch
from torch.utils.data import DataLoader, Subset


def make_loader(
    dataset,
    indices: Union[torch.Tensor, Sequence[int]],
    batch_size: int = 256,
    shuffle: bool = False,
    workers: int = 8,
    drop_last: bool = False,
    pin_memory: Optional[bool] = None,
    generator: Optional[torch.Generator] = None,
) -> DataLoader:
    """Subset + DataLoader over `indices`.

    `persistent_workers` matters on Python 3.14: the default start method
    pickles the in-memory CIFAR array into every worker, so respawning them each
    epoch is pure overhead.
    """
    if torch.is_tensor(indices):
        indices = indices.tolist()
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()
    return DataLoader(
        Subset(dataset, list(indices)),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=pin_memory,
        persistent_workers=workers > 0,
        prefetch_factor=4 if workers > 0 else None,
        drop_last=drop_last,
        generator=generator,
    )
