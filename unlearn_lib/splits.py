"""
unlearn_lib.splits
Forget/retain index construction: multi-class and subsampled forget sets.

Replaces the per-class precomputed .pt files for the experiments that need more
than one class or fewer than all of a class's samples. The existing single-class
files (data/<ds>/splits/forget_train_class_<c>.pt) remain valid and are still
used by the unchanged code paths; these helpers compute equivalents on the fly,
which is the only option for multi-class (CIFAR-100 has 2^100 subsets).

Forget-set-size semantics (E3, "data-efficiency reading")
---------------------------------------------------------
We still forget the *whole class*; we only vary how many class-c examples the
method is *shown*. So:

    forget_train = a size-n random subset of class c
    retain_train = every training example NOT in class c   (unchanged by n)
    full_train   = the entire training set                 (unchanged by n)

The class-c examples not sampled into forget_train belong to neither forget nor
retain: they are simply withheld from the method. This keeps D whole as n varies,
which matters because SSD's selection rule compares forget importance against
importance over D -- if D shrank with n, the comparison would drift for reasons
unrelated to forget-set size. Evaluation stays the standard full-class protocol.
"""
from __future__ import annotations

from typing import Optional, Sequence, Tuple, Union

import torch


def as_class_list(forget_classes: Union[int, Sequence[int]]) -> list:
    return [int(forget_classes)] if isinstance(forget_classes, int) else sorted(int(c) for c in forget_classes)


def _targets_tensor(dataset) -> torch.Tensor:
    targets = dataset.targets
    if not torch.is_tensor(targets):
        targets = torch.tensor(targets)
    return targets


def class_indices(
    dataset, forget_classes: Union[int, Sequence[int]]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """(forget_idx, retain_idx) for one or many classes.

    Uses torch.isin so the single-class and multi-class paths are identical;
    the existing `_class_split_indices` helpers only handle a scalar.
    """
    targets = _targets_tensor(dataset)
    classes = torch.tensor(as_class_list(forget_classes))
    mask = torch.isin(targets, classes)
    return mask.nonzero(as_tuple=True)[0], (~mask).nonzero(as_tuple=True)[0]


def subsample_indices(
    idx: torch.Tensor,
    n: Optional[int] = None,
    frac: Optional[float] = None,
    seed: int = 42,
) -> torch.Tensor:
    """A seeded random subset of `idx`, returned in sorted order.

    Sorted so the resulting loader order depends only on the chosen set, not on
    the permutation -- SSD's unshuffled importance pass is order-sensitive at the
    batch boundary, and we want a run to be reproducible from (n, seed) alone.

    Deliberately NOT ssd.py's subsample_dataset, which is dead code and uses a
    deterministic stride `int(1/perc)` that returns everything for frac > 0.5.
    """
    if n is None and frac is None:
        return idx
    if n is None:
        n = max(1, int(round(frac * len(idx))))
    n = min(int(n), len(idx))
    g = torch.Generator().manual_seed(int(seed))
    picked = torch.randperm(len(idx), generator=g)[:n]
    return idx[picked].sort().values


def build_split_indices(
    train_set,
    test_set,
    forget_classes: Union[int, Sequence[int]],
    forget_n: Optional[int] = None,
    forget_frac: Optional[float] = None,
    seed: int = 42,
) -> dict:
    """All index tensors an experiment needs, for any class set and forget size.

    Returns keys: forget_train, retain_train, full_train, forget_test,
    retain_test, test_all. See the module docstring for why retain_train and
    full_train do not depend on forget_n.
    """
    forget_train_all, retain_train = class_indices(train_set, forget_classes)
    forget_test, retain_test = class_indices(test_set, forget_classes)
    forget_train = subsample_indices(forget_train_all, forget_n, forget_frac, seed)

    # full_train is D: the complete training set, in the reference implementation's
    # order -- retain first, then the whole forget class
    # (SSD reference code, forget_full_class_main.py:
    # ConcatDataset((retain_train, forget_train))).
    # Order matters: SSD's importance pass is unshuffled and averages squared
    # gradients per batch, and we measured that swapping the two halves changes
    # the selected parameter count by ~1.4 points. Note this is the *whole* class
    # even when the method is only shown `forget_n` of it, so D does not shrink as
    # the forget-set size is swept.
    full_train = torch.cat([retain_train, forget_train_all])

    return {
        "forget_train": forget_train,
        "retain_train": retain_train,
        "full_train": full_train,
        "forget_test": forget_test,
        "retain_test": retain_test,
        "test_all": torch.arange(len(test_set)),
        # Bookkeeping for the results CSV.
        "forget_train_available": len(forget_train_all),
        "forget_train_used": len(forget_train),
    }
