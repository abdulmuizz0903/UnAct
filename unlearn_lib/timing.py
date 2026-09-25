"""
unlearn_lib.timing
Wall-clock measurement that is valid for CUDA work.

CUDA kernels launch asynchronously, so perf_counter() around a GPU call measures
queueing time, not execution time. SSD's reference timing
code has no synchronize() and is therefore optimistic: its
backward passes mostly force a sync, but modify_weight's in-place masked
multiplies do not. Since timing is a reported metric and the two methods must be
compared on equal terms, every method is timed through this helper.
"""
from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Any, Callable, Optional, Tuple


def _sync(device=None) -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize(device)
    except Exception:
        pass


def time_call(fn: Callable[..., Any], *args, device=None, **kwargs) -> Tuple[Any, float]:
    """Run fn(*args, **kwargs) and return (result, elapsed_seconds).

    Synchronizes before starting and before stopping, so the interval covers
    actual GPU execution rather than kernel-launch queueing.
    """
    _sync(device)
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    _sync(device)
    return result, time.perf_counter() - t0


@contextmanager
def timed(device=None):
    """Context manager form: `with timed() as t: ...` then `t.seconds`.

    Useful where the measured region is not a single call (e.g. unlearn-then-
    evaluate blocks in the relearn probe).
    """

    class _T:
        seconds: Optional[float] = None

    t = _T()
    _sync(device)
    t0 = time.perf_counter()
    try:
        yield t
    finally:
        _sync(device)
        t.seconds = time.perf_counter() - t0
