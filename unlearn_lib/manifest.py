"""
unlearn_lib.manifest
A registry of every checkpoint produced by the experiment programme.

Experiments should ask for a model by identity ("the retain-gold for cifar10
class 3") rather than reconstructing a path, so that a rename or a MODELS_ROOT
change does not silently produce a missing-file error weeks later. The manifest
also records which git commit and hyperparameters produced each checkpoint,
which is what makes a result reproducible months after the fact.

!! The d_r / d_f / mia / zrf columns here are AS-TRAINED values, measured on
!! whichever GPU trained the model (see train_device). They are provenance and
!! sanity checks -- NOT paper numbers. The same checkpoint scores differently on
!! the L4 and the A10 (measured: one test sample in 10,000 flips, from
!! floating-point reduction order), so a table built from this file would mix
!! devices. Every reported number is recomputed from the saved checkpoints by
!! the experiment scripts, which record eval_device per row (see README.md).
"""
from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime, timezone
from typing import Optional, Sequence, Union

from . import paths
from .io import upsert_rows

FIELDS = [
    "path", "role", "family", "arch", "dataset", "forget_spec",
    "seed", "hparams", "d_r", "d_f", "d_overall", "mia", "zrf",
    "train_time_s", "train_device", "git_commit", "timestamp",
]

#: A checkpoint is uniquely identified by what it is, not where it landed.
KEY = ["role", "family", "arch", "dataset", "forget_spec", "seed"]


def _device_name() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.get_device_name(torch.cuda.current_device())
    except Exception:
        pass
    return "cpu"


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=paths.REPO_ROOT, stderr=subprocess.DEVNULL, text=True,
        ).strip()
    except Exception:
        return "unknown"


def record(
    path: str,
    role: str,
    family: str,
    arch: str,
    dataset: Optional[str] = None,
    forget_classes: Union[int, Sequence[int], None] = None,
    seed: int = 42,
    hparams: Optional[dict] = None,
    metrics: Optional[dict] = None,
    train_time_s: Optional[float] = None,
) -> None:
    """Register a checkpoint. `role` is baseline | retain_gold | unlearned."""
    metrics = metrics or {}
    row = {
        "path": os.path.relpath(path, paths.models_root())
        if path.startswith(paths.models_root()) else path,
        "role": role,
        "family": family,
        "arch": arch,
        "dataset": dataset or "",
        "forget_spec": "" if forget_classes is None else paths.class_spec(forget_classes),
        "seed": seed,
        "hparams": json.dumps(hparams or {}, sort_keys=True),
        "d_r": metrics.get("d_r", ""),
        "d_f": metrics.get("d_f", ""),
        "d_overall": metrics.get("d_overall", ""),
        "mia": metrics.get("mia", ""),
        "zrf": metrics.get("zrf", ""),
        "train_time_s": "" if train_time_s is None else f"{train_time_s:.1f}",
        # Which card trained this. Training is split across GPUs for throughput
        # and the resulting models genuinely differ (within normal seed
        # variance), so the device is part of the provenance. Reported metrics
        # are always recomputed on the canonical eval GPU -- see README.md.
        "train_device": _device_name(),
        "git_commit": _git_commit(),
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    upsert_rows([row], paths.manifest_path(), FIELDS, KEY)


def lookup(role: str, family: str, arch: str, dataset: Optional[str] = None,
           forget_classes: Union[int, Sequence[int], None] = None,
           seed: int = 42) -> Optional[str]:
    """Absolute path of a registered checkpoint, or None if absent/missing on disk."""
    import csv

    mpath = paths.manifest_path()
    if not os.path.exists(mpath):
        return None
    want = (
        role, family, arch, dataset or "",
        "" if forget_classes is None else paths.class_spec(forget_classes), str(seed),
    )
    with open(mpath, newline="") as f:
        for row in csv.DictReader(f):
            if tuple(row.get(k, "") for k in KEY) == want:
                p = row["path"]
                full = p if os.path.isabs(p) else os.path.join(paths.models_root(), p)
                return full if os.path.exists(full) else None
    return None
