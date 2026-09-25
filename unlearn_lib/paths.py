"""
unlearn_lib.paths
Checkpoint locations and the naming scheme.

New checkpoints are written under UNLEARN_MODELS_ROOT (default: the in-repo
models/ directory). A ViT-B/16 checkpoint is ~346 MB and the full programme
trains dozens of retrained models, so point it at a disk with room to spare.

Reads fall back to the in-repo models/ directory, so the existing (immutable)
baseline checkpoints are still found when MODELS_ROOT points elsewhere. Writes
always go to MODELS_ROOT. Nothing needs to be moved.

Naming
------
    cifar/<ds>/resnet18.pt                   baseline, IMMUTABLE
    cifar/<ds>/resnet18_retain_c{spec}.pt    retrain gold
    vit/<ds>/vit_b16.pt                      ViT baseline
    vit/<ds>/vit_b16_retain_c{spec}.pt       ViT retrain gold
    unlearned/<method>/<arch>/<ds>/<cfghash>.pt
    manifest.csv

`spec` is the canonical forget-class string: "3" for a single class, "3-5-7" for
several (always sorted, so the same request maps to one path).
"""
from __future__ import annotations

import hashlib
import json
import os
from typing import Iterable, Optional, Sequence, Union

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_MODELS = os.path.join(REPO_ROOT, "models")


def models_root() -> str:
    """Where new checkpoints are written. Override with UNLEARN_MODELS_ROOT."""
    return os.environ.get("UNLEARN_MODELS_ROOT", REPO_MODELS)


def resolve_read(relpath: str) -> str:
    """Locate an existing checkpoint: MODELS_ROOT first, then the in-repo models/.

    Lets MODELS_ROOT point at another disk without relocating the baseline
    checkpoints.
    """
    primary = os.path.join(models_root(), relpath)
    if os.path.exists(primary):
        return primary
    fallback = os.path.join(REPO_MODELS, relpath)
    if os.path.exists(fallback):
        return fallback
    return primary  # non-existent; caller raises with the canonical path


def resolve_write(relpath: str) -> str:
    """Destination for a new checkpoint, creating the parent directory."""
    path = os.path.join(models_root(), relpath)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path


def class_spec(forget_classes: Union[int, Sequence[int]]) -> str:
    """Canonical forget-class string: 3 -> "3", [7,3,5] -> "3-5-7"."""
    if isinstance(forget_classes, int):
        return str(forget_classes)
    return "-".join(str(c) for c in sorted(forget_classes))


# --------------------------------------------------------------------------
# Relative paths (pass to resolve_read / resolve_write)
# --------------------------------------------------------------------------
def baseline_rel(family: str, dataset: Optional[str], arch: str) -> str:
    if family in ("cifar", "vit"):
        return os.path.join(family, dataset, f"{arch}.pt")
    if family == "cnn":
        return os.path.join("cnn", f"{arch}.pt")
    if family == "mlp":
        return os.path.join("mlp", f"{arch}.pt")
    raise ValueError(f"Unknown family {family!r}")


def gold_rel(family: str, dataset: Optional[str], arch: str,
             forget_classes: Union[int, Sequence[int]]) -> str:
    """Retrain-from-scratch gold model, trained on the retain set only."""
    tag = "d" if family in ("cnn", "mlp") else "c"
    spec = class_spec(forget_classes)
    base = baseline_rel(family, dataset, arch)
    stem, ext = os.path.splitext(base)
    return f"{stem}_retain_{tag}{spec}{ext}"


def config_hash(config: dict) -> str:
    """Short stable hash of a config dict, for unlearned-checkpoint filenames."""
    blob = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(blob.encode()).hexdigest()[:10]


def unlearned_rel(method: str, arch: str, dataset: Optional[str], config: dict) -> str:
    """Opt-in only. Unlearned models regenerate in seconds; saving every sweep
    config at ViT scale would be ~83 GB, so --save-unlearned defaults to off."""
    return os.path.join("unlearned", method, arch, dataset or "_",
                        f"{config_hash(config)}.pt")


def manifest_path() -> str:
    return os.path.join(models_root(), "manifest.csv")


def describe() -> str:
    """One-line summary for run logs, so every log records where models went."""
    root = models_root()
    default = " (default: in-repo)" if root == REPO_MODELS else ""
    return f"MODELS_ROOT={root}{default}"
