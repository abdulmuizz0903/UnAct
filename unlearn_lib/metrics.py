"""
unlearn_lib.metrics
The evaluation battery, applied identically to every unlearning method.

MIA and ZRF come from SSD/metrics.py (SSD's reference implementation), so every
method -- UnAct, SSD, LFSSD and the retrained model -- is scored by the same
code. This module makes that battery available to any method that returns a
model.

Interpretation note (from the SSD paper, arXiv 2308.07707): lower MIA is NOT
better. They explicitly warn that driving forget accuracy and MIA to zero is the
Streisand effect -- a model that is conspicuously ignorant of a class leaks that
the class was removed. The target is the *retrained* model's value, so report
|MIA_method - MIA_retrain| alongside the raw number.
"""
from __future__ import annotations

import os
import sys
from typing import Dict, Optional

import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SSD_DIR = os.path.join(_REPO_ROOT, "SSD")
if _SSD_DIR not in sys.path:
    sys.path.insert(0, _SSD_DIR)

from metrics import (  # noqa: E402  (SSD/metrics.py)
    UnLearningScore,
    get_membership_attack_prob,
)


@torch.no_grad()
def accuracy(model, loader, device) -> float:
    """Top-1 accuracy in percent, fp32.

    Every method is evaluated through this one function (never under autocast),
    so accuracies are comparable across methods.
    """
    model.eval()
    correct, total = 0, 0
    for batch in loader:
        x, y = (batch[0], batch[2]) if len(batch) == 3 else (batch[0], batch[1])
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        correct += (model(x).argmax(dim=1) == y).sum().item()
        total += y.size(0)
    return 100.0 * correct / max(total, 1)


def fraction_params_modified(before: Dict[str, torch.Tensor], model) -> float:
    """Fraction of parameters whose value changed, in percent.

    SSD reports 1.7% for CIFAR-100 rocket, so this is directly comparable. Call
    `snapshot_params` before unlearning and pass the result here after.
    """
    changed = total = 0
    for name, p in model.named_parameters():
        ref = before[name].to(p.device)
        changed += (p.detach() != ref).sum().item()
        total += p.numel()
    return 100.0 * changed / max(total, 1)


def snapshot_params(model) -> Dict[str, torch.Tensor]:
    return {n: p.detach().clone() for n, p in model.named_parameters()}


def evaluate_unlearning(
    model,
    loaders: Dict[str, object],
    device,
    teacher=None,
    with_mia: bool = True,
    with_zrf: bool = True,
) -> Dict[str, float]:
    """The full battery for one unlearned model.

    `loaders` must provide: valid, retain_valid, forget_valid, retain_train,
    forget_train. The last two are needed only for MIA.

    Loader roles follow SSD's reference code (strategies.get_metric_scores)
    exactly, so the
    numbers are comparable to theirs:
      ZRF uses the forget *test* split against a randomly initialised teacher;
      MIA trains on (retain_train = member, full test set = non-member) and is
      evaluated on forget_train.
    """
    out = {
        "d_overall": accuracy(model, loaders["valid"], device),
        "d_r": accuracy(model, loaders["retain_valid"], device),
        "d_f": accuracy(model, loaders["forget_valid"], device),
    }
    if with_zrf:
        if teacher is None:
            raise ValueError("ZRF needs a randomly initialised same-architecture teacher")
        out["zrf"] = float(
            UnLearningScore(model, teacher, loaders["forget_valid"], 128, device)
        )
    if with_mia:
        out["mia"] = float(
            get_membership_attack_prob(
                loaders["retain_train"], loaders["forget_train"], loaders["valid"], model
            )
        )
    return out


def gap_to_gold(metrics: Dict[str, float], gold: Optional[Dict[str, float]]) -> Dict[str, float]:
    """Signed distance to the retrained gold model, the quantity SSD actually ranks on.

    Returns {} when no gold model is available, so callers degrade gracefully
    rather than silently reporting absolute numbers as if they were gaps.
    """
    if not gold:
        return {}
    return {
        f"gap_{k}": metrics[k] - gold[k]
        for k in ("d_r", "d_f", "mia", "zrf")
        if k in metrics and k in gold
    }
