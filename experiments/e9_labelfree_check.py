"""
E9c: verify that LFSSD is genuinely label-free, by the same test used for UnAct.

Protocol (as described in the paper's appendix, "Label-free check"):
run the method twice on the same base model -- once with the true forget-set
labels, once with every forget label replaced by a constant -- and compare the
resulting parameters. A label-free method must return bit-identical weights.

Vanilla SSD is run as a control in the same script. It is expected to diverge,
which is what makes the test informative rather than vacuous: if SSD also came
out identical, the harness would be broken.

The constant is chosen as a class the model actually has, and deliberately NOT
the forget class, so the corrupted labels are wrong for every forget image.

Usage:
    python experiments/e9_labelfree_check.py --device cuda:0
"""
from __future__ import annotations

import argparse
import copy
import os
import sys

import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (_REPO_ROOT, os.path.join(_REPO_ROOT, "SSD"), os.path.join(_REPO_ROOT, "Our_Method"),
          os.path.join(_REPO_ROOT, "Baseline CIFAR Training")):
    if p not in sys.path:
        sys.path.insert(0, p)

import common as cifar_common  # noqa: E402
from lfssd import LFParameterPerturber  # noqa: E402
from ssd import ParameterPerturber  # noqa: E402
from unlearn_cnn import unlearn_blocks  # noqa: E402

from unlearn_lib.io import upsert_rows  # noqa: E402
from unlearn_lib.loaders import make_loader  # noqa: E402
from unlearn_lib.splits import build_split_indices  # noqa: E402

RESULTS_PATH = os.path.join(_REPO_ROOT, "results", "e9_labelfree_check.csv")
FIELDS = ["dataset", "arch", "forget_class", "method", "config", "seed",
          "checksum_true_labels", "checksum_constant_labels",
          "bit_identical", "n_params_differing", "max_abs_diff",
          "eval_device", "git_commit"]
KEY = ["dataset", "arch", "forget_class", "method", "config", "seed"]

# UnAct at its E6-selected configs (percentile, gamma, rounds).
UNACT_SELECTED = {"cifar10": (99.0, 0.01, 20), "cifar20": (99.0, 0.1, 20),
                  "cifar100": (90.0, 0.3, 1)}

SSD_FIXED = {"lower_bound": 1, "exponent": 1, "magnitude_diff": None,
             "min_layer": -1, "max_layer": -1, "forget_threshold": 1}


class ConstantLabelDataset(torch.utils.data.Dataset):
    """Wraps a dataset and replaces every label with `label`.

    Images are untouched, so any method that reads only x must be unaffected.
    """

    def __init__(self, base, label: int):
        self.base = base
        self.label = int(label)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        x = self.base[i][0]
        return x, self.label


def _git_commit() -> str:
    import subprocess
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       cwd=_REPO_ROOT, text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def checksum(model) -> float:
    return float(sum(p.detach().double().sum() for p in model.parameters()))


def run_once(perturber_cls, base_model, forget_loader, full_loader, device,
             alpha, lam):
    model = copy.deepcopy(base_model).eval()
    pdr = perturber_cls(model, torch.optim.SGD(model.parameters(), lr=0.1), device,
                        dict(SSD_FIXED, dampening_constant=lam,
                             selection_weighting=alpha))
    fimp = pdr.calc_importance(forget_loader)
    oimp = pdr.calc_importance(full_loader)
    pdr.modify_weight(oimp, fimp)
    return model


def compare(a, b):
    n_diff = 0
    max_abs = 0.0
    for (_, pa), (_, pb) in zip(a.named_parameters(), b.named_parameters()):
        d = (pa.detach() - pb.detach()).abs()
        n_diff += int((pa.detach() != pb.detach()).sum())
        max_abs = max(max_abs, float(d.max()))
    return n_diff, max_abs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["cifar10", "cifar20", "cifar100"])
    ap.add_argument("--classes", type=int, nargs="+", default=None)
    ap.add_argument("--methods", nargs="+", default=["lfssd", "ssd"])
    ap.add_argument("--alpha", type=float, default=10.0)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--results", default=RESULTS_PATH)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = cifar_common.pick_device(args.device)
    commit = _git_commit()
    dev_name = torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"
    default_cls = {"cifar10": [3], "cifar20": [3], "cifar100": [3]}

    rows = []
    for dataset in args.datasets:
        train_set, test_set = cifar_common.get_datasets(dataset, train_augment=False,
                                                        download=False)
        base_model, _ = cifar_common.load_model(cifar_common.checkpoint_path(dataset))
        for cls in (args.classes or default_cls[dataset]):
            idx = build_split_indices(train_set, test_set, cls)
            # A wrong-but-valid constant label: the next class, wrapping around.
            const = (cls + 1) % cifar_common.num_classes(dataset)

            f_true = make_loader(train_set, idx["forget_train"],
                                 batch_size=args.batch_size, workers=args.workers)
            f_const = make_loader(ConstantLabelDataset(train_set, const),
                                  idx["forget_train"],
                                  batch_size=args.batch_size, workers=args.workers)
            full = make_loader(train_set, idx["full_train"],
                               batch_size=args.batch_size, workers=args.workers)

            for method in args.methods:
                if method == "unact":
                    up, ug, uk = UNACT_SELECTED[dataset]
                    cfg = f"layer4_last_p{up}_g{ug}_k{uk}"
                    models = []
                    for fl in (f_true, f_const):
                        m = copy.deepcopy(base_model)
                        unlearn_blocks(m, fl, device, scope="layer4_last", penalty_scale=ug,
                                       threshold_percentile=up, iter_times=uk, verbose=False)
                        models.append(m)
                    m_true, m_const = models
                else:
                    cfg = f"a{args.alpha}_l{args.lam}_b{args.batch_size}"
                    cls_ = LFParameterPerturber if method == "lfssd" else ParameterPerturber
                    m_true = run_once(cls_, base_model, f_true, full, device,
                                      args.alpha, args.lam)
                    m_const = run_once(cls_, base_model, f_const, full, device,
                                       args.alpha, args.lam)
                n_diff, max_abs = compare(m_true, m_const)
                c_true, c_const = checksum(m_true), checksum(m_const)
                identical = n_diff == 0
                print(f"  {dataset} c{cls} {method:<6} "
                      f"true {c_true:.10f}  const {c_const:.10f}  "
                      f"{'BIT-IDENTICAL' if identical else f'DIFFERS ({n_diff} params, max {max_abs:.3e})'}",
                      flush=True)
                rows.append({
                    "dataset": dataset, "arch": "resnet18", "forget_class": cls,
                    "method": method,
                    "config": cfg,
                    "seed": args.seed,
                    "checksum_true_labels": f"{c_true:.10f}",
                    "checksum_constant_labels": f"{c_const:.10f}",
                    "bit_identical": int(identical),
                    "n_params_differing": n_diff,
                    "max_abs_diff": f"{max_abs:.6e}",
                    "eval_device": dev_name, "git_commit": commit,
                })
                del m_true, m_const
                if device.type == "cuda":
                    torch.cuda.empty_cache()

    upsert_rows(rows, args.results, FIELDS, KEY)
    print("\nE9c complete.")


if __name__ == "__main__":
    main()
