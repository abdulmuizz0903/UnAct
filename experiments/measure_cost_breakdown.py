"""
Cost breakdown: is UnAct's speed advantage real, or an artefact of not crediting
SSD its amortisation?

SSD's paper states that the Fisher matrix over the full training set D can be
computed once and reused across unlearning requests, saving 84.32% +- 0.21% of
its compute. Our E2 timings are *cold* -- every SSD run recomputes the full-D
pass -- so quoting them as SSD's cost overstates it for any deployment serving
more than one forget request.

This measures the three components separately so both numbers can be reported:

    cold      = forget pass + full-D pass + modify        (first request)
    amortised = forget pass + modify                      (subsequent requests)

UnAct has no amortisable component: it reads only the forget set, and its cost
is k forward passes over it. That is also why its cost falls as the label space
grows (CIFAR-100 has 500 images per class, CIFAR-10 has 5000) while SSD's full-D
pass is constant at 50,000 regardless.

Usage: python experiments/measure_cost_breakdown.py --device cuda:0
"""
from __future__ import annotations

import argparse
import copy
import os
import sys

import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (_REPO_ROOT, os.path.join(_REPO_ROOT, "Our_Method"),
          os.path.join(_REPO_ROOT, "SSD"),
          os.path.join(_REPO_ROOT, "Baseline CIFAR Training")):
    if p not in sys.path:
        sys.path.insert(0, p)

import common as cifar_common  # noqa: E402
from ssd import ParameterPerturber  # noqa: E402
from unlearn_cnn import unlearn_blocks  # noqa: E402

from unlearn_lib.io import upsert_rows  # noqa: E402
from unlearn_lib.loaders import make_loader  # noqa: E402
from unlearn_lib.splits import build_split_indices  # noqa: E402
from unlearn_lib.timing import time_call  # noqa: E402

RESULTS_PATH = os.path.join(_REPO_ROOT, "results", "cost_breakdown.csv")
FIELDS = ["dataset", "forget_class", "n_forget", "n_full", "repeat",
          "ssd_forget_s", "ssd_full_s", "ssd_modify_s", "ssd_cold_s", "ssd_amortised_s",
          "unact_config", "unact_s", "speedup_vs_cold", "speedup_vs_amortised",
          "eval_device", "git_commit"]
KEY = ["dataset", "forget_class", "repeat"]

SSD_SELECTED = {"cifar10": (8.0, 1.0, 256), "cifar20": (15.0, 1.0, 512),
                "cifar100": (50.0, 0.5, 64)}
UNACT_P = {"cifar10": 75.0, "cifar20": 95.0, "cifar100": 99.0}
FIXED = {"lower_bound": 1, "exponent": 1, "magnitude_diff": None,
         "min_layer": -1, "max_layer": -1, "forget_threshold": 1}


def _git_commit():
    import subprocess
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       cwd=_REPO_ROOT, text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["cifar10", "cifar20", "cifar100"])
    ap.add_argument("--forget-class", type=int, default=3)
    ap.add_argument("--repeats", type=int, default=3,
                    help="Timings are noisy; report mean over repeats.")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--unact-gamma", type=float, default=0.1)
    ap.add_argument("--unact-iters", type=int, default=5)
    # Defaults reproduce the original (pre-E6) measurement. The E6-selected
    # configs differ per dataset -- and UnAct's cost is linear in k -- so
    # timing them means one invocation per dataset with these overrides.
    ap.add_argument("--unact-p", type=float, default=None,
                    help="Override the per-dataset percentile in UNACT_P.")
    ap.add_argument("--results", default=RESULTS_PATH)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = cifar_common.pick_device(args.device)
    print(f"cost breakdown on {cifar_common.device_label(device)}  "
          f"(nothing else should be running on this card)")
    commit = _git_commit()
    dev_name = torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"
    rows = []

    for ds in args.datasets:
        alpha, lam, bs = SSD_SELECTED[ds]
        unact_p = UNACT_P[ds] if args.unact_p is None else args.unact_p
        train_set, test_set = cifar_common.get_datasets(ds, train_augment=False,
                                                        download=False)
        base, _ = cifar_common.load_model(cifar_common.checkpoint_path(ds))
        idx = build_split_indices(train_set, test_set, args.forget_class)
        ssd_f = make_loader(train_set, idx["forget_train"], batch_size=bs,
                            workers=args.workers)
        ssd_u = make_loader(train_set, idx["full_train"], batch_size=bs,
                            workers=args.workers)
        un_f = make_loader(train_set, idx["forget_train"], batch_size=256,
                           workers=args.workers)

        for rep in range(args.repeats):
            m = copy.deepcopy(base).eval()
            pdr = ParameterPerturber(m, torch.optim.SGD(m.parameters(), lr=0.1),
                                     device, dict(FIXED, dampening_constant=lam,
                                                  selection_weighting=alpha))
            fi, t_f = time_call(pdr.calc_importance, ssd_f, device=device)
            oi, t_u = time_call(pdr.calc_importance, ssd_u, device=device)
            _, t_m = time_call(pdr.modify_weight, oi, fi, device=device)
            del m, pdr, fi, oi
            torch.cuda.empty_cache() if device.type == "cuda" else None

            mu = copy.deepcopy(base)
            _, t_un = time_call(unlearn_blocks, mu, un_f, device,
                                scope="layer4_last", penalty_scale=args.unact_gamma,
                                threshold_percentile=unact_p,
                                iter_times=args.unact_iters, verbose=False,
                                device=device)
            del mu
            torch.cuda.empty_cache() if device.type == "cuda" else None

            cold, amort = t_f + t_u + t_m, t_f + t_m
            rows.append({
                "dataset": ds, "forget_class": args.forget_class,
                "n_forget": len(idx["forget_train"]), "n_full": len(idx["full_train"]),
                "repeat": rep,
                "ssd_forget_s": f"{t_f:.3f}", "ssd_full_s": f"{t_u:.3f}",
                "ssd_modify_s": f"{t_m:.3f}", "ssd_cold_s": f"{cold:.3f}",
                "ssd_amortised_s": f"{amort:.3f}",
                "unact_config": f"p{unact_p}_g{args.unact_gamma}_k{args.unact_iters}",
                "unact_s": f"{t_un:.3f}",
                "speedup_vs_cold": f"{cold / t_un:.2f}",
                "speedup_vs_amortised": f"{amort / t_un:.2f}",
                "eval_device": dev_name, "git_commit": commit,
            })
            print(f"  {ds:9s} rep{rep}  SSD forget {t_f:6.2f}s  full-D {t_u:6.2f}s  "
                  f"modify {t_m:5.2f}s | cold {cold:6.2f}s  amortised {amort:5.2f}s | "
                  f"UnAct {t_un:5.2f}s | speedup {cold/t_un:5.1f}x cold, "
                  f"{amort/t_un:4.1f}x amortised", flush=True)

    upsert_rows(rows, args.results, FIELDS, KEY)
    print(f"\nwrote {len(rows)} rows to {args.results}")


if __name__ == "__main__":
    main()
