"""
E3 for LFSSD: the forget-set-size sweep of e3_forget_size.py, for the LFSSD
baseline at its selected configuration (E9a), with no re-tuning at any n.

LFSSD is run as a baseline, as its paper runs it: one (alpha, lambda) per
dataset, alpha inside the paper's range. Semantics are identical to E3: the whole
class is forgotten, only the number of forget images the method is shown varies,
and LFSSD's denominator importance is computed once over the full training set
per (dataset, class) and reused at every n.

Usage: python experiments/e3_lfssd_forget_size.py --device cuda:0
"""
from __future__ import annotations

import argparse
import copy
import os
import sys

import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (_REPO_ROOT, os.path.join(_REPO_ROOT, "SSD"),
          os.path.join(_REPO_ROOT, "Baseline CIFAR Training")):
    if p not in sys.path:
        sys.path.insert(0, p)

import common as cifar_common  # noqa: E402
from lfssd import LFParameterPerturber  # noqa: E402

from unlearn_lib import manifest  # noqa: E402
from unlearn_lib.io import existing_keys, upsert_rows  # noqa: E402
from unlearn_lib.loaders import make_loader  # noqa: E402
from unlearn_lib.metrics import accuracy  # noqa: E402
from unlearn_lib.splits import build_split_indices  # noqa: E402
from unlearn_lib.timing import time_call  # noqa: E402

RESULTS_PATH = os.path.join(_REPO_ROOT, "results", "e3_lfssd_forget_size.csv")
FIELDS = ["dataset", "arch", "forget_class", "method", "config", "forget_n",
          "forget_available", "seed", "post_overall", "post_retain", "post_forget",
          "gold_retain", "gold_forget", "gap_retain", "score",
          "method_time_s", "marginal_time_s", "eval_device", "git_commit"]
KEY = ["dataset", "arch", "forget_class", "method", "config", "forget_n", "seed"]

FORGET_N = [1, 5, 10, 25, 50, 100, 500, 1000, 0]  # 0 = all available, as in E3
# E9a selections (results/e9_baselines.csv): alpha, lambda, importance batch size.
LFSSD_SELECTED = {"cifar10": (6.0, 0.1, 64), "cifar20": (10.0, 0.1, 64),
                  "cifar100": (30.0, 0.5, 64)}
FIXED = {"lower_bound": 1, "exponent": 1, "magnitude_diff": None,
         "min_layer": -1, "max_layer": -1, "forget_threshold": 1}
FORGET_CLASSES = {"cifar10": [0, 2, 3, 5, 8], "cifar20": [3, 4, 10, 14, 19],
                  "cifar100": [3, 20, 51, 69, 85]}


def _git_commit():
    import subprocess
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       cwd=_REPO_ROOT, text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def run_dataset(dataset, args, device, done):
    alpha, lam, bs = LFSSD_SELECTED[dataset]
    cfg = f"a{alpha}_l{lam}_b{bs}"
    train_set, test_set = cifar_common.get_datasets(dataset, train_augment=False,
                                                    download=False)
    base, _ = cifar_common.load_model(cifar_common.checkpoint_path(dataset))
    commit = _git_commit()
    dev_name = torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"
    params = dict(FIXED, dampening_constant=lam, selection_weighting=alpha)

    for cls in FORGET_CLASSES[dataset]:
        full_idx = build_split_indices(train_set, test_set, cls)
        ev = {k: make_loader(test_set, full_idx[v], batch_size=args.eval_batch_size,
                             workers=args.workers)
              for k, v in {"valid": "test_all", "retain_valid": "retain_test",
                           "forget_valid": "forget_test"}.items()}
        gm, _ = cifar_common.load_model(manifest.lookup("retain_gold", "cifar",
                                                        "resnet18", dataset, cls))
        gold_r = accuracy(gm, ev["retain_valid"], device)
        gold_f = accuracy(gm, ev["forget_valid"], device)
        del gm
        available = len(full_idx["forget_train"])
        print(f"\n  === {dataset} class {cls} ===  available {available}  "
              f"gold {gold_r:.2f}/{gold_f:.2f}", flush=True)

        # The full-D importance does not depend on n: once per class.
        probe = copy.deepcopy(base).eval()
        pr = LFParameterPerturber(probe, torch.optim.SGD(probe.parameters(), lr=0.1),
                                  device, params)
        full_loader = make_loader(train_set, full_idx["full_train"], batch_size=bs,
                                  workers=args.workers)
        oimp, t_full = time_call(pr.calc_importance, full_loader, device=device)
        del probe, pr, full_loader

        rows = []
        for n in FORGET_N:
            n_used = available if n == 0 else min(n, available)
            key = (dataset, "resnet18", str(cls), "lfssd", cfg, str(n_used), str(args.seed))
            if key in done:
                continue
            idx = build_split_indices(train_set, test_set, cls,
                                      forget_n=None if n == 0 else n, seed=args.seed)
            f_loader = make_loader(train_set, idx["forget_train"],
                                   batch_size=min(bs, n_used), workers=0)
            m = copy.deepcopy(base).eval()
            p2 = LFParameterPerturber(m, torch.optim.SGD(m.parameters(), lr=0.1),
                                      device, params)
            fimp, t_forget = time_call(p2.calc_importance, f_loader, device=device)
            _, t_mod = time_call(p2.modify_weight, oimp, fimp, device=device)
            dr = accuracy(m, ev["retain_valid"], device)
            df = accuracy(m, ev["forget_valid"], device)
            do = accuracy(m, ev["valid"], device)
            sc = abs(dr - gold_r) + abs(df - gold_f)
            rows.append(dict(
                dataset=dataset, arch="resnet18", forget_class=cls, method="lfssd",
                config=cfg, forget_n=n_used, forget_available=available, seed=args.seed,
                post_overall=f"{do:.4f}", post_retain=f"{dr:.4f}", post_forget=f"{df:.4f}",
                gold_retain=f"{gold_r:.4f}", gold_forget=f"{gold_f:.4f}",
                gap_retain=f"{dr-gold_r:+.4f}", score=f"{sc:.4f}",
                method_time_s=f"{t_full + t_forget + t_mod:.3f}",
                marginal_time_s=f"{t_forget + t_mod:.3f}",
                eval_device=dev_name, git_commit=commit))
            print(f"      n={n_used:<5} lfssd D_r {dr:6.2f} D_f {df:6.2f} score {sc:7.3f}",
                  flush=True)
            del m, p2, fimp
        if rows:
            upsert_rows(rows, args.results, FIELDS, KEY)
        del oimp
        if device.type == "cuda":
            torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["cifar10", "cifar20", "cifar100"])
    ap.add_argument("--eval-batch-size", type=int, default=512)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--results", default=RESULTS_PATH)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = cifar_common.pick_device(args.device)
    print(f"E3 (LFSSD) forget-set size on {cifar_common.device_label(device)}  n={FORGET_N}")
    done = existing_keys(args.results, KEY)
    for ds in args.datasets:
        run_dataset(ds, args, device, done)
    print("\nE3 (LFSSD) complete.")


if __name__ == "__main__":
    main()
