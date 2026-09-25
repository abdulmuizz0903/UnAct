"""
E4: multi-class and sequential forgetting.

SSD's paper forgets exactly one class (or one subclass, or one random 100-sample
set) in every experiment. It never forgets several classes, and never issues
repeated requests -- its Limitations section names repeat-forgetting degradation
as an open problem and predicts that a poorly chosen (alpha, lambda) will
"lead to significant model degradation" over successive requests. So both halves
of this experiment are beyond the baseline's published scope.

Two modes:

  simultaneous  forget k classes in one request (nested sets of size 1,2,3,5)
  sequential    forget the same k classes one at a time, re-applying the method
                to the already-unlearned model, measuring after every request

The reference for both is a retrain gold trained without all k classes. Note the
gold's D_r RISES with k (cifar10: 95.60 -> 95.51 -> 97.39 -> 97.92) because
fewer classes is an easier problem -- so a method that merely preserves its
original accuracy is falling further behind gold as k grows. Scoring against the
matching gold, rather than against the baseline, is what makes this visible.

Usage: python experiments/e4_multiclass.py --mode simultaneous --device cuda:0
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

from unlearn_lib import manifest  # noqa: E402
from unlearn_lib.io import existing_keys, upsert_rows  # noqa: E402
from unlearn_lib.loaders import make_loader  # noqa: E402
from unlearn_lib.metrics import accuracy  # noqa: E402
from unlearn_lib.paths import class_spec  # noqa: E402
from unlearn_lib.splits import build_split_indices  # noqa: E402
from unlearn_lib.timing import time_call  # noqa: E402

RESULTS_PATH = os.path.join(_REPO_ROOT, "results", "e4_multiclass.csv")
FIELDS = ["dataset", "arch", "mode", "method", "config", "forget_spec", "k",
          "request_idx", "seed", "post_overall", "post_retain", "post_forget",
          "gold_retain", "gold_forget", "gap_retain", "score",
          "cum_time_s", "params_damped_pct", "eval_device", "git_commit"]
KEY = ["dataset", "arch", "mode", "method", "config", "forget_spec", "request_idx", "seed"]

SSD_SELECTED = {"cifar10": (8.0, 1.0, 256), "cifar20": (15.0, 1.0, 512),
                "cifar100": (50.0, 0.5, 64)}
UNACT_SELECTED = {"cifar10": (99.0, 0.01, 20), "cifar20": (99.0, 0.1, 20),
                  "cifar100": (90.0, 0.3, 1)}
FIXED = {"lower_bound": 1, "exponent": 1, "magnitude_diff": None,
         "min_layer": -1, "max_layer": -1, "forget_threshold": 1}
# Nested prefixes: the k=1 point reuses the single-class golds already trained.
CLASS_ORDER = {"cifar10": [0, 2, 3, 5, 8], "cifar20": [3, 4, 10, 14, 19],
               "cifar100": [3, 20, 51, 69, 85]}
KS = [1, 2, 3, 5]
# Re-tuning grids for the multi-class / sequential task (Rule 3: symmetric tuning).
TUNE = {"ssd": [4.0, 8.0, 15.0, 30.0, 50.0], "unact": [90.0, 95.0, 99.0, 99.5]}


def _git():
    import subprocess
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       cwd=_REPO_ROOT, text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def apply_method(method, model, dataset, classes, train_set, test_set, device, workers,
                 override=None):
    """Apply one unlearning request in place; return elapsed seconds.

    `override` replaces the single-class-tuned hyperparameter (alpha for SSD,
    percentile for UnAct). Multi-class and sequential forgetting change the size
    and composition of D_f, so the k=1 operating points do not carry over; Rule 3
    (symmetric tuning) requires both methods to be re-tuned for this task.
    """
    idx = build_split_indices(train_set, test_set, classes)
    if method == "unact":
        p, g, k = UNACT_SELECTED[dataset]
        if override is not None:
            p = override
        fl = make_loader(train_set, idx["forget_train"], batch_size=256, workers=workers)
        _, t = time_call(unlearn_blocks, model, fl, device, scope="layer4_last",
                         penalty_scale=g, threshold_percentile=p, iter_times=k,
                         verbose=False, device=device)
        return t
    a, l, bs = SSD_SELECTED[dataset]
    if override is not None:
        a = override
    fl = make_loader(train_set, idx["forget_train"], batch_size=bs, workers=workers)
    ul = make_loader(train_set, idx["full_train"], batch_size=bs, workers=workers)
    model.eval()
    pdr = ParameterPerturber(model, torch.optim.SGD(model.parameters(), lr=0.1), device,
                             dict(FIXED, dampening_constant=l, selection_weighting=a))

    def run():
        # Importances are recomputed on the CURRENT model: after a previous
        # request the weights have changed, so reusing the original Fisher would
        # not be SSD applied to this model.
        fi = pdr.calc_importance(fl)
        oi = pdr.calc_importance(ul)
        pdr.modify_weight(oi, fi)

    _, t = time_call(run, device=device)
    return t


def evaluate_against_gold(model, dataset, classes, train_set, test_set, device, workers):
    idx = build_split_indices(train_set, test_set, classes)
    ev = {k: make_loader(test_set, idx[v], batch_size=512, workers=workers)
          for k, v in {"valid": "test_all", "retain_valid": "retain_test",
                       "forget_valid": "forget_test"}.items()}
    do = accuracy(model, ev["valid"], device)
    dr = accuracy(model, ev["retain_valid"], device)
    df = accuracy(model, ev["forget_valid"], device)
    gp = manifest.lookup("retain_gold", "cifar", "resnet18", dataset, classes)
    if gp:
        gm, _ = cifar_common.load_model(gp)
        gr = accuracy(gm, ev["retain_valid"], device)
        gf = accuracy(gm, ev["forget_valid"], device)
        del gm
    else:
        gr = gf = None
    return do, dr, df, gr, gf


def run_dataset(dataset, args, device, done):
    train_set, test_set = cifar_common.get_datasets(dataset, train_augment=False,
                                                    download=False)
    base, _ = cifar_common.load_model(cifar_common.checkpoint_path(dataset))
    commit = _git()
    dev = torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"
    order = CLASS_ORDER[dataset]
    a, l, bs = SSD_SELECTED[dataset]
    p, g, kk = UNACT_SELECTED[dataset]
    cfgs = {"unact": f"p{p}_g{g}_k{kk}", "ssd": f"a{a}_l{l}_b{bs}"}
    rows = []

    for method in args.methods:
        if args.mode == "simultaneous":
            for k in KS:
                classes = sorted(order[:k])
                spec = class_spec(classes)
                for ov in (TUNE[method] if args.tune else [None]):
                  cfg = cfgs[method] if ov is None else f"{cfgs[method]}_ov{ov}"
                  if (dataset, "resnet18", "simultaneous", method, cfg, spec,
                          "0", str(args.seed)) in done:
                    continue
                  m = copy.deepcopy(base)
                  t = apply_method(method, m, dataset, classes, train_set, test_set,
                                   device, args.workers, override=ov)
                  do, dr, df, gr, gf = evaluate_against_gold(m, dataset, classes, train_set,
                                                           test_set, device, args.workers)
                  sc = abs(dr - gr) + abs(df - gf) if gr is not None else None
                  rows.append(dict(
                    dataset=dataset, arch="resnet18", mode="simultaneous", method=method,
                    config=cfg, forget_spec=spec, k=k, request_idx=0,
                    seed=args.seed, post_overall=f"{do:.4f}", post_retain=f"{dr:.4f}",
                    post_forget=f"{df:.4f}",
                    gold_retain="" if gr is None else f"{gr:.4f}",
                    gold_forget="" if gf is None else f"{gf:.4f}",
                    gap_retain="" if gr is None else f"{dr-gr:+.4f}",
                    score="" if sc is None else f"{sc:.4f}",
                    cum_time_s=f"{t:.3f}", params_damped_pct="",
                    eval_device=dev, git_commit=commit))
                  print(f"    {method:6s} k={k} [{spec}] ov={ov}  D_r {dr:6.2f} "
                        f"(gold {gr:6.2f})  D_f {df:5.2f}  score {sc:6.3f}  t {t:6.2f}s",
                        flush=True)
                  del m
        else:  # sequential
          for ov in (TUNE[method] if args.tune else [None]):
            cfg = cfgs[method] if ov is None else f"{cfgs[method]}_ov{ov}"
            m = copy.deepcopy(base)
            cum_t = 0.0
            for i in range(len(order)):
                classes = sorted(order[:i + 1])
                spec = class_spec(classes)
                cum_t += apply_method(method, m, dataset, [order[i]], train_set,
                                      test_set, device, args.workers, override=ov)
                do, dr, df, gr, gf = evaluate_against_gold(m, dataset, classes, train_set,
                                                           test_set, device, args.workers)
                sc = abs(dr - gr) + abs(df - gf) if gr is not None else None
                rows.append(dict(
                    dataset=dataset, arch="resnet18", mode="sequential", method=method,
                    config=cfg, forget_spec=spec, k=i + 1, request_idx=i,
                    seed=args.seed, post_overall=f"{do:.4f}", post_retain=f"{dr:.4f}",
                    post_forget=f"{df:.4f}",
                    gold_retain="" if gr is None else f"{gr:.4f}",
                    gold_forget="" if gf is None else f"{gf:.4f}",
                    gap_retain="" if gr is None else f"{dr-gr:+.4f}",
                    score="" if sc is None else f"{sc:.4f}",
                    cum_time_s=f"{cum_t:.3f}", params_damped_pct="",
                    eval_device=dev, git_commit=commit))
                gs = "n/a" if gr is None else f"{gr:6.2f}"
                ss = "n/a" if sc is None else f"{sc:6.3f}"
                print(f"    {method:6s} ov={ov} req {i+1}/{len(order)} forgot {order[i]:3d} "
                      f"[{spec}]  D_r {dr:6.2f} (gold {gs})  D_f {df:5.2f}  "
                      f"score {ss}  cum_t {cum_t:6.2f}s", flush=True)
            del m
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if rows:
        upsert_rows(rows, RESULTS_PATH, FIELDS, KEY)


def main():
    global CLASS_ORDER, RESULTS_PATH
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["cifar10", "cifar20", "cifar100"])
    ap.add_argument("--mode", choices=["simultaneous", "sequential", "both"], default="both")
    ap.add_argument("--methods", nargs="+", default=["unact", "ssd"])
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--results", default=RESULTS_PATH)
    ap.add_argument("--order", type=int, nargs="+", default=None,
                    help="Class sequence for sequential mode (overrides CLASS_ORDER). "
                         "Prefixes without a trained gold get blank gold/score columns.")
    ap.add_argument("--tune", action="store_true",
                    help="Re-tune each method for this task (Rule 3). Without it, the "
                         "single-class operating points are used unchanged.")
    args = ap.parse_args()
    RESULTS_PATH = args.results
    if args.order:
        CLASS_ORDER = {d: list(args.order) for d in args.datasets}

    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = cifar_common.pick_device(args.device)
    print(f"E4 multi-class on {cifar_common.device_label(device)}  mode={args.mode}")
    done = existing_keys(RESULTS_PATH, KEY)
    print(f"{len(done)} row(s) already recorded")
    modes = ["simultaneous", "sequential"] if args.mode == "both" else [args.mode]
    for ds in args.datasets:
        for mode in modes:
            args.mode = mode
            print(f"\n  === {ds} / {mode} ===", flush=True)
            run_dataset(ds, args, device, done)
    print("\nE4 complete.")


if __name__ == "__main__":
    main()
