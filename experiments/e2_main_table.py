"""
E2: the full evaluation battery -- D_r, D_f, MIA, ZRF, time -- for every method.

This is the paper's main table. It also supplies the reference points without which the
other numbers cannot be read:

  baseline  the unmodified model. Its MIA is the *upper* anchor (SSD report
            0.873 on CIFAR-10): a model that has seen the forget class leaks it.
  gold      retrained without the class. Its MIA is the *target*. SSD's paper is
            explicit that D_f = 0 and MIA = 0 are NOT the goal -- that is the
            Streisand regime, where a conspicuously ignorant model reveals that
            the class was removed. Success is matching gold, so the quantity to
            report is |MIA_method - MIA_gold|, not MIA alone.

Methods are selected by their E1/E6 grids. UnAct's built-in defaults are NOT
the reported configuration: pass the E6 selection with --unact-* (the exact
commands are in scripts/run_all.sh).

Usage:
    python experiments/e2_main_table.py --methods baseline gold ssd --device cuda:0
    python experiments/e2_main_table.py --datasets cifar10 --methods unact \
        --unact-p-cifar10 99 --unact-gamma 0.01 --unact-iters 20 --device cuda:0
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
from unlearn_lib.metrics import accuracy, evaluate_unlearning  # noqa: E402
from unlearn_lib.splits import build_split_indices  # noqa: E402
from unlearn_lib.timing import time_call  # noqa: E402

RESULTS_PATH = os.path.join(_REPO_ROOT, "results", "e2_main_table.csv")
FIELDS = [
    "dataset", "arch", "forget_class", "method", "config", "seed",
    "d_overall", "d_r", "d_f", "mia", "zrf",
    "gold_d_r", "gold_d_f", "gold_mia",
    "gap_d_r", "gap_d_f", "gap_mia",
    "method_time_s", "params_damped_pct", "eval_device", "eval_batch_size", "git_commit",
]
KEY = ["dataset", "arch", "forget_class", "method", "config", "seed"]

# SSD operating points selected by E1: minimise
# |D_r - gold_D_r| + |D_f - gold_D_f| averaged over the 5 forget classes.
SSD_SELECTED = {
    "cifar10": {"alpha": 8.0, "lam": 1.0, "batch": 256},
    "cifar20": {"alpha": 15.0, "lam": 1.0, "batch": 512},
    "cifar100": {"alpha": 50.0, "lam": 0.5, "batch": 64},
}
SSD_FIXED = {"lower_bound": 1, "exponent": 1, "magnitude_diff": None,
             "min_layer": -1, "max_layer": -1, "forget_threshold": 1}

FORGET_CLASSES = {
    "cifar10": [0, 2, 3, 5, 8],
    "cifar20": [3, 4, 10, 14, 19],
    "cifar100": [3, 20, 51, 69, 85],
}


def _git_commit() -> str:
    import subprocess
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       cwd=_REPO_ROOT, text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def make_teacher(n_classes, device, seed):
    """Randomly initialised same-architecture network for ZRF.

    Seeded so ZRF is reproducible: the reference implementation leaves this
    unseeded, which makes their ZRF column irreproducible run to run.
    """
    torch.manual_seed(seed)
    return cifar_common.build_resnet18(n_classes=n_classes, device=device,
                                       channels_last=False).eval()


def build_method(method, dataset, cls, base_model, loaders, device, args):
    """Return (model, config_string, method_time_s). None if unavailable."""
    if method == "baseline":
        return copy.deepcopy(base_model), "none", 0.0

    if method == "gold":
        path = manifest.lookup("retain_gold", "cifar", "resnet18", dataset, cls)
        if path is None:
            return None, None, None
        model, _ = cifar_common.load_model(path)
        return model, "retrain", float("nan")

    if method == "ssd":
        cfg = SSD_SELECTED[dataset]
        model = copy.deepcopy(base_model).eval()
        pdr = ParameterPerturber(
            model, torch.optim.SGD(model.parameters(), lr=0.1), device,
            dict(SSD_FIXED, dampening_constant=cfg["lam"],
                 selection_weighting=cfg["alpha"]))

        def run():
            fi = pdr.calc_importance(loaders["ssd_forget"])
            oi = pdr.calc_importance(loaders["ssd_full"])
            pdr.modify_weight(oi, fi)

        _, t = time_call(run, device=device)
        return model, f"a{cfg['alpha']}_l{cfg['lam']}_b{cfg['batch']}", t

    if method == "unact":
        p = args.unact_percentile[dataset]
        model = copy.deepcopy(base_model)
        _, t = time_call(unlearn_blocks, model, loaders["forget_train"], device,
                         scope=args.unact_scope, penalty_scale=args.unact_gamma,
                         threshold_percentile=p, iter_times=args.unact_iters,
                         verbose=False, device=device)
        return model, f"{args.unact_scope}_p{p}_g{args.unact_gamma}_k{args.unact_iters}", t

    raise ValueError(method)


def run_dataset(dataset, args, device, done):
    n_classes = cifar_common.num_classes(dataset)
    train_set, test_set = cifar_common.get_datasets(dataset, train_augment=False,
                                                    download=False)
    base_model, _ = cifar_common.load_model(cifar_common.checkpoint_path(dataset))
    teacher = make_teacher(n_classes, device, args.seed)
    commit = _git_commit()
    dev_name = torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"

    for cls in FORGET_CLASSES[dataset]:
        idx = build_split_indices(train_set, test_set, cls)
        ssd_bs = SSD_SELECTED[dataset]["batch"]
        loaders = {
            # MIA needs retain_train; Our_Method never built this loader, which
            # is why UnAct had no MIA number before now.
            "retain_train": make_loader(train_set, idx["retain_train"],
                                        batch_size=args.eval_batch_size, workers=args.workers),
            "forget_train": make_loader(train_set, idx["forget_train"],
                                        batch_size=args.eval_batch_size, workers=args.workers),
            "valid": make_loader(test_set, idx["test_all"],
                                 batch_size=args.eval_batch_size, workers=args.workers),
            "retain_valid": make_loader(test_set, idx["retain_test"],
                                        batch_size=args.eval_batch_size, workers=args.workers),
            "forget_valid": make_loader(test_set, idx["forget_test"],
                                        batch_size=args.eval_batch_size, workers=args.workers),
            # SSD's importance passes use its own selected batch size.
            "ssd_forget": make_loader(train_set, idx["forget_train"],
                                      batch_size=ssd_bs, workers=args.workers),
            "ssd_full": make_loader(train_set, idx["full_train"],
                                    batch_size=ssd_bs, workers=args.workers),
        }

        # Gold first: it is the reference the other rows are scored against.
        gold_metrics = None
        gpath = manifest.lookup("retain_gold", "cifar", "resnet18", dataset, cls)
        if gpath:
            gmodel, _ = cifar_common.load_model(gpath)
            gold_metrics = evaluate_unlearning(gmodel, loaders, device, teacher=teacher)
            del gmodel

        rows = []
        for method in args.methods:
            if (dataset, "resnet18", str(cls), method, "*", str(args.seed)) in done:
                continue
            model, cfg, t = build_method(method, dataset, cls, base_model,
                                         loaders, device, args)
            if model is None:
                print(f"  {dataset} c{cls} {method}: unavailable, skipping", flush=True)
                continue
            if (dataset, "resnet18", str(cls), method, cfg, str(args.seed)) in done:
                del model
                continue

            snap = {n: p.detach().clone() for n, p in base_model.named_parameters()}
            changed = sum(int((p.detach() != snap[n].to(p.device)).sum())
                          for n, p in model.named_parameters())
            total = sum(p.numel() for p in model.parameters())

            m = evaluate_unlearning(model, loaders, device, teacher=teacher)
            g = gold_metrics or {}
            rows.append({
                "dataset": dataset, "arch": "resnet18", "forget_class": cls,
                "method": method, "config": cfg, "seed": args.seed,
                "d_overall": f"{m['d_overall']:.4f}", "d_r": f"{m['d_r']:.4f}",
                "d_f": f"{m['d_f']:.4f}", "mia": f"{m['mia']:.6f}",
                "zrf": f"{m['zrf']:.6f}",
                "gold_d_r": f"{g['d_r']:.4f}" if g else "",
                "gold_d_f": f"{g['d_f']:.4f}" if g else "",
                "gold_mia": f"{g['mia']:.6f}" if g else "",
                "gap_d_r": f"{m['d_r']-g['d_r']:+.4f}" if g else "",
                "gap_d_f": f"{m['d_f']-g['d_f']:+.4f}" if g else "",
                "gap_mia": f"{m['mia']-g['mia']:+.6f}" if g else "",
                "method_time_s": "" if t is None or t != t else f"{t:.3f}",
                "params_damped_pct": f"{100.0*changed/total:.4f}",
                "eval_device": dev_name, "eval_batch_size": args.eval_batch_size,
                "git_commit": commit,
            })
            print(f"  {dataset} c{cls:<3} {method:<9} {cfg:<22} "
                  f"D_r {m['d_r']:6.2f}  D_f {m['d_f']:6.2f}  "
                  f"MIA {m['mia']:.4f}  ZRF {m['zrf']:.4f}  "
                  f"t {'' if t is None or t!=t else f'{t:6.1f}s'}", flush=True)
            del model, snap

        if rows:
            upsert_rows(rows, RESULTS_PATH, FIELDS, KEY)
        if device.type == "cuda":
            torch.cuda.empty_cache()


def main():
    global FORGET_CLASSES, RESULTS_PATH
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["cifar10", "cifar20", "cifar100"])
    ap.add_argument("--methods", nargs="+", default=["baseline", "gold", "ssd", "unact"],
                    choices=["baseline", "gold", "ssd", "unact"])
    ap.add_argument("--eval-batch-size", type=int, default=512)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--classes", type=int, nargs="+", default=None)
    ap.add_argument("--results", default=RESULTS_PATH)
    # UnAct: pre-E6 defaults. Re-run with E6's selection once it lands.
    ap.add_argument("--unact-scope", default="layer4_last")
    ap.add_argument("--unact-gamma", type=float, default=0.1)
    ap.add_argument("--unact-iters", type=int, default=5)
    ap.add_argument("--unact-p-cifar10", type=float, default=75.0)
    ap.add_argument("--unact-p-cifar20", type=float, default=95.0)
    ap.add_argument("--unact-p-cifar100", type=float, default=99.0)
    args = ap.parse_args()
    args.unact_percentile = {"cifar10": args.unact_p_cifar10,
                             "cifar20": args.unact_p_cifar20,
                             "cifar100": args.unact_p_cifar100}
    if args.classes:
        FORGET_CLASSES = {d: list(args.classes) for d in args.datasets}
    RESULTS_PATH = args.results

    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = cifar_common.pick_device(args.device)
    print(f"E2 main table on {cifar_common.device_label(device)}  methods={args.methods}")
    done = existing_keys(RESULTS_PATH, KEY)
    print(f"{len(done)} row(s) already recorded")
    for ds in args.datasets:
        run_dataset(ds, args, device, done)
    print("\nE2 complete.")


if __name__ == "__main__":
    main()
