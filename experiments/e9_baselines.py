"""
E9: the LFSSD baseline, evaluated with the E2 battery.

Rows written here use exactly the schema of results/e2_main_table.csv, so the
two files concatenate into one table.

Methods
-------
  lfssd     Loss-Free SSD (ICLR 2024 Tiny Paper, Foster et al.), the label-free
            variant of SSD by the SSD authors. This is the baseline that decides
            whether "label-free" is still a contribution. Tuned in E9a.
  ssd       Re-run here, at its E1 operating point, *purely so the timings are
            comparable*: every timing in this file is measured in the same
            process, on the same card.
  unact     Re-run for the same reason, at its E6 operating point.

The accuracy/MIA/ZRF columns for ssd and unact are expected to reproduce E2 to
within one-test-sample device noise (README.md); if they do not,
something has changed and the discrepancy is the finding.

Usage:
    python experiments/e9_baselines.py --device cuda:0
    python experiments/e9_baselines.py --methods lfssd --datasets cifar10
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
from lfssd import LFParameterPerturber  # noqa: E402
from ssd import ParameterPerturber  # noqa: E402
from unlearn_cnn import unlearn_blocks  # noqa: E402

from unlearn_lib import manifest  # noqa: E402
from unlearn_lib.io import existing_keys, upsert_rows  # noqa: E402
from unlearn_lib.loaders import make_loader  # noqa: E402
from unlearn_lib.metrics import accuracy, evaluate_unlearning  # noqa: E402
from unlearn_lib.splits import build_split_indices  # noqa: E402
from unlearn_lib.timing import time_call  # noqa: E402

RESULTS_PATH = os.path.join(_REPO_ROOT, "results", "e9_baselines.csv")
# Identical to experiments/e2_main_table.py::FIELDS so the files concatenate.
FIELDS = [
    "dataset", "arch", "forget_class", "method", "config", "seed",
    "d_overall", "d_r", "d_f", "mia", "zrf",
    "gold_d_r", "gold_d_f", "gold_mia",
    "gap_d_r", "gap_d_f", "gap_mia",
    "method_time_s", "params_damped_pct", "eval_device", "eval_batch_size", "git_commit",
]
KEY = ["dataset", "arch", "forget_class", "method", "config", "seed"]

SSD_FIXED = {"lower_bound": 1, "exponent": 1, "magnitude_diff": None,
             "min_layer": -1, "max_layer": -1, "forget_threshold": 1}

# --- operating points -------------------------------------------------------
# SSD: E1 selection. UnAct: E6 selection.
SSD_SELECTED = {
    "cifar10": {"alpha": 8.0, "lam": 1.0, "batch": 256},
    "cifar20": {"alpha": 15.0, "lam": 1.0, "batch": 512},
    "cifar100": {"alpha": 50.0, "lam": 0.5, "batch": 64},
}
UNACT_SELECTED = {
    "cifar10": {"p": 99.0, "gamma": 0.01, "k": 20},
    "cifar20": {"p": 99.0, "gamma": 0.1, "k": 20},
    "cifar100": {"p": 90.0, "gamma": 0.3, "k": 1},
}
UNACT_SCOPE = "layer4_last"

# LFSSD: selected from the E9a grid by experiments/analyse_e9.py (printed by
# scripts/select_lfssd.py), using the same pre-registered selection rule.
# Overridable from the command line with --lfssd.
LFSSD_SELECTED = {
    "cifar10": {"alpha": 6.0, "lam": 0.1, "batch": 64},
    "cifar20": {"alpha": 10.0, "lam": 0.1, "batch": 64},
    "cifar100": {"alpha": 30.0, "lam": 0.5, "batch": 64},
}

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
    """Randomly initialised same-architecture network for ZRF, seeded as in E2."""
    torch.manual_seed(seed)
    return cifar_common.build_resnet18(n_classes=n_classes, device=device,
                                       channels_last=False).eval()


def _dampening_method(cls_, cfg, base_model, loaders, device, prefix):
    model = copy.deepcopy(base_model).eval()
    pdr = cls_(model, torch.optim.SGD(model.parameters(), lr=0.1), device,
               dict(SSD_FIXED, dampening_constant=cfg["lam"],
                    selection_weighting=cfg["alpha"]))

    def run():
        fi = pdr.calc_importance(loaders[f"{prefix}_forget"])
        oi = pdr.calc_importance(loaders[f"{prefix}_full"])
        pdr.modify_weight(oi, fi)

    _, t = time_call(run, device=device)
    return model, f"a{cfg['alpha']}_l{cfg['lam']}_b{cfg['batch']}", t


def build_method(method, dataset, cls, base_model, loaders, device, args):
    """Return (model, config_string, method_time_s)."""
    if method == "baseline":
        return copy.deepcopy(base_model), "none", 0.0

    if method == "gold":
        path = manifest.lookup("retain_gold", "cifar", "resnet18", dataset, cls)
        if path is None:
            return None, None, None
        model, _ = cifar_common.load_model(path)
        return model, "retrain", float("nan")

    if method == "ssd":
        return _dampening_method(ParameterPerturber, SSD_SELECTED[dataset],
                                 base_model, loaders, device, "ssd")

    if method == "lfssd":
        return _dampening_method(LFParameterPerturber, LFSSD_SELECTED[dataset],
                                 base_model, loaders, device, "lfssd")

    if method == "unact":
        cfg = UNACT_SELECTED[dataset]
        model = copy.deepcopy(base_model)
        _, t = time_call(unlearn_blocks, model, loaders["forget_train"], device,
                         scope=UNACT_SCOPE, penalty_scale=cfg["gamma"],
                         threshold_percentile=cfg["p"], iter_times=cfg["k"],
                         verbose=False, device=device)
        return model, f"{UNACT_SCOPE}_p{cfg['p']}_g{cfg['gamma']}_k{cfg['k']}", t

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
        lf_bs = LFSSD_SELECTED[dataset]["batch"]
        loaders = {
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
            "ssd_forget": make_loader(train_set, idx["forget_train"],
                                      batch_size=ssd_bs, workers=args.workers),
            "ssd_full": make_loader(train_set, idx["full_train"],
                                    batch_size=ssd_bs, workers=args.workers),
            "lfssd_forget": make_loader(train_set, idx["forget_train"],
                                        batch_size=lf_bs, workers=args.workers),
            "lfssd_full": make_loader(train_set, idx["full_train"],
                                      batch_size=lf_bs, workers=args.workers),
        }

        gold_metrics = None
        gpath = manifest.lookup("retain_gold", "cifar", "resnet18", dataset, cls)
        if gpath:
            gmodel, _ = cifar_common.load_model(gpath)
            gold_metrics = evaluate_unlearning(gmodel, loaders, device, teacher=teacher)
            del gmodel

        rows = []
        for method in args.methods:
            model, cfg, t = build_method(method, dataset, cls, base_model,
                                         loaders, device, args)
            if model is None:
                print(f"  {dataset} c{cls} {method}: unavailable, skipping", flush=True)
                continue
            if (dataset, "resnet18", str(cls), method, cfg, str(args.seed)) in done:
                print(f"  {dataset} c{cls} {method} {cfg}: already recorded", flush=True)
                del model
                continue

            snap = {n: p.detach().clone() for n, p in base_model.named_parameters()}
            changed = sum(int((p.detach() != snap[n].to(p.device)).sum())
                          for n, p in model.named_parameters())
            total = sum(p.numel() for p in model.parameters())

            m = evaluate_unlearning(model, loaders, device, teacher=teacher)
            gd = gold_metrics or {}
            rows.append({
                "dataset": dataset, "arch": "resnet18", "forget_class": cls,
                "method": method, "config": cfg, "seed": args.seed,
                "d_overall": f"{m['d_overall']:.4f}", "d_r": f"{m['d_r']:.4f}",
                "d_f": f"{m['d_f']:.4f}", "mia": f"{m['mia']:.6f}",
                "zrf": f"{m['zrf']:.6f}",
                "gold_d_r": f"{gd['d_r']:.4f}" if gd else "",
                "gold_d_f": f"{gd['d_f']:.4f}" if gd else "",
                "gold_mia": f"{gd['mia']:.6f}" if gd else "",
                "gap_d_r": f"{m['d_r']-gd['d_r']:+.4f}" if gd else "",
                "gap_d_f": f"{m['d_f']-gd['d_f']:+.4f}" if gd else "",
                "gap_mia": f"{m['mia']-gd['mia']:+.6f}" if gd else "",
                "method_time_s": "" if t is None or t != t else f"{t:.3f}",
                "params_damped_pct": f"{100.0*changed/total:.4f}",
                "eval_device": dev_name, "eval_batch_size": args.eval_batch_size,
                "git_commit": commit,
            })
            score = (abs(m['d_r']-gd['d_r']) + abs(m['d_f']-gd['d_f'])) if gd else float("nan")
            print(f"  {dataset} c{cls:<3} {method:<9} {cfg:<26} "
                  f"D_r {m['d_r']:6.2f}  D_f {m['d_f']:6.2f}  "
                  f"MIA {m['mia']:.4f}  ZRF {m['zrf']:.4f}  score {score:6.3f}  "
                  f"t {'' if t is None or t!=t else f'{t:7.2f}s'}", flush=True)
            del model, snap
            if device.type == "cuda":
                torch.cuda.empty_cache()

        if rows:
            upsert_rows(rows, RESULTS_PATH, FIELDS, KEY)
        del loaders
        if device.type == "cuda":
            torch.cuda.empty_cache()


def _parse_kv(spec, keys, cast):
    """'cifar10:6,0.1,64' -> ('cifar10', {k: cast[k](v)})."""
    ds, vals = spec.split(":")
    parts = vals.split(",")
    return ds, {k: cast[k](v) for k, v in zip(keys, parts)}


def main():
    global FORGET_CLASSES, RESULTS_PATH
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["cifar10", "cifar20", "cifar100"])
    ap.add_argument("--methods", nargs="+",
                    default=["lfssd", "ssd", "unact"],
                    choices=["baseline", "gold", "ssd", "unact", "lfssd"])
    ap.add_argument("--eval-batch-size", type=int, default=512)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--classes", type=int, nargs="+", default=None)
    ap.add_argument("--results", default=RESULTS_PATH)
    ap.add_argument("--lfssd", nargs="+", default=None,
                    metavar="DS:ALPHA,LAM,BATCH")
    args = ap.parse_args()

    for spec in args.lfssd or []:
        ds, d = _parse_kv(spec, ["alpha", "lam", "batch"],
                          {"alpha": float, "lam": float, "batch": int})
        LFSSD_SELECTED[ds] = d
    if args.classes:
        FORGET_CLASSES = {d: list(args.classes) for d in args.datasets}
    RESULTS_PATH = args.results

    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = cifar_common.pick_device(args.device)
    print(f"E9 baselines on {cifar_common.device_label(device)}  methods={args.methods}")
    for ds in args.datasets:
        print(f"  {ds}: lfssd={LFSSD_SELECTED.get(ds)}")
    done = existing_keys(RESULTS_PATH, KEY)
    print(f"{len(done)} row(s) already recorded")
    for ds in args.datasets:
        run_dataset(ds, args, device, done)
    print("\nE9 complete.")


if __name__ == "__main__":
    main()
