"""
E1: re-tune the SSD baseline, per dataset and per forget class.

Why this runs before anything else
----------------------------------
UnAct's hyperparameters are tuned per dataset, so SSD gets the same
per-dataset search budget here; running SSD at a single alpha on every
dataset would make the comparison asymmetric.

Efficiency
----------
The naive grid is 3 datasets x 5 classes x 3 batch sizes x 10 alphas x 3 lambdas
= 1350 SSD runs at ~60 s each, i.e. over 22 hours. But alpha and lambda only
enter modify_weight, which is milliseconds; the cost is the two importance
passes, which depend only on (dataset, class, batch_size). So importances are
computed once per (class, batch size) and replayed against all 30 (alpha,
lambda) pairs -- 45 importance passes instead of 1350, roughly 1.5 hours.

Timing reported per row is still SSD's *true* single-run cost --
importance_time + modify_time -- not the amortised cost, so the numbers stay
comparable to a real deployment and to UnAct.

MIA and ZRF are deliberately NOT computed here: MIA alone is 1-2 min per config,
which would put the grid back above 9 hours. The grid selects on accuracy; the
selected configurations then get the full battery in E2.

Usage:
    python experiments/e1_ssd_tune.py --datasets cifar10 --device cuda:0
    python experiments/e1_ssd_tune.py --datasets cifar10 cifar20 cifar100
"""
from __future__ import annotations

import argparse
import copy
import os
import sys
import time

import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (_REPO_ROOT, os.path.join(_REPO_ROOT, "SSD"),
          os.path.join(_REPO_ROOT, "Baseline CIFAR Training")):
    if p not in sys.path:
        sys.path.insert(0, p)

import common as cifar_common  # noqa: E402
from ssd import ParameterPerturber  # noqa: E402  (SSD/ssd.py)

from unlearn_lib.io import existing_keys, upsert_rows  # noqa: E402
from unlearn_lib.loaders import make_loader  # noqa: E402
from unlearn_lib.metrics import accuracy  # noqa: E402
from unlearn_lib.splits import build_split_indices  # noqa: E402
from unlearn_lib.timing import time_call  # noqa: E402

RESULTS_PATH = os.path.join(_REPO_ROOT, "results", "e1_ssd_tune.csv")
FIELDS = [
    "dataset", "arch", "forget_class", "alpha", "lambda", "batch_size", "seed",
    "base_overall", "base_retain", "base_forget",
    "post_overall", "post_retain", "post_forget",
    "gold_retain", "gold_forget", "gap_retain",
    "params_damped_pct", "importance_time_s", "modify_time_s", "method_time_s",
    "eval_device", "git_commit",
]
KEY = ["dataset", "arch", "forget_class", "alpha", "lambda", "batch_size", "seed"]

# Pre-registered grid (paper, appendix). SSD's paper uses alpha=10, lambda=1 for
# ResNet-18 on every CIFAR task; the grid brackets that by a wide margin in both
# directions so the selected value is interior rather than at an edge.
ALPHAS = [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 15.0, 20.0, 30.0, 50.0]
LAMBDAS = [0.1, 0.5, 1.0]

# Batch size is a genuine SSD hyperparameter, not an implementation detail:
# ssd.py:122 divides accumulated squared gradients by the number of BATCHES, so
# the importance scale -- and hence the meaning of alpha -- moves with it.
# Measured at alpha=9, lambda=1 on CIFAR-10 class 3: batch 512 -> D_f 0.00,
# batch 256 -> D_f 18.90. Fixing one arbitrary value would handicap SSD at every
# alpha, so it is searched. 64 is SSD's reference default; 512 is what our
# recorded results used. Importances cannot be cached across batch sizes.
BATCH_SIZES = [64, 256, 512]

FORGET_CLASSES = {
    "cifar10": [0, 2, 3, 5, 8],
    "cifar20": [3, 4, 10, 14, 19],
    "cifar100": [3, 20, 51, 69, 85],
}

SSD_FIXED = {
    "lower_bound": 1, "exponent": 1, "magnitude_diff": None,
    "min_layer": -1, "max_layer": -1, "forget_threshold": 1,
}


def _git_commit() -> str:
    import subprocess
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       cwd=_REPO_ROOT, text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def load_gold_metrics(dataset, forget_class, loaders, device):
    """Retrain gold model's (D_r, D_f), recomputed here on the evaluation device.

    The manifest's stored values were measured on whichever GPU trained the
    model; a table must not mix devices, so they are recomputed rather than read.
    """
    from unlearn_lib import manifest

    path = manifest.lookup("retain_gold", "cifar", "resnet18", dataset, forget_class)
    if path is None:
        return None, None
    model, _ = cifar_common.load_model(path)
    return (accuracy(model, loaders["retain_valid"], device),
            accuracy(model, loaders["forget_valid"], device))


def run_dataset(dataset, args, device, done_keys):
    n_classes = cifar_common.num_classes(dataset)
    train_set, test_set = cifar_common.get_datasets(dataset, train_augment=False,
                                                    download=False)
    base_model, _ = cifar_common.load_model(cifar_common.checkpoint_path(dataset))
    commit = _git_commit()
    rows = []

    for cls, batch_size in ((c, b) for c in FORGET_CLASSES[dataset] for b in BATCH_SIZES):
        pending = [
            (a, l) for a in ALPHAS for l in LAMBDAS
            if (dataset, "resnet18", str(cls), str(a), str(l),
                str(batch_size), str(args.seed)) not in done_keys
        ]
        if not pending:
            print(f"  {dataset} class {cls} bs={batch_size}: all "
                  f"{len(ALPHAS)*len(LAMBDAS)} configs done, skipping", flush=True)
            continue

        idx = build_split_indices(train_set, test_set, cls)
        loaders = {
            "forget_train": make_loader(train_set, idx["forget_train"],
                                        batch_size=batch_size, workers=args.workers),
            "full_train": make_loader(train_set, idx["full_train"],
                                      batch_size=batch_size, workers=args.workers),
            "valid": make_loader(test_set, idx["test_all"],
                                 batch_size=args.eval_batch_size, workers=args.workers),
            "retain_valid": make_loader(test_set, idx["retain_test"],
                                        batch_size=args.eval_batch_size, workers=args.workers),
            "forget_valid": make_loader(test_set, idx["forget_test"],
                                        batch_size=args.eval_batch_size, workers=args.workers),
        }

        base = (accuracy(base_model, loaders["valid"], device),
                accuracy(base_model, loaders["retain_valid"], device),
                accuracy(base_model, loaders["forget_valid"], device))
        gold_r, gold_f = load_gold_metrics(dataset, cls, loaders, device)
        gold_str = "n/a" if gold_r is None else f"{gold_r:.2f}/{gold_f:.2f}"
        print(f"\n  === {dataset} class {cls} bs={batch_size} ===  "
              f"base {base[0]:.2f}/{base[1]:.2f}/{base[2]:.2f}"
              f"  gold {gold_str}  ({len(pending)} configs)", flush=True)

        # Importances depend only on (dataset, class): compute once, replay for all
        # (alpha, lambda). Must be measured on the unmodified base model.
        probe = copy.deepcopy(base_model).eval()
        pdr = ParameterPerturber(
            probe, torch.optim.SGD(probe.parameters(), lr=0.1), device,
            dict(SSD_FIXED, dampening_constant=1.0, selection_weighting=1.0),
        )
        (fimp, oimp), t_imp = time_call(
            lambda: (pdr.calc_importance(loaders["forget_train"]),
                     pdr.calc_importance(loaders["full_train"])), device=device)
        print(f"      importances: {t_imp:.1f}s", flush=True)
        del probe, pdr

        for alpha, lamb in pending:
            model = copy.deepcopy(base_model)
            p2 = ParameterPerturber(
                model, torch.optim.SGD(model.parameters(), lr=0.1), device,
                dict(SSD_FIXED, dampening_constant=lamb, selection_weighting=alpha),
            )
            _, t_mod = time_call(p2.modify_weight, oimp, fimp, device=device)

            damped = total = 0
            for (_, p), (_, o), (_, f) in zip(model.named_parameters(),
                                              oimp.items(), fimp.items()):
                damped += int((f.to(p.device) > o.to(p.device) * alpha).sum())
                total += p.numel()

            post = (accuracy(model, loaders["valid"], device),
                    accuracy(model, loaders["retain_valid"], device),
                    accuracy(model, loaders["forget_valid"], device))
            rows.append({
                "dataset": dataset, "arch": "resnet18", "forget_class": cls,
                "alpha": alpha, "lambda": lamb, "batch_size": batch_size,
                "seed": args.seed,
                "base_overall": f"{base[0]:.4f}", "base_retain": f"{base[1]:.4f}",
                "base_forget": f"{base[2]:.4f}",
                "post_overall": f"{post[0]:.4f}", "post_retain": f"{post[1]:.4f}",
                "post_forget": f"{post[2]:.4f}",
                "gold_retain": "" if gold_r is None else f"{gold_r:.4f}",
                "gold_forget": "" if gold_f is None else f"{gold_f:.4f}",
                "gap_retain": "" if gold_r is None else f"{post[1] - gold_r:+.4f}",
                "params_damped_pct": f"{100.0 * damped / max(total,1):.4f}",
                "importance_time_s": f"{t_imp:.3f}",
                "modify_time_s": f"{t_mod:.3f}",
                "method_time_s": f"{t_imp + t_mod:.3f}",
                "eval_device": torch.cuda.get_device_name(device)
                if device.type == "cuda" else "cpu",
                "git_commit": commit,
            })
            gap = "" if gold_r is None else f"  gap_Dr {post[1]-gold_r:+6.2f}"
            print(f"      a={alpha:<5} l={lamb:<4} -> {post[0]:6.2f}/{post[1]:6.2f}/{post[2]:6.2f}"
                  f"  damped {100.0*damped/max(total,1):5.2f}%{gap}", flush=True)
            del model, p2

            # Flush per class so a crash loses at most one class's remaining work.
            if len(rows) >= len(pending):
                upsert_rows(rows, RESULTS_PATH, FIELDS, KEY)
                rows = []

        del fimp, oimp, loaders
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if rows:
        upsert_rows(rows, RESULTS_PATH, FIELDS, KEY)


def main():
    global ALPHAS, LAMBDAS, BATCH_SIZES, FORGET_CLASSES, RESULTS_PATH
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["cifar10", "cifar20", "cifar100"],
                    choices=["cifar10", "cifar20", "cifar100"])
    ap.add_argument("--batch-sizes", type=int, nargs="+", default=None,
                    help="Override the importance-pass batch-size axis.")
    ap.add_argument("--eval-batch-size", type=int, default=512)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda:0")
    # Overrides, for partial runs and smoke tests. Defaults are the
    # pre-registered grid; anything narrower is not the reported search.
    ap.add_argument("--alphas", type=float, nargs="+", default=None)
    ap.add_argument("--lambdas", type=float, nargs="+", default=None)
    ap.add_argument("--classes", type=int, nargs="+", default=None,
                    help="Override the forget classes (applies to every --datasets entry).")
    ap.add_argument("--results", default=RESULTS_PATH)
    args = ap.parse_args()

    if args.alphas:
        ALPHAS = args.alphas
    if args.lambdas:
        LAMBDAS = args.lambdas
    if args.batch_sizes:
        BATCH_SIZES = args.batch_sizes
    if args.classes:
        FORGET_CLASSES = {d: list(args.classes) for d in args.datasets}
    RESULTS_PATH = args.results

    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = cifar_common.pick_device(args.device)
    print(f"E1 SSD re-tuning on {cifar_common.device_label(device)}")
    print(f"grid: {len(ALPHAS)} alphas x {len(LAMBDAS)} lambdas x "
          f"{len(BATCH_SIZES)} batch sizes {BATCH_SIZES}")

    done = existing_keys(RESULTS_PATH, KEY)
    print(f"{len(done)} config(s) already recorded in {RESULTS_PATH}")

    for ds in args.datasets:
        run_dataset(ds, args, device, done)

    print("\nE1 complete.")


if __name__ == "__main__":
    main()
