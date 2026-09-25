"""
E8: is UnAct selecting class-SELECTIVE units, or merely high-magnitude ones?

Method-section Remark 1 concedes that the selection rule
(unlearn_cnn.py: top (100-p)% by mean activation over D_f) never consults the
retain set, so a unit that fires on everything ranks exactly like a unit that
fires only on the forget class. This measures how much that matters, and it is
the mechanistic explanation for why the percentile that works depends on the
size of the label space (10 -> 20 -> 100 classes).

Definitions, per output channel j of the target block:

    a_f[j]  mean activation over the forget-class training images
    a_r[j]  mean activation over the retain training images
    s[j] = (a_f[j] - a_r[j]) / (a_f[j] + a_r[j])        selectivity, in [-1, 1]

s = 1 means the unit responds only to the forget class; s = 0 means it responds
equally to both, so attenuating it is pure collateral damage.

Two summaries per (dataset, class, percentile):

  * mean s over the selected units -- how class-specific the selection is.
  * retain_mass_pct: the share of total retain activation carried by the
    selected units. This is the damage the intervention must inflict on D_r,
    predicted from forward statistics alone and before any weight is touched.

Hypothesis: selectivity falls and retain_mass rises with class count, which is
why p must rise to keep the selected set small enough to be safe.

Forward passes only; no weights are modified.

Usage:
    python experiments/e8_selectivity.py --datasets cifar10 --device cuda:0
"""
from __future__ import annotations

import argparse
import os
import sys

import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (_REPO_ROOT, os.path.join(_REPO_ROOT, "Our_Method"),
          os.path.join(_REPO_ROOT, "Baseline CIFAR Training")):
    if p not in sys.path:
        sys.path.insert(0, p)

import common as cifar_common  # noqa: E402
from unlearn_cnn import _mask_from, profile_block_activations  # noqa: E402

from unlearn_lib.io import existing_keys, upsert_rows  # noqa: E402
from unlearn_lib.loaders import make_loader  # noqa: E402
from unlearn_lib.splits import build_split_indices  # noqa: E402

RESULTS_PATH = os.path.join(_REPO_ROOT, "results", "e8_selectivity.csv")
FIELDS = [
    "dataset", "num_classes", "arch", "scope", "forget_class", "percentile",
    "n_units", "n_selected",
    "sel_mean_selectivity", "unsel_mean_selectivity",
    "sel_mean_act_forget", "sel_mean_act_retain",
    "retain_mass_pct", "forget_mass_pct", "mass_ratio",
    "frac_units_selectivity_gt_0.5", "eval_device", "git_commit",
]
KEY = ["dataset", "arch", "scope", "forget_class", "percentile"]

PERCENTILES = [50.0, 75.0, 90.0, 95.0, 99.0]
SCOPE = "layer4_last"
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


def run_dataset(dataset, args, device, done):
    n_classes = cifar_common.num_classes(dataset)
    train_set, test_set = cifar_common.get_datasets(dataset, train_augment=False,
                                                    download=False)
    model, _ = cifar_common.load_model(cifar_common.checkpoint_path(dataset))
    block = model.layer4[-1]
    commit = _git_commit()
    dev_name = torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"
    rows = []

    for cls in FORGET_CLASSES[dataset]:
        if all((dataset, "resnet18", SCOPE, str(cls), str(p)) in done for p in PERCENTILES):
            print(f"  {dataset} class {cls}: done, skipping", flush=True)
            continue

        idx = build_split_indices(train_set, test_set, cls)
        # Retain is ~9x larger than forget; subsample it to keep the pass cheap.
        # The statistic is a per-channel mean, so a large random subset is an
        # unbiased estimate -- and it is the same subset for every percentile.
        retain_idx = idx["retain_train"]
        if args.retain_sample and len(retain_idx) > args.retain_sample:
            g = torch.Generator().manual_seed(args.seed)
            retain_idx = retain_idx[torch.randperm(len(retain_idx), generator=g)[:args.retain_sample]].sort().values

        f_loader = make_loader(train_set, idx["forget_train"],
                               batch_size=args.batch_size, workers=args.workers)
        r_loader = make_loader(train_set, retain_idx,
                               batch_size=args.batch_size, workers=args.workers)

        a_f = profile_block_activations(model, block, f_loader, device)
        a_r = profile_block_activations(model, block, r_loader, device)
        n_units = a_f.numel()
        # Activations are post-ReLU and hence non-negative; guard the zero case.
        denom = (a_f + a_r).clamp_min(1e-12)
        s = (a_f - a_r) / denom

        print(f"\n  === {dataset} (C={n_classes}) class {cls} ===  "
              f"|D_f|={len(idx['forget_train'])} |D_r sample|={len(retain_idx)}", flush=True)

        for pct in PERCENTILES:
            if (dataset, "resnet18", SCOPE, str(cls), str(pct)) in done:
                continue
            mask = _mask_from(a_f, pct)
            sel, unsel = mask, ~mask
            retain_mass = 100.0 * a_r[sel].sum().item() / max(a_r.sum().item(), 1e-12)
            forget_mass = 100.0 * a_f[sel].sum().item() / max(a_f.sum().item(), 1e-12)
            rows.append({
                "dataset": dataset, "num_classes": n_classes, "arch": "resnet18",
                "scope": SCOPE, "forget_class": cls, "percentile": pct,
                "n_units": n_units, "n_selected": int(mask.sum()),
                "sel_mean_selectivity": f"{s[sel].mean().item():.4f}",
                "unsel_mean_selectivity": f"{s[unsel].mean().item():.4f}"
                if unsel.any() else "",
                "sel_mean_act_forget": f"{a_f[sel].mean().item():.4f}",
                "sel_mean_act_retain": f"{a_r[sel].mean().item():.4f}",
                "retain_mass_pct": f"{retain_mass:.2f}",
                "forget_mass_pct": f"{forget_mass:.2f}",
                "mass_ratio": f"{forget_mass / max(retain_mass, 1e-12):.4f}",
                "frac_units_selectivity_gt_0.5":
                    f"{100.0 * (s[sel] > 0.5).float().mean().item():.2f}",
                "eval_device": dev_name, "git_commit": commit,
            })
            print(f"      p={pct:<5} n_sel={int(mask.sum()):3d}  "
                  f"selectivity {s[sel].mean().item():+.3f}  "
                  f"forget_mass {forget_mass:5.1f}%  retain_mass {retain_mass:5.1f}%  "
                  f"ratio {forget_mass/max(retain_mass,1e-12):.2f}", flush=True)

        if rows:
            upsert_rows(rows, RESULTS_PATH, FIELDS, KEY)
            rows = []


def main():
    global PERCENTILES, FORGET_CLASSES, RESULTS_PATH
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["cifar10", "cifar20", "cifar100"])
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--retain-sample", type=int, default=10000,
                    help="Cap on retain images used for the mean (0 = all).")
    ap.add_argument("--percentiles", type=float, nargs="+", default=None)
    ap.add_argument("--classes", type=int, nargs="+", default=None)
    ap.add_argument("--results", default=RESULTS_PATH)
    args = ap.parse_args()

    if args.percentiles:
        PERCENTILES = args.percentiles
    if args.classes:
        FORGET_CLASSES = {d: list(args.classes) for d in args.datasets}
    RESULTS_PATH = args.results

    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = cifar_common.pick_device(args.device)
    print(f"E8 class-selectivity on {cifar_common.device_label(device)}")
    done = existing_keys(RESULTS_PATH, KEY)
    print(f"{len(done)} row(s) already recorded")
    for ds in args.datasets:
        run_dataset(ds, args, device, done)
    print("\nE8 complete.")


if __name__ == "__main__":
    main()
