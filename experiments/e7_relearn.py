"""
E7: relearn-time. Does UnAct delete the class, or only mute the readout?

This is the sharpest available attack on the method. UnAct attenuates the
classifier columns of the selected channels, so a sceptic's hypothesis is that
the representation survives intact and only the output pathway is suppressed --
in which case a handful of gradient steps on a few forget-class images would
restore the class immediately, and "unlearning" would be a misnomer.

The test: fine-tune each unlearned model on m forget-class images and measure
how fast D_f comes back. The reference is the RETRAINED gold model, which by
construction never saw the class -- whatever recovery rate it shows is what
"the knowledge is genuinely absent" looks like on this architecture and data.

  recovery much faster than gold  =>  the knowledge was still present (masked)
  recovery comparable to gold     =>  the knowledge was genuinely removed

Reporting the gold curve is what makes this interpretable; an absolute recovery
rate on its own says nothing, because even a from-scratch model relearns a
CIFAR class quickly from a good feature extractor.

D_r is tracked alongside, since a model can trivially "recover" D_f by
collapsing onto the forget class. A first version of this probe fine-tuned on
forget images ALONE and did exactly that: every method hit D_f = 100% by step 20
with D_r = 0.00%, which measures nothing. Each relearning batch therefore mixes
the m forget images with `--retain-per-step` retain images (default 4x m), so
the model must recover the class without discarding the rest -- the realistic
threat model, and the only one where the D_f curve is interpretable.

Usage: python experiments/e7_relearn.py --device cuda:0
"""
from __future__ import annotations

import argparse
import copy
import os
import sys

import torch
import torch.nn as nn

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
from unlearn_lib.splits import build_split_indices  # noqa: E402

RESULTS_PATH = os.path.join(_REPO_ROOT, "results", "e7_relearn.csv")
FIELDS = ["dataset", "arch", "forget_class", "method", "relearn_m", "retain_per_step",
          "step", "seed", "lr", "d_f", "d_r", "d_f_at_step0", "baseline_d_f",
          "recovery_pct", "eval_device", "git_commit"]
KEY = ["dataset", "arch", "forget_class", "method", "relearn_m", "step", "seed", "lr"]

SSD_SELECTED = {"cifar10": (8.0, 1.0, 256), "cifar20": (15.0, 1.0, 512),
                "cifar100": (50.0, 0.5, 64)}
UNACT_SELECTED = {"cifar10": (99.0, 0.01, 20), "cifar20": (99.0, 0.1, 20),
                  "cifar100": (90.0, 0.3, 1)}
FIXED = {"lower_bound": 1, "exponent": 1, "magnitude_diff": None,
         "min_layer": -1, "max_layer": -1, "forget_threshold": 1}
FORGET_CLASSES = {"cifar10": [0, 2, 3, 5, 8], "cifar20": [3, 4, 10, 14, 19],
                  "cifar100": [3, 20, 51, 69, 85]}
RELEARN_M = [1, 5, 25]
EVAL_STEPS = [0, 1, 2, 5, 10, 20, 50]


def _git_commit():
    import subprocess
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       cwd=_REPO_ROOT, text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def make_model(method, dataset, cls, base, loaders, device):
    if method == "gold":
        path = manifest.lookup("retain_gold", "cifar", "resnet18", dataset, cls)
        if path is None:
            return None
        m, _ = cifar_common.load_model(path)
        return m
    if method == "ssd":
        a, l, bs = SSD_SELECTED[dataset]
        m = copy.deepcopy(base).eval()
        pdr = ParameterPerturber(m, torch.optim.SGD(m.parameters(), lr=0.1), device,
                                 dict(FIXED, dampening_constant=l, selection_weighting=a))
        pdr.modify_weight(pdr.calc_importance(loaders["ssd_full"]),
                          pdr.calc_importance(loaders["ssd_forget"]))
        return m
    if method == "unact":
        p, g, k = UNACT_SELECTED[dataset]
        m = copy.deepcopy(base)
        unlearn_blocks(m, loaders["unact_forget"], device, scope="layer4_last",
                       penalty_scale=g, threshold_percentile=p, iter_times=k,
                       verbose=False)
        return m
    raise ValueError(method)


def run_dataset(dataset, args, device, done):
    train_set, test_set = cifar_common.get_datasets(dataset, train_augment=False,
                                                    download=False)
    base, _ = cifar_common.load_model(cifar_common.checkpoint_path(dataset))
    commit = _git_commit()
    dev_name = torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"
    a, l, ssd_bs = SSD_SELECTED[dataset]

    for cls in FORGET_CLASSES[dataset]:
        idx = build_split_indices(train_set, test_set, cls)
        ev_f = make_loader(test_set, idx["forget_test"], batch_size=512, workers=args.workers)
        ev_r = make_loader(test_set, idx["retain_test"], batch_size=512, workers=args.workers)
        loaders = {
            "ssd_forget": make_loader(train_set, idx["forget_train"], batch_size=ssd_bs,
                                      workers=args.workers),
            "ssd_full": make_loader(train_set, idx["full_train"], batch_size=ssd_bs,
                                    workers=args.workers),
            "unact_forget": make_loader(train_set, idx["forget_train"], batch_size=256,
                                        workers=args.workers),
        }
        baseline_df = accuracy(base, ev_f, device)
        print(f"\n  === {dataset} class {cls} ===  baseline D_f {baseline_df:.2f}", flush=True)

        rows = []
        for method in args.methods:
            model0 = make_model(method, dataset, cls, base, loaders, device)
            if model0 is None:
                print(f"      {method}: unavailable", flush=True)
                continue
            for m_n in RELEARN_M:
                if all((dataset, "resnet18", str(cls), method, str(m_n), str(s),
                        str(args.seed), str(args.lr)) in done for s in EVAL_STEPS):
                    continue
                # A fixed, seeded subset of forget-class TRAIN images to relearn from.
                sub = build_split_indices(train_set, test_set, cls,
                                          forget_n=m_n, seed=args.seed)["forget_train"]
                # Mix in retain images so the model cannot "recover" D_f by
                # collapsing onto the forget class (see module docstring).
                n_ret = int(round(args.retain_per_step * m_n))
                g = torch.Generator().manual_seed(args.seed)
                ridx = idx["retain_train"][
                    torch.randperm(len(idx["retain_train"]), generator=g)[:n_ret]]
                both = sub.tolist() + ridx.tolist()
                xs = torch.stack([train_set[i][0] for i in both]).to(device)
                ys = torch.tensor([train_set[i][1] for i in both]).to(device)

                model = copy.deepcopy(model0)
                model.train()
                opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
                crit = nn.CrossEntropyLoss()
                d_f0 = accuracy(model, ev_f, device)

                curve = []
                for step in range(max(EVAL_STEPS) + 1):
                    if step in EVAL_STEPS:
                        df = accuracy(model, ev_f, device)
                        dr = accuracy(model, ev_r, device)
                        curve.append((step, df, dr))
                        rows.append(dict(
                            dataset=dataset, arch="resnet18", forget_class=cls,
                            method=method, relearn_m=m_n,
                            retain_per_step=args.retain_per_step,
                            step=step, seed=args.seed,
                            lr=args.lr, d_f=f"{df:.4f}", d_r=f"{dr:.4f}",
                            d_f_at_step0=f"{d_f0:.4f}", baseline_d_f=f"{baseline_df:.4f}",
                            recovery_pct=f"{100.0*df/max(baseline_df,1e-9):.2f}",
                            eval_device=dev_name, git_commit=commit))
                        model.train()
                    if step == max(EVAL_STEPS):
                        break
                    opt.zero_grad(set_to_none=True)
                    crit(model(xs), ys).backward()
                    opt.step()

                pretty = "  ".join(f"s{s}:{df:.1f}" for s, df, _ in curve)
                print(f"      {method:<6} m={m_n:<3} D_f  {pretty}   "
                      f"(final D_r {curve[-1][2]:.2f})", flush=True)
                del model, xs, ys
            del model0
            if device.type == "cuda":
                torch.cuda.empty_cache()

        if rows:
            upsert_rows(rows, RESULTS_PATH, FIELDS, KEY)


def main():
    global FORGET_CLASSES, RESULTS_PATH, RELEARN_M
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["cifar10", "cifar20", "cifar100"])
    ap.add_argument("--methods", nargs="+", default=["gold", "ssd", "unact"])
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--retain-per-step", type=float, default=4.0,
                    help="Retain images per forget image in each relearning batch. "
                         "0 reproduces the degenerate forget-only protocol.")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--classes", type=int, nargs="+", default=None)
    ap.add_argument("--relearn-m", type=int, nargs="+", default=None)
    ap.add_argument("--results", default=RESULTS_PATH)
    args = ap.parse_args()
    if args.classes:
        FORGET_CLASSES = {d: list(args.classes) for d in args.datasets}
    if args.relearn_m:
        RELEARN_M = args.relearn_m
    RESULTS_PATH = args.results

    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = cifar_common.pick_device(args.device)
    print(f"E7 relearn probe on {cifar_common.device_label(device)}  "
          f"lr={args.lr}  m={RELEARN_M}  steps={EVAL_STEPS}")
    done = existing_keys(RESULTS_PATH, KEY)
    print(f"{len(done)} row(s) already recorded")
    for ds in args.datasets:
        run_dataset(ds, args, device, done)
    print("\nE7 complete.")


if __name__ == "__main__":
    main()
