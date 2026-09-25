"""
E5 (baseline half): Selective Synaptic Dampening on the same ViT-B/16 checkpoints.

`SSD/ssd.py::ParameterPerturber` is architecture-agnostic -- it iterates
`named_parameters()` and never mentions a layer type -- so it runs unchanged.
Only the data path changes: `unlearn_lib.vit.GpuBatches` supplies 224x224
batches, the same ones UnAct sees.

Grid
----
alpha in {2, 5, 10, 15, 25, 50} x lambda in {0.1, 0.5, 1}. SSD's own paper uses
alpha=5, lambda=1 for ViT class unlearning (against alpha=10 for ResNet-18), so
the grid brackets their ViT default by 2.5x in both directions.

**Disclosed asymmetry.** E1 swept SSD's importance batch size as a third axis,
because ssd.py:122 divides accumulated squared gradients by the number of
*batches*, making the scale of alpha depend on it. Here the batch size is fixed
(--batch-sizes, default 128): a single importance pass over 50,000 224x224
images costs ~8 minutes on these cards, so three batch sizes x 5 classes would
be 2 GPU-hours of importance passes alone. The 25x alpha range partly absorbs
the missing axis, but this is a real reduction relative to the ResNet protocol
and is recorded per row.

Efficiency: alpha and lambda enter only `modify_weight`, so importances are
computed once per (dataset, class, batch size) and replayed against all 18
pairs -- 1 importance pass instead of 18. Reported `method_time_s` is still the
true single-run cost (importance + modify), not the amortised one.

Usage:
    python experiments/e5_vit_ssd.py --datasets cifar10 --classes 3 --device cuda:0
"""
from __future__ import annotations

import argparse
import copy
import os
import subprocess
import sys

import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_REPO_ROOT, os.path.join(_REPO_ROOT, "SSD")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from ssd import ParameterPerturber  # noqa: E402  (SSD/ssd.py, unchanged)

from unlearn_lib import manifest, paths  # noqa: E402
from unlearn_lib.io import existing_keys, upsert_rows  # noqa: E402
from unlearn_lib.splits import build_split_indices, subsample_indices  # noqa: E402
from unlearn_lib.timing import time_call  # noqa: E402
from unlearn_lib.vit import ARCH, FAMILY, GpuBatches, get_vit_datasets, load_vit  # noqa: E402

sys.path.insert(0, os.path.join(_REPO_ROOT, "experiments"))
from e5_vit import _f, accuracy, build_eval_batches, gold_for, make_eval  # noqa: E402

RESULTS_PATH = os.path.join(_REPO_ROOT, "results", "e5_vit_ssd.csv")
FIELDS = [
    "dataset", "arch", "forget_class", "alpha", "lambda", "batch_size", "seed",
    "eval_retain_n", "eval_dtype",
    "base_overall", "base_retain", "base_forget",
    "post_overall", "post_retain", "post_forget",
    "gold_retain", "gold_forget", "gap_retain", "gap_forget", "score",
    "params_damped_pct", "importance_time_s", "modify_time_s", "method_time_s",
    "eval_device", "git_commit",
]
KEY = ["dataset", "arch", "forget_class", "alpha", "lambda", "batch_size", "seed",
       "eval_retain_n", "eval_dtype"]

ALPHAS = [2.0, 5.0, 10.0, 15.0, 25.0, 50.0]
LAMBDAS = [0.1, 0.5, 1.0]

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
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       cwd=_REPO_ROOT, text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def run_dataset(dataset, args, device, done):
    train_set, test_set = get_vit_datasets(dataset, download=False)
    path = paths.resolve_read(paths.baseline_rel(FAMILY, dataset, ARCH))
    if not os.path.exists(path):
        raise SystemExit(f"ViT baseline missing: {path}")
    base_model, _ = load_vit(path, device=device)
    commit = _git_commit()
    dev_name = torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"

    for cls in args.classes or FORGET_CLASSES[dataset]:
        idx = build_split_indices(train_set, test_set, cls)
        ev = build_eval_batches(test_set, idx, device, args)
        with_overall = args.eval_retain_n == 0
        base = make_eval(base_model, ev, device, args.eval_dtype, with_overall)
        gold_r, gold_f = gold_for(dataset, cls, ev, device, args.eval_dtype)
        print(f"\n  === {dataset} class {cls} ===  base "
              f"{base[1]:.2f}/{base[2]:.2f}  gold "
              f"{'n/a' if gold_r is None else f'{gold_r:.2f}/{gold_f:.2f}'}", flush=True)

        for bs in args.batch_sizes:
            pending = [(a, l) for a in args.alphas for l in args.lambdas
                       if (dataset, ARCH, str(cls), str(a), str(l), str(bs),
                           str(args.seed), str(args.eval_retain_n),
                           args.eval_dtype) not in done]
            if not pending:
                print(f"    bs={bs}: all configs done", flush=True)
                continue

            fb = GpuBatches(train_set, idx["forget_train"], device, batch_size=bs)
            db = GpuBatches(train_set, idx["full_train"], device, batch_size=bs)

            # Importances depend only on (class, batch size): computed once on
            # the unmodified model and replayed for every (alpha, lambda).
            probe = copy.deepcopy(base_model)
            probe.eval()
            pdr = ParameterPerturber(
                probe, torch.optim.SGD(probe.parameters(), lr=0.1), device,
                dict(SSD_FIXED, dampening_constant=1.0, selection_weighting=1.0),
                importance_device=torch.device("cpu") if args.importance_cpu else device,
            )
            (fimp, oimp), t_imp = time_call(
                lambda: (pdr.calc_importance(fb), pdr.calc_importance(db)), device=device)
            print(f"      bs={bs} importances: {t_imp:.1f}s "
                  f"({len(fb)}+{len(db)} batches)", flush=True)
            del probe, pdr
            if device.type == "cuda":
                torch.cuda.empty_cache()

            rows = []
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
                post = make_eval(model, ev, device, args.eval_dtype, with_overall)
                score = ("" if gold_r is None else
                         f"{abs(post[1]-gold_r)+abs(post[2]-gold_f):.4f}")
                rows.append({
                    "dataset": dataset, "arch": ARCH, "forget_class": cls,
                    "alpha": alpha, "lambda": lamb, "batch_size": bs, "seed": args.seed,
                    "eval_retain_n": args.eval_retain_n, "eval_dtype": args.eval_dtype,
                    "base_overall": _f(base[0]), "base_retain": f"{base[1]:.4f}",
                    "base_forget": f"{base[2]:.4f}",
                    "post_overall": _f(post[0]), "post_retain": f"{post[1]:.4f}",
                    "post_forget": f"{post[2]:.4f}",
                    "gold_retain": "" if gold_r is None else f"{gold_r:.4f}",
                    "gold_forget": "" if gold_f is None else f"{gold_f:.4f}",
                    "gap_retain": "" if gold_r is None else f"{post[1]-gold_r:+.4f}",
                    "gap_forget": "" if gold_f is None else f"{post[2]-gold_f:+.4f}",
                    "score": score,
                    "params_damped_pct": f"{100.0*damped/max(total,1):.4f}",
                    "importance_time_s": f"{t_imp:.3f}",
                    "modify_time_s": f"{t_mod:.3f}",
                    "method_time_s": f"{t_imp + t_mod:.3f}",
                    "eval_device": dev_name, "git_commit": commit,
                })
                print(f"      a={alpha:<5} l={lamb:<4} -> {post[1]:6.2f}/"
                      f"{post[2]:6.2f}  damped {100.0*damped/max(total,1):5.2f}%"
                      f"{'' if gold_r is None else f'  score {float(score):6.2f}'}", flush=True)
                del model, p2
            upsert_rows(rows, args.results, FIELDS, KEY)
            del fimp, oimp, fb, db
            if device.type == "cuda":
                torch.cuda.empty_cache()
        del ev
        if device.type == "cuda":
            torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["cifar10"])
    ap.add_argument("--classes", type=int, nargs="+", default=None)
    ap.add_argument("--alphas", type=float, nargs="+", default=ALPHAS)
    ap.add_argument("--lambdas", type=float, nargs="+", default=LAMBDAS)
    ap.add_argument("--batch-sizes", type=int, nargs="+", default=[128])
    ap.add_argument("--eval-retain-n", type=int, default=2000)
    ap.add_argument("--eval-dtype", default="bf16", choices=["bf16", "fp32"])
    ap.add_argument("--eval-batch-size", type=int, default=256)
    ap.add_argument("--importance-cpu", action="store_true",
                    help="Keep the two importance dicts (2 x 346 MB) on the CPU.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--results", default=RESULTS_PATH)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    print(f"E5 SSD on ViT, {device} "
          f"({torch.cuda.get_device_name(device) if device.type=='cuda' else 'cpu'})")
    print(f"grid: {len(args.alphas)} alphas x {len(args.lambdas)} lambdas x "
          f"{len(args.batch_sizes)} batch sizes {args.batch_sizes}")

    done = existing_keys(args.results, KEY)
    print(f"{len(done)} config(s) already recorded")
    for ds in args.datasets:
        run_dataset(ds, args, device, done)
    print("\nE5 SSD complete.")


if __name__ == "__main__":
    main()
