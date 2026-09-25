"""
E6: the UnAct hyperparameter grid -- p x gamma x k, per dataset and forget class.

This is the counterpart to E1: SSD gets a per-dataset search over its grid,
and UnAct gets a search of comparable size over its own, with the same forget
classes, selection rule and gold models.

It also produces two things the paper needs independently:

  * sensitivity to the attenuation strength gamma.
  * the cumulative coverage Gamma_k. Method-section Eq. (8) claims iteration
    *broadens* the modified set rather than deepening it, bounded by
    min(1, k(1-p/100)). This measures the realised value.

Efficiency
----------
Iterations are nested: a k=20 run passes through the k=1, 5 and 10 states on its
way. So each (p, gamma) is run once to max(k) and evaluated at every k in the
grid, cutting the k axis 4x -- 25 runs per (dataset, class) instead of 100.

Usage:
    python experiments/e6_unact_grid.py --datasets cifar10 --device cuda:0
"""
from __future__ import annotations

import argparse
import copy
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
from unlearn_lib.metrics import accuracy  # noqa: E402
from unlearn_lib.splits import build_split_indices  # noqa: E402
from unlearn_lib.timing import time_call  # noqa: E402

RESULTS_PATH = os.path.join(_REPO_ROOT, "results", "e6_unact_grid.csv")
FIELDS = [
    "dataset", "arch", "scope", "forget_class", "percentile", "gamma", "iters", "seed",
    "base_overall", "base_retain", "base_forget",
    "post_overall", "post_retain", "post_forget",
    "gold_retain", "gold_forget", "gap_retain",
    "units_per_round", "cum_units", "cum_coverage_pct", "coverage_bound_pct",
    "overlap_prev", "params_damped_pct", "method_time_s", "eval_device", "git_commit",
]
KEY = ["dataset", "arch", "scope", "forget_class", "percentile", "gamma", "iters", "seed"]

# Pre-registered grid (paper, appendix).
PERCENTILES = [50.0, 75.0, 90.0, 95.0, 99.0]
GAMMAS = [0.5, 0.3, 0.1, 0.03, 0.01]
ITERS = [1, 5, 10, 20]
SCOPE = "layer4_last"  # the configuration the CIFAR results use

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


def unlearn_tracked(model, forget_loader, device, percentile, gamma, max_iters,
                    eval_at, evaluate_fn):
    """Run UnAct for max_iters rounds, recording state at each k in eval_at.

    Semantics are identical to Our_Method/unlearn_cnn.py::unlearn_blocks with
    scope='layer4_last': profile the final block's output channels, select the
    top (100-p)%, attenuate conv2 / bn2 affine / the classifier columns. The only
    additions are intermediate evaluation and mask bookkeeping.
    """
    block = model.layer4[-1]
    n_units = block.conv2.weight.shape[0]
    seen = torch.zeros(n_units, dtype=torch.bool, device=device)
    prev = None
    out = {}
    elapsed = 0.0

    for t in range(1, max_iters + 1):
        def one_round():
            nonlocal prev, seen
            act = profile_block_activations(model, block, forget_loader, device)
            mask = _mask_from(act, percentile)
            ov = int((mask & prev).sum()) if prev is not None else -1
            seen |= mask
            with torch.no_grad():
                block.conv2.weight[mask] *= gamma
                block.bn2.weight[mask] *= gamma
                block.bn2.bias[mask] *= gamma
                model.fc.weight[:, mask] *= gamma
            prev = mask
            return int(mask.sum()), ov

        (n_sel, overlap), dt = time_call(one_round, device=device)
        elapsed += dt
        if t in eval_at:
            out[t] = {
                "metrics": evaluate_fn(model),
                "units_per_round": n_sel,
                "cum_units": int(seen.sum()),
                "cum_coverage_pct": 100.0 * int(seen.sum()) / n_units,
                "overlap_prev": overlap,
                "time_s": elapsed,
            }
    return out


def run_dataset(dataset, args, device, done_keys):
    train_set, test_set = cifar_common.get_datasets(dataset, train_augment=False,
                                                    download=False)
    base_model, _ = cifar_common.load_model(cifar_common.checkpoint_path(dataset))
    commit = _git_commit()
    dev_name = torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"

    for cls in FORGET_CLASSES[dataset]:
        idx = build_split_indices(train_set, test_set, cls)
        forget_loader = make_loader(train_set, idx["forget_train"],
                                    batch_size=args.batch_size, workers=args.workers)
        ev = {k: make_loader(test_set, idx[v], batch_size=args.eval_batch_size,
                             workers=args.workers)
              for k, v in {"valid": "test_all", "retain_valid": "retain_test",
                           "forget_valid": "forget_test"}.items()}

        base = (accuracy(base_model, ev["valid"], device),
                accuracy(base_model, ev["retain_valid"], device),
                accuracy(base_model, ev["forget_valid"], device))

        from unlearn_lib import manifest
        gold_path = manifest.lookup("retain_gold", "cifar", "resnet18", dataset, cls)
        if gold_path:
            gm, _ = cifar_common.load_model(gold_path)
            gold_r = accuracy(gm, ev["retain_valid"], device)
            gold_f = accuracy(gm, ev["forget_valid"], device)
            del gm
        else:
            gold_r = gold_f = None

        print(f"\n  === {dataset} class {cls} ===  base "
              f"{base[0]:.2f}/{base[1]:.2f}/{base[2]:.2f}  gold "
              f"{'n/a' if gold_r is None else f'{gold_r:.2f}/{gold_f:.2f}'}", flush=True)

        rows = []
        for pct in PERCENTILES:
            for gamma in GAMMAS:
                need = [k for k in ITERS
                        if (dataset, "resnet18", SCOPE, str(cls), str(pct), str(gamma),
                            str(k), str(args.seed)) not in done_keys]
                if not need:
                    continue
                torch.manual_seed(args.seed)
                model = copy.deepcopy(base_model)
                snap = {n: p.detach().clone() for n, p in model.named_parameters()}

                def ev_fn(m):
                    return (accuracy(m, ev["valid"], device),
                            accuracy(m, ev["retain_valid"], device),
                            accuracy(m, ev["forget_valid"], device))

                res = unlearn_tracked(model, forget_loader, device, pct, gamma,
                                      max(ITERS), set(ITERS), ev_fn)
                for k in need:
                    r = res[k]
                    post = r["metrics"]
                    changed = sum(int((p.detach() != snap[n].to(p.device)).sum())
                                  for n, p in model.named_parameters())
                    total = sum(p.numel() for p in model.parameters())
                    rows.append({
                        "dataset": dataset, "arch": "resnet18", "scope": SCOPE,
                        "forget_class": cls, "percentile": pct, "gamma": gamma,
                        "iters": k, "seed": args.seed,
                        "base_overall": f"{base[0]:.4f}", "base_retain": f"{base[1]:.4f}",
                        "base_forget": f"{base[2]:.4f}",
                        "post_overall": f"{post[0]:.4f}", "post_retain": f"{post[1]:.4f}",
                        "post_forget": f"{post[2]:.4f}",
                        "gold_retain": "" if gold_r is None else f"{gold_r:.4f}",
                        "gold_forget": "" if gold_f is None else f"{gold_f:.4f}",
                        "gap_retain": "" if gold_r is None else f"{post[1]-gold_r:+.4f}",
                        "units_per_round": r["units_per_round"],
                        "cum_units": r["cum_units"],
                        "cum_coverage_pct": f"{r['cum_coverage_pct']:.2f}",
                        "coverage_bound_pct": f"{min(100.0, k*(100.0-pct)):.2f}",
                        "overlap_prev": r["overlap_prev"],
                        "params_damped_pct": f"{100.0*changed/total:.4f}",
                        "method_time_s": f"{r['time_s']:.3f}",
                        "eval_device": dev_name, "git_commit": commit,
                    })
                best = res[max(ITERS)]
                print(f"      p={pct:<5} g={gamma:<5} -> k=20: "
                      f"{best['metrics'][1]:6.2f}/{best['metrics'][2]:6.2f}  "
                      f"cov {best['cum_coverage_pct']:5.1f}% "
                      f"(bound {min(100.0, 20*(100.0-pct)):5.1f}%)", flush=True)
                del model, snap

            if rows:
                upsert_rows(rows, RESULTS_PATH, FIELDS, KEY)
                rows = []
        if device.type == "cuda":
            torch.cuda.empty_cache()


def main():
    global PERCENTILES, GAMMAS, ITERS, FORGET_CLASSES, RESULTS_PATH, SCOPE
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["cifar10", "cifar20", "cifar100"])
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--eval-batch-size", type=int, default=512)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--percentiles", type=float, nargs="+", default=None)
    ap.add_argument("--gammas", type=float, nargs="+", default=None)
    ap.add_argument("--iters", type=int, nargs="+", default=None)
    ap.add_argument("--classes", type=int, nargs="+", default=None)
    ap.add_argument("--scope", default=SCOPE)
    ap.add_argument("--results", default=RESULTS_PATH)
    args = ap.parse_args()

    if args.percentiles:
        PERCENTILES = args.percentiles
    if args.gammas:
        GAMMAS = args.gammas
    if args.iters:
        ITERS = args.iters
    if args.classes:
        FORGET_CLASSES = {d: list(args.classes) for d in args.datasets}
    SCOPE = args.scope
    RESULTS_PATH = args.results

    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = cifar_common.pick_device(args.device)
    print(f"E6 UnAct grid on {cifar_common.device_label(device)}")
    print(f"grid: {len(PERCENTILES)} p x {len(GAMMAS)} gamma x {len(ITERS)} k, scope={SCOPE}")

    done = existing_keys(RESULTS_PATH, KEY)
    print(f"{len(done)} config(s) already recorded")
    for ds in args.datasets:
        run_dataset(ds, args, device, done)
    print("\nE6 complete.")


if __name__ == "__main__":
    main()
