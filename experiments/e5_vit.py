"""
E5: UnAct on ViT-B/16 -- scope ablation and the (p, gamma, k) grid.

Two things are genuinely new at ViT scale and neither transfers from the ResNet
experiments:

  * **What the unit is.** There is no conv channel. The analogue is the FFN
    hidden neuron (3072 per block), so `p` is being applied to a 6x wider layer
    than the 512-channel ResNet block E6 tuned on. A percentile that selects 6
    units there selects 31 here, so p cannot be carried over and is re-searched.
  * **Where to apply it.** Twelve identical blocks offer a depth axis the ResNet
    ablation (fc_only / layer4_last / layer4_all / all_layers) only crudely
    approximated, plus a second unit kind -- the attention head.

Cost, and the two pre-registered efficiency choices
---------------------------------------------------
A ViT-B/16 forward pass at 224x224 costs ~70x a CIFAR ResNet-18 one. On the
L4/A10 GPUs this measures at 240-400 img/s, so the full pre-registered
design (6 scopes x 5 p x 5 gamma, k-nested to 20, on 5 forget classes, full
profiling set, full test set) costs ~78 GPU-hours. Two reductions were fixed
**before any E5 result was inspected**, and both are recorded in the CSV so any
row's provenance is explicit:

  1. `--profile-n 500`: activations are profiled on a seeded 500-image subset of
     the forget class rather than all 5000. E3 established that UnAct is within
     2x of its best score from n=25 upwards on CIFAR-10, so 500 is 20x inside
     the regime where forget-set size stopped mattering. Verified directly at
     the selected configuration by re-running with `--profile-n 0` (all).
  2. `--eval-retain-n 2000`: during the grid, retain accuracy is measured on a
     seeded 2000-image subset of the 9000-image retain test split (forget test
     is always complete). Selected configurations are re-run with
     `--eval-retain-n 0` for the reported table.

Neither touches the search space, the selection rule, or which configurations
are compared; both are recorded per row and undone for the reported numbers.

Evaluation dtype: bfloat16 autocast, recorded as `eval_dtype`. The ResNet
programme reports fp32, but fp32 ViT inference on these cards is ~3x slower and
would put the grid out of reach; the final table rows are re-run with
`--eval-dtype fp32` so the two are directly comparable.

Efficiency: iterations are nested exactly as in E6 -- one run to max(k),
evaluated at every k in the grid -- which cuts the k axis 4x.

Usage:
    # scope ablation + grid, CIFAR-10 class 3
    python experiments/e5_vit.py --datasets cifar10 --classes 3 --device cuda:0
    # confirm a selected configuration on all 5 classes, full data
    python experiments/e5_vit.py --datasets cifar10 --scopes ffn_last3 \
        --percentiles 99 --gammas 0.1 --iters 5 --profile-n 0 \
        --eval-retain-n 0 --eval-dtype fp32
    # the GELU attenuation measurement
    python experiments/e5_vit.py --mode attenuation --datasets cifar10 --classes 3
"""
from __future__ import annotations

import argparse
import copy
import os
import subprocess
import sys

import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_REPO_ROOT, os.path.join(_REPO_ROOT, "Our_Method")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from unlearn_cnn import _mask_from  # noqa: E402
from unlearn_vit import (  # noqa: E402
    ALL_SCOPES, SCOPES, infer_forget_class, measure_attenuation, n_units, profile,
    scope_rule, scope_spec, select_mask,
    attenuate_attn, attenuate_ffn, attenuate_head,
)

from unlearn_lib import manifest  # noqa: E402
from unlearn_lib.io import existing_keys, upsert_rows  # noqa: E402
from unlearn_lib.splits import build_split_indices, subsample_indices  # noqa: E402
from unlearn_lib.timing import time_call  # noqa: E402
from unlearn_lib.vit import (  # noqa: E402
    ARCH, FAMILY, GpuBatches, build_vit, get_vit_datasets, load_vit,
)
from unlearn_lib import paths  # noqa: E402

GRID_PATH = os.path.join(_REPO_ROOT, "results", "e5_vit_unact.csv")
ATTEN_PATH = os.path.join(_REPO_ROOT, "results", "e5_vit_attenuation.csv")

FIELDS = [
    "dataset", "arch", "scope", "token_pool", "forget_class", "percentile", "gamma",
    "iters", "seed", "profile_n", "eval_retain_n", "eval_dtype",
    "base_overall", "base_retain", "base_forget",
    "post_overall", "post_retain", "post_forget",
    "gold_retain", "gold_forget", "gap_retain", "gap_forget", "score",
    "n_blocks_touched", "units_total", "units_per_round", "cum_units",
    "cum_coverage_pct", "coverage_bound_pct", "overlap_prev",
    "params_damped_pct", "method_time_s", "eval_device", "git_commit",
]
KEY = ["dataset", "arch", "scope", "token_pool", "forget_class", "percentile",
       "gamma", "iters", "seed", "profile_n", "eval_retain_n", "eval_dtype"]

ATTEN_FIELDS = [
    "dataset", "arch", "scope", "token_pool", "forget_class", "percentile", "gamma",
    "seed", "profile_n", "block", "n_masked",
    "gelu_before", "gelu_after", "gelu_ratio", "gelu_expected",
    "contrib_before", "contrib_after", "contrib_ratio", "contrib_expected",
    "eval_device", "git_commit",
]
ATTEN_KEY = ["dataset", "arch", "scope", "token_pool", "forget_class", "percentile",
             "gamma", "seed", "profile_n", "block"]

# Pre-registered grid for E5 (see the module docstring for why p is re-searched).
PERCENTILES = [90.0, 95.0, 99.0, 99.5, 99.9]
GAMMAS = [0.5, 0.3, 0.1, 0.03, 0.01]
ITERS = [1, 5, 10, 20]

FORGET_CLASSES = {
    "cifar10": [0, 2, 3, 5, 8],
    "cifar20": [3, 4, 10, 14, 19],
    "cifar100": [3, 20, 51, 69, 85],
}
DATASET_CLASSES = {"cifar10": 10, "cifar20": 20, "cifar100": 100}


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       cwd=_REPO_ROOT, text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def _f(v):
    """Format an accuracy that may be absent (overall is skipped during the grid)."""
    return "" if v == "" else f"{v:.4f}"


@torch.no_grad()
def accuracy(model, batches, device, dtype="bf16") -> float:
    """Top-1 accuracy in percent over a GpuBatches iterable."""
    model.eval()
    correct = total = 0
    for x, y in batches:
        with torch.autocast("cuda", dtype=torch.bfloat16,
                            enabled=(dtype == "bf16" and device.type == "cuda")):
            pred = model(x).argmax(dim=1)
        correct += int((pred == y).sum())
        total += int(y.numel())
    return 100.0 * correct / max(total, 1)


def make_eval(model, ev, device, dtype, with_overall=True):
    """(overall, retain, forget) accuracies; overall is skipped during the grid.

    The selection rule only uses D_r and D_f. Overall accuracy would add the
    whole 10,000-image test split to every one of the grid's ~600 evaluations --
    4.3x the work for a column nothing is selected on -- so it is computed only
    for the full-evaluation rows that go in the table.
    """
    return ("" if not with_overall else accuracy(model, ev["all"], device, dtype),
            accuracy(model, ev["retain"], device, dtype),
            accuracy(model, ev["forget"], device, dtype))


def unlearn_tracked(model, scope, forget_batches, device, percentile, gamma,
                    max_iters, eval_at, evaluate_fn, token_pool, amp, damped_fn=None):
    """Run UnAct for max_iters rounds, recording state at each k in eval_at.

    Identical in semantics to Our_Method/unlearn_vit.unlearn_vit; the additions
    are intermediate evaluation and the per-block mask bookkeeping Eq. (8) needs.
    """
    kind, blocks_idx = scope_spec(scope)
    layers = list(model.encoder.layers)
    total_units = n_units(scope)
    keys = ["head"] if kind == "head" else list(blocks_idx)
    seen = {k: torch.zeros(total_units, dtype=torch.bool, device=device) for k in keys}
    prev = {k: None for k in keys}
    out, elapsed = {}, 0.0
    # DLA scopes infer the forget class once, from the unedited model; the time is
    # part of the method's cost.
    c = None
    if scope_rule(scope) == "dla":
        c, elapsed = time_call(infer_forget_class, model, forget_batches, device, amp,
                               device=device)

    for t in range(1, max_iters + 1):
        def one_round():
            acts = profile(model, scope, forget_batches, device, token_pool, amp, c)
            n_sel, overlap = 0, 0
            has_prev = all(prev[k] is not None for k in keys)
            for k in keys:
                a = acts if kind == "head" else acts[k]
                mask = select_mask(a, percentile, scope)
                if kind == "head":
                    attenuate_head(model, mask, gamma)
                elif kind == "ffn":
                    attenuate_ffn(layers[k], mask, gamma)
                else:
                    attenuate_attn(layers[k], mask, gamma)
                n_sel += int(mask.sum())
                if has_prev:
                    overlap += int((mask & prev[k]).sum())
                seen[k] |= mask
                prev[k] = mask
            return n_sel, (overlap if has_prev else -1)

        (n_sel, overlap), dt = time_call(one_round, device=device)
        elapsed += dt
        if t in eval_at:
            cum = sum(int(seen[k].sum()) for k in keys)
            out[t] = {
                "metrics": evaluate_fn(model),
                # Measured at this k, not at max(k): the modified set broadens
                # with iteration, so one value for every k row would be wrong.
                "params_damped_pct": None if damped_fn is None else damped_fn(model),
                "units_per_round": n_sel / len(keys),
                "cum_units": cum / len(keys),
                "cum_coverage_pct": 100.0 * cum / (len(keys) * total_units),
                "overlap_prev": overlap,
                "time_s": elapsed,
            }
    return out


def load_base(dataset, device):
    path = paths.resolve_read(paths.baseline_rel(FAMILY, dataset, ARCH))
    if not os.path.exists(path):
        raise SystemExit(
            f"ViT baseline missing: {path}\n"
            f'  Train it first:  scripts/launch.sh 0 vit-base-{dataset} python '
            f'"Baseline ViT Training/train_vit_cifar.py" --dataset {dataset} --device cuda:0')
    model, _ = load_vit(path, device=device)
    return model


def build_eval_batches(test_set, idx, device, args):
    retain_idx = idx["retain_test"]
    if args.eval_retain_n:
        retain_idx = subsample_indices(retain_idx, n=args.eval_retain_n, seed=args.seed)
    return {
        "all": GpuBatches(test_set, idx["test_all"], device, batch_size=args.eval_batch_size),
        "retain": GpuBatches(test_set, retain_idx, device, batch_size=args.eval_batch_size),
        "forget": GpuBatches(test_set, idx["forget_test"], device, batch_size=args.eval_batch_size),
    }


def gold_for(dataset, cls, ev, device, dtype):
    """Retrain gold (D_r, D_f), recomputed here rather than read from the
    manifest: the manifest's values were measured on whichever card trained the
    model, and a table must not mix devices."""
    path = manifest.lookup("retain_gold", FAMILY, ARCH, dataset, cls)
    if path is None:
        return None, None
    gm, _ = load_vit(path, device=device)
    r = accuracy(gm, ev["retain"], device, dtype)
    f = accuracy(gm, ev["forget"], device, dtype)
    del gm
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return r, f


def run_grid(dataset, args, device, done):
    train_set, test_set = get_vit_datasets(dataset, download=False)
    base_model = load_base(dataset, device)
    commit = _git_commit()
    dev_name = torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"
    amp = args.eval_dtype == "bf16"
    with_overall = args.eval_retain_n == 0

    for cls in args.classes or FORGET_CLASSES[dataset]:
        idx = build_split_indices(train_set, test_set, cls)
        prof_idx = idx["forget_train"]
        if args.profile_n:
            prof_idx = subsample_indices(prof_idx, n=args.profile_n, seed=args.seed)
        fb = GpuBatches(train_set, prof_idx, device, batch_size=args.batch_size)
        ev = build_eval_batches(test_set, idx, device, args)

        base = make_eval(base_model, ev, device, args.eval_dtype, with_overall)
        gold_r, gold_f = gold_for(dataset, cls, ev, device, args.eval_dtype)
        print(f"\n  === {dataset} class {cls} ===  profile n={fb.n_samples}  "
              f"eval retain n={ev['retain'].n_samples}  base "
              f"{base[1]:.2f}/{base[2]:.2f}  gold "
              f"{'n/a' if gold_r is None else f'{gold_r:.2f}/{gold_f:.2f}'}", flush=True)

        for scope in args.scopes:
            kind, blocks_idx = scope_spec(scope)
            n_blk = max(len(blocks_idx), 1)
            total_units = n_units(scope)
            rows = []
            for pct in args.percentiles:
                for gamma in args.gammas:
                    need = [k for k in args.iters
                            if (dataset, ARCH, scope, args.token_pool, str(cls), str(pct),
                                str(gamma), str(k), str(args.seed), str(args.profile_n),
                                str(args.eval_retain_n), args.eval_dtype) not in done]
                    if not need:
                        continue
                    torch.manual_seed(args.seed)
                    model = copy.deepcopy(base_model)
                    snap = {n: p.detach().clone() for n, p in model.named_parameters()}
                    total = sum(p.numel() for p in model.parameters())

                    def damped_fn(m, _snap=snap, _total=total):
                        changed = sum(int((p.detach() != _snap[n]).sum())
                                      for n, p in m.named_parameters())
                        return 100.0 * changed / _total

                    res = unlearn_tracked(
                        model, scope, fb, device, pct, gamma, max(args.iters),
                        set(args.iters),
                        lambda m: make_eval(m, ev, device, args.eval_dtype, with_overall),
                        args.token_pool, amp, damped_fn)
                    for k in need:
                        r = res[k]
                        post = r["metrics"]
                        score = ("" if gold_r is None else
                                 f"{abs(post[1]-gold_r)+abs(post[2]-gold_f):.4f}")
                        rows.append({
                            "dataset": dataset, "arch": ARCH, "scope": scope,
                            "token_pool": args.token_pool, "forget_class": cls,
                            "percentile": pct, "gamma": gamma, "iters": k,
                            "seed": args.seed, "profile_n": args.profile_n,
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
                            "n_blocks_touched": n_blk, "units_total": total_units,
                            "units_per_round": f"{r['units_per_round']:.2f}",
                            "cum_units": f"{r['cum_units']:.2f}",
                            "cum_coverage_pct": f"{r['cum_coverage_pct']:.2f}",
                            # Corrected Eq. (8): Gamma_k <= min(1, k|M|/n), not
                            # min(1, k(1-p/100)).
                            "coverage_bound_pct":
                                f"{min(100.0, k*r['units_per_round']/total_units*100.0):.2f}",
                            "overlap_prev": r["overlap_prev"],
                            "params_damped_pct": f"{r['params_damped_pct']:.4f}",
                            "method_time_s": f"{r['time_s']:.3f}",
                            "eval_device": dev_name, "git_commit": commit,
                        })
                    b = res[max(args.iters)]
                    print(f"      {scope:11s} p={pct:<5} g={gamma:<5} -> k={max(args.iters)}: "
                          f"{b['metrics'][1]:6.2f}/{b['metrics'][2]:6.2f}  "
                          f"cov {b['cum_coverage_pct']:5.1f}%  "
                          f"{r['time_s']:.0f}s", flush=True)
                    del model, snap
                if rows:
                    upsert_rows(rows, args.results, FIELDS, KEY)
                    rows = []
            if rows:
                upsert_rows(rows, args.results, FIELDS, KEY)
        del ev, fb
        if device.type == "cuda":
            torch.cuda.empty_cache()


def run_attenuation(dataset, args, device, done):
    """Measure the realised attenuation of one UnAct round on GELU units.

    Proposition 1 predicts gamma for the activation and gamma^2 for the unit's
    contribution, both exact for a positively homogeneous activation. GELU is
    not one, so this records what actually happens.
    """
    train_set, test_set = get_vit_datasets(dataset, download=False)
    base_model = load_base(dataset, device)
    commit = _git_commit()
    dev_name = torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"
    amp = args.eval_dtype == "bf16"
    rows = []

    for cls in args.classes or FORGET_CLASSES[dataset]:
        idx = build_split_indices(train_set, test_set, cls)
        prof_idx = idx["forget_train"]
        if args.profile_n:
            prof_idx = subsample_indices(prof_idx, n=args.profile_n, seed=args.seed)
        fb = GpuBatches(train_set, prof_idx, device, batch_size=args.batch_size)

        for scope in args.scopes:
            kind, blocks_idx = scope_spec(scope)
            if kind != "ffn":
                continue
            for pct in args.percentiles:
                for gamma in args.gammas:
                    k = (dataset, ARCH, scope, args.token_pool, str(cls), str(pct),
                         str(gamma), str(args.seed), str(args.profile_n),
                         str(blocks_idx[-1]))
                    if k in done:
                        continue
                    model = copy.deepcopy(base_model)
                    acts = profile(model, scope, fb, device, args.token_pool, amp)
                    b = blocks_idx[-1]
                    mask = _mask_from(acts[b], pct)
                    layers = list(model.encoder.layers)
                    for bb in blocks_idx:
                        attenuate_ffn(layers[bb], _mask_from(acts[bb], pct), gamma)
                    m = measure_attenuation(base_model, model, b, mask, fb, device, amp)
                    rows.append({
                        "dataset": dataset, "arch": ARCH, "scope": scope,
                        "token_pool": args.token_pool, "forget_class": cls,
                        "percentile": pct, "gamma": gamma, "seed": args.seed,
                        "profile_n": args.profile_n, "block": b,
                        "n_masked": int(mask.sum()),
                        "gelu_before": f"{m['gelu_before']:.6f}",
                        "gelu_after": f"{m['gelu_after']:.6f}",
                        "gelu_ratio": f"{m['gelu_ratio']:.6f}",
                        "gelu_expected": f"{gamma:.6f}",
                        "contrib_before": f"{m['contrib_before']:.6f}",
                        "contrib_after": f"{m['contrib_after']:.6f}",
                        "contrib_ratio": f"{m['contrib_ratio']:.6f}",
                        "contrib_expected": f"{gamma**2:.6f}",
                        "eval_device": dev_name, "git_commit": commit,
                    })
                    print(f"      {scope} p={pct} g={gamma}: gelu {m['gelu_ratio']:.4f} "
                          f"(expect {gamma}), contrib {m['contrib_ratio']:.5f} "
                          f"(expect {gamma**2:.4f})", flush=True)
                    del model
            upsert_rows(rows, args.results, ATTEN_FIELDS, ATTEN_KEY)
            rows = []
        del fb
        if device.type == "cuda":
            torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="grid", choices=["grid", "attenuation"])
    ap.add_argument("--datasets", nargs="+", default=["cifar10"], choices=list(DATASET_CLASSES))
    ap.add_argument("--classes", type=int, nargs="+", default=None)
    ap.add_argument("--scopes", nargs="+", default=list(SCOPES), choices=list(ALL_SCOPES))
    ap.add_argument("--percentiles", type=float, nargs="+", default=PERCENTILES)
    ap.add_argument("--gammas", type=float, nargs="+", default=GAMMAS)
    ap.add_argument("--iters", type=int, nargs="+", default=ITERS)
    ap.add_argument("--token-pool", default="mean", choices=["mean", "cls"])
    ap.add_argument("--profile-n", type=int, default=500,
                    help="Forget images used for activation profiling; 0 = the whole class.")
    ap.add_argument("--eval-retain-n", type=int, default=2000,
                    help="Retain-test images used for D_r; 0 = the whole split.")
    ap.add_argument("--eval-dtype", default="bf16", choices=["bf16", "fp32"])
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--eval-batch-size", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--results", default=None)
    args = ap.parse_args()

    if args.results is None:
        args.results = GRID_PATH if args.mode == "grid" else ATTEN_PATH

    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    print(f"E5 ViT ({args.mode}) on {device} "
          f"({torch.cuda.get_device_name(device) if device.type=='cuda' else 'cpu'})")
    print(f"scopes={args.scopes}  p={args.percentiles}  gamma={args.gammas}  k={args.iters}")
    print(f"profile_n={args.profile_n}  eval_retain_n={args.eval_retain_n}  "
          f"eval_dtype={args.eval_dtype}  -> {args.results}")

    key = KEY if args.mode == "grid" else ATTEN_KEY
    done = existing_keys(args.results, key)
    print(f"{len(done)} config(s) already recorded")

    for ds in args.datasets:
        (run_grid if args.mode == "grid" else run_attenuation)(ds, args, device, done)
    print(f"\nE5 {args.mode} complete.")


if __name__ == "__main__":
    main()
