"""
E5c: relearn-time on ViT-B/16. Does the ViT variant delete the class, or mute it?

The ViT port of E7 (experiments/e7_relearn.py), with the same protocol so the two
are directly comparable: fine-tune each unlearned model on m forget-class train
images mixed with 4m retain images (so recovery cannot come from collapsing onto
the class), and record D_f / D_r after each step in EVAL_STEPS. The reference is
the retrain gold model, which never saw the class.

Methods and configurations are passed in rather than hard-coded, because on ViT
they come from the E5/E5b selections:
  --ssd   ALPHA,LAMBDA,BATCH          e.g. 5,0.1,128
  --unact SCOPE,P,GAMMA,K             e.g. ffn_dla_last4,85,0.1,5
UnAct profiles on the whole forget class (profile_n 0), as in E5b stage 2.

Evaluation uses the E5 grid shortcuts (bf16, a seeded 2000-image retain subset,
the full forget test split): this probe compares recovery *curves* between
methods, so it needs the same yardstick for every row, not table-grade D_r.

Usage:
  python experiments/e5c_vit_relearn.py --ssd 5,0.1,128 --unact ffn_dla_last4,85,0.1,5
"""
from __future__ import annotations

import argparse
import copy
import os
import subprocess
import sys

import torch
import torch.nn as nn

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_REPO_ROOT, os.path.join(_REPO_ROOT, "Our_Method"),
           os.path.join(_REPO_ROOT, "SSD"), os.path.join(_REPO_ROOT, "experiments")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from ssd import ParameterPerturber  # noqa: E402
from unlearn_vit import unlearn_vit  # noqa: E402
from e5_vit import accuracy, load_base  # noqa: E402

from unlearn_lib import manifest  # noqa: E402
from unlearn_lib.io import existing_keys, upsert_rows  # noqa: E402
from unlearn_lib.splits import build_split_indices, subsample_indices  # noqa: E402
from unlearn_lib.vit import ARCH, FAMILY, GpuBatches, get_vit_datasets, load_vit  # noqa: E402

RESULTS_PATH = os.path.join(_REPO_ROOT, "results", "e5c_vit_relearn.csv")
FIELDS = ["dataset", "arch", "forget_class", "method", "config", "relearn_m",
          "retain_per_step", "step", "seed", "lr", "d_f", "d_r", "d_f_at_step0",
          "baseline_d_f", "recovery_pct", "eval_retain_n", "eval_dtype",
          "eval_device", "git_commit"]
KEY = ["dataset", "arch", "forget_class", "method", "config", "relearn_m", "step",
       "seed", "lr"]

SSD_FIXED = {"lower_bound": 1, "exponent": 1, "magnitude_diff": None,
             "min_layer": -1, "max_layer": -1, "forget_threshold": 1}
FORGET_CLASSES = {"cifar10": [0, 2, 3, 5, 8]}
RELEARN_M = [5, 25]
EVAL_STEPS = [0, 1, 2, 5, 10, 20, 50]


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       cwd=_REPO_ROOT, text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def make_model(method, cfg, dataset, cls, base, train_set, idx, device):
    if method == "gold":
        path = manifest.lookup("retain_gold", FAMILY, ARCH, dataset, cls)
        if path is None:
            return None
        model, _ = load_vit(path, device=device)
        return model
    if method == "ssd":
        a, lam, bs = cfg
        model = copy.deepcopy(base).eval()
        pdr = ParameterPerturber(model, torch.optim.SGD(model.parameters(), lr=0.1), device,
                                 dict(SSD_FIXED, dampening_constant=lam, selection_weighting=a))
        fb = GpuBatches(train_set, idx["forget_train"], device, batch_size=bs)
        db = GpuBatches(train_set, idx["full_train"], device, batch_size=bs)
        pdr.modify_weight(pdr.calc_importance(db), pdr.calc_importance(fb))
        return model
    if method == "unact":
        scope, p, g, k = cfg
        model = copy.deepcopy(base)
        fb = GpuBatches(train_set, idx["forget_train"], device, batch_size=128)
        unlearn_vit(model, fb, device, scope=scope, percentile=p, gamma=g, iters=k,
                    token_pool="cls")
        return model
    raise ValueError(method)


def run(args, device, done):
    dataset = "cifar10"
    train_set, test_set = get_vit_datasets(dataset, download=False)
    base = load_base(dataset, device)
    commit = _git_commit()
    dev_name = torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"
    configs = {"gold": None, "ssd": args.ssd, "unact": args.unact}
    cfg_str = {"gold": "retrain",
               "ssd": "a{}_l{}_b{}".format(*args.ssd),
               "unact": "{}_p{}_g{}_k{}".format(*args.unact)}

    for cls in args.classes or FORGET_CLASSES[dataset]:
        idx = build_split_indices(train_set, test_set, cls)
        ev_f = GpuBatches(test_set, idx["forget_test"], device, batch_size=256)
        ev_r = GpuBatches(test_set, subsample_indices(idx["retain_test"], n=2000,
                                                      seed=args.seed),
                          device, batch_size=256)
        baseline_df = accuracy(base, ev_f, device)
        print(f"\n  === {dataset} class {cls} ===  baseline D_f {baseline_df:.2f}", flush=True)

        for method in args.methods:
            pending = [m for m in RELEARN_M
                       if not all((dataset, ARCH, str(cls), method, cfg_str[method], str(m),
                                   str(s), str(args.seed), str(args.lr)) in done
                                  for s in EVAL_STEPS)]
            if not pending:
                continue
            model0 = make_model(method, configs[method], dataset, cls, base,
                                train_set, idx, device)
            if model0 is None:
                print(f"      {method}: unavailable", flush=True)
                continue
            rows = []
            for m_n in pending:
                sub = subsample_indices(idx["forget_train"], n=m_n, seed=args.seed)
                n_ret = int(round(args.retain_per_step * m_n))
                g = torch.Generator().manual_seed(args.seed)
                ridx = idx["retain_train"][
                    torch.randperm(len(idx["retain_train"]), generator=g)[:n_ret]]
                both = torch.cat([sub, ridx])
                # One fixed batch, re-used every step, exactly as in E7.
                xs, ys = next(iter(GpuBatches(train_set, both, device,
                                              batch_size=len(both))))

                model = copy.deepcopy(model0)
                opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
                crit = nn.CrossEntropyLoss()
                d_f0 = accuracy(model, ev_f, device)
                curve = []
                for step in range(max(EVAL_STEPS) + 1):
                    if step in EVAL_STEPS:
                        df, dr = accuracy(model, ev_f, device), accuracy(model, ev_r, device)
                        curve.append((step, df, dr))
                        rows.append(dict(
                            dataset=dataset, arch=ARCH, forget_class=cls, method=method,
                            config=cfg_str[method], relearn_m=m_n,
                            retain_per_step=args.retain_per_step, step=step,
                            seed=args.seed, lr=args.lr, d_f=f"{df:.4f}", d_r=f"{dr:.4f}",
                            d_f_at_step0=f"{d_f0:.4f}", baseline_d_f=f"{baseline_df:.4f}",
                            recovery_pct=f"{100.0*df/max(baseline_df,1e-9):.2f}",
                            eval_retain_n=2000, eval_dtype="bf16",
                            eval_device=dev_name, git_commit=commit))
                    if step == max(EVAL_STEPS):
                        break
                    model.train()
                    opt.zero_grad(set_to_none=True)
                    with torch.autocast("cuda", dtype=torch.bfloat16,
                                        enabled=device.type == "cuda"):
                        loss = crit(model(xs), ys)
                    loss.backward()
                    opt.step()
                pretty = "  ".join(f"s{s}:{df:.1f}" for s, df, _ in curve)
                print(f"      {method:<6} m={m_n:<3} D_f  {pretty}   "
                      f"(final D_r {curve[-1][2]:.2f})", flush=True)
                del model, xs, ys
            upsert_rows(rows, args.results, FIELDS, KEY)
            del model0
            if device.type == "cuda":
                torch.cuda.empty_cache()


def _parse(spec, casts):
    parts = spec.split(",")
    if len(parts) != len(casts):
        raise argparse.ArgumentTypeError(f"expected {len(casts)} comma-separated values")
    return tuple(c(v) for c, v in zip(casts, parts))


def main():
    global RELEARN_M
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", nargs="+", default=["gold", "ssd", "unact"])
    ap.add_argument("--ssd", required=True, type=lambda s: _parse(s, (float, float, int)),
                    metavar="ALPHA,LAMBDA,BATCH")
    ap.add_argument("--unact", required=True,
                    type=lambda s: _parse(s, (str, float, float, int)),
                    metavar="SCOPE,P,GAMMA,K")
    ap.add_argument("--classes", type=int, nargs="+", default=None)
    ap.add_argument("--relearn-m", type=int, nargs="+", default=None)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--retain-per-step", type=float, default=4.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--results", default=RESULTS_PATH)
    args = ap.parse_args()
    if args.relearn_m:
        RELEARN_M = args.relearn_m

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    print(f"E5c ViT relearn on {device}  ssd={args.ssd}  unact={args.unact}  m={RELEARN_M}")
    done = existing_keys(args.results, KEY)
    print(f"{len(done)} row(s) already recorded")
    run(args, device, done)
    print("\nE5c complete.")


if __name__ == "__main__":
    main()
