"""
E5d: forget-set size on ViT-B/16 -- the ViT counterpart of E3.

Same semantics as experiments/e3_forget_size.py: the whole class is forgotten and
evaluated; only the number n of forget images the method is SHOWN varies. SSD's
denominator importance over the full training set does not depend on n, so it is
computed once per class and reused at every n. Both methods run at their selected
ViT configurations with no re-tuning:
  UnAct  ffn_dla_last4, p=85, gamma=0.1, k=5, token_pool=cls
  SSD    alpha=5, lambda=0.1, importance batch 128
Evaluation: fp32, full retain test split, as in the ViT main table.

Usage: python experiments/e5d_vit_forget_size.py --device cuda:0
"""
from __future__ import annotations

import argparse
import copy
import os
import sys

import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_REPO_ROOT, os.path.join(_REPO_ROOT, "Our_Method"),
           os.path.join(_REPO_ROOT, "SSD"), os.path.join(_REPO_ROOT, "experiments")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from ssd import ParameterPerturber  # noqa: E402
from unlearn_vit import unlearn_vit  # noqa: E402
from e5_vit import build_eval_batches, gold_for, load_base, make_eval  # noqa: E402

from unlearn_lib.io import existing_keys, upsert_rows  # noqa: E402
from unlearn_lib.splits import build_split_indices, subsample_indices  # noqa: E402
from unlearn_lib.timing import time_call  # noqa: E402
from unlearn_lib.vit import ARCH, GpuBatches, get_vit_datasets  # noqa: E402

RESULTS_PATH = os.path.join(_REPO_ROOT, "results", "e5d_vit_forget_size.csv")
FIELDS = ["dataset", "arch", "forget_class", "method", "config", "forget_n",
          "forget_available", "seed", "eval_retain_n", "eval_dtype",
          "post_overall", "post_retain", "post_forget", "gold_retain", "gold_forget",
          "gap_retain", "score", "method_time_s", "eval_device", "git_commit"]
KEY = ["dataset", "arch", "forget_class", "method", "config", "forget_n", "seed",
       "eval_retain_n", "eval_dtype"]

FORGET_N = [1, 5, 10, 25, 100, 500, 0]  # 0 = the whole class
FORGET_CLASSES = [0, 2, 3, 5, 8]
UNACT = ("ffn_dla_last4", 85.0, 0.1, 5)
SSD = (5.0, 0.1, 128)
SSD_FIXED = {"lower_bound": 1, "exponent": 1, "magnitude_diff": None,
             "min_layer": -1, "max_layer": -1, "forget_threshold": 1}


def _git_commit() -> str:
    import subprocess
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       cwd=_REPO_ROOT, text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--classes", type=int, nargs="+", default=None)
    ap.add_argument("--methods", nargs="+", default=["unact", "ssd"])
    ap.add_argument("--eval-retain-n", type=int, default=0)
    ap.add_argument("--eval-dtype", default="fp32", choices=["bf16", "fp32"])
    ap.add_argument("--eval-batch-size", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--results", default=RESULTS_PATH)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    dataset = "cifar10"
    train_set, test_set = get_vit_datasets(dataset, download=False)
    base = load_base(dataset, device)
    commit = _git_commit()
    dev_name = torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"
    done = existing_keys(args.results, KEY)
    with_overall = args.eval_retain_n == 0
    cfg = {"unact": "{}_p{}_g{}_k{}".format(*UNACT), "ssd": "a{}_l{}_b{}".format(*SSD)}
    print(f"E5d ViT forget-set size on {dev_name}  n={FORGET_N}  {len(done)} rows recorded")

    for cls in args.classes or FORGET_CLASSES:
        idx = build_split_indices(train_set, test_set, cls)
        ev = build_eval_batches(test_set, idx, device, args)
        gold_r, gold_f = gold_for(dataset, cls, ev, device, args.eval_dtype)
        available = len(idx["forget_train"])
        print(f"\n  === class {cls} ===  gold {gold_r:.2f}/{gold_f:.2f}", flush=True)

        def pending(m):
            return [n for n in FORGET_N
                    if (dataset, ARCH, str(cls), m, cfg[m],
                        str(available if n == 0 else min(n, available)), str(args.seed),
                        str(args.eval_retain_n), args.eval_dtype) not in done]

        oimp = t_full = None
        if "ssd" in args.methods and pending("ssd"):
            probe = copy.deepcopy(base).eval()
            pr = ParameterPerturber(probe, torch.optim.SGD(probe.parameters(), lr=0.1), device,
                                    dict(SSD_FIXED, dampening_constant=SSD[1],
                                         selection_weighting=SSD[0]))
            db = GpuBatches(train_set, idx["full_train"], device, batch_size=SSD[2])
            oimp, t_full = time_call(pr.calc_importance, db, device=device)
            print(f"      SSD full-D importance {t_full:.0f}s (shared across n)", flush=True)
            del probe, pr, db

        rows = []
        for method in args.methods:
            for n in pending(method):
                n_used = available if n == 0 else min(n, available)
                sub = idx["forget_train"] if n == 0 else \
                    subsample_indices(idx["forget_train"], n=n_used, seed=args.seed)
                m = copy.deepcopy(base)
                if method == "unact":
                    fb = GpuBatches(train_set, sub, device, batch_size=min(128, n_used))
                    scope, p, g, k = UNACT
                    _, t = time_call(unlearn_vit, m, fb, device, scope=scope, percentile=p,
                                     gamma=g, iters=k, token_pool="cls", device=device)
                else:
                    m.eval()
                    p2 = ParameterPerturber(m, torch.optim.SGD(m.parameters(), lr=0.1), device,
                                            dict(SSD_FIXED, dampening_constant=SSD[1],
                                                 selection_weighting=SSD[0]))
                    fb = GpuBatches(train_set, sub, device, batch_size=min(SSD[2], n_used))
                    fimp, t_f = time_call(p2.calc_importance, fb, device=device)
                    _, t_m = time_call(p2.modify_weight, oimp, fimp, device=device)
                    t = t_full + t_f + t_m
                    del p2, fimp
                do, dr, df = make_eval(m, ev, device, args.eval_dtype, with_overall)
                sc = abs(dr - gold_r) + abs(df - gold_f)
                rows.append(dict(
                    dataset=dataset, arch=ARCH, forget_class=cls, method=method,
                    config=cfg[method], forget_n=n_used, forget_available=available,
                    seed=args.seed, eval_retain_n=args.eval_retain_n,
                    eval_dtype=args.eval_dtype,
                    post_overall="" if do == "" else f"{do:.4f}",
                    post_retain=f"{dr:.4f}", post_forget=f"{df:.4f}",
                    gold_retain=f"{gold_r:.4f}", gold_forget=f"{gold_f:.4f}",
                    gap_retain=f"{dr-gold_r:+.4f}", score=f"{sc:.4f}",
                    method_time_s=f"{t:.3f}", eval_device=dev_name, git_commit=commit))
                print(f"      n={n_used:<5} {method:<6} D_r {dr:6.2f} D_f {df:6.2f} "
                      f"score {sc:7.3f}  t {t:6.1f}s", flush=True)
                del m
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            upsert_rows(rows, args.results, FIELDS, KEY)
            rows = []
        del oimp, ev
    print("\nE5d complete.")


if __name__ == "__main__":
    main()
