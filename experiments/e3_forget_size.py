"""
E3: how many forget-set examples does each method actually need?

SSD's paper evaluates a single forget-set size (100 random CIFAR-10 samples,
0.2%) and names scaling this as future work; their Limitations section says the
same. So this axis is open.

Semantics (the "data-efficiency reading", fixed before the run): we still forget
the WHOLE class. We vary only how many class-c examples the method is *shown*.
D_r is every non-c image and D (for SSD's importance denominator) is the whole
training set -- neither changes with n. Evaluation is the standard full-class
protocol against the same retrain gold models. So the question is precisely
"how much evidence does the method need to locate the class?", not "what
happens if you forget less".

Why UnAct should win: its entire input is a per-unit mean over the forget set,
which is a 512-dimensional statistic. A mean converges in O(1/sqrt(n)), so it
should saturate at small n. SSD must estimate a per-PARAMETER Fisher diagonal --
11.7M numbers -- from the same n samples, and additionally needs a pass over all
50,000 training images regardless of n.

Fairness provisions:
  * SSD's full-D importance does not depend on n, so it is computed once per
    (dataset, class) and reused for every n. This is exactly the amortisation
    its paper claims, and withholding it would inflate SSD's cost 10-100x at
    small n.
  * At each n, SSD is additionally re-tuned over the alpha grid (replaying the
    cached importances, which is nearly free). Its per-batch-mean importance
    makes alpha batch-count sensitive, so holding alpha fixed as n shrinks would
    handicap it for reasons unrelated to data efficiency. Both the fixed-alpha
    and best-alpha curves are recorded.

Usage: python experiments/e3_forget_size.py --device cuda:0
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
from unlearn_lib.metrics import accuracy  # noqa: E402
from unlearn_lib.splits import build_split_indices  # noqa: E402
from unlearn_lib.timing import time_call  # noqa: E402

RESULTS_PATH = os.path.join(_REPO_ROOT, "results", "e3_forget_size.csv")
FIELDS = [
    "dataset", "arch", "forget_class", "method", "config", "forget_n",
    "forget_available", "seed",
    "post_overall", "post_retain", "post_forget",
    "gold_retain", "gold_forget", "gap_retain", "score",
    "method_time_s", "marginal_time_s", "params_damped_pct",
    "eval_device", "git_commit",
]
KEY = ["dataset", "arch", "forget_class", "method", "config", "forget_n", "seed"]

FORGET_N = [1, 5, 10, 25, 50, 100, 500, 1000, 0]  # 0 = all available

# Operating points selected by E1 / E6.
SSD_SELECTED = {"cifar10": (8.0, 1.0, 256), "cifar20": (15.0, 1.0, 512),
                "cifar100": (50.0, 0.5, 64)}
UNACT_SELECTED = {"cifar10": (99.0, 0.01, 20), "cifar20": (99.0, 0.1, 20),
                  "cifar100": (90.0, 0.3, 1)}
ALPHA_GRID = [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 15.0, 20.0, 30.0, 50.0]
FIXED = {"lower_bound": 1, "exponent": 1, "magnitude_diff": None,
         "min_layer": -1, "max_layer": -1, "forget_threshold": 1}
FORGET_CLASSES = {"cifar10": [0, 2, 3, 5, 8], "cifar20": [3, 4, 10, 14, 19],
                  "cifar100": [3, 20, 51, 69, 85]}


def _git_commit():
    import subprocess
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       cwd=_REPO_ROOT, text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def run_dataset(dataset, args, device, done):
    alpha0, lam, ssd_bs = SSD_SELECTED[dataset]
    p0, g0, k0 = UNACT_SELECTED[dataset]
    train_set, test_set = cifar_common.get_datasets(dataset, train_augment=False,
                                                    download=False)
    base, _ = cifar_common.load_model(cifar_common.checkpoint_path(dataset))
    commit = _git_commit()
    dev_name = torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"
    unact_cfg = f"p{p0}_g{g0}_k{k0}"

    for cls in FORGET_CLASSES[dataset]:
        full_idx = build_split_indices(train_set, test_set, cls)
        ev = {k: make_loader(test_set, full_idx[v], batch_size=args.eval_batch_size,
                             workers=args.workers)
              for k, v in {"valid": "test_all", "retain_valid": "retain_test",
                           "forget_valid": "forget_test"}.items()}
        gpath = manifest.lookup("retain_gold", "cifar", "resnet18", dataset, cls)
        if gpath:
            gm, _ = cifar_common.load_model(gpath)
            gold_r = accuracy(gm, ev["retain_valid"], device)
            gold_f = accuracy(gm, ev["forget_valid"], device)
            del gm
        else:
            gold_r = gold_f = None

        available = len(full_idx["forget_train"])
        print(f"\n  === {dataset} class {cls} ===  |D_f| available = {available}  "
              f"gold {gold_r:.2f}/{gold_f:.2f}", flush=True)

        # SSD's full-D importance is independent of n: compute once, reuse.
        probe = copy.deepcopy(base).eval()
        pr = ParameterPerturber(probe, torch.optim.SGD(probe.parameters(), lr=0.1),
                                device, dict(FIXED, dampening_constant=lam,
                                             selection_weighting=alpha0))
        full_loader = make_loader(train_set, full_idx["full_train"],
                                  batch_size=ssd_bs, workers=args.workers)
        oimp, t_full = time_call(pr.calc_importance, full_loader, device=device)
        print(f"      SSD full-D importance (shared across all n): {t_full:.1f}s", flush=True)
        del probe, pr, full_loader

        rows = []
        for n in FORGET_N:
            n_used = available if n == 0 else min(n, available)
            idx = build_split_indices(train_set, test_set, cls,
                                      forget_n=None if n == 0 else n, seed=args.seed)
            # Small n with a large batch size would give a single tiny batch;
            # cap the batch so the loader shape stays sane.
            f_ssd = make_loader(train_set, idx["forget_train"],
                                batch_size=min(ssd_bs, n_used), workers=0)
            f_una = make_loader(train_set, idx["forget_train"],
                                batch_size=min(256, n_used), workers=0)

            def score(dr, df):
                return (abs(dr - gold_r) + abs(df - gold_f)) if gold_r is not None else None

            # ---- UnAct at its selected configuration ----
            key = (dataset, "resnet18", str(cls), "unact", unact_cfg, str(n_used),
                   str(args.seed))
            if key not in done:
                m = copy.deepcopy(base)
                _, t = time_call(unlearn_blocks, m, f_una, device, scope="layer4_last",
                                 penalty_scale=g0, threshold_percentile=p0,
                                 iter_times=k0, verbose=False, device=device)
                dr, df = accuracy(m, ev["retain_valid"], device), accuracy(m, ev["forget_valid"], device)
                do = accuracy(m, ev["valid"], device)
                changed = sum(int((q.detach() != w.to(q.device)).sum())
                              for (nm, q), w in zip(m.named_parameters(),
                                                    [x.detach() for x in base.parameters()]))
                rows.append(dict(
                    dataset=dataset, arch="resnet18", forget_class=cls, method="unact",
                    config=unact_cfg, forget_n=n_used, forget_available=available,
                    seed=args.seed, post_overall=f"{do:.4f}", post_retain=f"{dr:.4f}",
                    post_forget=f"{df:.4f}",
                    gold_retain=f"{gold_r:.4f}", gold_forget=f"{gold_f:.4f}",
                    gap_retain=f"{dr-gold_r:+.4f}", score=f"{score(dr,df):.4f}",
                    method_time_s=f"{t:.3f}", marginal_time_s=f"{t:.3f}",
                    params_damped_pct=f"{100.0*changed/sum(q.numel() for q in m.parameters()):.4f}",
                    eval_device=dev_name, git_commit=commit))
                print(f"      n={n_used:<5} unact              D_r {dr:6.2f} D_f {df:6.2f} "
                      f"score {score(dr,df):6.3f}  t {t:5.2f}s", flush=True)
                del m

            # ---- SSD: forget importance at this n, then replay the alpha grid ----
            probe = copy.deepcopy(base).eval()
            pr = ParameterPerturber(probe, torch.optim.SGD(probe.parameters(), lr=0.1),
                                    device, dict(FIXED, dampening_constant=lam,
                                                 selection_weighting=alpha0))
            fimp, t_forget = time_call(pr.calc_importance, f_ssd, device=device)
            del probe, pr

            best = None
            for a in ALPHA_GRID:
                m = copy.deepcopy(base)
                p2 = ParameterPerturber(m, torch.optim.SGD(m.parameters(), lr=0.1),
                                        device, dict(FIXED, dampening_constant=lam,
                                                     selection_weighting=a))
                _, t_mod = time_call(p2.modify_weight, oimp, fimp, device=device)
                dr, df = accuracy(m, ev["retain_valid"], device), accuracy(m, ev["forget_valid"], device)
                do = accuracy(m, ev["valid"], device)
                sc = score(dr, df)
                cfg = f"a{a}_l{lam}_b{ssd_bs}"
                tag = "ssd" if a == alpha0 else "ssd_alphagrid"
                key = (dataset, "resnet18", str(cls), tag, cfg, str(n_used), str(args.seed))
                if key not in done:
                    rows.append(dict(
                        dataset=dataset, arch="resnet18", forget_class=cls, method=tag,
                        config=cfg, forget_n=n_used, forget_available=available,
                        seed=args.seed, post_overall=f"{do:.4f}", post_retain=f"{dr:.4f}",
                        post_forget=f"{df:.4f}", gold_retain=f"{gold_r:.4f}",
                        gold_forget=f"{gold_f:.4f}", gap_retain=f"{dr-gold_r:+.4f}",
                        score=f"{sc:.4f}",
                        # cold cost includes the shared full-D pass; marginal does not
                        method_time_s=f"{t_full + t_forget + t_mod:.3f}",
                        marginal_time_s=f"{t_forget + t_mod:.3f}",
                        params_damped_pct="", eval_device=dev_name, git_commit=commit))
                if best is None or sc < best[1]:
                    best = (a, sc, dr, df)
                del m, p2
            print(f"      n={n_used:<5} ssd a={alpha0:<5}(fixed)  best-alpha={best[0]:<5} "
                  f"D_r {best[2]:6.2f} D_f {best[3]:6.2f} score {best[1]:6.3f}  "
                  f"t_marginal {t_forget:5.2f}s", flush=True)
            del fimp

            if rows:
                upsert_rows(rows, RESULTS_PATH, FIELDS, KEY)
                rows = []

        del oimp
        if device.type == "cuda":
            torch.cuda.empty_cache()


def main():
    global FORGET_CLASSES, RESULTS_PATH, FORGET_N
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["cifar10", "cifar20", "cifar100"])
    ap.add_argument("--eval-batch-size", type=int, default=512)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--classes", type=int, nargs="+", default=None)
    ap.add_argument("--forget-n", type=int, nargs="+", default=None)
    ap.add_argument("--results", default=RESULTS_PATH)
    args = ap.parse_args()
    if args.classes:
        FORGET_CLASSES = {d: list(args.classes) for d in args.datasets}
    if args.forget_n:
        FORGET_N = args.forget_n
    RESULTS_PATH = args.results

    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = cifar_common.pick_device(args.device)
    print(f"E3 forget-set size on {cifar_common.device_label(device)}  n={FORGET_N}")
    done = existing_keys(RESULTS_PATH, KEY)
    print(f"{len(done)} row(s) already recorded")
    for ds in args.datasets:
        run_dataset(ds, args, device, done)
    print("\nE3 complete.")


if __name__ == "__main__":
    main()
