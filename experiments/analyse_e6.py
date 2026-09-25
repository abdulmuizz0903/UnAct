"""
Select UnAct's operating point from the E6 grid, and compare like-for-like
against the SSD point selected from E1.

Identical selection rule to analyse_e1.py, fixed before inspecting either grid:

    score = |D_r - gold_D_r| + |D_f - gold_D_f|,  averaged over the 5 forget classes

Same rule, same forget classes, same gold models, same evaluation. This is the
first comparison in which both methods have received a per-dataset search of
comparable size (SSD: 10 alpha x 3 lambda x 3 batch = 90 configs; UnAct: 5 p x
5 gamma x 4 k = 100 configs).

Also reports:
  * sensitivity to gamma at the selected (p, k).
  * cumulative coverage Gamma_k against the Eq. (8) bound min(1, k(1-p/100)).

Usage: python experiments/analyse_e6.py
"""
from __future__ import annotations

import argparse
import csv
import os
import statistics
from collections import defaultdict

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASETS = ("cifar10", "cifar20", "cifar100")


def load(path, keyfields, numeric):
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            if not r.get("gold_retain"):
                continue
            d = {k: r[k] for k in keyfields}
            for k in numeric:
                d[k] = float(r[k])
            d["dataset"] = r["dataset"]
            d["cls"] = int(r["forget_class"])
            rows.append(d)
    return rows


def score_of(rs):
    return statistics.mean(
        abs(r["post_retain"] - r["gold_retain"]) + abs(r["post_forget"] - r["gold_forget"])
        for r in rs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--e6", default=os.path.join(_REPO_ROOT, "results", "e6_unact_grid.csv"))
    ap.add_argument("--e1", default=os.path.join(_REPO_ROOT, "results", "e1_ssd_tune.csv"))
    args = ap.parse_args()

    num = ["post_retain", "post_forget", "gold_retain", "gold_forget",
           "percentile", "gamma", "iters", "cum_coverage_pct",
           "coverage_bound_pct", "params_damped_pct", "method_time_s"]
    e6 = load(args.e6, ["dataset", "forget_class"], num)
    e1 = load(args.e1, ["dataset", "forget_class"],
              ["post_retain", "post_forget", "gold_retain", "gold_forget",
               "alpha", "lambda", "batch_size", "params_damped_pct", "method_time_s"])
    print(f"E6: {len(e6)} rows   E1: {len(e1)} rows\n")

    selections = {}
    for ds in DATASETS:
        cfgs = defaultdict(list)
        for r in e6:
            if r["dataset"] == ds:
                cfgs[(r["percentile"], r["gamma"], r["iters"])].append(r)
        stats = []
        for (p, gm, k), rs in cfgs.items():
            if len(rs) < 5:
                continue
            stats.append({
                "p": p, "gamma": gm, "k": k, "score": score_of(rs),
                "d_r": statistics.mean(r["post_retain"] for r in rs),
                "d_r_sd": statistics.stdev(r["post_retain"] for r in rs),
                "d_f": statistics.mean(r["post_forget"] for r in rs),
                "gold_r": statistics.mean(r["gold_retain"] for r in rs),
                "cov": statistics.mean(r["cum_coverage_pct"] for r in rs),
                "bound": statistics.mean(r["coverage_bound_pct"] for r in rs),
                "damp": statistics.mean(r["params_damped_pct"] for r in rs),
                "t": statistics.mean(r["method_time_s"] for r in rs),
            })
        best = min(stats, key=lambda s: s["score"])
        selections[ds] = best
        print(f"=== {ds} ===  gold D_r {best['gold_r']:.2f}   ({len(stats)} configs)")
        print(f"  {'p':>5} {'gamma':>6} {'k':>3} {'D_r':>14} {'D_f':>7} {'score':>7} "
              f"{'cov%':>6} {'bound%':>7} {'damp%':>7} {'t(s)':>6}")
        for s in sorted(stats, key=lambda s: s["score"])[:6]:
            mark = "  <- SELECTED" if s is best else ""
            print(f"  {s['p']:5.0f} {s['gamma']:6.2f} {s['k']:3.0f} "
                  f"{s['d_r']:7.2f}+-{s['d_r_sd']:<5.2f} {s['d_f']:7.2f} {s['score']:7.3f} "
                  f"{s['cov']:6.1f} {s['bound']:7.1f} {s['damp']:7.2f} {s['t']:6.1f}{mark}")

        # gamma robustness at the selected (p, k)
        same = sorted([s for s in stats if s["p"] == best["p"] and s["k"] == best["k"]],
                      key=lambda s: -s["gamma"])
        print(f"  gamma sweep at p={best['p']:.0f}, k={best['k']:.0f}: " +
              "  ".join(f"g={s['gamma']}:{s['score']:.2f}" for s in same))
        # previous default, for contrast
        prev_p = {"cifar10": 75.0, "cifar20": 95.0, "cifar100": 99.0}[ds]
        old = [s for s in stats if s["p"] == prev_p and s["gamma"] == 0.1 and s["k"] == 5]
        if old:
            print(f"  previous default (p={prev_p:.0f}, g=0.1, k=5): "
                  f"D_r {old[0]['d_r']:.2f}  D_f {old[0]['d_f']:.2f}  score {old[0]['score']:.3f}")
        print()

    # Head-to-head against E1's SSD selection.
    ssd_sel = {"cifar10": (8.0, 1.0, 256), "cifar20": (15.0, 1.0, 512),
               "cifar100": (50.0, 0.5, 64)}
    print("=== head-to-head, identical selection rule ===")
    print(f"  {'dataset':9s} {'UnAct score':>12} {'SSD score':>11} {'winner':>8}   "
          f"{'UnAct cfg':<22} {'SSD cfg'}")
    for ds in DATASETS:
        a, l, b = ssd_sel[ds]
        rs = [r for r in e1 if r["dataset"] == ds and r["alpha"] == a
              and r["lambda"] == l and r["batch_size"] == b]
        ssd_score = score_of(rs)
        u = selections[ds]
        win = "UnAct" if u["score"] < ssd_score else "SSD"
        print(f"  {ds:9s} {u['score']:12.3f} {ssd_score:11.3f} {win:>8}   "
              f"p={u['p']:.0f} g={u['gamma']} k={u['k']:.0f}{'':<8} a={a} l={l} b={b}")


if __name__ == "__main__":
    main()
