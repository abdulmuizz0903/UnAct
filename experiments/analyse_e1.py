"""
Select SSD's operating point per dataset from the E1 grid.

Selection rule (fixed before inspecting the grid):

    score = |D_r - gold_D_r| + |D_f - gold_D_f|,   averaged over the 5 forget classes

Both terms are percentage points, equally weighted. This is the scalar form of
SSD's own "closest to the retrained gold model" criterion. It deliberately does
not privilege either axis: rewarding low D_f alone would select the Streisand
regime their paper warns about, and rewarding high D_r alone would select doing
nothing at all.

A second rule is reported as a robustness check:

    among configs with mean D_f <= 5, pick the highest mean D_r

If the two rules disagree, both selections are reported rather than the more
convenient one.

Usage: python experiments/analyse_e1.py [--results results/e1_ssd_tune.csv]
"""
from __future__ import annotations

import argparse
import csv
import os
import statistics
import sys
from collections import defaultdict

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load(path):
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            if not r.get("gold_retain"):
                continue  # no gold model -> cannot score
            rows.append({
                "dataset": r["dataset"],
                "cls": int(r["forget_class"]),
                "alpha": float(r["alpha"]),
                "lam": float(r["lambda"]),
                "bs": int(r["batch_size"]),
                "d_r": float(r["post_retain"]),
                "d_f": float(r["post_forget"]),
                "gold_r": float(r["gold_retain"]),
                "gold_f": float(r["gold_forget"]),
                "damped": float(r["params_damped_pct"]),
                "t": float(r["method_time_s"]),
            })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=os.path.join(_REPO_ROOT, "results", "e1_ssd_tune.csv"))
    args = ap.parse_args()

    rows = load(args.results)
    print(f"{len(rows)} scored rows from {args.results}\n")

    # Group by (dataset, config) across forget classes.
    grouped = defaultdict(list)
    for r in rows:
        grouped[(r["dataset"], r["alpha"], r["lam"], r["bs"])].append(r)

    for dataset in ("cifar10", "cifar20", "cifar100"):
        configs = {k: v for k, v in grouped.items() if k[0] == dataset}
        if not configs:
            continue
        stats = []
        for (ds, alpha, lam, bs), rs in configs.items():
            if len(rs) < 5:
                continue  # incomplete config
            d_r = [x["d_r"] for x in rs]
            d_f = [x["d_f"] for x in rs]
            score = statistics.mean(
                abs(x["d_r"] - x["gold_r"]) + abs(x["d_f"] - x["gold_f"]) for x in rs
            )
            stats.append({
                "alpha": alpha, "lam": lam, "bs": bs, "score": score,
                "d_r": statistics.mean(d_r), "d_r_sd": statistics.stdev(d_r),
                "d_f": statistics.mean(d_f), "d_f_sd": statistics.stdev(d_f),
                "gold_r": statistics.mean(x["gold_r"] for x in rs),
                "damped": statistics.mean(x["damped"] for x in rs),
                "t": statistics.mean(x["t"] for x in rs),
            })

        best = min(stats, key=lambda s: s["score"])
        feasible = [s for s in stats if s["d_f"] <= 5.0]
        alt = max(feasible, key=lambda s: s["d_r"]) if feasible else None

        print(f"=== {dataset} ===   gold D_r = {best['gold_r']:.2f}, gold D_f = 0.00"
              f"   ({len(stats)} complete configs)")
        print(f"  {'alpha':>6} {'lam':>5} {'bs':>5} {'D_r':>14} {'D_f':>14} "
              f"{'score':>7} {'damp%':>7} {'t(s)':>7}")
        for s in sorted(stats, key=lambda s: s["score"])[:8]:
            mark = "  <- SELECTED" if s is best else ""
            print(f"  {s['alpha']:6.1f} {s['lam']:5.2f} {s['bs']:5d} "
                  f"{s['d_r']:7.2f}+-{s['d_r_sd']:<5.2f} {s['d_f']:7.2f}+-{s['d_f_sd']:<5.2f} "
                  f"{s['score']:7.2f} {s['damped']:7.2f} {s['t']:7.1f}{mark}")
        if alt and (alt["alpha"], alt["lam"], alt["bs"]) != (best["alpha"], best["lam"], best["bs"]):
            print(f"  robustness rule (D_f<=5, max D_r) picks a DIFFERENT config: "
                  f"alpha={alt['alpha']}, lam={alt['lam']}, bs={alt['bs']} -> "
                  f"D_r {alt['d_r']:.2f}, D_f {alt['d_f']:.2f}, score {alt['score']:.2f}")
        elif alt:
            print("  robustness rule (D_f<=5, max D_r) agrees with the selection")
        else:
            print("  robustness rule: NO config reaches D_f <= 5")

        # What the paper's own defaults would have given, for contrast.
        for label, (a, l) in {"SSD paper (a=10, l=1)": (10.0, 1.0),
                              "our old default (a=9, l=1)": (9.0, 1.0)}.items():
            cand = [s for s in stats if s["alpha"] == a and s["lam"] == l]
            if cand:
                b = min(cand, key=lambda s: s["score"])
                print(f"  {label:28s} best over bs: bs={b['bs']:<5d} "
                      f"D_r {b['d_r']:6.2f}  D_f {b['d_f']:6.2f}  score {b['score']:.2f}")
        print()


if __name__ == "__main__":
    main()
