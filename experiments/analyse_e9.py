"""
Select the LFSSD operating point (E9) from its tuning grid.

Same pre-registered rule as E1 (SSD) and E6 (UnAct), applied unchanged:

    score = |D_r - gold_D_r| + |D_f - gold_D_f|,  averaged over the 5 forget classes

A configuration is only eligible if it was measured on all 5 forget classes, so
a partial sweep cannot win by being evaluated on an easy subset. The robustness
rule (mean D_f <= 5, maximise mean D_r) is reported alongside, and where the two
disagree both are printed rather than the more convenient one.

Usage:
    python experiments/analyse_e9.py
"""
from __future__ import annotations

import argparse
import csv
import os
import statistics
from collections import defaultdict

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

GRIDS = {
    "lfssd": {
        "path": os.path.join(_REPO_ROOT, "results", "e9_lfssd_tune.csv"),
        "config": lambda r: (float(r["alpha"]), float(r["lambda"]), int(r["batch_size"])),
        "label": lambda c: f"alpha={c[0]:g} lambda={c[1]:g} batch={c[2]}",
        "method_of": lambda r: "lfssd",
    },
}


def load(spec):
    rows = []
    if not os.path.exists(spec["path"]):
        return rows
    with open(spec["path"], newline="") as f:
        for r in csv.DictReader(f):
            if not r.get("gold_retain"):
                continue
            rows.append({
                "dataset": r["dataset"],
                "method": spec["method_of"](r),
                "cls": int(r["forget_class"]),
                "cfg": spec["config"](r),
                "d_r": float(r["post_retain"]),
                "d_f": float(r["post_forget"]),
                "gold_r": float(r["gold_retain"]),
                "gold_f": float(r["gold_forget"]),
                "damped": float(r["params_damped_pct"]),
                "t": float(r["method_time_s"]),
            })
    return rows


def summarise(rows, spec, name):
    by = defaultdict(list)
    for r in rows:
        by[(r["dataset"], r["method"], r["cfg"])].append(r)

    n_classes = defaultdict(set)
    for r in rows:
        n_classes[(r["dataset"], r["method"])].add(r["cls"])

    out = {}
    for (ds, method), classes in sorted(n_classes.items()):
        need = len(classes)
        cands = []
        for (d, m, cfg), rs in by.items():
            if (d, m) != (ds, method) or len(rs) < need:
                continue
            scores = [abs(r["d_r"] - r["gold_r"]) + abs(r["d_f"] - r["gold_f"]) for r in rs]
            cands.append({
                "cfg": cfg,
                "score": statistics.fmean(scores),
                "d_r": statistics.fmean(r["d_r"] for r in rs),
                "d_r_sd": statistics.pstdev([r["d_r"] for r in rs]),
                "d_f": statistics.fmean(r["d_f"] for r in rs),
                "d_f_sd": statistics.pstdev([r["d_f"] for r in rs]),
                "gold_r": statistics.fmean(r["gold_r"] for r in rs),
                "damped": statistics.fmean(r["damped"] for r in rs),
                "t": statistics.fmean(r["t"] for r in rs),
            })
        if not cands:
            continue
        cands.sort(key=lambda c: c["score"])
        best = cands[0]
        robust = max((c for c in cands if c["d_f"] <= 5.0),
                     key=lambda c: c["d_r"], default=None)
        out[(ds, method)] = best

        print(f"\n=== {name}: {ds} / {method} "
              f"({len(cands)} complete configs over {need} classes, gold D_r "
              f"{best['gold_r']:.2f}) ===")
        print("  top 5 by score:")
        for c in cands[:5]:
            print(f"    {spec['label'](c['cfg']):<34} score {c['score']:8.4f}  "
                  f"D_r {c['d_r']:6.2f}+-{c['d_r_sd']:4.2f}  "
                  f"D_f {c['d_f']:6.2f}+-{c['d_f_sd']:4.2f}  "
                  f"damped {c['damped']:5.2f}%  t {c['t']:7.2f}s")
        if robust and robust["cfg"] != best["cfg"]:
            print(f"  robustness rule (D_f<=5, max D_r) prefers: "
                  f"{spec['label'](robust['cfg'])}  score {robust['score']:.4f}  "
                  f"D_r {robust['d_r']:.2f}  D_f {robust['d_f']:.2f}")
        else:
            print("  robustness rule agrees with the score rule.")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", nargs="+", default=list(GRIDS), choices=list(GRIDS))
    args = ap.parse_args()

    selected = {}
    for name in args.grid:
        spec = GRIDS[name]
        rows = load(spec)
        if not rows:
            print(f"\n=== {name}: no rows at {spec['path']} ===")
            continue
        selected.update(summarise(rows, spec, name))

    print("\n=== selections (paste into experiments/e9_baselines.py) ===")
    for (ds, method), best in sorted(selected.items()):
        print(f"  {method:<9} {ds:<9} {GRIDS['lfssd']['label'](best['cfg']) if method=='lfssd' else best['cfg'][0]}"
              f"   score {best['score']:.4f}")


if __name__ == "__main__":
    main()
