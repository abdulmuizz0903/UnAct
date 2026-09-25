"""e5_select_ssd.py -- print SSD's selected ViT config as `ALPHA,LAMBDA,BATCH`.

Pools the pre-registered E5 grid (results/e5_vit_ssd.csv) with the matching
exploratory search (results/e5b_vit_ssd_explore.csv), at the shared grid
evaluation (bf16, 2000-image retain subset), and applies the usual rule:
lowest mean |dD_r|+|dD_f| over configs measured on all 5 forget classes.
"""
import csv
import os
import statistics
import sys
from collections import defaultdict

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
by = defaultdict(dict)
for name in ("e5_vit_ssd.csv", "e5b_vit_ssd_explore.csv"):
    path = os.path.join(_REPO_ROOT, "results", name)
    if not os.path.exists(path):
        continue
    for r in csv.DictReader(open(path)):
        if (r["eval_retain_n"], r["eval_dtype"]) != ("2000", "bf16") or not r["score"]:
            continue
        cfg = (float(r["alpha"]), float(r["lambda"]), int(r["batch_size"]))
        by[cfg][r["forget_class"]] = float(r["score"])
cands = [(statistics.fmean(v.values()), c) for c, v in by.items() if len(v) == 5]
if not cands:
    sys.exit("no SSD ViT config is complete on all 5 classes")
score, (a, lam, bs) = min(cands)
print(f"  SSD ViT selected alpha={a:g} lambda={lam:g} batch={bs}  mean score {score:.3f}",
      file=sys.stderr)
print(f"{a:g},{lam:g},{bs}")
