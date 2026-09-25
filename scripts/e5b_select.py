"""e5b_select.py -- print each E5b scope's best stage-1 config on cifar10 class 3,
one line per scope: `<scope> <p> <gamma> <k>`. Lower score (|dD_r|+|dD_f|) wins,
ties broken toward fewer iterations (cheaper)."""
import csv
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO_ROOT, "Our_Method"))
from unlearn_vit import EXPLORATORY_SCOPES  # noqa: E402

path = os.path.join(_REPO_ROOT, "results", "e5b_vit_explore.csv")
best = {}
for r in csv.DictReader(open(path)):
    if (r["forget_class"], r["profile_n"], r["eval_retain_n"], r["eval_dtype"]) != ("3", "500", "2000", "bf16"):
        continue
    if r["scope"] not in EXPLORATORY_SCOPES or not r["score"]:
        continue
    key = (float(r["score"]), int(r["iters"]))
    if r["scope"] not in best or key < best[r["scope"]][0]:
        best[r["scope"]] = (key, r)
for scope in EXPLORATORY_SCOPES:
    if scope in best:
        (sc, _), r = best[scope]
        print(scope, r["percentile"], r["gamma"], r["iters"])
        print(f"  {scope}: score {sc:.3f}  D_r {r['post_retain']}  D_f {r['post_forget']}", file=sys.stderr)
