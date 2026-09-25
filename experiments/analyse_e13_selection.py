"""
E13: can the operating point be chosen without a retrained model? (analysis only)

Uses the existing full grids (E1 SSD, E9a LFSSD, E6 UnAct); no new runs.
For class removal the retrained model's forget accuracy is 0 by construction
(gold_forget == 0 in every row), so Delta = |D_r - D_r^gold| + D_f and the only
quantity that needs a retrained model is D_r^gold. Rules compared, each applied
per dataset to the mean over the five forget classes, identically for all methods:

  oracle     argmin mean Delta                        (the paper's rule)
  rf         argmax mean D_r  s.t. mean D_f <= TAU (TAU=0) (no retrained model)
  lodo       argmin mean Delta on the OTHER two datasets (cross-dataset transfer)
  single     one configuration for all three datasets: argmin of the mean
             over datasets of mean Delta (reported per dataset)

Writes results/e13_selection.csv.
"""
import os
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
TAU = float(os.environ.get("E13_TAU", 0.0))
DATASETS = ["cifar10", "cifar20", "cifar100"]
SPECS = {
    "ssd": ("e1_ssd_tune.csv", ["alpha", "lambda", "batch_size"]),
    "lfssd": ("e9_lfssd_tune.csv", ["alpha", "lambda", "batch_size"]),
    "unact": ("e6_unact_grid.csv", ["percentile", "gamma", "iters"]),
}


def cfg_str(row, keys):
    return "/".join(f"{row[k]:g}" for k in keys)


rows = []
for meth, (f, keys) in SPECS.items():
    d = pd.read_csv(os.path.join(RES, f))
    d["delta"] = (d.post_retain - d.gold_retain).abs() + (d.post_forget - d.gold_forget).abs()
    g = d.groupby(["dataset"] + keys).agg(d_r=("post_retain", "mean"), d_f=("post_forget", "mean"),
                                          delta=("delta", "mean"), n=("delta", "size")).reset_index()
    assert (g.n == 5).all()
    wide = g.pivot_table(index=keys, columns="dataset", values="delta")
    for ds in DATASETS:
        gd = g[g.dataset == ds].set_index(keys)
        oracle = gd.delta.idxmin()
        feas = gd[gd.d_f <= TAU]
        rf = feas.d_r.idxmax() if len(feas) else None
        others = [o for o in DATASETS if o != ds]
        lodo = wide[others].mean(1).idxmin()
        single = wide[DATASETS].mean(1).idxmin()
        for rule, c in (("oracle", oracle), ("rf", rf), ("lodo", lodo), ("single", single)):
            r = gd.loc[c]
            rows.append({"method": meth, "dataset": ds, "rule": rule,
                         "config": "/".join(f"{v:g}" for v in (c if isinstance(c, tuple) else (c,))),
                         "d_r": round(r.d_r, 4), "d_f": round(r.d_f, 4), "delta": round(r.delta, 4),
                         "same_as_oracle": c == oracle})
out = pd.DataFrame(rows)
if "E13_TAU" not in os.environ:
    out.to_csv(os.path.join(RES, "e13_selection.csv"), index=False)
print(out.to_string())
