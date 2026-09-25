"""
Per-n oracle for SSD, from EXISTING E3 rows only.

E3 already replayed SSD's alpha grid at every forget-set size n with lambda and the
importance batch fixed at the selected values (method ssd / ssd_alphagrid). Here we
pick, separately at each n, the alpha with the lowest mean Delta over the five forget
classes -- an oracle that needs a retrained model per n -- restricted to the range
the SSD paper reports (alpha in [5, 50]). UnAct keeps its single full-class
configuration (no per-n tuning). Writes results/e3_ssd_oracle.csv.
"""
import os
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
d = pd.read_csv(os.path.join(ROOT, "results", "e3_forget_size.csv"))
d["alpha"] = d.config.str.extract(r"a([\d.]+)_").astype(float)
d["delta"] = (d.post_retain - d.gold_retain).abs() + (d.post_forget - d.gold_forget).abs()
d["n"] = d.forget_n.where(d.forget_n < d.forget_available, -1)  # -1 = whole class
drop = (d.gold_retain - d.post_retain)
d["drop"] = drop
rows = []
for (ds, n), g in d.groupby(["dataset", "n"]):
    ua = g[g.method == "unact"]
    ssd = g[g.method.isin(["ssd", "ssd_alphagrid"]) & g.alpha.between(5, 50)]
    per_a = ssd.groupby("alpha").agg(delta=("delta", "mean"), worst=("drop", "max"), k=("delta", "size"))
    assert (per_a.k == 5).all(), (ds, n)
    a_star = per_a.delta.idxmin()
    fixed = g[g.method == "ssd"]
    rows.append(dict(dataset=ds, n=n, unact_delta=ua.delta.mean(), unact_worst_drop=ua["drop"].max(),
                     ssd_fixed_alpha=fixed.alpha.iloc[0], ssd_fixed_delta=fixed.delta.mean(),
                     ssd_fixed_worst_drop=fixed["drop"].max(),
                     ssd_oracle_alpha=a_star, ssd_oracle_delta=per_a.delta.min(),
                     ssd_oracle_worst_drop=per_a.loc[a_star, "worst"]))
out = pd.DataFrame(rows).round(4)
out.to_csv(os.path.join(ROOT, "results", "e3_ssd_oracle.csv"), index=False)
print(out.to_string())
