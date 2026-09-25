"""select_lfssd.py <dataset> -- print LFSSD's selected operating point as
`<dataset>:<alpha>,<lambda>,<batch>`, the format e9_baselines.py --lfssd takes.

Applies analyse_e9's pre-registered rule unchanged, but refuses (exit 1) unless
the grid is complete on all 5 pre-registered forget classes: analyse_e9 only
requires the configs to cover whichever classes are present, so a partial
sweep would otherwise select on an easy subset.
"""
import contextlib
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO_ROOT, "experiments"))
import analyse_e9  # noqa: E402

N_CLASSES = 5
ds = sys.argv[1]
spec = analyse_e9.GRIDS["lfssd"]
rows = [r for r in analyse_e9.load(spec) if r["dataset"] == ds]
classes = {r["cls"] for r in rows}
if len(classes) < N_CLASSES:
    sys.exit(f"{ds}: LFSSD grid covers classes {sorted(classes)}, need {N_CLASSES}")
with contextlib.redirect_stdout(sys.stderr):  # keep stdout for the spec only
    sel = analyse_e9.summarise(rows, spec, "lfssd")
a, lam, bs = sel[(ds, "lfssd")]["cfg"]
print(f"{ds}:{a:g},{lam:g},{bs}")
