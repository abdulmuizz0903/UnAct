"""Cost figure: UnAct's speed/accuracy trade-off in the number of rounds k.

fig_cost_k : distance to retraining against wall-clock time. UnAct is shown at
             the best configuration of its pre-registered grid for each timed k
             (k = 1, 5 and the selected k); SSD at its selected configuration,
             both cold (including the full-training-set importance pass) and
             amortised (that pass precomputed and reused, as SSD's paper allows).

Timings: serialised L4 re-times only (results/cost_breakdown_l4*.csv), warm-up
repeat discarded, one forget class (3), full forget class. SSD's point is the mean
over the timing sessions with the session range as a whisker; the paired
same-session ratios quoted in the text are in tab_cost. Delta values come from the
grids (E1 for SSD, E6 for UnAct, E9 for LFSSD), mean over the five forget classes.
LFSSD was not re-timed under the serialised protocol, so it appears as its Delta
only.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from common import (C, DATASETS, DATASET_LABEL, LFSSD_SELECTED, MARK, SSD_SELECTED,
                    TEXTW, UNACT_SELECTED, load_cost_l4, load_cost_ssd_pooled, load_e1,
                    load_e9_grid, save, set_style)


def _grid_score(grid, ds, sel):
    a, l, b = sel
    r = grid[(grid.dataset == ds) & (grid.alpha == a) & (grid["lambda"] == l)
             & (grid.batch_size == b)]
    return float(r.score.iloc[0])


def fig_cost_k():
    cost = load_cost_l4()
    ssd = load_cost_ssd_pooled()
    e1, e9 = load_e1(), load_e9_grid()
    fig, axes = plt.subplots(1, 3, figsize=(TEXTW, 2.05), sharey=True)
    for j, ds in enumerate(DATASETS):
        ax = axes[j]
        u = cost[cost.dataset == ds].sort_values("iters")
        k_sel = UNACT_SELECTED[ds][2]
        lf = _grid_score(e9, ds, LFSSD_SELECTED[ds])
        ax.plot(u.unact_s, u.score, color=C["unact"], lw=1.2, zorder=3)
        for _, r in u.iterrows():
            sel = int(r.iters) == k_sel
            ax.plot([r.unact_s], [r.score], marker=MARK["unact"], ms=5 if sel else 4.2,
                    color=C["unact"], mfc=C["unact"] if sel else "white", mew=1.1,
                    zorder=5, ls="none")
            below = r.score < 1.15 * lf      # keep labels off the LFSSD line
            ax.annotate(f"$k={int(r.iters)}$", (r.unact_s, r.score),
                        xytext=(0, -6 if below else 5), textcoords="offset points",
                        ha="center", va="top" if below else "bottom", fontsize=6.5,
                        color="#333333")
        # SSD: amortised (filled) and cold (open) at the same Delta.
        s = ssd.loc[ds]
        sc = _grid_score(e1, ds, SSD_SELECTED[ds])
        ax.plot([s.amort, s.cold], [sc, sc], color=C["ssd"], lw=0.8, ls=(0, (1, 1.5)),
                zorder=2)
        for x, lo, hi, filled in [(s.amort, s.amort_min, s.amort_max, True),
                                  (s.cold, s.cold_min, s.cold_max, False)]:
            ax.errorbar([x], [sc], xerr=[[x - lo], [hi - x]], fmt=MARK["ssd"], ms=4.5,
                        color=C["ssd"], mfc=C["ssd"] if filled else "white", mew=1.1,
                        elinewidth=0.8, capsize=1.8, zorder=4)
        # LFSSD: Delta only (not timed under the serialised protocol).
        ax.axhline(lf, color=C["lfssd"], lw=1.0, ls=(0, (5, 2)), zorder=1)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(0.06, 80)
        ax.set_xticks([0.1, 1, 10])
        ax.set_xticklabels(["0.1", "1", "10"])
        ax.set_ylim(0.045, 4)
        ax.set_yticks([0.1, 0.3, 1, 3])
        ax.set_yticklabels(["0.1", "0.3", "1", "3"])
        ax.set_title(DATASET_LABEL[ds], pad=3)
        ax.set_xlabel("Wall-clock time (s)")
    axes[0].set_ylabel("Distance to retrain $\\Delta$")
    h = [plt.Line2D([], [], color=C["unact"], marker=MARK["unact"], mfc="white", mew=1.1,
                    lw=1.2, label="UnAct, $k$ rounds (filled: selected)"),
         plt.Line2D([], [], color=C["ssd"], marker=MARK["ssd"], mfc=C["ssd"], ls="none",
                    label="SSD, importance precomputed"),
         plt.Line2D([], [], color=C["ssd"], marker=MARK["ssd"], mfc="white", mew=1.1,
                    ls="none", label="SSD, cold"),
         plt.Line2D([], [], color=C["lfssd"], lw=1.0, ls=(0, (5, 2)),
                    label="LFSSD (untimed)")]
    fig.subplots_adjust(wspace=0.08, left=0.09, right=0.99, top=0.78, bottom=0.2)
    fig.legend(handles=h, loc="lower center", ncol=4, bbox_to_anchor=(0.5, 0.87),
               columnspacing=1.2, handlelength=1.8, borderaxespad=0.1)
    save(fig, "fig_cost_k")


if __name__ == "__main__":
    set_style()
    fig_cost_k()
