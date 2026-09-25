"""Hyperparameter sensitivity (appendix): E1 (SSD), E9a (LFSSD), E6 (UnAct).

fig4_sensitivity_profiles : 1-D sweeps along each method's own axes.
fig5_tuning_robustness    : fraction of the searched grid reaching a given quality.
figA1_grids               : the complete grids as heatmaps.

All three grids were fixed in advance with comparable budgets: 90 configurations
per dataset for SSD and LFSSD, 100 for UnAct. Every method is brittle somewhere;
the figures are drawn so that this is visible for UnAct too (its k axis is as
sharp as SSD's alpha axis on CIFAR-10).
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

from common import (C, DATASETS, DATASET_LABEL, LFSSD_SELECTED, METHOD_LABEL,
                    RAMP_LFSSD, RAMP_SSD, RAMP_UNACT, SSD_SELECTED, TEXTW,
                    UNACT_SELECTED, load_e1, load_e6, load_e9_grid, save, set_style)

FLOOR = 0.05  # log-axis floor; no measured score is below this
BATCHES = [64, 256, 512]
KS = [1, 5, 10, 20]


def _star(ax, x, y):
    ax.plot([x], [max(y, FLOOR)], marker="*", ms=9, color="black", mfc="#FFD23F",
            mew=0.6, zorder=10, ls="none")


def _logy(ax):
    ax.set_yscale("log")
    ax.set_ylim(FLOOR, 300)
    ax.set_yticks([0.1, 1, 10, 100])
    ax.set_yticklabels(["0.1", "1", "10", "100"])


def _alpha_row(axes_row, grid, selected, ramp):
    for j, ds in enumerate(DATASETS):
        ax = axes_row[j]
        a_sel, l_sel, b_sel = selected[ds]
        g = grid[(grid.dataset == ds) & (grid["lambda"] == l_sel)]
        for b, col in zip(BATCHES, ramp):
            s = g[g.batch_size == b].sort_values("alpha")
            ax.plot(s.alpha, np.maximum(s.score, FLOOR), marker="o", ms=2.6, color=col,
                    lw=1.2, label=f"batch {b}")
        sel = g[(g.alpha == a_sel) & (g.batch_size == b_sel)]
        _star(ax, a_sel, sel.score.iloc[0])
        # Largest one-step jump along alpha at the selected batch size.
        s = g[g.batch_size == b_sel].sort_values("alpha").reset_index(drop=True)
        r = s.score.values[1:] / np.maximum(s.score.values[:-1], 1e-9)
        i = int(np.argmax(r))
        if r[i] > 50:
            lo, hi = s.alpha[i], s.alpha[i + 1]
            ax.axvspan(lo, hi, color="#BBBBBB", alpha=0.35, lw=0, zorder=0)
            ax.text(np.sqrt(lo * hi), 180, f"$\\times{r[i]:.0f}$", fontsize=6.5,
                    color="#333333", ha="center", va="center")
        ax.set_xscale("log")
        ax.set_xticks([2, 5, 10, 20, 50])
        ax.set_xticklabels(["2", "5", "10", "20", "50"])
        ax.xaxis.set_minor_locator(plt.NullLocator())
        # Shade alpha outside SSD's published range [5, 50].
        ax.axvspan(1.7, 5, color="#F2F2F2", lw=0, zorder=0)
        ax.text(2.9, 0.075, "$\\alpha<5$", fontsize=6, color="#777777", ha="center",
                va="center")
        ax.set_xlim(1.7, 60)
        _logy(ax)
        ax.set_xlabel("$\\alpha$", labelpad=1)
    axes_row[2].legend(loc="lower left", fontsize=6.5, handlelength=1.4,
                       labelspacing=0.2, borderaxespad=0.2, bbox_to_anchor=(0.0, 0.08))


def fig_sensitivity_profiles():
    e1, e9, e6 = load_e1(), load_e9_grid(), load_e6()
    fig, axes = plt.subplots(3, 3, figsize=(TEXTW, 4.9), sharey=True)
    _alpha_row(axes[0], e1, SSD_SELECTED, RAMP_SSD)
    _alpha_row(axes[1], e9, LFSSD_SELECTED, RAMP_LFSSD)
    for j, ds in enumerate(DATASETS):
        ax = axes[2, j]
        p_sel, g_sel, k_sel = UNACT_SELECTED[ds]
        g = e6[(e6.dataset == ds) & (e6.percentile == p_sel)]
        for k, col in zip(KS, RAMP_UNACT):
            s = g[g.iters == k].sort_values("gamma")
            ax.plot(s.gamma, np.maximum(s.score, FLOOR), marker="o", ms=2.6, color=col,
                    lw=1.2, label=f"$k={k}$")
        sel = g[(g.gamma == g_sel) & (g.iters == k_sel)]
        _star(ax, g_sel, sel.score.iloc[0])
        ax.set_xscale("log")
        ax.set_xticks([0.01, 0.03, 0.1, 0.3, 0.5])
        ax.set_xticklabels(["0.01", "0.03", "0.1", "0.3", "0.5"])
        ax.xaxis.set_minor_locator(plt.NullLocator())
        _logy(ax)
        ax.set_xlabel(f"$\\gamma$  ($p={p_sel:g}$)", labelpad=1)
    axes[2, 2].legend(loc="lower left", fontsize=6.5, handlelength=1.4,
                      labelspacing=0.2, ncol=2, columnspacing=0.8, borderaxespad=0.2)
    for j, ds in enumerate(DATASETS):
        axes[0, j].set_title(DATASET_LABEL[ds], pad=3)
    for i, key in enumerate(["ssd", "lfssd", "unact"]):
        axes[i, 0].set_ylabel(f"{METHOD_LABEL[key]}\ndistance $\\Delta$")
    fig.subplots_adjust(hspace=0.5, wspace=0.08, left=0.11, right=0.99, top=0.95,
                        bottom=0.07)
    for i, letter in enumerate("abc"):
        bb = axes[i, 0].get_position()
        fig.text(0.0, bb.y1 + 0.008, letter, fontsize=9, fontweight="bold",
                 ha="left", va="bottom")
    save(fig, "fig4_sensitivity_profiles")


def fig_tuning_robustness():
    """Fraction of each method's own search grid within tau of retraining."""
    e1, e9, e6 = load_e1(), load_e9_grid(), load_e6()
    taus = np.logspace(-1.1, 2, 300)
    fig, axes = plt.subplots(1, 3, figsize=(TEXTW, 1.95), sharey=True)
    for j, ds in enumerate(DATASETS):
        ax = axes[j]
        for src, key in [(e1, "ssd"), (e9, "lfssd"), (e6, "unact")]:
            s = src[src.dataset == ds].score.values
            ax.plot(taus, [(s <= t).mean() for t in taus], color=C[key], lw=1.4,
                    label=f"{METHOD_LABEL[key]} ($N={len(s)}$)",
                    zorder=4 if key == "unact" else 3)
        ax.set_xscale("log")
        ax.set_xlim(0.08, 100)
        ax.set_xticks([0.1, 1, 10, 100])
        ax.set_xticklabels(["0.1", "1", "10", "100"])
        ax.set_ylim(0, 1.0)
        ax.set_title(DATASET_LABEL[ds], pad=3)
        ax.set_xlabel("Threshold $\\tau$ on $\\Delta$")
    axes[0].set_ylabel("Fraction of grid\nwith $\\Delta \\leq \\tau$")
    h, l = axes[0].get_legend_handles_labels()
    fig.subplots_adjust(wspace=0.08, left=0.11, right=0.99, top=0.76, bottom=0.21)
    fig.legend(h, [x.rsplit(" ($N", 1)[0] for x in l], loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, 0.87), borderaxespad=0.1)
    save(fig, "fig5_tuning_robustness")


def fig_grids():
    """Complete E1 / E9a / E6 grids (appendix; the full grid, not the selected point)."""
    e1, e9, e6 = load_e1(), load_e9_grid(), load_e6()
    fig, axes = plt.subplots(3, 3, figsize=(TEXTW, 5.6),
                             gridspec_kw={"hspace": 0.6, "wspace": 0.28,
                                          "top": 0.95, "bottom": 0.07, "left": 0.12,
                                          "right": 0.88})
    norm = LogNorm(vmin=0.05, vmax=100)
    cmap = "magma_r"
    im = None
    for row, (grid, sel, key) in enumerate([(e1, SSD_SELECTED, "ssd"),
                                            (e9, LFSSD_SELECTED, "lfssd")]):
        for j, ds in enumerate(DATASETS):
            a_sel, l_sel, b_sel = sel[ds]
            p = (grid[(grid.dataset == ds) & (grid["lambda"] == l_sel)]
                 .pivot(index="alpha", columns="batch_size", values="score"))
            ax = axes[row, j]
            im = ax.imshow(np.maximum(p.values, 0.05), aspect="auto", cmap=cmap,
                           norm=norm, origin="lower")
            ax.set_xticks(range(len(p.columns)), [str(c) for c in p.columns])
            ax.set_yticks(range(len(p.index)), [f"{a:g}" for a in p.index])
            ax.tick_params(labelsize=6.5)
            ax.set_xlabel("importance batch size", labelpad=1)
            ax.set_title(f"{DATASET_LABEL[ds]}, $\\lambda={l_sel:g}$", pad=3)
            if j == 0:
                ax.set_ylabel(f"{METHOD_LABEL[key]} $\\alpha$")
            ax.plot([list(p.columns).index(b_sel)], [list(p.index).index(a_sel)],
                    marker="*", ms=9, color="black", mfc="#00D4FF", mew=0.6)
            ax.grid(False)
    for j, ds in enumerate(DATASETS):
        p_sel, g_sel, k_sel = UNACT_SELECTED[ds]
        q = (e6[(e6.dataset == ds) & (e6.gamma == g_sel)]
             .pivot(index="percentile", columns="iters", values="score"))
        ax = axes[2, j]
        ax.imshow(np.maximum(q.values, 0.05), aspect="auto", cmap=cmap, norm=norm,
                  origin="lower")
        ax.set_xticks(range(len(q.columns)), [str(c) for c in q.columns])
        ax.set_yticks(range(len(q.index)), [f"{v:g}" for v in q.index])
        ax.tick_params(labelsize=6.5)
        ax.set_xlabel("rounds $k$", labelpad=1)
        ax.set_title(f"{DATASET_LABEL[ds]}, $\\gamma={g_sel:g}$", pad=3)
        if j == 0:
            ax.set_ylabel(f"{METHOD_LABEL['unact']} $p$")
        ax.plot([list(q.columns).index(k_sel)], [list(q.index).index(p_sel)],
                marker="*", ms=9, color="black", mfc="#00D4FF", mew=0.6)
        ax.grid(False)
    cax = fig.add_axes([0.9, 0.25, 0.018, 0.5])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("Distance to retrain $\\Delta$ (lower is better)")
    cb.ax.tick_params(labelsize=6.5)
    for i, letter in enumerate("abc"):
        bb = axes[i, 0].get_position()
        fig.text(0.0, bb.y1 + 0.012, letter, fontsize=9, fontweight="bold",
                 ha="left", va="bottom")
    save(fig, "figA1_grids")


if __name__ == "__main__":
    set_style()
    fig_sensitivity_profiles()
    fig_tuning_robustness()
    fig_grids()
