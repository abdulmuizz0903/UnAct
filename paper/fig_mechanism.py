"""Mechanism figures (appendix), UnAct only.

fig6_selectivity : E8 (75 rows). Class selectivity of the selected channels, from
    forward statistics only (no weight modified). Datasets are coloured by an
    ordinal blue ramp in class count (10 -> 20 -> 100).
fig7_coverage    : E6 (1500 rows). Cumulative coverage Gamma_k organises UnAct's
    (p, k) grid; the bound in the Method section, Gamma_k <= min(1, k m / N),
    holds in every run, whereas k(1 - p/100) under-counts the selected set.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import (DATASETS, DATASET_LABEL, RAMP_DS, RESULTS, TEXTW, load_e6, load_e8,
                    panel_label, save, set_style)

DS_MARK = {"cifar10": "o", "cifar20": "s", "cifar100": "^"}
N_CLASSES = {"cifar10": 10, "cifar20": 20, "cifar100": 100}


def fig_selectivity():
    e8 = load_e8()
    fig, axes = plt.subplots(1, 3, figsize=(TEXTW, 1.9))

    ax = axes[0]
    for ds in DATASETS:
        s = e8[e8.dataset == ds].sort_values("percentile")
        ax.errorbar(s.percentile, s.sel, yerr=s.sel_sd, marker=DS_MARK[ds], ms=3.5,
                    color=RAMP_DS[ds], mfc="white", mew=1.0, capsize=1.5, lw=1.2,
                    elinewidth=0.7)
    ax.axhline(0, color="#999999", lw=0.6)
    ax.set_xlabel("Percentile $p$")
    ax.set_ylabel("Selectivity $s$")

    ax = axes[1]
    for ds in DATASETS:
        s = e8[e8.dataset == ds].sort_values("percentile")
        ax.errorbar(s.percentile, s.ratio, yerr=s.ratio_sd, marker=DS_MARK[ds], ms=3.5,
                    color=RAMP_DS[ds], mfc="white", mew=1.0, capsize=1.5, lw=1.2,
                    elinewidth=0.7)
    ax.set_xlabel("Percentile $p$")
    ax.set_ylabel("Forget / retain mass")

    ax = axes[2]
    s75 = e8[e8.percentile == 75].set_index("dataset").loc[DATASETS]
    x = np.arange(len(DATASETS))
    for xi, ds in enumerate(DATASETS):
        r = s75.loc[ds]
        ax.plot([xi, xi], [r.retain_mass, r.forget_mass], color="#BBBBBB", lw=1.0,
                zorder=1)
        ax.plot([xi], [r.forget_mass], marker="o", ms=5, color=RAMP_DS[ds], zorder=3,
                ls="none")
        ax.plot([xi], [r.retain_mass], marker="o", ms=5, color="white",
                mec=RAMP_DS[ds], mew=1.2, zorder=3, ls="none")
        ax.text(xi + 0.12, r.forget_mass, f"{r.forget_mass:.0f}", fontsize=6.5,
                va="center", color="#333333")
        ax.text(xi + 0.12, r.retain_mass, f"{r.retain_mass:.0f}", fontsize=6.5,
                va="center", color="#333333")
        if xi == 0:   # direct labels instead of a legend
            ax.text(xi - 0.14, r.forget_mass, "forget", fontsize=6.5, va="center",
                    ha="right", color="#333333")
            ax.text(xi - 0.14, r.retain_mass, "retain", fontsize=6.5, va="center",
                    ha="right", color="#333333")
    ax.set_xticks(x, [f"{N_CLASSES[d]}" for d in DATASETS])
    ax.set_xlim(-0.95, 2.55)
    ax.set_xlabel("Classes ($p=75$)")
    ax.set_ylabel("Activation mass (%)")
    ax.set_ylim(0, 100)

    h = [plt.Line2D([], [], color=RAMP_DS[d], marker=DS_MARK[d], mfc="white", mew=1.0,
                    ms=3.5, label=f"{DATASET_LABEL[d]}") for d in DATASETS]
    fig.subplots_adjust(wspace=0.45, left=0.08, right=0.99, top=0.8, bottom=0.2)
    fig.legend(handles=h, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.88),
               borderaxespad=0.1)
    for ax, letter in zip(axes, "abc"):
        panel_label(ax, letter, x=-0.2, y=1.0)
    save(fig, "fig6_selectivity")


WINDOWS = {"cifar10": (22.5, 50.0), "cifar20": (22.5, 45.0), "cifar100": (5.9, 11.7)}


def fig_coverage():
    e6 = load_e6()
    raw = pd.read_csv(f"{RESULTS}/e6_unact_grid.csv")
    fig, axes = plt.subplots(1, 3, figsize=(TEXTW, 1.95))

    # (a) Delta against cumulative coverage, gamma <= 0.1
    ax = axes[0]
    safe = e6[e6.gamma <= 0.1]
    for ds in DATASETS:
        s = safe[safe.dataset == ds].sort_values("coverage")
        ax.plot(s.coverage, np.maximum(s.score, 0.05), DS_MARK[ds], ms=3.0,
                color=RAMP_DS[ds], mfc="none", mew=0.9, ls="none")
    ax.axhspan(0.05, 2.0, color="#EEEEEE", lw=0, zorder=0)
    ax.set_yscale("log"); ax.set_xscale("log")
    ax.set_yticks([0.1, 1, 10, 100]); ax.set_yticklabels(["0.1", "1", "10", "100"])
    ax.set_xticks([1, 2, 5, 10, 20, 50, 100])
    ax.set_xticklabels(["1", "2", "5", "10", "20", "50", "100"])
    ax.xaxis.set_minor_locator(plt.NullLocator())
    ax.set_xlabel("Coverage $\\Gamma_k$ (%)")
    ax.set_ylabel("Distance $\\Delta$")

    # (b) usable coverage window per dataset (fitted on this grid: descriptive)
    ax = axes[1]
    for i, ds in enumerate(DATASETS):
        lo, hi = WINDOWS[ds]
        s = safe[safe.dataset == ds]
        good = s.score <= 2.0
        ax.plot(s.coverage[good], np.full(good.sum(), i) + 0.2, "o", ms=2.6,
                color=RAMP_DS[ds], alpha=0.8, mew=0)
        ax.plot(s.coverage[~good], np.full((~good).sum(), i) + 0.2, "x", ms=2.6,
                color="#AAAAAA", mew=0.7)
        ax.barh(i - 0.12, hi - lo, left=lo, height=0.24, color=RAMP_DS[ds])
        acc = ((s.coverage >= lo) & (s.coverage <= hi) == good).mean()
        ax.text(hi + 2.5, i - 0.12, f"{acc * 100:.0f}%", va="center", ha="left",
                fontsize=6.5, color="#333333")
    ax.set_yticks(range(3), [f"{N_CLASSES[d]} cls" for d in DATASETS])
    ax.set_xlim(0, 101)
    ax.set_ylim(2.6, -0.6)
    ax.set_xlabel("Coverage $\\Gamma_k$ (%)")
    ax.grid(axis="y", visible=False)

    # (c) measured coverage over two candidate bounds
    ax = axes[2]
    meas = raw.cum_coverage_pct.values
    ratios = [
        ("$k(1-p/100)$", meas / raw.coverage_bound_pct.values, "#BBBBBB"),
        ("$km/N$", meas / np.minimum(100.0, raw.iters.values * raw.units_per_round.values
                                     / 512 * 100), "#0173B2"),
    ]
    bins = np.linspace(0.55, 1.25, 36)
    for lab, r, col in ratios:
        ax.hist(r, bins=bins, color=col, alpha=0.75, label=lab, lw=0)
    ax.axvline(1.0, color="#333333", lw=0.8)
    ax.set_xlabel("Measured $\\Gamma_k$ / bound")
    ax.set_ylabel("Runs")
    ax.legend(loc="upper left", fontsize=6.5, handlelength=1.0, borderaxespad=0.1)

    h = [plt.Line2D([], [], color=RAMP_DS[d], marker=DS_MARK[d], mfc="white", mew=1.0,
                    ms=3.5, ls="none", label=DATASET_LABEL[d]) for d in DATASETS]
    fig.subplots_adjust(wspace=0.42, left=0.08, right=0.99, top=0.8, bottom=0.21)
    fig.legend(handles=h, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.88),
               borderaxespad=0.1)
    for ax, letter in zip(axes, "abc"):
        panel_label(ax, letter, x=-0.2, y=1.0)
    save(fig, "fig7_coverage")
    n_bad = int((meas > raw.coverage_bound_pct.values + 1e-6).sum())
    print(f"  coverage: {n_bad}/{len(meas)} runs exceed k(1-p/100)")


if __name__ == "__main__":
    set_style()
    fig_selectivity()
    fig_coverage()
