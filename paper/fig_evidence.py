"""Relearning figures (appendix).

fig8_relearn     : E7 (945 rows), ResNet-18. Forget accuracy during relearning on
                   m forget images per step mixed 4:1 with retain images. The
                   retrained model's own retain accuracy is drawn too: at lr 0.01
                   and m = 1 the probe itself degrades every model (the retrained
                   model's D_r falls from 95.6 to 40.6 on CIFAR-10 by step 50), so
                   late steps at m = 1 measure damage rather than relearning.
fig_vit_relearn  : E5c, ViT-B/16, lr 5e-4 rows only (lr 0.01 is a broken
                   protocol: the retrained model's retain accuracy collapses).
                   Plots whatever classes are present and says how many.

"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from common import (C, DATASETS, DATASET_LABEL, GOLD_LS, MARK, TEXTW, load_e5c,
                    load_e7, method_handle, save, set_style)

M_VALUES = [1, 5, 25]
STEPS = [0, 1, 2, 5, 10, 20, 50]


def _relearn_axes(ax):
    ax.set_xscale("symlog", linthresh=1)
    ax.set_xticks(STEPS)
    ax.set_xticklabels([str(s) for s in STEPS])
    ax.xaxis.set_minor_locator(plt.NullLocator())
    ax.set_ylim(-4, 104)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_xlim(-0.2, 62)


def _curve(ax, t, key, col="d_f"):
    ax.plot(t.step, t[col], marker=MARK[key], ms=3.2, color=C[key], mfc="white",
            mew=1.0, ls=GOLD_LS if key == "gold" else "-", lw=1.3,
            zorder=2 if key == "gold" else 3)


def _handles():
    return [method_handle("unact", ms=3.2), method_handle("ssd", ms=3.2),
            method_handle("gold", ms=3.2, ls=GOLD_LS, label="Retrain (forget acc.)"),
            plt.Line2D([], [], color="#9A9A9A", lw=0.9, ls=(0, (1, 1.2)),
                       label="Retrain, retain accuracy")]


def fig_relearn():
    e7 = load_e7()
    fig, axes = plt.subplots(3, 3, figsize=(TEXTW, 4.4), sharex=True, sharey=True)
    for i, m in enumerate(M_VALUES):
        for j, ds in enumerate(DATASETS):
            ax = axes[i, j]
            s = e7[(e7.dataset == ds) & (e7.relearn_m == m)]
            g = s[s.method == "gold"].sort_values("step")
            ax.plot(g.step, g.d_r, color="#9A9A9A", lw=0.9, ls=(0, (1, 1.2)), zorder=1)
            for key in ("gold", "ssd", "unact"):
                _curve(ax, s[s.method == key].sort_values("step"), key)
            _relearn_axes(ax)
            if i == 0:
                ax.set_title(DATASET_LABEL[ds], pad=3)
            if i == 2:
                ax.set_xlabel("Relearning steps")
            if j == 0:
                ax.set_ylabel(f"$m={m}$\nforget accuracy (%)")
    fig.subplots_adjust(hspace=0.14, wspace=0.08, left=0.11, right=0.99, top=0.9,
                        bottom=0.1)
    fig.legend(handles=_handles(), loc="lower center", ncol=4,
               bbox_to_anchor=(0.5, 0.94), columnspacing=1.2, handlelength=2.0,
               borderaxespad=0.1)
    save(fig, "fig8_relearn")


def fig_vit_relearn():
    d = load_e5c()
    if d is None or len(d) == 0:
        print("  skip fig_vit_relearn: no lr 5e-4 rows yet")
        return None
    ms = sorted(d.relearn_m.unique())
    fig, axes = plt.subplots(1, len(ms), figsize=(TEXTW * 0.7, 1.9), sharey=True,
                             squeeze=False)
    status = {}
    for j, m in enumerate(ms):
        ax = axes[0, j]
        s = d[d.relearn_m == m]
        g = s[s.method == "gold"].groupby("step", as_index=False).d_r.mean()
        ax.plot(g.step, g.d_r, color="#9A9A9A", lw=0.9, ls=(0, (1, 1.2)), zorder=1)
        for key in ("gold", "ssd", "unact"):
            t = s[s.method == key]
            if len(t) == 0:
                continue
            # Only classes with the full step range, so the mean is over a fixed set.
            full = t.groupby("forget_class").step.nunique()
            cls = full[full == len(STEPS)].index
            status[(m, key)] = len(cls)
            t = t[t.forget_class.isin(cls)].groupby("step", as_index=False)[["d_f"]].mean()
            if len(t):
                _curve(ax, t, key)
        _relearn_axes(ax)
        n_un = status.get((m, "unact"), 0)
        if n_un < 5:
            ax.text(0.97, 0.05, f"preliminary: {n_un} of 5 classes", transform=ax.transAxes,
                    ha="right", va="bottom", fontsize=6.5, color="#B00020")
        ax.set_title(f"$m={m}$", pad=3)
        ax.set_xlabel("Relearning steps")
    axes[0, 0].set_ylabel("Forget accuracy (%)")
    fig.subplots_adjust(wspace=0.08, left=0.14, right=0.99, top=0.72, bottom=0.2)
    fig.legend(handles=_handles(), loc="lower center", ncol=2,
               bbox_to_anchor=(0.55, 0.8), columnspacing=1.2, handlelength=2.0,
               borderaxespad=0.1)
    save(fig, "fig_vit_relearn")
    print("  fig_vit_relearn classes per (m, method):", status)
    return status


if __name__ == "__main__":
    set_style()
    fig_relearn()
    fig_vit_relearn()
