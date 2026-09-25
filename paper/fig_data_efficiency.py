"""E3 figures: how many forget images does each method need?

fig_forget_size_main : main body. Two rows against forget-set size n:
                       (a) distance to retraining Delta, mean over the five
                           forget classes;
                       (b) retain accuracy D_r: mean over the five classes
                           (solid) and the WORST of the five classes (light,
                           dotted), with the retrained model's mean D_r dashed.
fig_forget_size_full : appendix. Delta, D_r and D_f; every per-class value is
                       drawn as a small faint marker behind the mean line, so the
                       bimodal behaviour of SSD / LFSSD (collapse on some classes,
                       fine on others) is visible instead of an sd bar that
                       spans 0-100.

Why not mean +/- sd: at small n SSD and LFSSD are bimodal across the five
classes (e.g. SSD, CIFAR-100, n = 5: D_r 1.0 / 77.2 / 75.4 / 68.4 / 27.8), so the
sd bar covers the whole axis and says nothing about either mode. The worst-class
line states the safety property directly (does any class collapse?), and the
per-class markers in the appendix show every run.

The shaded band marks n <= 10 in every panel, the regime the text discusses.
It is drawn identically for every dataset and method; it does not select data.

SSD and LFSSD run at their single selected operating point at every n (E1 / E9),
as their papers run them; SSD's per-class alpha oracle ("ssd_alphagrid")
is not used. Their importance denominator is always the full training set.

Forget accuracy is never plotted on its own in the main figure: at small n SSD
and LFSSD reach D_f = 0 by destroying the network (e.g. LFSSD, CIFAR-10, n = 1:
D_f 0.0 with D_r 11.3), so a D_f-only panel would read as success exactly where
they fail. The appendix D_f row sits under the D_r row for that reason.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_hex, to_rgb
from matplotlib.ticker import FixedLocator, NullFormatter, NullLocator

from common import (C, DATASETS, DATASET_LABEL, GOLD_LS, MARK, TEXTW, gold_handle,
                    load_e3_all, load_e3_raw, method_handle, save, set_style)

ORDER = ["ssd", "lfssd", "unact"]           # UnAct drawn last, on top
DODGE = {"ssd": 1.10, "lfssd": 1.0, "unact": 1 / 1.10}
FLOOR = 0.05
SMALL_N = 10                                # upper edge of the shaded regime
SHADE = "#F1F1F1"
WORST_LS = (0, (1.2, 1.1))


def _tint(hex_, w):
    """Mix a colour with white (w = share of white)."""
    r = np.array(to_rgb(hex_))
    return to_hex(r * (1 - w) + w)


def _series(data, key, ds):
    if key not in data:
        return None
    s = data[key]
    s = s[s.dataset == ds].sort_values("forget_n")
    return s if len(s) else None


def _xaxis(ax, nmax, labels=True):
    ax.set_xscale("log")
    ax.set_xlim(0.7, nmax * 1.5)
    ticks = [1, 10, 100] + ([1000] if nmax >= 4000 else []) + [nmax]
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_minor_locator(NullLocator())
    if labels:
        ax.set_xticklabels([f"{t:g}" for t in ticks])
    else:
        ax.xaxis.set_major_formatter(NullFormatter())


def _shade_small_n(ax):
    ax.axvspan(0.7, SMALL_N * 1.35, color=SHADE, lw=0, zorder=0)


def _dots(ax, raw, key, ds, col, floor=None):
    """Every per-class value as a small faint marker, jittered in log-x."""
    r = raw[(raw.method == key) & (raw.dataset == ds)]
    if not len(r):
        return
    rng = np.random.default_rng(0)
    classes = sorted(r.forget_class.unique())
    off = {c: f for c, f in zip(classes, np.linspace(-0.035, 0.035, len(classes)))}
    x = r.forget_n.values * DODGE[key] * 10 ** np.array([off[c] for c in r.forget_class])
    x = x * 10 ** rng.uniform(-0.004, 0.004, len(x))
    y = r[col].values if floor is None else np.maximum(r[col].values, floor)
    ax.scatter(x, y, s=7, marker=MARK[key], facecolor=_tint(C[key], 0.3),
               edgecolor="none", alpha=0.6, zorder=2, clip_on=False)


def _plot_score(ax, data, ds, raw=None):
    for key in ORDER:
        s = _series(data, key, ds)
        if s is None:
            continue
        if raw is not None:
            _dots(ax, raw, key, ds, "score", floor=FLOOR)
        x = s.forget_n.values * (DODGE[key] if raw is not None else 1.0)
        ax.plot(x, np.maximum(s.score, FLOOR), marker=MARK[key], color=C[key],
                mfc="white", mew=1.1, ms=4, lw=1.4, zorder=4 if key == "unact" else 3)
    ax.set_yscale("log")
    ax.set_ylim(FLOOR, 200)
    ax.set_yticks([0.1, 1, 10, 100])
    ax.set_yticklabels(["0.1", "1", "10", "100"])


def _plot_acc(ax, data, ds, col, gold=None, worst=False, raw=None):
    """Mean over the five classes; optionally the worst class (min for D_r)
    as a light dotted line, or every class as faint markers (raw)."""
    if gold is not None:
        ax.axhline(gold, color=C["gold"], ls=GOLD_LS, lw=1.0, zorder=1)
    for key in ORDER:
        s = _series(data, key, ds)
        if s is None:
            continue
        dodge = DODGE[key] if raw is not None else 1.0
        x = s.forget_n.values * dodge
        if raw is not None:
            _dots(ax, raw, key, ds, raw_col(col))
        if worst:
            ax.plot(x, s[f"{col}_min"].values, color=_tint(C[key], 0.3), ls=WORST_LS,
                    lw=1.0, marker=MARK[key], ms=2.6, mfc=_tint(C[key], 0.3), mew=0,
                    zorder=2.5)
        ax.plot(x, s[col].values, marker=MARK[key], color=C[key], mfc="white", mew=1.1,
                ms=4, lw=1.4, zorder=4 if key == "unact" else 3)
    ax.set_ylim(-4, 104)
    ax.set_yticks([0, 25, 50, 75, 100])


def raw_col(col):
    return {"d_r": "post_retain", "d_f": "post_forget", "score": "score"}[col]


def _legend(fig, data, y=1.0, extra=()):
    h = [method_handle(k) for k in ("unact", "ssd", "lfssd") if k in data]
    fig.legend(handles=h + [gold_handle()] + list(extra), loc="lower center",
               ncol=4 if not extra else 4 + len(extra), bbox_to_anchor=(0.5, y),
               columnspacing=1.5, handlelength=2.2, borderaxespad=0.1)


def _raw():
    raw = load_e3_raw()
    raw["score"] = ((raw.post_retain - raw.gold_retain).abs()
                    + (raw.post_forget - raw.gold_forget).abs())
    return raw


def fig_forget_size_main():
    data = load_e3_all()
    fig, axes = plt.subplots(2, 3, figsize=(TEXTW, 3.45), sharex="col")
    for j, ds in enumerate(DATASETS):
        nmax = int(data["unact"][data["unact"].dataset == ds].forget_n.max())
        _plot_score(axes[0, j], data, ds)
        gold = data["unact"][data["unact"].dataset == ds].gold_r.iloc[0]
        _plot_acc(axes[1, j], data, ds, "d_r", gold=gold, worst=True)
        for i in range(2):
            _shade_small_n(axes[i, j])
            _xaxis(axes[i, j], nmax, labels=(i == 1))
        axes[0, j].set_title(DATASET_LABEL[ds], pad=3)
        axes[1, j].set_xlabel("Forget images $n$")
        if j:
            for ax in axes[:, j]:
                ax.tick_params(labelleft=False)
    axes[0, 0].set_ylabel("Distance to retrain $\\Delta$")
    axes[1, 0].set_ylabel("Retain accuracy $D_r$ (%)")
    axes[0, 0].text(1.0, 0.075, f"$n\\leq{SMALL_N}$", fontsize=6.5, color="#6B6B6B",
                    ha="left", va="bottom")
    fig.subplots_adjust(hspace=0.16, wspace=0.08, left=0.1, right=0.99, top=0.855,
                        bottom=0.115)
    for i, letter in enumerate("ab"):
        bb = axes[i, 0].get_position()
        fig.text(0.0, bb.y1 + 0.01, letter, fontsize=9, fontweight="bold",
                 ha="left", va="bottom")
    _legend(fig, data, y=0.945)
    style = [plt.Line2D([], [], color="#555555", lw=1.4, marker="o", ms=4, mfc="white",
                        mew=1.1, label="mean of 5 forget classes"),
             plt.Line2D([], [], color=_tint("#555555", 0.3), lw=1.0, ls=WORST_LS,
                        marker="o", ms=2.6, mfc=_tint("#555555", 0.3), mew=0,
                        label="worst of the 5 classes (row b)")]
    fig.legend(handles=style, loc="lower center", ncol=2, bbox_to_anchor=(0.5, 0.895),
               columnspacing=1.8, handlelength=2.2, borderaxespad=0.1)
    save(fig, "fig_forget_size_main")


def fig_forget_size_full():
    """Appendix: Delta, D_r and D_f; mean line plus every per-class value."""
    data = load_e3_all()
    raw = _raw()
    fig, axes = plt.subplots(3, 3, figsize=(TEXTW, 4.9), sharex="col")
    for j, ds in enumerate(DATASETS):
        u = data["unact"][data["unact"].dataset == ds]
        nmax = int(u.forget_n.max())
        _plot_score(axes[0, j], data, ds, raw=raw)
        _plot_acc(axes[1, j], data, ds, "d_r", gold=u.gold_r.iloc[0], raw=raw)
        _plot_acc(axes[2, j], data, ds, "d_f", gold=0.0, raw=raw)
        for i in range(3):
            _shade_small_n(axes[i, j])
            _xaxis(axes[i, j], nmax, labels=(i == 2))
        axes[0, j].set_title(DATASET_LABEL[ds], pad=3)
        axes[2, j].set_xlabel("Forget images $n$")
        if j:
            for ax in axes[:, j]:
                ax.tick_params(labelleft=False)
    axes[0, 0].set_ylabel("Distance to\nretrain $\\Delta$")
    axes[1, 0].set_ylabel("Retain\naccuracy $D_r$ (%)")
    axes[2, 0].set_ylabel("Forget\naccuracy $D_f$ (%)")
    axes[1, 2].text(1.0, 32, f"$n\\leq{SMALL_N}$", fontsize=6.5, color="#6B6B6B",
                    ha="left", va="bottom")
    fig.subplots_adjust(hspace=0.14, wspace=0.08, left=0.11, right=0.99, top=0.885,
                        bottom=0.085)
    for i, letter in enumerate("abc"):
        bb = axes[i, 0].get_position()
        fig.text(0.0, bb.y1 + 0.008, letter, fontsize=9, fontweight="bold",
                 ha="left", va="bottom")
    _legend(fig, data, y=0.955)
    style = [plt.Line2D([], [], color="#555555", lw=1.4, marker="o", ms=4, mfc="white",
                        mew=1.1, label="mean of 5 forget classes"),
             plt.Line2D([], [], color=_tint("#555555", 0.35), lw=0, marker="o", ms=2.4,
                        mew=0, alpha=0.7, label="one forget class")]
    fig.legend(handles=style, loc="lower center", ncol=2, bbox_to_anchor=(0.5, 0.918),
               columnspacing=1.8, handlelength=2.2, borderaxespad=0.1)
    save(fig, "fig_forget_size_full")


def fig_vit_forget_size():
    """Appendix: ViT-B/16 counterpart of Fig. 2 (results/e5d_vit_forget_size.csv).
    Drawn only when all 70 rows are present (common.load_e5d)."""
    from common import load_e5d
    agg, _ = load_e5d()
    if agg is None:
        print("  skip fig_vit_forget_size: e5d incomplete")
        return None
    fig, axes = plt.subplots(1, 3, figsize=(TEXTW, 1.9))
    nmax = int(agg.forget_n.max())
    gold = agg.gold_r.mean()
    ax_s, ax_r, ax_f = axes
    ax_r.axhline(gold, color=C["gold"], ls=GOLD_LS, lw=1.0, zorder=1)
    ax_f.axhline(0.0, color=C["gold"], ls=GOLD_LS, lw=1.0, zorder=1)
    for key in ("ssd", "unact"):
        s = agg[agg.method == key].sort_values("forget_n")
        x = s.forget_n.values
        ax_s.plot(x, np.maximum(s.score, FLOOR), marker=MARK[key], color=C[key],
                  mfc="white", mew=1.1, ms=4, lw=1.4, zorder=4 if key == "unact" else 3)
        ax_r.plot(x, s.d_r_min.values, color=_tint(C[key], 0.3), ls=WORST_LS, lw=1.0,
                  marker=MARK[key], ms=2.6, mfc=_tint(C[key], 0.3), mew=0, zorder=2.5)
        ax_r.plot(x, s.d_r.values, marker=MARK[key], color=C[key], mfc="white", mew=1.1,
                  ms=4, lw=1.4, zorder=4 if key == "unact" else 3)
        # Forget accuracy: mean, and the WORST class (highest D_f), dotted.
        ax_f.plot(x, s.d_f_max.values, color=_tint(C[key], 0.3), ls=WORST_LS, lw=1.0,
                  marker=MARK[key], ms=2.6, mfc=_tint(C[key], 0.3), mew=0, zorder=2.5)
        ax_f.plot(x, s.d_f.values, marker=MARK[key], color=C[key], mfc="white", mew=1.1,
                  ms=4, lw=1.4, zorder=4 if key == "unact" else 3)
    ax_s.set_yscale("log")
    ax_s.set_ylim(FLOOR, 200)
    ax_s.set_yticks([0.1, 1, 10, 100])
    ax_s.set_yticklabels(["0.1", "1", "10", "100"])
    ax_r.set_ylim(-4, 104)
    ax_r.set_yticks([0, 25, 50, 75, 100])
    ax_f.set_ylim(-4, 104)
    ax_f.set_yticks([0, 25, 50, 75, 100])
    for ax in axes:
        _shade_small_n(ax)
        _xaxis(ax, nmax)
        ax.set_xlabel("Forget images $n$")
    ax_s.set_ylabel("Distance to retrain $\\Delta$")
    ax_r.set_ylabel("Retain accuracy $D_r$ (%)")
    ax_f.set_ylabel("Forget accuracy $D_f$ (%)")
    fig.subplots_adjust(wspace=0.42, left=0.08, right=0.99, top=0.78, bottom=0.2)
    for ax, letter in zip(axes, "abc"):
        ax.text(-0.02, 1.02, letter, transform=ax.transAxes, fontsize=9,
                fontweight="bold", ha="right", va="bottom")
    h = [method_handle("unact"), method_handle("ssd"), gold_handle(),
         plt.Line2D([], [], color=_tint("#555555", 0.3), lw=1.0, ls=WORST_LS,
                    marker="o", ms=2.6, mfc=_tint("#555555", 0.3), mew=0,
                    label="worst of 5 classes (b, c)")]
    fig.legend(handles=h, loc="lower center", ncol=4, bbox_to_anchor=(0.53, 0.86),
               columnspacing=1.2, handlelength=2.0, borderaxespad=0.1)
    save(fig, "fig_vit_forget_size")
    return agg


if __name__ == "__main__":
    set_style()
    fig_forget_size_main()
    fig_forget_size_full()
    fig_vit_forget_size()
