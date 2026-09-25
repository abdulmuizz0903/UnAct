"""Shared plotting style and result loaders for the paper figures.

Every figure in paper/figures is produced from a committed results CSV by one of
the scripts in this directory. Nothing here re-runs an experiment or invents a
number: the loaders below only filter and aggregate.

Aggregation conventions:
  * score (Delta in the paper) = |D_r - gold_D_r| + |D_f - gold_D_f|, both in
    percentage points, computed per forget class and then averaged.
  * avg_gap = mean of |dD_f|, |dD_r| and |dMIA| (MIA in points), the three
    Avg. Gap terms of Fan et al. (2024) that we record.
  * Spread is mean +/- sd (or min-max) across the 5 pre-registered forget classes.
  * Baselines are run as their papers run them: ONE operating point per dataset,
    alpha inside the published range [5, 50].
      SSD   (E1): alpha 8/15/50,  lambda 1/1/0.5,   batch 256/512/64
      LFSSD (E9): alpha 6/10/30,  lambda 0.1/0.1/0.5, batch 64
    Rows that re-tune alpha per setting are never used: e3 "ssd_alphagrid"
    and the e4 override rows other than the one equal to the selected
    single-class value.
  * Timings come from the serialised L4 cost files only (cost_breakdown_l4*.csv),
    with the warm-up repeat (repeat == 0) discarded.
"""
from __future__ import annotations

import os
import re

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(REPO, "results")
FIGDIR = os.path.join(REPO, "paper", "figures")
TABDIR = os.path.join(REPO, "paper", "tables")

DATASETS = ["cifar10", "cifar20", "cifar100"]
DATASET_LABEL = {"cifar10": "CIFAR-10", "cifar20": "CIFAR-20", "cifar100": "CIFAR-100"}

# ICLR text block is 5.5 in wide. Figures are drawn at that width so that the
# font sizes set below are the sizes printed on the page (no down-scaling).
TEXTW = 5.5

# Okabe-Ito hues, one per method, used identically in every figure.
# Validated with the dataviz validator (all-pairs, light surface): worst CVD
# dE 11.0, worst normal-vision dE 18.5, all >= 3:1 contrast. Methods also carry
# distinct markers (MARK) so identity never rests on colour alone.
C = {
    "unact": "#0173B2",
    "ssd": "#D55E00",
    "lfssd": "#029E73",
    "gold": "#3A3A3A",
    "baseline": "#9A9A9A",
    "accent": "#CC78BC",
}
MARK = {"unact": "o", "ssd": "s", "lfssd": "^", "gold": "D", "baseline": "x"}
METHOD_LABEL = {
    "unact": "UnAct (ours)",
    "ssd": "SSD",
    "lfssd": "LFSSD",
    "gold": "Retrain",
    "baseline": "Original model",
}
# Single-hue ordinal ramps for series *within* one method (e.g. k for UnAct,
# batch size for SSD, class count). Both pass the validator's --ordinal checks.
RAMP_UNACT = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
RAMP_SSD = ["#EE9A62", "#D55E00", "#8A3C00"]
RAMP_LFSSD = ["#5CC49B", "#029E73", "#01573F"]
RAMP_DS = {"cifar10": "#86b6ef", "cifar20": "#2a78d6", "cifar100": "#104281"}
GOLD_LS = (0, (4, 2))


def set_style():
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["STIXGeneral", "Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 8,
        "axes.titlesize": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "axes.linewidth": 0.6,
        "axes.edgecolor": "#555555",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": "#E3E3E3",
        "grid.linewidth": 0.5,
        "grid.linestyle": "-",
        "axes.axisbelow": True,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.minor.width": 0.4,
        "ytick.minor.width": 0.4,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "xtick.minor.size": 1.5,
        "ytick.minor.size": 1.5,
        "xtick.color": "#333333",
        "ytick.color": "#333333",
        "lines.linewidth": 1.4,
        "lines.markersize": 4,
        "legend.frameon": False,
        "legend.handlelength": 1.8,
        "legend.columnspacing": 1.4,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def save(fig, name):
    os.makedirs(FIGDIR, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIGDIR, f"{name}.{ext}"))
    plt.close(fig)
    print(f"  wrote figures/{name}.pdf (+.png)")


def panel_label(ax, letter, x=-0.02, y=1.02, **kw):
    """Bold lower-case panel letter just outside the top-left corner."""
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=9, fontweight="bold",
            ha="right", va="bottom", **kw)


def row_label(fig, ax, letter, dx=0.0):
    """Panel letter for a row of small multiples, placed left of its y-label."""
    bb = ax.get_position()
    fig.text(0.0 + dx, bb.y1 + 0.005, letter, fontsize=9, fontweight="bold",
             ha="left", va="bottom")


def method_handle(key, **kw):
    style = dict(color=C[key], marker=MARK[key], mfc="white", mew=1.1,
                 label=METHOD_LABEL[key])
    style.update(kw)
    return plt.Line2D([], [], **style)


def gold_handle(label=None):
    return plt.Line2D([], [], color=C["gold"], ls=GOLD_LS, lw=1.0,
                      label=label or METHOD_LABEL["gold"])


def _score(df, post_r="post_retain", post_f="post_forget"):
    return (df[post_r] - df.gold_retain).abs() + (df[post_f] - df.gold_forget).abs()


def _read(name):
    path = os.path.join(RESULTS, name)
    return pd.read_csv(path) if os.path.exists(path) else None


# --------------------------------------------------------------------------
# Selected operating points (E1 / E6 / E9), fixed by the pre-registered rule.
# --------------------------------------------------------------------------
SSD_SELECTED = {"cifar10": (8.0, 1.0, 256), "cifar20": (15.0, 1.0, 512),
                "cifar100": (50.0, 0.5, 64)}
LFSSD_SELECTED = {"cifar10": (6.0, 0.1, 64), "cifar20": (10.0, 0.1, 64),
                  "cifar100": (30.0, 0.5, 64)}
UNACT_SELECTED = {"cifar10": (99.0, 0.01, 20), "cifar20": (99.0, 0.1, 20),
                  "cifar100": (90.0, 0.3, 1)}
GOLD_D_R = {"cifar10": 95.60, "cifar20": 85.69, "cifar100": 77.71}
# ViT (E5b stage 2, exploratory): UnAct class-token DLA score; SSD selected in E5.
VIT_UNACT = ("ffn_dla_last4", 85.0, 0.1, 5)
VIT_SSD = (5.0, 0.1, 128)


# --------------------------------------------------------------------------
# Grids
# --------------------------------------------------------------------------
def load_e1():
    """SSD tuning grid, 1350 rows -> per-config means over the 5 forget classes."""
    d = pd.read_csv(os.path.join(RESULTS, "e1_ssd_tune.csv"))
    d["score"] = _score(d)
    return d.groupby(["dataset", "alpha", "lambda", "batch_size"], as_index=False).agg(
        score=("score", "mean"), score_sd=("score", "std"),
        d_r=("post_retain", "mean"), d_f=("post_forget", "mean"),
        damped=("params_damped_pct", "mean"),
    )


def load_e9_grid():
    """LFSSD tuning grid (E9a), 1350 rows, same axes and budget as E1."""
    d = pd.read_csv(os.path.join(RESULTS, "e9_lfssd_tune.csv"))
    d["score"] = _score(d)
    return d.groupby(["dataset", "alpha", "lambda", "batch_size"], as_index=False).agg(
        score=("score", "mean"), score_sd=("score", "std"),
        d_r=("post_retain", "mean"), d_f=("post_forget", "mean"),
        n=("score", "size"),
    )


def load_e6():
    """UnAct grid, 1500 rows -> per-config means over the 5 forget classes."""
    d = pd.read_csv(os.path.join(RESULTS, "e6_unact_grid.csv"))
    d["score"] = _score(d)
    return d.groupby(["dataset", "percentile", "gamma", "iters"], as_index=False).agg(
        score=("score", "mean"), score_sd=("score", "std"),
        d_r=("post_retain", "mean"), d_f=("post_forget", "mean"),
        coverage=("cum_coverage_pct", "mean"),
        bound=("coverage_bound_pct", "mean"),
        units=("units_per_round", "mean"),
        damped=("params_damped_pct", "mean"),
    )


def best_per_k(e6=None):
    """Best UnAct configuration at each number of rounds k (same rule, k fixed)."""
    e6 = load_e6() if e6 is None else e6
    idx = e6.groupby(["dataset", "iters"]).score.idxmin()
    return e6.loc[idx].reset_index(drop=True)


# --------------------------------------------------------------------------
# Main table (full forget set)
# --------------------------------------------------------------------------
def _agg_main(d):
    d = d.copy()
    d["score"] = d.gap_d_r.abs() + d.gap_d_f.abs()
    d["avg_gap"] = (d.gap_d_f.abs() + d.gap_d_r.abs() + 100 * d.gap_mia.abs()) / 3
    return d.groupby(["dataset", "method", "config"], as_index=False).agg(
        score=("score", "mean"), score_sd=("score", "std"),
        avg_gap=("avg_gap", "mean"),
        overall=("d_overall", "mean"),
        d_r=("d_r", "mean"), d_r_sd=("d_r", "std"),
        d_f=("d_f", "mean"), d_f_sd=("d_f", "std"),
        mia=("mia", "mean"), mia_sd=("mia", "std"),
        gold_mia=("gold_mia", "mean"),
        zrf=("zrf", "mean"),
        gap_d_r=("gap_d_r", lambda s: s.abs().mean()),
        gap_d_f=("gap_d_f", lambda s: s.abs().mean()),
        time_s=("method_time_s", "mean"),
        damped=("params_damped_pct", "mean"),
        devices=("eval_device", lambda s: ",".join(sorted(set(s)))),
        n=("d_r", "size"),
    )


def load_main():
    """Main table: E2 re-run on the L4 (baseline, gold, SSD, UnAct at selected
    configs) plus LFSSD at its E9 selection (results/e9_baselines.csv)."""
    e2 = pd.read_csv(os.path.join(RESULTS, "e2_main_table_l4.csv"))
    e9 = pd.read_csv(os.path.join(RESULTS, "e9_baselines.csv"))
    e9 = e9[e9.method == "lfssd"]
    return _agg_main(pd.concat([e2, e9], ignore_index=True))


# --------------------------------------------------------------------------
# Forget-set size (E3)
# --------------------------------------------------------------------------
def _agg_e3(x):
    return x.groupby(["dataset", "forget_n"], as_index=False).agg(
        score=("score", "mean"), score_sd=("score", "std"),
        score_min=("score", "min"), score_max=("score", "max"),
        d_f=("post_forget", "mean"), d_f_sd=("post_forget", "std"),
        d_f_min=("post_forget", "min"), d_f_max=("post_forget", "max"),
        d_r=("post_retain", "mean"), d_r_sd=("post_retain", "std"),
        d_r_min=("post_retain", "min"), d_r_max=("post_retain", "max"),
        gold_r=("gold_retain", "mean"),
        time_s=("method_time_s", "mean"),
        n_classes=("score", "size"),
    )


def load_e3():
    """Forget-set-size sweep, (unact, ssd) at their selected configs at every n.
    The per-class alpha oracle (method == 'ssd_alphagrid') is never used."""
    d = pd.read_csv(os.path.join(RESULTS, "e3_forget_size.csv"))
    return _agg_e3(d[d.method == "unact"]), _agg_e3(d[d.method == "ssd"])


def load_e3_all():
    """dict method -> aggregate, including LFSSD from e3_lfssd_forget_size.csv
    (its E9 selection at every n). Only cells with all 5 classes are kept, so a
    partially written file never yields a mean over an easy subset."""
    unact, ssd = load_e3()
    out = {"unact": unact, "ssd": ssd}
    lf = _read("e3_lfssd_forget_size.csv")
    if lf is not None and len(lf):
        lf = lf[lf.method == "lfssd"]
        a = _agg_e3(lf)
        out["lfssd"] = a[a.n_classes == 5].reset_index(drop=True)
    return out


def load_e3_raw():
    d = pd.read_csv(os.path.join(RESULTS, "e3_forget_size.csv"))
    d = d[d.method.isin(["unact", "ssd"])]
    lf = _read("e3_lfssd_forget_size.csv")
    if lf is not None:
        d = pd.concat([d, lf[lf.method == "lfssd"]], ignore_index=True)
    return d


# --------------------------------------------------------------------------
# Multi-class and sequential requests (E4)
# --------------------------------------------------------------------------
_SEL_OVERRIDE = {"ssd": {d: v[0] for d, v in SSD_SELECTED.items()},
                 "unact": {d: v[0] for d, v in UNACT_SELECTED.items()}}


def load_e4():
    """E4 rows at each method's SELECTED single-class configuration only.

    e4_multiclass.py was run with --tune: every row's config ends in
    `_ov<value>`, the value replacing alpha (SSD) or p (UnAct). The row whose
    override equals the selected single-class value is exactly the selected
    configuration; all other overrides are a per-task re-tuning (SSD's grid also
    contains alpha = 4, outside the published range) and are excluded for both
    methods symmetrically.
    """
    d = pd.read_csv(os.path.join(RESULTS, "e4_multiclass.csv"))
    d["override"] = d.config.str.extract(r"_ov([0-9.]+)$", expand=False).astype(float)
    sel = d.apply(lambda r: r.override == _SEL_OVERRIDE[r.method][r.dataset], axis=1)
    d = d[sel].copy()
    d["drop_r"] = d.gold_retain - d.post_retain
    return d.sort_values(["dataset", "mode", "method", "k"]).reset_index(drop=True)


# Pre-registered class orders for the sequential experiment. The first order of
# each dataset is the original E4 run (e4_multiclass.csv, selected `_ov` rows);
# the other four are E4b (e4b_seq_orders.csv, selected configs, no override).
SEQ_ORDERS = {
    "cifar10": [(0, 2, 3, 5, 8), (5, 6, 1, 2, 0), (8, 7, 1, 5, 6), (6, 0, 3, 7, 8),
                (0, 4, 9, 6, 7)],
    "cifar20": [(3, 4, 10, 14, 19), (5, 13, 2, 11, 19), (8, 1, 19, 0, 10),
                (6, 12, 3, 7, 1), (10, 14, 15, 2, 5)],
    "cifar100": [(3, 20, 51, 69, 85), (45, 15, 90, 32, 35), (48, 97, 1, 81, 90),
                 (86, 42, 89, 92, 3), (30, 13, 49, 91, 37)],
}
_SEQ_BASE_CFG = {"ssd": {d: f"a{v[0]:.1f}_l{v[1]:.1f}_b{v[2]}" for d, v in SSD_SELECTED.items()},
                 "unact": {d: f"p{v[0]:.1f}_g{v[1]:g}_k{v[2]}" for d, v in UNACT_SELECTED.items()}}


def _spec(classes):
    return "-".join(str(c) for c in sorted(classes))


def load_e4_sequences(verbose=True):
    """Sequential requests over every pre-registered class order.

    Returns (rows, missing). `rows` has one row per (dataset, order, method,
    request) with request = 1..5, post_retain / post_forget (forget accuracy on
    ALL classes requested so far) and gold_retain where a retrained model exists
    (only for prefixes of the original order and for single classes that were
    pre-registered). Only orders with all 5 requests for BOTH methods are
    returned, so a half-written file never yields a mean over an easy prefix.

    e4b_seq_orders.csv is kept sorted by upsert_rows, so its rows are not in run
    order: each request is found by its forget_spec, which is the sorted prefix
    of the known order (request_idx = i for the prefix of length i + 1).
    """
    e4 = load_e4()
    e4 = e4[e4["mode"] == "sequential"].copy()
    e4b = _read("e4b_seq_orders.csv")
    if e4b is not None:
        e4b = e4b[e4b["mode"] == "sequential"].copy()
        ok = e4b.apply(lambda r: r.config == _SEQ_BASE_CFG[r.method][r.dataset], axis=1)
        e4b = e4b[ok]
    rows, missing = [], []
    for ds, orders in SEQ_ORDERS.items():
        for oi, order in enumerate(orders):
            src = e4 if oi == 0 else e4b
            got = []
            for meth in ("ssd", "unact"):
                for i in range(5):
                    spec = _spec(order[:i + 1])
                    if src is None:
                        r = []
                    else:
                        r = src[(src.dataset == ds) & (src.method == meth)
                                & (src.forget_spec == spec) & (src.request_idx == i)]
                    if len(r) != 1:
                        break
                    r = r.iloc[0]
                    got.append(dict(dataset=ds, order=",".join(map(str, order)),
                                    order_idx=oi, method=meth, request=i + 1,
                                    post_retain=r.post_retain, post_forget=r.post_forget,
                                    gold_retain=r.gold_retain, gold_forget=r.gold_forget,
                                    device=r.eval_device,
                                    source="e4_multiclass.csv" if oi == 0
                                    else "e4b_seq_orders.csv"))
            if len(got) == 10:
                rows.extend(got)
            else:
                missing.append((ds, order, len(got)))
    if verbose:
        for ds, order, n in missing:
            print(f"  [sequential] incomplete order {ds} {order}: {n}/10 rows, not used")
    return pd.DataFrame(rows), missing


def orig_accuracy():
    """Original (pre-unlearning) model's test accuracy over all classes, from
    the baseline rows of the L4 main table."""
    m = load_main()
    b = m[m.method == "baseline"].set_index("dataset")
    return b.overall.to_dict()


# --------------------------------------------------------------------------
# Cost (serialised L4 re-times)
# --------------------------------------------------------------------------
COST_FILES = ["cost_breakdown_l4.csv", "cost_breakdown_l4_k1.csv",
              "cost_breakdown_l4_k5.csv"]


def _parse_unact_cfg(s):
    m = re.match(r"p([0-9.]+)_g([0-9.]+)_k(\d+)", s)
    return float(m.group(1)), float(m.group(2)), int(m.group(3))


def load_cost_l4(drop_warmup=True):
    """One row per (dataset, UnAct config): UnAct time and SSD time measured in
    the SAME serialised session (paired), warm-up repeat discarded. SSD always
    runs its selected configuration on the full forget class (class 3)."""
    frames = []
    for f in COST_FILES:
        d = _read(f)
        if d is None:
            continue
        d = d.assign(source=f)
        frames.append(d)
    d = pd.concat(frames, ignore_index=True)
    if drop_warmup:
        d = d[d.repeat > 0]
    g = d.groupby(["dataset", "source", "unact_config"], as_index=False).agg(
        unact_s=("unact_s", "mean"), unact_min=("unact_s", "min"),
        unact_max=("unact_s", "max"),
        ssd_forget=("ssd_forget_s", "mean"), ssd_full=("ssd_full_s", "mean"),
        ssd_modify=("ssd_modify_s", "mean"), ssd_cold=("ssd_cold_s", "mean"),
        ssd_amort=("ssd_amortised_s", "mean"), repeats=("unact_s", "size"),
        forget_class=("forget_class", "first"), n_forget=("n_forget", "first"),
    )
    pk = g.unact_config.map(_parse_unact_cfg)
    g["percentile"] = [x[0] for x in pk]
    g["gamma"] = [x[1] for x in pk]
    g["iters"] = [x[2] for x in pk]
    e6 = load_e6()
    g = g.merge(e6[["dataset", "percentile", "gamma", "iters", "score"]],
                on=["dataset", "percentile", "gamma", "iters"], how="left")
    return g.sort_values(["dataset", "iters"]).reset_index(drop=True)


def load_cost_ssd_pooled(drop_warmup=True):
    """SSD cost pooled over every serialised L4 session (identical computation),
    with the session-to-session range, for plotting a single SSD reference."""
    frames = [x for x in (_read(f) for f in COST_FILES) if x is not None]
    d = pd.concat([f.assign(source=n) for f, n in zip(frames, COST_FILES)],
                  ignore_index=True)
    if drop_warmup:
        d = d[d.repeat > 0]
    s = d.groupby(["dataset", "source"], as_index=False)[
        ["ssd_cold_s", "ssd_amortised_s", "ssd_full_s"]].mean()
    return s.groupby("dataset").agg(
        cold=("ssd_cold_s", "mean"), cold_min=("ssd_cold_s", "min"),
        cold_max=("ssd_cold_s", "max"),
        amort=("ssd_amortised_s", "mean"), amort_min=("ssd_amortised_s", "min"),
        amort_max=("ssd_amortised_s", "max"),
        full=("ssd_full_s", "mean"), sessions=("source", "size"),
    )


# --------------------------------------------------------------------------
# ViT-B/16 (E5b stage 2 and E5 SSD selection)
# --------------------------------------------------------------------------
def load_vit():
    """Per-class ViT rows at fp32 on the full retain split.

    UnAct: e5b_vit_explore.csv, profile_n == 0 (stage 2), every scope variant.
    SSD:   e5b_vit_ssd_explore.csv, eval_retain_n == 0 (selected alpha 5,
           lambda 0.1, batch 128). The failed pre-registered magnitude grid
           (e5_vit_unact.csv) is not read.
    """
    u = pd.read_csv(os.path.join(RESULTS, "e5b_vit_explore.csv"))
    u = u[(u.profile_n == 0) & (u.eval_retain_n == 0) & (u.eval_dtype == "fp32")].copy()
    u["method"] = "unact"
    s = pd.read_csv(os.path.join(RESULTS, "e5b_vit_ssd_explore.csv"))
    s = s[(s.eval_retain_n == 0) & (s.eval_dtype == "fp32")].copy()
    s["method"] = "ssd"
    s["scope"] = "ssd"
    for d in (u, s):
        d["score"] = _score(d)
    return u, s


E5D_EXPECTED = {"classes": 5, "methods": ("unact", "ssd"), "n": 7}


def load_e5d(verbose=True):
    """ViT forget-set-size sweep (results/e5d_vit_forget_size.csv): UnAct and SSD
    at their selected ViT configurations at every n, fp32, full retain split.

    forget_n == 0 in the file means the whole class; it is mapped to the number
    of forget images available. Returns (agg, raw) only when every (method, n)
    cell has all five forget classes (5 x 2 x 7 = 70 rows), otherwise
    (None, None), so a half-written file never yields a mean over an easy subset.
    """
    d = _read("e5d_vit_forget_size.csv")
    if d is None or not len(d):
        return None, None
    d = d[(d.eval_dtype == "fp32") & (d.eval_retain_n == 0)
          & d.method.isin(E5D_EXPECTED["methods"])].copy()
    d["forget_n"] = np.where(d.forget_n == 0, d.forget_available, d.forget_n).astype(int)
    d["score"] = _score(d)
    cells = d.groupby(["method", "forget_n"]).forget_class.nunique()
    need = len(E5D_EXPECTED["methods"]) * E5D_EXPECTED["n"]
    if len(cells) < need or (cells < E5D_EXPECTED["classes"]).any():
        if verbose:
            print(f"  [e5d] incomplete: {int((cells == 5).sum())}/{need} complete cells, "
                  f"{len(d)} rows")
        return None, None
    agg = d.groupby(["method", "forget_n"], as_index=False).agg(
        score=("score", "mean"), d_r=("post_retain", "mean"),
        d_r_min=("post_retain", "min"), d_f=("post_forget", "mean"),
        d_f_max=("post_forget", "max"),
        gold_r=("gold_retain", "mean"), n_classes=("score", "size"))
    return agg, d


def load_e5c(lr=0.0005):
    """ViT relearn probe. Only lr == 5e-4 is valid: at lr 0.01 the retrained
    model's own retain accuracy collapses, so those rows measure damage."""
    d = _read("e5c_vit_relearn.csv")
    if d is None:
        return None
    return d[np.isclose(d.lr, lr)].copy()


# --------------------------------------------------------------------------
# Other experiments
# --------------------------------------------------------------------------
def load_e7():
    d = pd.read_csv(os.path.join(RESULTS, "e7_relearn.csv"))
    return d.groupby(["dataset", "relearn_m", "method", "step"], as_index=False).agg(
        d_f=("d_f", "mean"), d_f_sd=("d_f", "std"), d_r=("d_r", "mean"),
        baseline_d_f=("baseline_d_f", "mean"),
    )


def load_e8():
    d = pd.read_csv(os.path.join(RESULTS, "e8_selectivity.csv"))
    return d.groupby(["dataset", "num_classes", "percentile"], as_index=False).agg(
        sel=("sel_mean_selectivity", "mean"), sel_sd=("sel_mean_selectivity", "std"),
        unsel=("unsel_mean_selectivity", "mean"),
        forget_mass=("forget_mass_pct", "mean"), forget_mass_sd=("forget_mass_pct", "std"),
        retain_mass=("retain_mass_pct", "mean"), retain_mass_sd=("retain_mass_pct", "std"),
        ratio=("mass_ratio", "mean"), ratio_sd=("mass_ratio", "std"),
        n_sel=("n_selected", "mean"),
    )


def load_e11():
    """Held-out classes SSD's own paper reports failures on. Returns one frame
    with every method's row at its already-selected 5-class configuration, plus
    each method's per-class grid optimum (an oracle, labelled as such)."""
    keys = {"ssd": ["alpha", "lambda", "batch_size"],
            "lfssd": ["alpha", "lambda", "batch_size"],
            "unact": ["percentile", "gamma", "iters"]}
    sel = {"ssd": SSD_SELECTED, "lfssd": LFSSD_SELECTED, "unact": UNACT_SELECTED}
    rows = []
    for m in ("ssd", "lfssd", "unact"):
        d = pd.read_csv(os.path.join(RESULTS, f"e11_hard_classes_{m}.csv"))
        d["score"] = _score(d)
        k = keys[m]
        for (ds, c), g in d.groupby(["dataset", "forget_class"]):
            v = sel[m][ds]
            r = g[(g[k[0]] == v[0]) & (g[k[1]] == v[1]) & (g[k[2]] == v[2])]
            b = g.loc[g.score.idxmin()]
            rows.append(dict(dataset=ds, forget_class=int(c), method=m,
                             d_r=r.post_retain.iloc[0], d_f=r.post_forget.iloc[0],
                             gold_r=r.gold_retain.iloc[0], score=r.score.iloc[0],
                             oracle_score=b.score,
                             oracle_cfg=tuple(b[x] for x in k), n_grid=len(g)))
    return pd.DataFrame(rows)
