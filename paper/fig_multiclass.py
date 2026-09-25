"""E4 figure: repeated (sequential) deletion requests.

fig_sequential : forget accuracy on every class requested so far (row a) and
                 retain accuracy (row b) after each of five one-class requests
                 applied to the same model, for five pre-registered class orders
                 per dataset. Thick line: mean over the orders; thin faint lines:
                 one order each. Both methods run at their SELECTED single-class
                 configuration with no re-tuning (common.load_e4_sequences).

Reference: most prefixes of the new orders have no retrained model, so the
dashed grey line is the ORIGINAL model's test accuracy over all classes (before
any request), not a retrained model. The retrained references that exist (the
original order, requests 1/2/3/5) are in tab_multiclass.

Only orders complete for both methods are drawn; incomplete ones are printed.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_hex, to_rgb

from common import (C, DATASETS, DATASET_LABEL, MARK, TEXTW, load_e4_sequences,
                    method_handle, orig_accuracy, save, set_style)

ORIG_LS = (0, (4, 2))



def _tint(hex_, w):
    r = np.array(to_rgb(hex_))
    return to_hex(r * (1 - w) + w)


def fig_sequential():
    seq, _ = load_e4_sequences()
    orig = orig_accuracy()
    fig, axes = plt.subplots(2, 3, figsize=(TEXTW, 3.2), sharex=True)
    # Same D_r span (points) in every panel, so equal gaps look equal.
    tops = {ds: min(100.0, np.ceil((max(seq[seq.dataset == ds].post_retain.max(),
                                        orig[ds]) + 1.5) / 2) * 2) for ds in DATASETS}
    span = max(32, max(np.ceil((tops[ds] - seq[seq.dataset == ds].post_retain.min() + 3) / 4) * 4
                       for ds in DATASETS))
    for j, ds in enumerate(DATASETS):
        s = seq[seq.dataset == ds]
        n_orders = s.order.nunique()
        for key in ("ssd", "unact"):
            t = s[s.method == key]
            z = 4 if key == "unact" else 3
            for i, col in enumerate(("post_forget", "post_retain")):
                ax = axes[i, j]
                for _, g in t.groupby("order"):
                    g = g.sort_values("request")
                    ax.plot(g.request, g[col], color=_tint(C[key], 0.45), lw=0.7,
                            alpha=0.9, zorder=z - 1.5, solid_capstyle="round")
                m = t.groupby("request")[col].mean()
                ax.plot(m.index, m.values, marker=MARK[key], color=C[key], mfc="white",
                        mew=1.1, ms=4, lw=1.5, zorder=z)
        axes[0, j].axhline(0, color=C["gold"], lw=0.6, zorder=1)
        axes[0, j].set_ylim(-4, 104)
        axes[0, j].set_yticks([0, 25, 50, 75, 100])
        axes[1, j].axhline(orig[ds], color=C["gold"], ls=ORIG_LS, lw=1.0, zorder=1)
        axes[1, j].set_ylim(tops[ds] - span, tops[ds])
        axes[0, j].set_title(DATASET_LABEL[ds], pad=3)
        axes[1, j].set_xlabel("Deletion request")
        axes[1, j].set_xticks([1, 2, 3, 4, 5])
        axes[1, j].set_xlim(0.7, 5.3)
        if n_orders < 5:
            print(f"  [sequential] {ds}: {n_orders} of 5 orders drawn")
    axes[0, 0].set_ylabel("Forget accuracy on all\nrequested classes (%)")
    axes[1, 0].set_ylabel("Retain\naccuracy $D_r$ (%)")
    for j in (1, 2):
        axes[0, j].tick_params(labelleft=False)
    fig.subplots_adjust(hspace=0.16, wspace=0.22, left=0.12, right=0.99, top=0.855,
                        bottom=0.125)
    for i, letter in enumerate("ab"):
        bb = axes[i, 0].get_position()
        fig.text(0.0, bb.y1 + 0.01, letter, fontsize=9, fontweight="bold",
                 ha="left", va="bottom")
    h = [method_handle("unact"), method_handle("ssd"),
         plt.Line2D([], [], color=C["gold"], ls=ORIG_LS, lw=1.0,
                    label="Original model (before any request)")]
    fig.legend(handles=h, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.945),
               columnspacing=1.8, handlelength=2.2, borderaxespad=0.1)
    style = [plt.Line2D([], [], color="#555555", lw=1.5, marker="o", ms=4, mfc="white",
                        mew=1.1, label="mean over class orders"),
             plt.Line2D([], [], color=_tint("#555555", 0.45), lw=0.7,
                        label="one class order")]
    fig.legend(handles=style, loc="lower center", ncol=2, bbox_to_anchor=(0.5, 0.895),
               columnspacing=1.8, handlelength=2.2, borderaxespad=0.1)
    save(fig, "fig_sequential")


if __name__ == "__main__":
    set_style()
    fig_sequential()
