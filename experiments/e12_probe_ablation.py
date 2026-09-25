"""
E12: what does Delta measure? Trivial output-layer baselines, an in/out ablation
of UnAct, and representation-level evaluations.

Per dataset and forget class, at every method's SELECTED configuration:

  original     the unmodified model
  gold         retrained without the class
  unact        UnAct as reported (layer4_last, In_j and Out_j attenuated)
  unact_in     the same per-round channel masks, replayed on In_j only
               (conv2 filters + bn2 affine of the last block); fc untouched
  unact_out    the same masks replayed on Out_j only (fc columns); every
               internal representation unchanged
  logit_mask   the class logit is removed (fc row c zeroed, bias -1e4):
               the trivial output-suppression baseline
  ssd, lfssd   as in E2/E9

Metrics: D_r, D_f (test), Delta; top-1 agreement with the retrained model and
KL(gold || model) on the test set; and a linear probe for the forget class on
the penultimate (pooled, 512-d) features: a class-balanced logistic regression
(forget class vs rest) trained on the training-set features of that model and
evaluated on its test-set features (AUROC and balanced accuracy).

In/out replay uses the masks recorded from the full UnAct run so that the three
UnAct rows edit exactly the same channels; only the path differs.

Usage:
    python experiments/e12_probe_ablation.py --datasets cifar10 --device cuda:1
"""
from __future__ import annotations

import argparse
import copy
import os
import subprocess
import sys

import torch
import torch.nn.functional as F

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (_REPO_ROOT, os.path.join(_REPO_ROOT, "Our_Method"),
          os.path.join(_REPO_ROOT, "SSD"),
          os.path.join(_REPO_ROOT, "Baseline CIFAR Training")):
    if p not in sys.path:
        sys.path.insert(0, p)

import common as cifar_common  # noqa: E402
from lfssd import LFParameterPerturber  # noqa: E402
from ssd import ParameterPerturber  # noqa: E402
from unlearn_cnn import _mask_from, profile_block_activations  # noqa: E402

from unlearn_lib import manifest  # noqa: E402
from unlearn_lib.io import upsert_rows  # noqa: E402
from unlearn_lib.loaders import make_loader  # noqa: E402
from unlearn_lib.splits import build_split_indices  # noqa: E402

RESULTS_PATH = os.path.join(_REPO_ROOT, "results", "e12_probe_ablation.csv")
FIELDS = ["dataset", "arch", "forget_class", "method", "config", "seed",
          "d_r", "d_f", "gold_d_r", "gold_d_f", "delta",
          "agree_retain", "agree_forget", "kl_gold",
          "probe_auroc", "probe_bacc", "params_damped_pct",
          "eval_device", "git_commit"]
KEY = ["dataset", "arch", "forget_class", "method", "config", "seed"]

# Selected operating points (paper/common.py).
SSD_SELECTED = {"cifar10": (8.0, 1.0, 256), "cifar20": (15.0, 1.0, 512),
                "cifar100": (50.0, 0.5, 64)}
LFSSD_SELECTED = {"cifar10": (6.0, 0.1, 64), "cifar20": (10.0, 0.1, 64),
                  "cifar100": (30.0, 0.5, 64)}
UNACT_SELECTED = {"cifar10": (99.0, 0.01, 20), "cifar20": (99.0, 0.1, 20),
                  "cifar100": (90.0, 0.3, 1)}
SSD_FIXED = {"lower_bound": 1, "exponent": 1, "magnitude_diff": None,
             "min_layer": -1, "max_layer": -1, "forget_threshold": 1}
FORGET_CLASSES = {"cifar10": [0, 2, 3, 5, 8], "cifar20": [3, 4, 10, 14, 19],
                  "cifar100": [3, 20, 51, 69, 85]}


def _git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       cwd=_REPO_ROOT, text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def unact_with_masks(model, loader, device, p, gamma, k):
    """UnAct (layer4_last) exactly as unlearn_cnn.unlearn_blocks, recording masks."""
    block = model.layer4[-1]
    masks = []
    for _ in range(k):
        mask = _mask_from(profile_block_activations(model, block, loader, device), p)
        masks.append(mask.clone())
        with torch.no_grad():
            block.conv2.weight[mask] *= gamma
            block.bn2.weight[mask] *= gamma
            block.bn2.bias[mask] *= gamma
            model.fc.weight[:, mask] *= gamma
    return masks


def replay(model, masks, gamma, path):
    block = model.layer4[-1]
    with torch.no_grad():
        for mask in masks:
            if path == "in":
                block.conv2.weight[mask] *= gamma
                block.bn2.weight[mask] *= gamma
                block.bn2.bias[mask] *= gamma
            else:
                model.fc.weight[:, mask] *= gamma


def dampen(cls_, cfg, model, forget_loader, full_loader, device):
    alpha, lam, _ = cfg
    pdr = cls_(model, torch.optim.SGD(model.parameters(), lr=0.1), device,
               dict(SSD_FIXED, dampening_constant=lam, selection_weighting=alpha))
    fi = pdr.calc_importance(forget_loader)
    oi = pdr.calc_importance(full_loader)
    pdr.modify_weight(oi, fi)


@torch.no_grad()
def feats_logits(model, loader, device):
    model.eval()
    feats, logits, ys = [], [], []
    h = model.fc.register_forward_hook(lambda m, i, o: feats.append(i[0].detach().float()))
    for batch in loader:
        x, y = batch[0].to(device), batch[1]
        logits.append(model(x).float())
        ys.append(y)
    h.remove()
    return torch.cat(feats), torch.cat(logits), torch.cat(ys).to(device)


def auroc(scores, labels):
    order = torch.argsort(scores)
    ranks = torch.empty_like(scores)
    ranks[order] = torch.arange(1, len(scores) + 1, device=scores.device, dtype=scores.dtype)
    pos = labels.bool()
    n_pos, n_neg = pos.sum().item(), (~pos).sum().item()
    return ((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)).item()


def linear_probe(ftr, ytr, fte, yte, iters=300, l2=1e-4):
    """Class-balanced binary logistic regression, full-batch L-BFGS on GPU."""
    mu, sd = ftr.mean(0), ftr.std(0) + 1e-6
    xtr, xte = (ftr - mu) / sd, (fte - mu) / sd
    ytr, yte = ytr.float(), yte.float()
    pos_w = (1 - ytr).sum() / ytr.sum().clamp(min=1)
    w = torch.zeros(xtr.shape[1], device=xtr.device, requires_grad=True)
    b = torch.zeros(1, device=xtr.device, requires_grad=True)
    opt = torch.optim.LBFGS([w, b], lr=1, max_iter=iters, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = F.binary_cross_entropy_with_logits(xtr @ w + b, ytr, pos_weight=pos_w) \
            + l2 * (w * w).sum()
        loss.backward()
        return loss

    opt.step(closure)
    with torch.no_grad():
        s = xte @ w + b
        pred = (s > 0).float()
        tpr = (pred[yte == 1] == 1).float().mean().item()
        tnr = (pred[yte == 0] == 0).float().mean().item()
        return auroc(s, yte), 50.0 * (tpr + tnr)


def run_dataset(ds, args, device):
    torch.manual_seed(args.seed)
    train_set, test_set = cifar_common.get_datasets(ds, train_augment=False, download=False)
    base, _ = cifar_common.load_model(cifar_common.checkpoint_path(ds))
    base = base.to(device).eval()
    commit = _git_commit()
    dev = torch.cuda.get_device_name(device)
    p, gamma, k = UNACT_SELECTED[ds]
    bs = args.eval_batch_size
    train_loader = make_loader(train_set, torch.arange(len(train_set)), batch_size=bs, workers=args.workers)
    test_loader = make_loader(test_set, torch.arange(len(test_set)), batch_size=bs, workers=args.workers)

    for cls in FORGET_CLASSES[ds]:
        idx = build_split_indices(train_set, test_set, cls)
        forget_loader = make_loader(train_set, idx["forget_train"], batch_size=bs, workers=args.workers)
        gpath = manifest.lookup("retain_gold", "cifar", "resnet18", ds, cls)
        gold, _ = cifar_common.load_model(gpath)
        gold = gold.to(device).eval()

        models = {"original": (copy.deepcopy(base), "none"), "gold": (gold, "retrain")}
        m = copy.deepcopy(base)
        masks = unact_with_masks(m, forget_loader, device, p, gamma, k)
        cfg = f"p{p}_g{gamma}_k{k}"
        models["unact"] = (m, cfg)
        for path in ("in", "out"):
            m2 = copy.deepcopy(base)
            replay(m2, masks, gamma, path)
            models[f"unact_{path}"] = (m2, cfg)
        m = copy.deepcopy(base)
        with torch.no_grad():
            m.fc.weight[cls] = 0
            m.fc.bias[cls] = -1e4
        models["logit_mask"] = (m, "fc_row_c")
        for name, cls_, sel in (("ssd", ParameterPerturber, SSD_SELECTED),
                                ("lfssd", LFParameterPerturber, LFSSD_SELECTED)):
            m = copy.deepcopy(base).eval()
            b = sel[ds][2]
            dampen(cls_, sel[ds], m,
                   make_loader(train_set, idx["forget_train"], batch_size=b, workers=args.workers),
                   make_loader(train_set, idx["full_train"], batch_size=b, workers=args.workers),
                   device)
            models[name] = (m, f"a{sel[ds][0]}_l{sel[ds][1]}_b{b}")

        _, g_te_logits, yte = feats_logits(gold, test_loader, device)
        g_pred = g_te_logits.argmax(1)
        g_logp = F.log_softmax(g_te_logits, 1)
        fmask = yte == cls
        g_dr = (g_pred[~fmask] == yte[~fmask]).float().mean().item() * 100
        g_df = (g_pred[fmask] == yte[fmask]).float().mean().item() * 100
        base_params = {n: q.detach().clone() for n, q in base.named_parameters()}

        rows = []
        for name, (model, cfg) in models.items():
            model.eval()
            ftr, _, ytr = feats_logits(model, train_loader, device)
            fte, te_logits, _ = feats_logits(model, test_loader, device)
            pred = te_logits.argmax(1)
            d_r = (pred[~fmask] == yte[~fmask]).float().mean().item() * 100
            d_f = (pred[fmask] == yte[fmask]).float().mean().item() * 100
            agree_r = (pred[~fmask] == g_pred[~fmask]).float().mean().item() * 100
            agree_f = (pred[fmask] == g_pred[fmask]).float().mean().item() * 100
            logq = F.log_softmax(te_logits.clamp(min=-1e3), 1)
            kl = (g_logp.exp() * (g_logp - logq)).sum(1).mean().item()
            au, bacc = linear_probe(ftr, (ytr == cls), fte, (yte == cls))
            if name == "gold":
                changed = float("nan")
            else:
                changed = 100.0 * sum(int((q.detach() != base_params[n]).sum())
                                      for n, q in model.named_parameters()) \
                    / sum(q.numel() for q in model.parameters())
            rows.append({
                "dataset": ds, "arch": "resnet18", "forget_class": cls, "method": name,
                "config": cfg, "seed": args.seed,
                "d_r": f"{d_r:.4f}", "d_f": f"{d_f:.4f}",
                "gold_d_r": f"{g_dr:.4f}", "gold_d_f": f"{g_df:.4f}",
                "delta": f"{abs(d_r - g_dr) + abs(d_f - g_df):.4f}",
                "agree_retain": f"{agree_r:.4f}", "agree_forget": f"{agree_f:.4f}",
                "kl_gold": f"{kl:.6f}", "probe_auroc": f"{au:.6f}", "probe_bacc": f"{bacc:.4f}",
                "params_damped_pct": f"{changed:.4f}", "eval_device": dev, "git_commit": commit,
            })
            print(f"  {ds} c{cls:<3} {name:<11} D_r {d_r:6.2f} D_f {d_f:6.2f} "
                  f"Delta {abs(d_r-g_dr)+abs(d_f-g_df):6.2f} agreeR {agree_r:6.2f} "
                  f"KL {kl:7.4f} probe AUROC {au:.4f} bAcc {bacc:6.2f}", flush=True)
        upsert_rows(rows, args.results, FIELDS, KEY)
        del models
        torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["cifar10", "cifar20", "cifar100"])
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--eval-batch-size", type=int, default=512)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--results", default=RESULTS_PATH)
    args = ap.parse_args()
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device(args.device)
    for ds in args.datasets:
        run_dataset(ds, args, device)
    print("E12 complete.")


if __name__ == "__main__":
    main()
