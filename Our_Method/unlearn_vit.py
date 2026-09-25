"""
unlearn_vit.py

UnAct (activation-magnitude dampening) for ViT-B/16. Written from scratch rather
than retrofitted onto unlearn_cnn.py, because the unit being selected is
different in kind -- but the *selection rule* is shared verbatim
(`unlearn_cnn._mask_from`), so the semantics of `p` are identical across
architectures.

What plays the role of a conv channel
-------------------------------------
The FFN hidden neuron: the 3072-dimensional intermediate inside each encoder
block's `mlp` = Sequential(Linear(768->3072), GELU, Dropout, Linear(3072->768),
Dropout). Unit j has

    incoming  mlp[0].weight[j, :], mlp[0].bias[j]
    outgoing  mlp[3].weight[:, j]

exactly the incoming/outgoing pair that `conv2.weight[c]` / `bn2.{weight,bias}[c]`
/ `fc.weight[:, c]` form for a ResNet channel, and -- unlike a conv channel --
it is genuinely block-local: it is written once and read once.

**Residual-stream dimensions are deliberately NOT used.** All 12 blocks read and
write the same 768-d stream, so masking dimension d is not a block-local edit at
all; and `encoder.ln` re-normalises the stream before the classifier, which
restores the scale of anything attenuated there. Both properties break the
method's premise, so the 3072-d FFN interior is the only faithful analogue.

Attention heads (scope `attn_last3`) are the second natural unit: head h owns
the column slice `self_attention.out_proj.weight[:, 64h:64(h+1)]`, so
attenuating that slice scales head h's contribution to the residual stream and
nothing else.

Profiling
---------
Mean over the forget set of the GELU output inside `mlp` (the output of
`mlp[1]`), pooled over tokens. `token_pool='mean'` is the default and is the
direct analogue of the spatial mean used for conv feature maps; `token_pool='cls'`
reads the class token only, which is what the classifier ultimately consumes.

Proposition 1 does NOT hold exactly here
----------------------------------------
For a ReLU/conv block, scaling incoming and outgoing weights by gamma scales the
unit's contribution by exactly gamma^2, because ReLU is positively homogeneous:
ReLU(gamma z) = gamma ReLU(z) for gamma > 0. GELU is not
(GELU(gamma z) != gamma GELU(z)), so on ViT the realised attenuation is only
approximately gamma^2 -- and, because GELU saturates towards the identity for
large positive z and towards 0 for small z, shrinking the pre-activation moves a
unit into the *sub*-linear part of the curve, so the realised factor is smaller
than gamma^2 (more attenuation than the proposition predicts, not less).
`measure_attenuation()` measures the realised ratio so this is reported rather
than assumed.

Usage:
    from unlearn_vit import unlearn_vit, SCOPES
    unlearn_vit(model, forget_batches, device, scope="ffn_last3",
                percentile=99.0, gamma=0.1, iters=5)
"""
from __future__ import annotations

import os
import sys
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

# The selection rule is shared verbatim so that `p` means the same thing on both
# architectures, including its off-by-one behaviour: `>=` against an
# interpolated quantile selects ceil(n(1-p/100)) units, not n(1-p/100).
from unlearn_cnn import _mask_from  # noqa: E402

SCOPES = ("head_only", "ffn_last1", "ffn_last3", "ffn_last6", "ffn_all", "attn_last3")
# E5b (exploratory, designed AFTER the pre-registered E5 grid failed). Kept out of
# SCOPES so the pre-registered default grid is unchanged.
#   ffn_pos_last4   the original magnitude rule on blocks 8-11, but profiling the
#                   positive part of GELU. Plain GELU means are negative for ~99% of
#                   late-block units, so an attenuated unit (output -> 0) ranks
#                   *higher* next round and is re-selected forever: k>1 was a no-op.
#   ffn_dla_last4/2 FFN units scored by direct logit attribution to the forget class
#                   at the CLS token (the only token the classifier reads).
#   head_dla        the 768 classifier-input dims, scored the same way.
# All three remain retain-free, gradient-free and label-free (the forget class is
# the argmax of the unedited model's mean forget-set logits).
EXPLORATORY_SCOPES = ("ffn_pos_last4", "ffn_dla_last4", "ffn_dla_last2", "head_dla")
ALL_SCOPES = SCOPES + EXPLORATORY_SCOPES
TOKEN_POOLS = ("mean", "cls")

N_BLOCKS = 12
MLP_DIM = 3072
HIDDEN_DIM = 768
N_HEADS = 12
HEAD_DIM = HIDDEN_DIM // N_HEADS


# --------------------------------------------------------------------------
# Scope -> which blocks, and which unit kind
# --------------------------------------------------------------------------
def scope_spec(scope: str) -> Tuple[str, List[int]]:
    """(unit_kind, block indices) for a scope name.

    unit_kind is 'ffn' (3072 hidden neurons per block), 'attn' (12 heads per
    block) or 'head' (the 768-d feature feeding the classifier).
    """
    if scope == "head_only":
        return "head", []
    if scope == "ffn_last1":
        return "ffn", [N_BLOCKS - 1]
    if scope == "ffn_last3":
        return "ffn", list(range(N_BLOCKS - 3, N_BLOCKS))
    if scope == "ffn_last6":
        return "ffn", list(range(N_BLOCKS - 6, N_BLOCKS))
    if scope == "ffn_all":
        return "ffn", list(range(N_BLOCKS))
    if scope == "attn_last3":
        return "attn", list(range(N_BLOCKS - 3, N_BLOCKS))
    if scope in ("ffn_pos_last4", "ffn_dla_last4"):
        return "ffn", list(range(N_BLOCKS - 4, N_BLOCKS))
    if scope == "ffn_dla_last2":
        return "ffn", list(range(N_BLOCKS - 2, N_BLOCKS))
    if scope == "head_dla":
        return "head", []
    raise ValueError(f"Unknown scope {scope!r}; expected one of {ALL_SCOPES}")


def scope_rule(scope: str) -> str:
    """'mag' (original), 'pos' (positive-part magnitude) or 'dla' (logit attribution)."""
    if scope == "ffn_pos_last4":
        return "pos"
    if scope in ("ffn_dla_last4", "ffn_dla_last2", "head_dla"):
        return "dla"
    return "mag"


def _mask_topk(scores: torch.Tensor, percentile: float) -> torch.Tensor:
    """Exactly the count `_mask_from` selects on tie-free scores, ties broken by index.

    `_mask_from` uses `>=` against the interpolated quantile, which selects
    n - ceil((n-1) q) units when the scores are distinct -- but EVERY tied unit
    when the quantile lands on a tie. Positive-part GELU at the CLS token is
    exactly 0 for most units, so `>=` would select the whole layer.
    """
    import math
    n = scores.numel()
    k = max(1, n - math.ceil((n - 1) * percentile / 100.0 - 1e-9))
    mask = torch.zeros(n, dtype=torch.bool, device=scores.device)
    mask[torch.topk(scores, k).indices] = True
    return mask


def select_mask(scores: torch.Tensor, percentile: float, scope: str) -> torch.Tensor:
    """The pre-registered rule for the original scopes; tie-safe top-k for E5b."""
    return _mask_from(scores, percentile) if scope_rule(scope) == "mag" else \
        _mask_topk(scores, percentile)


def n_units(scope: str) -> int:
    kind, _ = scope_spec(scope)
    return {"ffn": MLP_DIM, "attn": N_HEADS, "head": HIDDEN_DIM}[kind]


# --------------------------------------------------------------------------
# Activation profiling
# --------------------------------------------------------------------------
def _pool(t: torch.Tensor, token_pool: str) -> torch.Tensor:
    """[B, T, U] -> [B, U]. 'mean' pools over tokens, 'cls' takes token 0."""
    if token_pool == "mean":
        return t.mean(dim=1)
    if token_pool == "cls":
        return t[:, 0]
    raise ValueError(f"Unknown token_pool {token_pool!r}; expected one of {TOKEN_POOLS}")


@torch.no_grad()
def profile_ffn(model, blocks_idx: Sequence[int], forget_batches, device,
                token_pool: str = "mean", amp: bool = True,
                positive: bool = False) -> Dict[int, torch.Tensor]:
    """Mean GELU output per FFN hidden unit, per requested block.

    One forward pass serves every block: hooks are registered on all of them at
    once, so scope breadth costs nothing extra at profiling time.
    """
    layers = list(model.encoder.layers)
    sums = {b: None for b in blocks_idx}
    count = 0
    handles = []

    def make_hook(b):
        def hook(_mod, _inp, out):
            o = out.detach().float()
            if positive:
                o = o.clamp_min(0)
            v = _pool(o, token_pool).sum(dim=0)  # [3072]
            sums[b] = v if sums[b] is None else sums[b] + v
        return hook

    for b in blocks_idx:
        handles.append(layers[b].mlp[1].register_forward_hook(make_hook(b)))
    model.eval()
    try:
        for x, _ in forget_batches:
            x = x.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16,
                                enabled=amp and device.type == "cuda"):
                model(x)
            count += x.shape[0]
    finally:
        for h in handles:
            h.remove()
    return {b: sums[b] / max(count, 1) for b in blocks_idx}


@torch.no_grad()
def profile_attn(model, blocks_idx: Sequence[int], forget_batches, device,
                 token_pool: str = "mean", amp: bool = True) -> Dict[int, torch.Tensor]:
    """Mean L2 norm of each attention head's output, per requested block.

    The quantity wanted is the per-head 64-dim output *before* `out_proj` mixes
    the heads. It cannot be read with a hook: `nn.MultiheadAttention` calls
    `F.multi_head_attention_forward`, which uses `out_proj.weight` as a plain
    tensor, so `out_proj` is never invoked as a module and a hook on it never
    fires (verified on torch 2.11, with and without the fused fast path). It is
    therefore recomputed from the block's `ln_1` output -- the exact tensor the
    attention module is called on -- using that module's own `in_proj_weight`.
    Only the requested blocks are recomputed, so this costs a few percent of one
    forward pass.
    """
    import torch.nn.functional as F

    layers = list(model.encoder.layers)
    sums = {b: None for b in blocks_idx}
    count = 0
    handles = []

    def make_hook(b):
        attn = layers[b].self_attention

        def hook(_mod, _inp, out):
            h = out.detach()                                   # [B, T, 768]
            B, T, _ = h.shape
            qkv = F.linear(h, attn.in_proj_weight, attn.in_proj_bias)
            q, k, v = qkv.chunk(3, dim=-1)
            shape = (B, T, N_HEADS, HEAD_DIM)
            q = q.view(shape).transpose(1, 2)
            k = k.view(shape).transpose(1, 2)
            v = v.view(shape).transpose(1, 2)
            o = F.scaled_dot_product_attention(q, k, v)        # [B, H, T, 64]
            norms = o.float().norm(dim=-1).transpose(1, 2)     # [B, T, H]
            val = _pool(norms, token_pool).sum(dim=0)          # [H]
            sums[b] = val if sums[b] is None else sums[b] + val
        return hook

    for b in blocks_idx:
        handles.append(layers[b].ln_1.register_forward_hook(make_hook(b)))

    model.eval()
    try:
        for x, _ in forget_batches:
            x = x.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16,
                                enabled=amp and device.type == "cuda"):
                model(x)
            count += x.shape[0]
    finally:
        for h in handles:
            h.remove()

    if any(v is None for v in sums.values()):
        raise RuntimeError("attention profiling captured nothing")
    return {b: sums[b] / max(count, 1) for b in blocks_idx}


@torch.no_grad()
def profile_head(model, forget_batches, device, amp: bool = True) -> torch.Tensor:
    """Mean activation of the 768-d feature entering `heads.head`.

    The direct analogue of unlearn_cnn.profile_activations, which profiles the
    512-d pooled feature entering `fc`.
    """
    total = None
    count = 0
    acts = []

    def hook(_mod, inp, _out):
        acts.append(inp[0].detach().float().sum(dim=0))

    handle = model.heads.head.register_forward_hook(hook)
    model.eval()
    try:
        for x, _ in forget_batches:
            x = x.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16,
                                enabled=amp and device.type == "cuda"):
                model(x)
            count += x.shape[0]
            total = acts[-1] if total is None else total + acts[-1]
            acts.clear()
    finally:
        handle.remove()
    return total / max(count, 1)


@torch.no_grad()
def infer_forget_class(model, forget_batches, device, amp: bool = True) -> int:
    """The forget class, label-free: argmax of the mean logits over the forget set.

    Must be called on the model BEFORE any attenuation -- afterwards the forget
    class is, by design, no longer the argmax.
    """
    total = None
    model.eval()
    for x, _ in forget_batches:
        x = x.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16,
                            enabled=amp and device.type == "cuda"):
            s = model(x).float().sum(dim=0)
        total = s if total is None else total + s
    return int(total.argmax())


def _class_direction(model, c: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """(wd, u) for class c, from the CURRENT weights.

    wd = W[c] - mean_k W[k]: how a head-input dim moves logit c relative to the
    other logits (the mean subtraction removes the component that shifts every
    logit equally, which softmax ignores).
    u  = the same direction pulled back through encoder.ln's affine and its mean
    subtraction, so that a residual-stream write v changes the margin by
    (v . u) / sigma, sigma being the LayerNorm's per-sample std.
    """
    W = model.heads.head.weight.float()
    wd = W[c] - W.mean(dim=0)
    u = model.encoder.ln.weight.float() * wd
    return wd, u - u.mean()


@torch.no_grad()
def profile_dla(model, scope: str, forget_batches, device, forget_class: int,
                amp: bool = True):
    """Mean direct logit attribution to the forget class, per unit, at the CLS token.

    FFN unit j of block b:  E_F[ GELU_j(x_cls) * (W3[:, j] . u) / sigma ]
    head-input dim d:       E_F[ x_d * wd_d ]
    Positive = the unit pushes the forget-class logit up on the forget set. Only
    the forget set and the model's weights are read.
    """
    kind, blocks_idx = scope_spec(scope)
    layers = list(model.encoder.layers)
    wd, u = _class_direction(model, forget_class)
    w3u = {b: layers[b].mlp[3].weight.float().t() @ u for b in blocks_idx}  # [3072]
    cache, acc, n, handles = {}, {}, 0, []

    for b in blocks_idx:
        def hook(_m, _i, out, b=b):
            cache[b] = out[:, 0].float()
        handles.append(layers[b].mlp[1].register_forward_hook(hook))

    def ln_hook(_m, inp, out):
        x = inp[0][:, 0].float()
        cache["sigma"] = x.var(dim=-1, unbiased=False).add(model.encoder.ln.eps).sqrt()
        cache["headin"] = out[:, 0].float()
    handles.append(model.encoder.ln.register_forward_hook(ln_hook))

    model.eval()
    try:
        for x, _ in forget_batches:
            x = x.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16,
                                enabled=amp and device.type == "cuda"):
                model(x)
            n += x.shape[0]
            if kind == "head":
                v = (cache["headin"] * wd).sum(dim=0)
                acc["head"] = v if "head" not in acc else acc["head"] + v
                continue
            for b in blocks_idx:
                v = (cache[b] * w3u[b] / cache["sigma"][:, None]).sum(dim=0)
                acc[b] = v if b not in acc else acc[b] + v
    finally:
        for h in handles:
            h.remove()
    n = max(n, 1)
    if kind == "head":
        return acc["head"] / n
    return {b: acc[b] / n for b in blocks_idx}


def profile(model, scope: str, forget_batches, device, token_pool: str = "mean",
            amp: bool = True, forget_class: Optional[int] = None):
    """Dispatch to the right profiler. Returns {block: [U]} or a bare [U] tensor."""
    kind, blocks_idx = scope_spec(scope)
    rule = scope_rule(scope)
    if rule == "dla":
        if token_pool != "cls":
            raise ValueError(f"{scope} reads the CLS token only; pass token_pool='cls'")
        if forget_class is None:
            raise ValueError(f"{scope} needs forget_class (see infer_forget_class)")
        return profile_dla(model, scope, forget_batches, device, forget_class, amp)
    if rule == "pos":
        return profile_ffn(model, blocks_idx, forget_batches, device, token_pool, amp,
                           positive=True)
    if kind == "ffn":
        return profile_ffn(model, blocks_idx, forget_batches, device, token_pool, amp)
    if kind == "attn":
        return profile_attn(model, blocks_idx, forget_batches, device, token_pool, amp)
    return profile_head(model, forget_batches, device, amp)


# --------------------------------------------------------------------------
# Attenuation
# --------------------------------------------------------------------------
@torch.no_grad()
def attenuate_ffn(block, mask: torch.Tensor, gamma: float) -> None:
    """Scale FFN unit j's incoming and outgoing weights by gamma."""
    block.mlp[0].weight[mask, :] *= gamma
    block.mlp[0].bias[mask] *= gamma
    block.mlp[3].weight[:, mask] *= gamma


@torch.no_grad()
def attenuate_attn(block, mask: torch.Tensor, gamma: float) -> None:
    """Scale head h's out_proj column slice by gamma, for every selected h."""
    w = block.self_attention.out_proj.weight
    for h in mask.nonzero(as_tuple=True)[0].tolist():
        w[:, h * HEAD_DIM:(h + 1) * HEAD_DIM] *= gamma


@torch.no_grad()
def attenuate_head(model, mask: torch.Tensor, gamma: float) -> None:
    """Scale the classifier's incoming weights for the selected feature dims."""
    model.heads.head.weight[:, mask] *= gamma


# --------------------------------------------------------------------------
# The method
# --------------------------------------------------------------------------
def unlearn_round(model, scope: str, forget_batches, device, percentile: float,
                  gamma: float, token_pool: str = "mean", amp: bool = True,
                  forget_class: Optional[int] = None) -> Dict[str, torch.Tensor]:
    """One profile-select-attenuate round. Returns {block_key: mask}."""
    kind, blocks_idx = scope_spec(scope)
    layers = list(model.encoder.layers)
    acts = profile(model, scope, forget_batches, device, token_pool, amp, forget_class)
    masks = {}
    if kind == "head":
        mask = select_mask(acts, percentile, scope)
        attenuate_head(model, mask, gamma)
        masks["head"] = mask
        return masks
    for b in blocks_idx:
        mask = select_mask(acts[b], percentile, scope)
        (attenuate_ffn if kind == "ffn" else attenuate_attn)(layers[b], mask, gamma)
        masks[b] = mask
    return masks


def unlearn_vit(model, forget_batches, device, scope: str = "ffn_last3",
                percentile: float = 99.0, gamma: float = 0.1, iters: int = 5,
                token_pool: str = "mean", amp: bool = True, verbose: bool = False):
    """UnAct on ViT-B/16, in place. Returns the model."""
    c = (infer_forget_class(model, forget_batches, device, amp)
         if scope_rule(scope) == "dla" else None)
    for i in range(iters):
        masks = unlearn_round(model, scope, forget_batches, device, percentile,
                              gamma, token_pool, amp, c)
        if verbose:
            sel = {k: int(v.sum()) for k, v in masks.items()}
            print(f"  [{scope}] round {i+1}/{iters}: selected {sel}", flush=True)
    return model


# --------------------------------------------------------------------------
# Measuring the realised attenuation (Proposition 1 on a non-homogeneous GELU)
# --------------------------------------------------------------------------
@torch.no_grad()
def measure_attenuation(model_before, model_after, block_idx: int,
                        mask: torch.Tensor, forget_batches, device,
                        amp: bool = True) -> Dict[str, float]:
    """Realised attenuation of the masked FFN units, before vs after one round.

    Two ratios, both restricted to the masked units M of one block:

      gelu_ratio    mean |GELU(W0 x + b0)[M]| after / before.
                    Exactly gamma for a positively homogeneous activation;
                    GELU is not, so this is what it actually is.
      contrib_ratio mean || W3[:, M] GELU(...)[M] || after / before -- the
                    masked units' whole contribution to the block's output.
                    This is the quantity Proposition 1 claims equals gamma^2.

    Both models are run on the same batches, so the ratio is not confounded by
    sampling. Note the *after* model's earlier blocks may also have been
    modified (scopes wider than one block), which is part of what is measured.
    """
    def collect(model):
        blk = list(model.encoder.layers)[block_idx]
        stats = {"gelu": 0.0, "contrib": 0.0, "n": 0}
        w3 = blk.mlp[3].weight

        def hook(_m, _i, out):
            h = out.detach().float()                      # [B, T, 3072]
            hm = h[..., mask]
            stats["gelu"] += float(hm.abs().mean(dim=(1, 2)).sum())
            contrib = hm @ w3.float()[:, mask].t()        # [B, T, 768]
            stats["contrib"] += float(contrib.norm(dim=-1).mean(dim=1).sum())
            stats["n"] += h.shape[0]

        hd = blk.mlp[1].register_forward_hook(hook)
        model.eval()
        try:
            for x, _ in forget_batches:
                x = x.to(device, non_blocking=True)
                with torch.autocast("cuda", dtype=torch.bfloat16,
                                    enabled=amp and device.type == "cuda"):
                    model(x)
        finally:
            hd.remove()
        n = max(stats["n"], 1)
        return stats["gelu"] / n, stats["contrib"] / n

    g_b, c_b = collect(model_before)
    g_a, c_a = collect(model_after)
    return {
        "gelu_before": g_b, "gelu_after": g_a,
        "gelu_ratio": g_a / g_b if g_b else float("nan"),
        "contrib_before": c_b, "contrib_after": c_a,
        "contrib_ratio": c_a / c_b if c_b else float("nan"),
    }
