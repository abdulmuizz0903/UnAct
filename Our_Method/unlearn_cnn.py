"""
unlearn_cnn.py

UnAct on ResNet-18. Unlike an MLP, ResNet-18 has no simple "last hidden
Linear layer" - its activations live in spatial conv feature maps at multiple
depths (layer1..layer4), and its 512-dim pooled feature vector (input to
`fc`) is produced by conv + global average pooling, not a Linear layer.

For each BasicBlock we profile activations as the per-channel mean (over
the forget set AND over spatial positions) of that block's output feature
map, then mask/penalize the top-percentile ("highly active") channels.
The paper's reported variant is layer4_last.

Variants implemented:

  * fc_only     - profile the 512-dim pooled feature (input to `fc`);
                  only scale down the *outgoing* weights `fc.weight[:, mask]`.
                  No conv/BatchNorm layers are touched.
  * layer4_last - also scale down the *incoming* path for the masked
                  channels: layer4's LAST BasicBlock's conv2 filters +
                  bn2 affine params, plus fc.weight[:, mask].
  * layer4_all  - same as layer4_last, but profiles/penalizes BOTH
                  BasicBlocks in layer4 (fc is penalized using the mask
                  from the last block only, since that's what feeds
                  pooling -> fc).
  * all_layers  - profiles/penalizes every BasicBlock in layer1..layer4
                  (8 blocks total for ResNet-18), i.e. checks activations
                  at every depth in the network, not just the final one.
                  fc is still penalized using the last layer4 block's mask.

Caveat: ResNet's residual/identity shortcuts still carry a masked
channel's signal forward from earlier blocks, so - unlike in an MLP - full
suppression of a channel's contribution is not guaranteed even with the
most thorough (`all_layers`) variant. This is an expected, reportable
limitation of adapting the method to convolutional/residual architectures.
"""
from functools import partial

import torch


# --------------------------------------------------------------------------
# Activation profiling
# --------------------------------------------------------------------------
def profile_activations(model, forget_loader, device):
    """Mean activation (over the forget set) of each of the 512 pooled
    feature channels feeding into `model.fc`."""
    activations = []

    def hook(module, inputs, output):
        activations.append(inputs[0].detach())

    handle = model.fc.register_forward_hook(hook)
    model.eval()
    with torch.no_grad():
        for data, _ in forget_loader:
            data = data.to(device)
            model(data)
    handle.remove()

    all_activations = torch.cat(activations, dim=0)
    return all_activations.mean(dim=0)


def profile_block_activations(model, block, forget_loader, device):
    """Mean activation (over the forget set AND spatial positions) of each
    output channel of a given BasicBlock's feature map [B, C, H, W] -> [C]."""
    activations = []

    def hook(module, inputs, output):
        activations.append(output.detach().mean(dim=(2, 3)))  # [B, C]

    handle = block.register_forward_hook(hook)
    model.eval()
    with torch.no_grad():
        for data, _ in forget_loader:
            data = data.to(device)
            model(data)
    handle.remove()

    all_activations = torch.cat(activations, dim=0)  # [N, C]
    return all_activations.mean(dim=0)  # [C]


def _mask_from(mean_activations, threshold_percentile):
    threshold = torch.quantile(mean_activations, threshold_percentile / 100.0)
    return mean_activations >= threshold


# --------------------------------------------------------------------------
# Variant: fc_only (outgoing-only, profiles the pooled feature vector)
# --------------------------------------------------------------------------
def unlearn_fc_only(model, forget_loader, device, penalty_scale=0.1, threshold_percentile=90, iter_times=5, verbose=True):
    for i in range(iter_times):
        if verbose:
            print(f"  [fc_only] Unlearning iteration {i + 1}/{iter_times}")
        mean_activations = profile_activations(model, forget_loader, device)
        mask = _mask_from(mean_activations, threshold_percentile)
        with torch.no_grad():
            model.fc.weight[:, mask] *= penalty_scale
    return model


# --------------------------------------------------------------------------
# Variants: layer4_last / layer4_all / all_layers (incoming+outgoing,
# profile per-block conv feature maps at increasing depth coverage)
# --------------------------------------------------------------------------
def _blocks_for_scope(model, scope):
    if scope == "layer4_last":
        return [model.layer4[-1]]
    if scope == "layer4_all":
        return list(model.layer4)
    if scope == "all_layers":
        blocks = []
        for layer_name in ("layer1", "layer2", "layer3", "layer4"):
            blocks.extend(list(getattr(model, layer_name)))
        return blocks
    raise ValueError(f"Unknown scope: {scope}")


def unlearn_blocks(model, forget_loader, device, scope, penalty_scale=0.1, threshold_percentile=90, iter_times=5, verbose=True):
    blocks = _blocks_for_scope(model, scope)
    final_block = model.layer4[-1]  # this one's output feeds avgpool -> fc

    for i in range(iter_times):
        if verbose:
            print(f"  [{scope}] Unlearning iteration {i + 1}/{iter_times} ({len(blocks)} block(s))")
        for block in blocks:
            mean_activations = profile_block_activations(model, block, forget_loader, device)
            mask = _mask_from(mean_activations, threshold_percentile)
            with torch.no_grad():
                block.conv2.weight[mask, :, :, :] *= penalty_scale
                block.bn2.weight[mask] *= penalty_scale
                block.bn2.bias[mask] *= penalty_scale
                if block is final_block:
                    model.fc.weight[:, mask] *= penalty_scale
    return model


VARIANTS = {
    "fc_only": unlearn_fc_only,
    "layer4_last": partial(unlearn_blocks, scope="layer4_last"),
    "layer4_all": partial(unlearn_blocks, scope="layer4_all"),
    "all_layers": partial(unlearn_blocks, scope="all_layers"),
}
