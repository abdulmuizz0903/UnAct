"""
ssd.py
Selective Synaptic Dampening (SSD) — ParameterPerturber.

Faithful port of if-loops/selective-synaptic-dampening/src/ssd.py
(arXiv 2308.07707), adapted to our (image, label) 2-tuple DataLoaders.

Core idea:
  1. Compute Fisher-style importances (squared CE gradients) on the forget
     set and on the full training set D.
  2. Where forget_imp > alpha * full_imp, multiply the weight by
     min(1, (lambda * full_imp) / forget_imp). Weights can only shrink.
"""
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset


class ParameterPerturber:
    def __init__(
        self,
        model,
        opt,
        device="cuda" if torch.cuda.is_available() else "cpu",
        parameters=None,
        importance_device=None,
    ):
        self.model = model
        self.opt = opt
        self.device = device
        # Importances are two full copies of the parameter set; keeping them on
        # the CPU frees ~180 MB of VRAM on small cards.
        self.importance_device = importance_device or device
        self.alpha = None
        self.xmin = None

        parameters = parameters or {}
        self.lower_bound = parameters.get("lower_bound", 1)
        self.exponent = parameters.get("exponent", 1)
        self.magnitude_diff = parameters.get("magnitude_diff", None)
        self.min_layer = parameters.get("min_layer", -1)
        self.max_layer = parameters.get("max_layer", -1)
        self.forget_threshold = parameters.get("forget_threshold", 1)
        self.dampening_constant = parameters["dampening_constant"]
        self.selection_weighting = parameters["selection_weighting"]

    def get_layer_num(self, layer_name: str) -> int:
        layer_id = layer_name.split(".")[1]
        if layer_id.isnumeric():
            return int(layer_id)
        return -1

    def zerolike_params_dict(self, model: torch.nn.Module) -> Dict[str, torch.Tensor]:
        """Named-parameters dict filled with zeros (Avalanche-style)."""
        return {
            k: torch.zeros_like(p, device=self.importance_device)
            for k, p in model.named_parameters()
        }

    def fulllike_params_dict(
        self, model: torch.nn.Module, fill_value, as_tensor: bool = False
    ) -> Dict[str, torch.Tensor]:
        def full_like_tensor(fillval, shape: list):
            if len(shape) > 1:
                fillval = full_like_tensor(fillval, shape[1:])
            return [fillval for _ in range(shape[0])]

        dictionary = {}
        for n, p in model.named_parameters():
            if as_tensor:
                dictionary[n] = torch.tensor(
                    full_like_tensor(fill_value, list(p.shape)),
                    device=self.device,
                )
            else:
                dictionary[n] = full_like_tensor(fill_value, list(p.shape))
        return dictionary

    def subsample_dataset(self, dataset, sample_perc: float) -> Subset:
        sample_idxs = np.arange(0, len(dataset), step=int(1 / sample_perc))
        return Subset(dataset, sample_idxs)

    def split_dataset_by_class(self, dataset) -> List[Subset]:
        n_classes = len({target for _, target in dataset})
        subset_idxs = [[] for _ in range(n_classes)]
        for idx, (_, y) in enumerate(dataset):
            subset_idxs[y].append(idx)
        return [Subset(dataset, subset_idxs[idx]) for idx in range(n_classes)]

    def calc_importance(self, dataloader: DataLoader) -> Dict[str, torch.Tensor]:
        """
        Per-parameter importance = mean over batches of (grad CE)^2.

        Adapted for our (x, y) 2-tuple loaders. The reference repo unpacks
        (x, _, y) because its datasets yield (img, label, clabel); numerically
        identical when the third element equals the second.
        """
        criterion = nn.CrossEntropyLoss()
        importances = self.zerolike_params_dict(self.model)
        for batch in dataloader:
            if len(batch) == 3:
                x, _, y = batch
            else:
                x, y = batch
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)
            self.opt.zero_grad(set_to_none=True)
            out = self.model(x)
            loss = criterion(out, y)
            loss.backward()

            for (k1, p), (k2, imp) in zip(
                self.model.named_parameters(), importances.items()
            ):
                if p.grad is not None:
                    imp.data += p.grad.data.detach().to(imp.device).pow(2)

        for _, imp in importances.items():
            imp.data /= float(len(dataloader))
        return importances

    def modify_weight(
        self,
        original_importance: Dict[str, torch.Tensor],
        forget_importance: Dict[str, torch.Tensor],
    ) -> None:
        """
        Synapse selection (alpha) + dampening (lambda), applied in place.

        locations = where forget_imp > alpha * original_imp
        update    = min(lower_bound, (lambda * original_imp / forget_imp) ** exponent)
        p[locations] *= update
        """
        with torch.no_grad():
            for (n, p), (oimp_n, oimp), (fimp_n, fimp) in zip(
                self.model.named_parameters(),
                original_importance.items(),
                forget_importance.items(),
            ):
                oimp = oimp.to(p.device)
                fimp = fimp.to(p.device)

                # Synapse selection with parameter alpha
                oimp_norm = oimp.mul(self.selection_weighting)
                locations = torch.where(fimp > oimp_norm)

                # Synapse dampening with parameter lambda
                weight = ((oimp.mul(self.dampening_constant)).div(fimp)).pow(
                    self.exponent
                )
                update = weight[locations]
                # Bound by 1 to prevent parameter values from increasing.
                min_locs = torch.where(update > self.lower_bound)
                update[min_locs] = self.lower_bound
                p[locations] = p[locations].mul(update)
