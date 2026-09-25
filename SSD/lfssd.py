"""
lfssd.py
Label-Free Selective Synaptic Dampening (LFSSD).

"Loss-Free Machine Unlearning", Foster, Schoepf & Brintrup, ICLR 2024 Tiny
Paper (arXiv 2402.19308), by the SSD authors. It is SSD with the supervised
importance objective replaced by a label-free one, so no labels and no loss
function are needed -- only the model's own output.

Per the official repository, the change is exactly two lines inside
`calc_importance`:

    vanilla SSD   loss = criterion(out, y)
                  imp.data += p.grad.data.clone().pow(2)

    LFSSD         loss = torch.norm(out, p="fro", dim=1).pow(2).mean()
                  imp.data += p.grad.data.clone().abs()

Everything else -- the alpha selection rule, the lambda dampening, the
lower bound, the per-batch averaging -- is inherited unchanged from
`ssd.ParameterPerturber`. This file therefore subclasses it rather than
copying it: `SSD/ssd.py` is a verified faithful port of the reference and must
not be edited.

Two consequences worth stating, because they are what separates LFSSD from
UnAct:

  * LFSSD needs no labels, but it still needs a *backward pass* through the
    network, and it still needs the whole of D for the denominator importance.
    UnAct needs neither.
  * Because the objective no longer involves y, the importance is invariant to
    any relabelling of the data. That is verified empirically rather than
    asserted -- see experiments/e9_labelfree_check.py.
"""
from typing import Dict

import torch

from ssd import ParameterPerturber


class LFParameterPerturber(ParameterPerturber):
    """SSD's perturber with the loss-free importance estimator."""

    def calc_importance(self, dataloader) -> Dict[str, torch.Tensor]:
        """Per-parameter importance = mean over batches of |grad of ||f(x)||_F^2|.

        Labels are read from the loader (our loaders yield them) but are never
        used, which is the entire point of the method.
        """
        importances = self.zerolike_params_dict(self.model)
        for batch in dataloader:
            x = batch[0]
            x = x.to(self.device, non_blocking=True)
            self.opt.zero_grad(set_to_none=True)
            out = self.model(x)
            # Loss-free objective: the squared Frobenius norm of the logits,
            # averaged over the batch. No y, no criterion.
            loss = torch.norm(out, p="fro", dim=1).pow(2).mean()
            loss.backward()

            for (k1, p), (k2, imp) in zip(
                self.model.named_parameters(), importances.items()
            ):
                if p.grad is not None:
                    # abs(), not pow(2): the reference's second changed line.
                    imp.data += p.grad.data.detach().to(imp.device).abs()

        for _, imp in importances.items():
            imp.data /= float(len(dataloader))
        return importances
