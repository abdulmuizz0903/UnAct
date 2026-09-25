"""
metrics.py
Evaluation metrics used by Selective Synaptic Dampening.

Ports of the helpers from if-loops/selective-synaptic-dampening/src/metrics.py
(themselves inherited from the Bad Teaching Unlearning paper, arXiv 2205.08096):

  * JSDiv / UnLearningScore  -> ZRF (Zero Retrain Forgetting)
  * entropy / collect_prob / get_membership_attack_prob -> MIA

Deviation from the reference: collect_prob batches the forward pass instead of
rebuilding the loader with batch_size=1. Softmax is per-sample, so the MIA
number is identical and CIFAR runs finish in seconds rather than minutes.
"""
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression


def JSDiv(p, q):
    """Jensen-Shannon divergence between two probability tensors.

    Matches the reference: F.kl_div is called with the default reduction='mean',
    so the absolute ZRF value scales with the number of classes. Keep this quirk
    if you want numbers comparable to the paper's tables.
    """
    m = (p + q) / 2
    return 0.5 * F.kl_div(torch.log(p), m) + 0.5 * F.kl_div(torch.log(q), m)


@torch.no_grad()
def UnLearningScore(tmodel, gold_model, forget_dl, batch_size, device):
    """ZRF: 1 - JS(softmax(unlearned), softmax(random teacher)) on the forget set.

    Higher means the unlearned model's forget-set outputs look more like a
    randomly initialised network. `batch_size` is accepted for API parity with
    the reference and is unused (the loader already has its own batch size).
    """
    del batch_size
    model_preds = []
    gold_model_preds = []
    tmodel.eval()
    gold_model.eval()
    for batch in forget_dl:
        if len(batch) == 3:
            x, _, _ = batch
        else:
            x, _ = batch
        x = x.to(device, non_blocking=True)
        model_preds.append(F.softmax(tmodel(x), dim=1).detach().cpu())
        gold_model_preds.append(F.softmax(gold_model(x), dim=1).detach().cpu())

    model_preds = torch.cat(model_preds, dim=0)
    gold_model_preds = torch.cat(gold_model_preds, dim=0)
    return float(1 - JSDiv(model_preds, gold_model_preds))


def entropy(p, dim=-1, keepdim=False):
    """Shannon entropy with a zero-safe log, used as the single MIA feature."""
    return -torch.where(p > 0, p * p.log(), p.new_zeros(())).sum(
        dim=dim, keepdim=keepdim
    )


@torch.no_grad()
def collect_prob(data_loader, model):
    """Concatenate per-sample softmax probabilities over a loader.

    Unlike the reference (which rebuilds the loader at batch_size=1), we keep
    the caller's batch size. Softmax is element-wise over the class dimension,
    so the resulting (N, C) tensor is identical.
    """
    model.eval()
    device = next(model.parameters()).device
    probs = []
    for batch in data_loader:
        if len(batch) == 3:
            x, _, _ = batch
        else:
            x, _ = batch
        x = x.to(device, non_blocking=True)
        probs.append(F.softmax(model(x), dim=1).detach().cpu())
    return torch.cat(probs, dim=0)


def get_membership_attack_data(retain_loader, forget_loader, test_loader, model):
    """Build the entropy-based MIA feature matrices.

    Retain samples -> label 1 (member), test samples -> label 0 (non-member).
    Forget samples become the attack evaluation set X_f.
    """
    retain_prob = collect_prob(retain_loader, model)
    forget_prob = collect_prob(forget_loader, model)
    test_prob = collect_prob(test_loader, model)

    X_r = torch.cat([entropy(retain_prob), entropy(test_prob)], dim=0).reshape(-1, 1)
    Y_r = torch.cat(
        [
            torch.ones(retain_prob.shape[0]),
            torch.zeros(test_prob.shape[0]),
        ],
        dim=0,
    )
    X_f = entropy(forget_prob).reshape(-1, 1)
    Y_f = torch.ones(forget_prob.shape[0])
    return X_f.numpy(), Y_f.numpy(), X_r.numpy(), Y_r.numpy()


def get_membership_attack_prob(retain_loader, forget_loader, test_loader, model):
    """Fraction of forget-set samples classified as members by a balanced LR.

    Lower is better after unlearning. Trains on (retain_train vs test) using
    prediction entropy as the sole feature, then predicts on the forget set.
    """
    X_f, Y_f, X_r, Y_r = get_membership_attack_data(
        retain_loader, forget_loader, test_loader, model
    )
    del Y_f

    # The reference offers SVC(C=3, gamma="auto", kernel="rbf") here but uses this.
    clf = LogisticRegression(class_weight="balanced", solver="lbfgs")
    clf.fit(X_r, Y_r)
    results = clf.predict(X_f)
    return float(results.mean())
