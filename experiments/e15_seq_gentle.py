"""
E15: CIFAR-100 sequential requests with gentler UnAct configurations.

Replays the five class orders of E4/E4b (the default order plus the four random
orders) with UnAct at its selected CIFAR-100 configuration (sanity check against
E4b) and at two configurations that edit fewer parameters. Measures retain and
forget test accuracy on the requested prefix after every request. No retrained
references exist for most prefixes, so no Delta is reported.
"""
import copy, os, sys
import torch
_R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (_R, os.path.join(_R, "Our_Method"), os.path.join(_R, "Baseline CIFAR Training")):
    sys.path.insert(0, p)
import common as cifar_common  # noqa
from unlearn_cnn import unlearn_blocks  # noqa
from unlearn_lib.io import upsert_rows  # noqa
from unlearn_lib.loaders import make_loader  # noqa
from unlearn_lib.metrics import accuracy  # noqa
from unlearn_lib.splits import build_split_indices  # noqa

ORDERS = [[3, 20, 51, 69, 85], [30, 13, 49, 91, 37], [86, 42, 89, 92, 3],
          [48, 97, 1, 81, 90], [45, 15, 90, 32, 35]]
CONFIGS = [(90.0, 0.3, 1), (99.0, 0.3, 10), (99.0, 0.01, 20)]
OUT = os.path.join(_R, "results", "e15_seq_gentle.csv")
FIELDS = ["dataset", "order", "config", "request_idx", "post_retain", "post_forget", "eval_device"]
KEY = ["dataset", "order", "config", "request_idx"]


def main():
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    dev = torch.device("cuda:0")
    tr, te = cifar_common.get_datasets("cifar100", train_augment=False, download=False)
    base, _ = cifar_common.load_model(cifar_common.checkpoint_path("cifar100"))
    name = torch.cuda.get_device_name(dev)
    for (p, g, k) in CONFIGS:
        for order in ORDERS:
            m = copy.deepcopy(base)
            rows = []
            for i, c in enumerate(order):
                idx = build_split_indices(tr, te, [c])
                unlearn_blocks(m, make_loader(tr, idx["forget_train"], batch_size=256, workers=2),
                               dev, scope="layer4_last", penalty_scale=g,
                               threshold_percentile=p, iter_times=k, verbose=False)
                pidx = build_split_indices(tr, te, order[:i + 1])
                dr = accuracy(m, make_loader(te, pidx["retain_test"], batch_size=512, workers=2), dev)
                df = accuracy(m, make_loader(te, pidx["forget_test"], batch_size=512, workers=2), dev)
                rows.append(dict(dataset="cifar100", order="-".join(map(str, order)),
                                 config=f"p{p}_g{g}_k{k}", request_idx=i,
                                 post_retain=f"{dr:.4f}", post_forget=f"{df:.4f}", eval_device=name))
                print(f"p{p} g{g} k{k} order {order} req {i+1}: D_r {dr:.2f} D_f {df:.2f}", flush=True)
            upsert_rows(rows, OUT, FIELDS, KEY)


if __name__ == "__main__":
    main()
