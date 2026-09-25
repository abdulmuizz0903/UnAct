"""
common.py
Shared data / model / evaluation utilities for training baseline ResNet-18
models on CIFAR-10, CIFAR-20 (CIFAR-100 coarse superclasses) and CIFAR-100,
and for persisting reusable per-class dataset splits.

Outputs are written to the repository root, regardless of the working
directory this script is invoked from:
    data/                       raw CIFAR-10 / CIFAR-100 downloads
    data/<dataset>/splits/      saved split index tensors
    models/cifar/<dataset>/     trained ResNet-18 checkpoint
    models/cifar/accuracies.csv summary of test accuracies
"""
import os
import pickle

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.models import resnet18

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Lets cuDNN pick the fastest conv algorithms for these fixed 32x32 shapes.
torch.backends.cudnn.benchmark = True

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(REPO_ROOT, "data")
MODELS_DIR = os.path.join(REPO_ROOT, "models", "cifar")

CIFAR10_STATS = ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
CIFAR100_STATS = ((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))


# --------------------------------------------------------------------------
# Device selection
# --------------------------------------------------------------------------
def pick_device(spec="auto"):
    """Resolves and installs the module-level DEVICE used by the train/eval
    helpers.

    'auto' prefers the CUDA device with the most free memory, so a card that
    someone else is already using is avoided. Free memory is bucketed to the
    nearest GiB so that two idle cards count as tied, and ties are broken by SM
    count - on this machine that picks the A10 (72 SMs, ~9.6k img/s) over the
    L4 (58 SMs, ~8.4k img/s). Pass e.g. 'cuda:0' or 'cpu' to override.
    """
    global DEVICE
    if spec != "auto":
        DEVICE = torch.device(spec)
    elif not torch.cuda.is_available():
        DEVICE = torch.device("cpu")
    else:
        candidates = []
        for i in range(torch.cuda.device_count()):
            free_bytes, _ = torch.cuda.mem_get_info(i)
            props = torch.cuda.get_device_properties(i)
            candidates.append((free_bytes // (1 << 30), props.multi_processor_count, i))
        DEVICE = torch.device(f"cuda:{max(candidates)[2]}")
    if DEVICE.type == "cuda":
        torch.cuda.set_device(DEVICE)
    return DEVICE


def device_label(device=None):
    device = device or DEVICE
    if device.type != "cuda":
        return str(device)
    return f"{device} ({torch.cuda.get_device_name(device)})"


def autocast_enabled(amp):
    return bool(amp) and DEVICE.type == "cuda"


# --------------------------------------------------------------------------
# Datasets
# --------------------------------------------------------------------------
class CIFAR20(datasets.CIFAR100):
    """CIFAR-100 relabelled with its 20 coarse superclass labels.

    torchvision's CIFAR100 only reads the `fine_labels` field of the raw
    pickles, so the coarse labels are re-read here and substituted for
    `targets`. Overriding `meta["key"]` makes `classes` report the 20
    superclass names.
    """

    meta = {
        "filename": "meta",
        "key": "coarse_label_names",
        "md5": "7973b15100ade9c7d40fb424638fde48",
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        file_list = self.train_list if self.train else self.test_list
        coarse_targets = []
        for file_name, _ in file_list:
            with open(os.path.join(self.root, self.base_folder, file_name), "rb") as f:
                coarse_targets.extend(pickle.load(f, encoding="latin1")["coarse_labels"])
        self.targets = coarse_targets


DATASETS = {
    "cifar10": {"cls": datasets.CIFAR10, "num_classes": 10, "stats": CIFAR10_STATS},
    "cifar20": {"cls": CIFAR20, "num_classes": 20, "stats": CIFAR100_STATS},
    "cifar100": {"cls": datasets.CIFAR100, "num_classes": 100, "stats": CIFAR100_STATS},
}


def dataset_names():
    return list(DATASETS)


def num_classes(name):
    return DATASETS[name]["num_classes"]


def splits_dir(name):
    return os.path.join(DATA_DIR, name, "splits")


def get_datasets(name, train_augment=True, download=True):
    """Returns the (train, test) datasets for one of cifar10/cifar20/cifar100.

    `train_augment=False` yields a train set with the evaluation transform,
    which is what activation profiling during unlearning must use - random
    crops and flips would make the per-channel activation means noisy.
    """
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset {name!r}; expected one of {dataset_names()}")
    spec = DATASETS[name]
    mean, std = spec["stats"]
    normalize = [transforms.ToTensor(), transforms.Normalize(mean, std)]
    augment = [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()]

    train_transform = transforms.Compose((augment if train_augment else []) + normalize)
    test_transform = transforms.Compose(normalize)

    train_set = spec["cls"](DATA_DIR, train=True, download=download, transform=train_transform)
    test_set = spec["cls"](DATA_DIR, train=False, download=download, transform=test_transform)
    return train_set, test_set


# --------------------------------------------------------------------------
# Splits
# --------------------------------------------------------------------------
def _class_split_indices(dataset, class_idx):
    """Returns (forget_indices, retain_indices) for a given class."""
    targets = dataset.targets
    if not torch.is_tensor(targets):
        targets = torch.tensor(targets)
    forget = (targets == class_idx).nonzero(as_tuple=True)[0]
    retain = (targets != class_idx).nonzero(as_tuple=True)[0]
    return forget, retain


def build_and_save_splits(name, train_set, test_set, force=False):
    """Saves index tensors for the train/test/forget/retain splits under
    data/<name>/splits/, for every class in the dataset. Skips files that
    already exist unless force=True."""
    out_dir = splits_dir(name)
    os.makedirs(out_dir, exist_ok=True)

    def save(file_name, tensor):
        path = os.path.join(out_dir, file_name)
        if force or not os.path.exists(path):
            torch.save(tensor, path)

    save("train_indices.pt", torch.arange(len(train_set)))
    save("test_indices.pt", torch.arange(len(test_set)))

    for class_idx in range(num_classes(name)):
        forget_train, retain_train = _class_split_indices(train_set, class_idx)
        save(f"forget_train_class_{class_idx}.pt", forget_train)
        save(f"retain_train_class_{class_idx}.pt", retain_train)

        forget_test, retain_test = _class_split_indices(test_set, class_idx)
        save(f"forget_test_class_{class_idx}.pt", forget_test)
        save(f"retain_test_class_{class_idx}.pt", retain_test)


def load_split_loader(name, dataset, split_file, batch_size=256, shuffle=False,
                      workers=8, drop_last=False):
    """Wraps the given dataset in a Subset + DataLoader using a saved index
    tensor from data/<name>/splits/.

    `persistent_workers` matters here: on Python 3.14 the default start method
    pickles the whole in-memory CIFAR array into every worker, so respawning
    them each epoch is pure overhead.
    """
    path = os.path.join(splits_dir(name), split_file)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing split file {path}. Run 'python prepare_data.py' first."
        )
    indices = torch.load(path)
    subset = Subset(dataset, indices.tolist())
    return DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=DEVICE.type == "cuda",
        persistent_workers=workers > 0,
        prefetch_factor=4 if workers > 0 else None,
        drop_last=drop_last,
    )


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------
def build_resnet18(n_classes=10, device=None, channels_last=True):
    """ResNet-18 with the standard CIFAR stem: a 3x3 stride-1 first conv and
    no max-pool, so 32x32 inputs are not downsampled to 8x8 before the first
    residual block. Trained from scratch (no ImageNet weights).

    `channels_last` is worth ~1.4x on these GPUs because it lets cuDNN use its
    tensor-core conv kernels; it changes only the memory layout, not results.
    """
    model = resnet18(weights=None, num_classes=n_classes)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    if device is not None:
        model = model.to(device)
        if channels_last and device.type == "cuda":
            model = model.to(memory_format=torch.channels_last)
    return model


# --------------------------------------------------------------------------
# Train / Eval
# --------------------------------------------------------------------------
def _to_device(x, y):
    x = x.to(DEVICE, non_blocking=True)
    if DEVICE.type == "cuda":
        x = x.contiguous(memory_format=torch.channels_last)
    return x, y.to(DEVICE, non_blocking=True)


def train_one_epoch(model, loader, optimizer, criterion, amp=True):
    """One epoch of SGD. Uses bfloat16 autocast on CUDA (~2x faster, and
    unlike fp16 it needs no GradScaler), and keeps the running loss on the GPU
    so no per-step `.item()` call stalls the pipeline."""
    model.train()
    use_amp = autocast_enabled(amp)
    running_loss = torch.zeros((), device=DEVICE)
    steps = 0
    for x, y in loader:
        x, y = _to_device(x, y)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
            loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        running_loss += loss.detach()
        steps += 1
    return (running_loss / max(steps, 1)).item()


@torch.no_grad()
def evaluate(model, loader, amp=True):
    model.eval()
    use_amp = autocast_enabled(amp)
    correct = torch.zeros((), device=DEVICE, dtype=torch.long)
    total = 0
    for x, y in loader:
        x, y = _to_device(x, y)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
            preds = model(x).argmax(dim=1)
        correct += (preds == y).sum()
        total += y.size(0)
    return 100.0 * correct.item() / max(total, 1)


# --------------------------------------------------------------------------
# Checkpointing
# --------------------------------------------------------------------------
def checkpoint_path(name):
    return os.path.join(MODELS_DIR, name, "resnet18.pt")


def save_model(model, name, test_accuracy):
    path = checkpoint_path(name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "architecture": "resnet18_cifar",
        "dataset": name,
        "num_classes": num_classes(name),
        "test_accuracy": test_accuracy,
    }, path)
    return path


def load_model(path):
    checkpoint = torch.load(path, map_location=DEVICE)
    model = build_resnet18(n_classes=checkpoint["num_classes"], device=DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint
