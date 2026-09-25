"""
unlearn_lib.vit
ViT-B/16 model construction and a GPU-resident CIFAR pipeline.

Why a separate data path
------------------------
Every CIFAR experiment so far runs at 32x32, where the DataLoader is free and
the GPU is the bottleneck. ViT-B/16 needs 224x224 inputs, so a PIL-side
`Resize(224)` turns each 3 KB image into a 588 KB tensor *on the CPU* and then
ships it over PCIe: measured, that makes the loader -- not the GPU -- the limit,
and it is pure waste because the upsample is a fixed linear map.

Instead the raw uint8 CIFAR array is held on the GPU (50,000 images = 150 MB)
and each batch is upsampled and normalised there. This is used identically by
the trainer, the unlearning method and every evaluation, so training and
inference see exactly the same preprocessing.

Normalisation uses **ImageNet** statistics, not the CIFAR statistics that
`Baseline CIFAR Training/common.py` applies, because the backbone is
IMAGENET1K_V1-pretrained and its first LayerNorm-free patch embedding expects
that input distribution. `common.py::get_datasets` is still the source of the
datasets (and hence of CIFAR-20's coarse relabelling); only the transform is
replaced, and that file is not modified.

Provenance note for the paper
-----------------------------
We use torchvision's `vit_b_16(IMAGENET1K_V1)`, trained on ImageNet-1k only.
SSD's paper uses HuggingFace `google/vit-base-patch16-224`, which is
ImageNet-21k pretrained and then fine-tuned on ImageNet-1k. Same architecture,
same resolution, same patch size, but a different (stronger) pretraining corpus,
so absolute accuracies are not directly comparable to their table.
"""
from __future__ import annotations

import os
from typing import Optional, Sequence, Union

# torch.hub caches the 330 MB pretrained checkpoint under TORCH_HOME. Keep it
# next to the other checkpoints when UNLEARN_MODELS_ROOT is set.
# Must happen before the weights are fetched.
if "TORCH_HOME" not in os.environ:
    _root = os.environ.get("UNLEARN_MODELS_ROOT")
    if _root:
        os.environ["TORCH_HOME"] = os.path.join(_root, "torch-home")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ViT_B_16_Weights, vit_b_16

ARCH = "vit_b16"
FAMILY = "vit"
RESOLUTION = 224
HIDDEN_DIM = 768
MLP_DIM = 3072
N_BLOCKS = 12
N_HEADS = 12

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------
def build_vit(n_classes: int, device=None, pretrained: bool = True, seed: int = 42):
    """ViT-B/16 with a fresh `heads.head` Linear(768, n_classes).

    The head is re-created under an explicit seed so two runs of the same
    command start from the same parameters.
    """
    weights = ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None
    model = vit_b_16(weights=weights)
    torch.manual_seed(seed)
    model.heads.head = nn.Linear(model.hidden_dim, n_classes)
    if device is not None:
        model = model.to(device)
    return model


def load_vit(path: str, device=None):
    """Load a checkpoint written by train_vit_cifar.py. Returns (model, ckpt)."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = build_vit(ckpt["num_classes"], device=None, pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    if device is not None:
        model = model.to(device)
    model.eval()
    return model, ckpt


# --------------------------------------------------------------------------
# GPU-resident CIFAR batches
# --------------------------------------------------------------------------
def _norm_stats(device, dtype=torch.float32):
    mean = torch.tensor(IMAGENET_MEAN, device=device, dtype=dtype).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device, dtype=dtype).view(1, 3, 1, 1)
    return mean, std


def preprocess(x_uint8: torch.Tensor, mean: torch.Tensor, std: torch.Tensor,
               resolution: int = RESOLUTION) -> torch.Tensor:
    """[B,3,32,32] uint8 -> [B,3,224,224] float32, ImageNet-normalised.

    Bilinear upsampling; `antialias` is a no-op when upsampling. The same call
    is used for training, profiling and evaluation, so there is no train/test
    preprocessing mismatch.
    """
    x = x_uint8.float().div_(255.0)
    x = F.interpolate(x, size=resolution, mode="bilinear", align_corners=False)
    return (x - mean) / std


class GpuBatches:
    """Iterable of preprocessed (x, y) batches held entirely on the GPU.

    Reads the dataset's raw `.data` / `.targets` arrays rather than going
    through its torchvision transform, which is exactly `ToTensor()` on the
    uint8 array. `augment=True` adds the CIFAR recipe's random crop + horizontal
    flip, applied on the GPU at 32x32 *before* upsampling, so the augmentation
    is identical in kind to the ResNet baselines'.
    """

    def __init__(self, dataset, indices, device, batch_size: int = 128,
                 shuffle: bool = False, augment: bool = False,
                 resolution: int = RESOLUTION, drop_last: bool = False):
        import numpy as np

        if torch.is_tensor(indices):
            indices = indices.tolist()
        idx = torch.as_tensor(list(indices), dtype=torch.long)
        data = torch.from_numpy(np.ascontiguousarray(dataset.data))  # [N,32,32,3] uint8
        targets = torch.as_tensor(dataset.targets, dtype=torch.long)
        self.x = data[idx].permute(0, 3, 1, 2).contiguous().to(device)
        self.y = targets[idx].to(device)
        self.device = device
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augment = augment
        self.resolution = resolution
        self.drop_last = drop_last
        self.generator = torch.Generator(device=device)
        self._mean, self._std = _norm_stats(device)

    def __len__(self):
        n = self.x.shape[0]
        if self.drop_last:
            return n // self.batch_size
        return (n + self.batch_size - 1) // self.batch_size

    @property
    def n_samples(self):
        return int(self.x.shape[0])

    def set_epoch(self, seed: int):
        self.generator.manual_seed(int(seed))

    def _augment(self, xb):
        b = xb.shape[0]
        # Random crop with 4px reflect-free zero padding, matching
        # transforms.RandomCrop(32, padding=4).
        pad = F.pad(xb, (4, 4, 4, 4))
        ox = torch.randint(0, 9, (b,), device=xb.device, generator=self.generator)
        oy = torch.randint(0, 9, (b,), device=xb.device, generator=self.generator)
        ar = torch.arange(32, device=xb.device)
        rows = (oy[:, None] + ar[None, :])          # [B,32]
        cols = (ox[:, None] + ar[None, :])          # [B,32]
        bi = torch.arange(b, device=xb.device)[:, None, None]
        xb = pad[bi, :, rows[:, :, None], cols[:, None, :]]   # [B,32,32,3]
        xb = xb.permute(0, 3, 1, 2).contiguous()
        flip = torch.rand(b, device=xb.device, generator=self.generator) < 0.5
        xb = torch.where(flip[:, None, None, None], xb.flip(-1), xb)
        return xb

    def __iter__(self):
        n = self.x.shape[0]
        order = (torch.randperm(n, device=self.device, generator=self.generator)
                 if self.shuffle else torch.arange(n, device=self.device))
        last = (n // self.batch_size) * self.batch_size if self.drop_last else n
        for i in range(0, last, self.batch_size):
            sel = order[i:i + self.batch_size]
            xb = self.x[sel]
            if self.augment:
                xb = self._augment(xb)
            yield preprocess(xb, self._mean, self._std, self.resolution), self.y[sel]


def get_vit_datasets(name: str, download: bool = False):
    """(train_set, test_set) for a CIFAR variant, transform-free.

    `GpuBatches` reads `.data`/`.targets` directly, so the torchvision transform
    is never invoked; `train_augment=False` is passed only to avoid constructing
    a random-crop pipeline that would be dead code.
    """
    import sys
    _repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    _bct = os.path.join(_repo, "Baseline CIFAR Training")
    if _bct not in sys.path:
        sys.path.insert(0, _bct)
    import common as cifar_common  # noqa: E402

    return cifar_common.get_datasets(name, train_augment=False, download=download)


# --------------------------------------------------------------------------
# Structure accessors (used by Our_Method/unlearn_vit.py and the SSD runner)
# --------------------------------------------------------------------------
def blocks(model):
    """The 12 encoder blocks, in depth order."""
    return list(model.encoder.layers)


def n_classes_of(model) -> int:
    return model.heads.head.out_features
