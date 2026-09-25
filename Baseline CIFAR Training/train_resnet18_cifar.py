"""
train_resnet18_cifar.py

Trains a ResNet-18 (CIFAR stem: 3x3 stride-1 conv1, no max-pool) from scratch
on the full training set of CIFAR-10, CIFAR-20 (coarse superclasses) or
CIFAR-100, and saves the checkpoint to models/cifar/<dataset>/resnet18.pt.
Also updates models/cifar/accuracies.csv.

Recipe: SGD (momentum 0.9, weight decay 5e-4), reference lr 0.1 at batch size
128 scaled linearly with the batch size, linear warmup then a cosine schedule,
100 epochs, random crop + horizontal flip augmentation.

Speed: bfloat16 autocast + channels_last + cuDNN benchmark, which together cut
the epoch time roughly 2.7x versus plain fp32.

Usage:
    python train_resnet18_cifar.py --dataset cifar10
    python train_resnet18_cifar.py --dataset cifar100 --batch-size 512 --device cuda:1
    python train_all.py                      # all three datasets across both GPUs
"""
import argparse
import csv
import os
import time

import torch
import torch.nn as nn

import common
from common import (
    MODELS_DIR,
    build_and_save_splits,
    build_resnet18,
    dataset_names,
    evaluate,
    get_datasets,
    load_split_loader,
    num_classes,
    save_model,
    train_one_epoch,
)

REFERENCE_BATCH_SIZE = 128
REFERENCE_LR = 0.1

ACCURACY_CSV_FIELDS = [
    "dataset", "model_name", "architecture", "num_classes",
    "epochs", "batch_size", "lr", "test_accuracy",
]


def build_scheduler(optimizer, epochs, warmup_epochs):
    """Linear warmup then cosine decay. Warmup exists because the lr is scaled
    up with the batch size, and a large lr applied from step 0 can diverge."""
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(epochs - warmup_epochs, 1)
    )
    if warmup_epochs <= 0:
        return cosine
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs
    )
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer, [warmup, cosine], milestones=[warmup_epochs]
    )


def update_accuracy_csv(row):
    """Upserts one row (keyed on dataset) in models/cifar/accuracies.csv so
    training the three datasets in any order or in parallel-ish sequence
    accumulates rather than overwrites."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    csv_path = os.path.join(MODELS_DIR, "accuracies.csv")

    rows = {}
    if os.path.exists(csv_path):
        with open(csv_path, newline="") as f:
            for existing in csv.DictReader(f):
                rows[existing["dataset"]] = existing
    rows[row["dataset"]] = row
    for existing in rows.values():
        for field in ACCURACY_CSV_FIELDS:
            existing.setdefault(field, "")

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=ACCURACY_CSV_FIELDS)
        writer.writeheader()
        for name in sorted(rows):
            writer.writerow(rows[name])
    return csv_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=dataset_names())
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=None,
                        help="Absolute lr. Default: 0.1 scaled linearly from batch size 128.")
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--warmup-epochs", type=int, default=None,
                        help="Linear lr warmup. Default: 5 when the batch size exceeds 128, else 0.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="auto",
                        help="'auto' (most free memory), 'cuda:0', 'cuda:1' or 'cpu'")
    parser.add_argument("--no-amp", action="store_true", help="Disable bfloat16 autocast")
    parser.add_argument("--eval-every", type=int, default=10, help="Epoch interval for interim test evaluation")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = common.pick_device(args.device)
    amp = not args.no_amp

    lr = args.lr if args.lr is not None else REFERENCE_LR * args.batch_size / REFERENCE_BATCH_SIZE
    warmup_epochs = (
        args.warmup_epochs if args.warmup_epochs is not None
        else (5 if args.batch_size > REFERENCE_BATCH_SIZE else 0)
    )

    print(f"Device: {common.device_label(device)}")
    if device.type != "cuda":
        print(
            "WARNING: CUDA is unavailable, so this will train on CPU and take many "
            "hours. Check that the installed torch build matches your CUDA driver."
        )
    print(f"Dataset: {args.dataset} ({num_classes(args.dataset)} classes)")
    print(f"Batch size: {args.batch_size}  lr: {lr:g}  warmup: {warmup_epochs} epoch(s)  "
          f"bf16 autocast: {amp}")

    train_set, test_set = get_datasets(args.dataset, train_augment=True)
    build_and_save_splits(args.dataset, train_set, test_set)

    train_loader = load_split_loader(
        args.dataset, train_set, "train_indices.pt",
        batch_size=args.batch_size, shuffle=True, workers=args.workers, drop_last=True,
    )
    test_loader = load_split_loader(
        args.dataset, test_set, "test_indices.pt",
        batch_size=args.batch_size, shuffle=False, workers=min(args.workers, 4),
    )

    print(f"=== Training ResNet-18 on {args.dataset} for {args.epochs} epoch(s) ===")
    model = build_resnet18(n_classes=num_classes(args.dataset), device=device)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=lr, momentum=args.momentum, weight_decay=args.weight_decay
    )
    scheduler = build_scheduler(optimizer, args.epochs, warmup_epochs)
    criterion = nn.CrossEntropyLoss()

    start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.perf_counter()
        loss = train_one_epoch(model, train_loader, optimizer, criterion, amp=amp)
        scheduler.step()
        epoch_time = time.perf_counter() - epoch_start
        eta = (args.epochs - epoch) * (time.perf_counter() - start) / epoch
        line = (f"  Epoch {epoch}/{args.epochs} - loss: {loss:.4f} - "
                f"lr: {scheduler.get_last_lr()[0]:.4f} - {epoch_time:.1f}s - eta {eta / 60:.1f}m")
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            line += f" - test acc: {evaluate(model, test_loader, amp=amp):.2f}%"
        print(line, flush=True)

    acc = evaluate(model, test_loader, amp=amp)
    total_min = (time.perf_counter() - start) / 60
    print(f"  Final test accuracy: {acc:.2f}%  (trained in {total_min:.1f} min)")

    path = save_model(model, args.dataset, acc)
    print(f"  Saved checkpoint to {path}")

    csv_path = update_accuracy_csv({
        "dataset": args.dataset,
        "model_name": "resnet18",
        "architecture": "resnet18_cifar",
        "num_classes": num_classes(args.dataset),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": f"{lr:g}",
        "test_accuracy": f"{acc:.4f}",
    })
    print(f"  Updated accuracy summary at {csv_path}")


if __name__ == "__main__":
    main()
