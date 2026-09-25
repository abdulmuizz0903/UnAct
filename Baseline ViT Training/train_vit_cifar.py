"""
train_vit_cifar.py

Fine-tunes torchvision's ImageNet-1k ViT-B/16 on CIFAR-10 / CIFAR-20 / CIFAR-100
at 224x224, producing either

  * the E5 *baseline* (trained on the full training set), or
  * an E5 *retrain gold model* (--forget-class C ...: trained on the retain set
    only, i.e. the forget class is never seen).

Both roles live in one script because the recipe must be identical -- a gold
model whose optimiser or schedule differed from the baseline would not be a
valid reference point. The ResNet pair (train_resnet18_cifar.py /
train_retain_cifar.py) shares one recipe the same way, by import.

Recipe
------
The ViT paper's transfer recipe: SGD momentum 0.9, no weight decay, gradient
clipping at global norm 1.0, short linear warmup then cosine decay, 8 epochs.
bfloat16 autocast. Deterministic kernels and seed 42, as for every other
checkpoint in this programme.

Data path: `unlearn_lib.vit.GpuBatches` -- the raw uint8 CIFAR array is held on
the GPU and upsampled to 224 there. A CPU-side Resize(224) would make the
DataLoader the bottleneck (588 KB per image over PCIe) and is a fixed linear map
anyway. Training, profiling and evaluation all use the same call, so there is no
preprocessing mismatch anywhere in E5.

Durability: checkpoints every epoch (model, optimiser, scheduler, RNG) via an
atomic write to <target>.pt.resume and resumes from the last completed epoch, so
an interrupted 45-minute run costs one epoch. Re-running a completed command is
a no-op unless --force.

Usage:
    python "Baseline ViT Training/train_vit_cifar.py" --dataset cifar10 --device cuda:0
    python "Baseline ViT Training/train_vit_cifar.py" --dataset cifar10 --forget-class 3
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import torch
import torch.nn as nn

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_REPO_ROOT, os.path.join(_REPO_ROOT, "Baseline CIFAR Training")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from unlearn_lib import manifest, paths  # noqa: E402
from unlearn_lib.io import save_checkpoint_atomic  # noqa: E402
from unlearn_lib.splits import build_split_indices  # noqa: E402
from unlearn_lib.vit import (  # noqa: E402
    ARCH, FAMILY, GpuBatches, build_vit, get_vit_datasets,
)

DATASET_CLASSES = {"cifar10": 10, "cifar20": 20, "cifar100": 100}

#: CIFAR-10 must clear this after fine-tuning; below ~90 means the upsampling or
#: the normalisation is wrong (the brief's stop-and-debug condition).
SANITY_MIN = {"cifar10": 97.0}


def build_scheduler(optimizer, total_steps, warmup_steps):
    """Linear warmup then cosine decay, stepped per optimiser step.

    Per-step rather than per-epoch because the run is only 8 epochs long: a
    per-epoch warmup would spend an eighth of the budget at a reduced lr.
    """
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(total_steps - warmup_steps, 1))
    if warmup_steps <= 0:
        return cosine
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.02, end_factor=1.0, total_iters=warmup_steps)
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer, [warmup, cosine], milestones=[warmup_steps])


@torch.no_grad()
def evaluate(model, batches, amp=True, device=None):
    """Top-1 accuracy in percent. fp32 when amp=False -- the reported numbers."""
    model.eval()
    correct = total = 0
    for x, y in batches:
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            pred = model(x).argmax(dim=1)
        correct += int((pred == y).sum())
        total += int(y.numel())
    return 100.0 * correct / max(total, 1)


def _save_resume(path, epoch, model, optimizer, scheduler):
    save_checkpoint_atomic({
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "cpu_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }, path)


def _load_resume(path, model, optimizer, scheduler):
    if not os.path.exists(path):
        return 0
    try:
        state = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:
        print(f"  Ignoring unreadable resume file ({exc}); starting from epoch 1")
        return 0
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    torch.set_rng_state(state["cpu_rng"].cpu().to(torch.uint8))
    if state.get("cuda_rng") is not None and torch.cuda.is_available():
        try:
            torch.cuda.set_rng_state_all([s.cpu().to(torch.uint8) for s in state["cuda_rng"]])
        except Exception:
            pass
    return int(state["epoch"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(DATASET_CLASSES))
    ap.add_argument("--forget-class", type=int, nargs="+", default=None,
                    help="Train the retain-gold model: these classes are held out entirely.")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=0.01,
                    help="SGD lr at --batch-size 128; scaled linearly otherwise.")
    ap.add_argument("--momentum", type=float, default=0.9)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--warmup-frac", type=float, default=0.06)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--eval-batch-size", type=int, default=256)
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--nondeterministic", action="store_true")
    ap.add_argument("--skip-sanity", action="store_true",
                    help="Do not abort when CIFAR-10 falls short of 97%% (for short probe runs).")
    args = ap.parse_args()

    forget = args.forget_class
    is_gold = forget is not None
    rel = (paths.gold_rel(FAMILY, args.dataset, ARCH, forget) if is_gold
           else paths.baseline_rel(FAMILY, args.dataset, ARCH))
    out_path = paths.resolve_write(rel)
    if os.path.exists(out_path) and not args.force:
        print(f"Checkpoint already exists, nothing to do: {out_path}\n  (pass --force to retrain)")
        return

    torch.manual_seed(args.seed)
    if not args.nondeterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)
        if os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in (":4096:8", ":16:8"):
            print("  NOTE: CUBLAS_WORKSPACE_CONFIG unset; launch via scripts/launch.sh.")

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    amp = not args.no_amp
    n_classes = DATASET_CLASSES[args.dataset]
    lr = args.lr * args.batch_size / 128.0

    print(f"Device: {device} ({torch.cuda.get_device_name(device)})   {paths.describe()}")
    print(f"Dataset: {args.dataset} ({n_classes} classes)   "
          f"role: {'retain_gold forget=' + paths.class_spec(forget) if is_gold else 'baseline'}")

    train_set, test_set = get_vit_datasets(args.dataset, download=False)
    if is_gold:
        idx = build_split_indices(train_set, test_set, forget)
        train_idx = idx["retain_train"]
        eval_splits = {"retain": idx["retain_test"], "forget": idx["forget_test"],
                       "all": idx["test_all"]}
        print(f"  retain train: {len(train_idx)}   withheld: {len(idx['forget_train'])}")
    else:
        train_idx = torch.arange(len(train_set.targets))
        eval_splits = {"all": torch.arange(len(test_set.targets))}

    train_batches = GpuBatches(train_set, train_idx, device, batch_size=args.batch_size,
                               shuffle=True, augment=True, drop_last=True)
    eval_batches = {k: GpuBatches(test_set, v, device, batch_size=args.eval_batch_size)
                    for k, v in eval_splits.items()}

    model = build_vit(n_classes, device=device, pretrained=True, seed=args.seed)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=args.momentum,
                                weight_decay=args.weight_decay)
    steps_per_epoch = len(train_batches)
    total_steps = steps_per_epoch * args.epochs
    scheduler = build_scheduler(optimizer, total_steps,
                                int(round(args.warmup_frac * total_steps)))
    criterion = nn.CrossEntropyLoss()

    resume_path = out_path + ".resume"
    done = _load_resume(resume_path, model, optimizer, scheduler)
    if done:
        print(f"  Resuming from epoch {done + 1}/{args.epochs}")

    print(f"=== Fine-tuning ViT-B/16 (IMAGENET1K_V1) on {args.dataset} for "
          f"{args.epochs} epoch(s), {steps_per_epoch} steps/epoch ===", flush=True)
    start = time.perf_counter()
    for epoch in range(done + 1, args.epochs + 1):
        # Re-seeded from (seed, epoch) so a resumed run matches an uninterrupted
        # one: the batch order and augmentation depend on the epoch index only.
        train_batches.set_epoch(args.seed * 100_000 + epoch)
        model.train()
        epoch_start = time.perf_counter()
        running = torch.zeros((), device=device)
        for step, (x, y) in enumerate(train_batches):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                loss = criterion(model(x), y)
            loss.backward()
            if args.grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            scheduler.step()
            running += loss.detach()
        elapsed = time.perf_counter() - start
        eta = (args.epochs - epoch) * elapsed / max(epoch - done, 1)
        acc = evaluate(model, eval_batches[list(eval_splits)[0]], amp=amp)
        print(f"  Epoch {epoch}/{args.epochs} - loss: {running.item()/steps_per_epoch:.4f}"
              f" - lr: {scheduler.get_last_lr()[0]:.5f}"
              f" - {time.perf_counter() - epoch_start:.1f}s - eta {eta/60:.1f}m"
              f" - acc: {acc:.2f}%", flush=True)
        _save_resume(resume_path, epoch, model, optimizer, scheduler)

    train_time = time.perf_counter() - start
    # fp32 for the reported numbers, as everywhere else in this programme.
    metrics = {k: evaluate(model, b, amp=False) for k, b in eval_batches.items()}
    if is_gold:
        d_r, d_f, d_all = metrics["retain"], metrics["forget"], metrics["all"]
        print(f"  Retain test acc: {d_r:.2f}%   Forget test acc: {d_f:.2f}%   "
              f"Overall: {d_all:.2f}%   ({train_time/60:.1f} min)")
        if d_f > 5.0:
            print(f"  WARNING: a gold model should be near 0% on the forgotten class, got {d_f:.2f}%")
        rec_metrics = {"d_r": f"{d_r:.4f}", "d_f": f"{d_f:.4f}", "d_overall": f"{d_all:.4f}"}
        sanity_value = d_r
    else:
        d_all = metrics["all"]
        print(f"  Test acc: {d_all:.2f}%   ({train_time/60:.1f} min)")
        rec_metrics = {"d_overall": f"{d_all:.4f}"}
        sanity_value = d_all

    floor = SANITY_MIN.get(args.dataset)
    if floor is not None and sanity_value < floor and not args.skip_sanity:
        print(f"  SANITY CHECK FAILED: {args.dataset} reached {sanity_value:.2f}% < {floor}%.")
        print("  Below ~90% means the upsampling or normalisation is wrong. "
              "The checkpoint was NOT saved; the .resume file is kept so the run "
              "can be continued or inspected.")
        raise SystemExit(3)

    save_checkpoint_atomic({
        "state_dict": model.state_dict(),
        "architecture": "vit_b16",
        "pretrained": "torchvision ViT_B_16_Weights.IMAGENET1K_V1",
        "dataset": args.dataset,
        "num_classes": n_classes,
        "resolution": 224,
        "normalisation": "imagenet",
        "role": "retain_gold" if is_gold else "baseline",
        "forget_classes": forget,
        "seed": args.seed,
        "epochs": args.epochs,
        **{f"{k}_accuracy": v for k, v in metrics.items()},
    }, out_path)
    print(f"  Saved to {out_path}")

    manifest.record(
        out_path, role="retain_gold" if is_gold else "baseline", family=FAMILY, arch=ARCH,
        dataset=args.dataset, forget_classes=forget, seed=args.seed,
        hparams={"epochs": args.epochs, "batch_size": args.batch_size, "lr": lr,
                 "momentum": args.momentum, "weight_decay": args.weight_decay,
                 "grad_clip": args.grad_clip, "warmup_frac": args.warmup_frac,
                 "resolution": 224, "pretrained": "IMAGENET1K_V1"},
        metrics=rec_metrics, train_time_s=train_time,
    )
    if os.path.exists(resume_path):
        os.remove(resume_path)
    print(f"  Recorded in {paths.manifest_path()}")


if __name__ == "__main__":
    main()
