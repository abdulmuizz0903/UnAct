"""
train_retain_cifar.py

Trains the retrain-from-scratch GOLD MODEL: a ResNet-18 trained on the retain
set only (every class except the forget class(es)), using exactly the recipe
that produced the baselines in train_resnet18_cifar.py.

This is the reference point the whole evaluation hangs on. SSD's paper judges
unlearning by *closeness to retrain*, not by how far forget accuracy is driven
down -- they explicitly warn that D_f=0 and MIA=0 invite the Streisand effect.
Without this model, a forget accuracy or MIA value cannot be interpreted.

Kept separate from train_resnet18_cifar.py, which trains the original models;
the training recipe is imported from there, so the two cannot drift apart.

Durability: the run checkpoints every epoch (model, optimizer, scheduler and RNG
state) via an atomic write, and resumes from the last completed epoch if
restarted. A 16-minute run interrupted by a reboot then costs one epoch, not the
whole run. Re-running the identical command after completion is a no-op unless
--force is given.

Usage:
    python train_retain_cifar.py --dataset cifar10 --forget-class 3
    python train_retain_cifar.py --dataset cifar100 --forget-class 69 --device cuda:1
    python train_retain_cifar.py --dataset cifar10 --forget-class 3 5 7   # multi-class
"""
import argparse
import os
import sys
import time

import torch
import torch.nn as nn

import common
from common import (
    build_and_save_splits,
    build_resnet18,
    dataset_names,
    evaluate,
    get_datasets,
    num_classes,
)
from train_resnet18_cifar import REFERENCE_BATCH_SIZE, REFERENCE_LR, build_scheduler

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from unlearn_lib import manifest, paths  # noqa: E402
from unlearn_lib.io import save_checkpoint_atomic  # noqa: E402
from unlearn_lib.loaders import make_loader  # noqa: E402
from unlearn_lib.splits import build_split_indices  # noqa: E402


def _resume_path(target_path):
    return target_path + ".resume"


def _save_resume(path, epoch, model, optimizer, scheduler):
    save_checkpoint_atomic(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "cpu_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
        path,
    )


def _load_resume(path, model, optimizer, scheduler):
    """Returns the number of completed epochs, or 0 if there is nothing to resume."""
    if not os.path.exists(path):
        return 0
    try:
        state = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:  # truncated/corrupt -> start over rather than crash
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
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[2])
    parser.add_argument("--dataset", required=True, choices=dataset_names())
    parser.add_argument("--forget-class", type=int, nargs="+", required=True,
                        help="Class index/indices held out of training entirely.")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--warmup-epochs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--eval-every", type=int, default=20)
    parser.add_argument("--force", action="store_true",
                        help="Retrain even if the gold checkpoint already exists.")
    parser.add_argument("--nondeterministic", action="store_true",
                        help="Allow cuDNN autotuning and non-deterministic kernels "
                             "(faster, but same-seed runs differ by ~1 point).")
    args = parser.parse_args()

    forget = args.forget_class
    spec = paths.class_spec(forget)
    rel = paths.gold_rel("cifar", args.dataset, "resnet18", forget)
    out_path = paths.resolve_write(rel)

    if os.path.exists(out_path) and not args.force:
        print(f"Gold model already exists, nothing to do: {out_path}")
        print("  (pass --force to retrain)")
        return

    torch.manual_seed(args.seed)

    # The gold model is the reference every other number is compared against, so
    # it is worth making bit-reproducible. common.py sets cudnn.benchmark=True at
    # import time for throughput; with it on, three same-seed 8-epoch runs gave
    # 88.92 / 88.28 / 90.00 retain accuracy. Deterministic kernels remove that.
    if not args.nondeterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        # warn_only: a few ops have no deterministic kernel; warn rather than abort
        # a multi-hour run. CUBLAS_WORKSPACE_CONFIG must be set before CUDA init,
        # which scripts/launch.sh does.
        torch.use_deterministic_algorithms(True, warn_only=True)
        if os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in (":4096:8", ":16:8"):
            print("  NOTE: CUBLAS_WORKSPACE_CONFIG is unset; some cuBLAS reductions "
                  "may stay nondeterministic. Launch via scripts/launch.sh to set it.")

    device = common.pick_device(args.device)
    amp = not args.no_amp
    lr = args.lr if args.lr is not None else REFERENCE_LR * args.batch_size / REFERENCE_BATCH_SIZE
    warmup_epochs = (
        args.warmup_epochs if args.warmup_epochs is not None
        else (5 if args.batch_size > REFERENCE_BATCH_SIZE else 0)
    )

    print(f"Device: {common.device_label(device)}   {paths.describe()}")
    print(f"Dataset: {args.dataset} ({num_classes(args.dataset)} classes)  "
          f"forget class(es): {spec}")

    # Augmented train set (matching the baseline recipe); non-augmented for eval.
    train_set, test_set = get_datasets(args.dataset, train_augment=True)
    build_and_save_splits(args.dataset, train_set, test_set)
    eval_train_set, _ = get_datasets(args.dataset, train_augment=False)
    idx = build_split_indices(train_set, test_set, forget)

    print(f"  retain train: {len(idx['retain_train'])}   "
          f"withheld (class {spec}): {len(idx['forget_train'])}")

    # The shuffle generator is re-seeded from (seed, epoch) at the top of every
    # epoch, so the batch order and the worker augmentation seeds are a pure
    # function of the epoch index. Without this, a resumed run diverges from an
    # uninterrupted one (measured: 89.21% vs 87.80% over 8 epochs) because the
    # generator's position depends on how many epochs this *process* has run.
    shuffle_gen = torch.Generator()
    train_loader = make_loader(train_set, idx["retain_train"], batch_size=args.batch_size,
                               shuffle=True, workers=args.workers, drop_last=True,
                               generator=shuffle_gen)
    retain_test_loader = make_loader(test_set, idx["retain_test"], batch_size=args.batch_size,
                                     workers=min(args.workers, 4))
    forget_test_loader = make_loader(test_set, idx["forget_test"], batch_size=args.batch_size,
                                     workers=min(args.workers, 4))

    model = build_resnet18(n_classes=num_classes(args.dataset), device=device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=args.momentum,
                                weight_decay=args.weight_decay)
    scheduler = build_scheduler(optimizer, args.epochs, warmup_epochs)
    criterion = nn.CrossEntropyLoss()

    resume_path = _resume_path(out_path)
    done = _load_resume(resume_path, model, optimizer, scheduler)
    if done:
        print(f"  Resuming from epoch {done + 1}/{args.epochs}")

    print(f"=== Retraining ResNet-18 on {args.dataset} minus class {spec} "
          f"for {args.epochs} epoch(s) ===")
    start = time.perf_counter()
    for epoch in range(done + 1, args.epochs + 1):
        epoch_start = time.perf_counter()
        shuffle_gen.manual_seed(args.seed * 100_000 + epoch)
        loss = common.train_one_epoch(model, train_loader, optimizer, criterion, amp=amp)
        scheduler.step()
        elapsed = time.perf_counter() - start
        eta = (args.epochs - epoch) * elapsed / max(epoch - done, 1)
        line = (f"  Epoch {epoch}/{args.epochs} - loss: {loss:.4f} - "
                f"lr: {scheduler.get_last_lr()[0]:.4f} - "
                f"{time.perf_counter() - epoch_start:.1f}s - eta {eta / 60:.1f}m")
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            line += f" - retain acc: {evaluate(model, retain_test_loader, amp=amp):.2f}%"
        print(line, flush=True)
        _save_resume(resume_path, epoch, model, optimizer, scheduler)

    # Report in fp32: these are the numbers the paper quotes.
    d_r = evaluate(model, retain_test_loader, amp=False)
    d_f = evaluate(model, forget_test_loader, amp=False)
    train_time = time.perf_counter() - start
    print(f"  Retain test acc: {d_r:.2f}%   Forget test acc: {d_f:.2f}%   "
          f"({train_time / 60:.1f} min)")
    if d_f > 5.0:
        print(f"  WARNING: a gold model should be near 0% on the forgotten class, got {d_f:.2f}%")

    save_checkpoint_atomic({
        "state_dict": model.state_dict(),
        "architecture": "resnet18_cifar",
        "dataset": args.dataset,
        "num_classes": num_classes(args.dataset),
        "role": "retain_gold",
        "forget_classes": forget,
        "seed": args.seed,
        "retain_accuracy": d_r,
        "forget_accuracy": d_f,
        "epochs": args.epochs,
    }, out_path)
    print(f"  Saved gold model to {out_path}")

    manifest.record(
        out_path, role="retain_gold", family="cifar", arch="resnet18",
        dataset=args.dataset, forget_classes=forget, seed=args.seed,
        hparams={"epochs": args.epochs, "batch_size": args.batch_size, "lr": lr,
                 "momentum": args.momentum, "weight_decay": args.weight_decay,
                 "warmup_epochs": warmup_epochs},
        metrics={"d_r": f"{d_r:.4f}", "d_f": f"{d_f:.4f}"},
        train_time_s=train_time,
    )
    os.remove(resume_path)
    print(f"  Recorded in {paths.manifest_path()}")


if __name__ == "__main__":
    main()
