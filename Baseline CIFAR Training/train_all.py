"""
train_all.py

Trains the CIFAR baselines for several datasets in parallel, one process per
GPU. ResNet-18 on 32x32 inputs is too small for multi-GPU data parallelism to
pay off (the gradient all-reduce costs more than the extra compute saves), so
the two cards are used to run *different datasets* concurrently instead.

With two GPUs and three datasets this runs two at a time and starts the third
as soon as a card frees up.

Usage:
    python train_all.py
    python train_all.py --datasets cifar10 cifar100 --epochs 100
    python train_all.py --gpus 1                       # single card
    python train_all.py --extra-args --batch-size 512
"""
import argparse
import os
import subprocess
import sys
import time

import torch

import common

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_SCRIPT = os.path.join(_THIS_DIR, "train_resnet18_cifar.py")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=common.dataset_names(),
                        choices=common.dataset_names())
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--gpus", type=int, nargs="+", default=None,
                        help="GPU indices to use. Default: all visible GPUs.")
    parser.add_argument("--log-dir", default=os.path.join(_THIS_DIR, "logs"))
    parser.add_argument("--extra-args", nargs=argparse.REMAINDER, default=[],
                        help="Everything after this flag is forwarded to train_resnet18_cifar.py")
    args = parser.parse_args()

    gpus = args.gpus if args.gpus is not None else list(range(torch.cuda.device_count()))
    if not gpus:
        parser.error("No GPUs available; run train_resnet18_cifar.py directly to train on CPU.")

    os.makedirs(args.log_dir, exist_ok=True)
    print(f"Datasets: {args.datasets}")
    print(f"GPUs: {[f'{g}: {torch.cuda.get_device_name(g)}' for g in gpus]}")
    print(f"Logs:  {args.log_dir}")

    pending = list(args.datasets)
    running = {}  # gpu index -> (dataset, Popen, log file handle, start time)
    failures = []

    while pending or running:
        for gpu in gpus:
            if not pending or gpu in running:
                continue
            dataset = pending.pop(0)
            log_path = os.path.join(args.log_dir, f"{dataset}.log")
            log_file = open(log_path, "w")
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
            cmd = [
                sys.executable, TRAIN_SCRIPT,
                "--dataset", dataset,
                "--epochs", str(args.epochs),
            ] + args.extra_args
            proc = subprocess.Popen(cmd, env=env, stdout=log_file, stderr=subprocess.STDOUT)
            running[gpu] = (dataset, proc, log_file, time.perf_counter())
            print(f"  launched {dataset} on GPU {gpu} (pid {proc.pid}) -> {log_path}", flush=True)

        time.sleep(5)

        for gpu, (dataset, proc, log_file, started) in list(running.items()):
            if proc.poll() is None:
                continue
            log_file.close()
            del running[gpu]
            mins = (time.perf_counter() - started) / 60
            if proc.returncode == 0:
                print(f"  finished {dataset} on GPU {gpu} in {mins:.1f} min", flush=True)
            else:
                failures.append(dataset)
                print(f"  FAILED {dataset} on GPU {gpu} (exit {proc.returncode}) - "
                      f"see {os.path.join(args.log_dir, dataset + '.log')}", flush=True)

    csv_path = os.path.join(common.MODELS_DIR, "accuracies.csv")
    if os.path.exists(csv_path):
        print(f"\nAccuracy summary ({csv_path}):")
        with open(csv_path) as f:
            for line in f:
                print("  " + line.rstrip())

    if failures:
        print(f"\n{len(failures)} run(s) failed: {failures}")
        sys.exit(1)
    print("\nAll runs completed.")


if __name__ == "__main__":
    main()
