"""
prepare_data.py

Downloads CIFAR-10 and CIFAR-100 into data/ and builds the reusable per-class
split index files under data/<dataset>/splits/ for all three dataset views
(cifar10, cifar20, cifar100). CIFAR-20 shares the CIFAR-100 download but is
labelled with the 20 coarse superclasses.

Idempotent: existing archives and split files are left alone unless --force
is passed.

Usage:
    python prepare_data.py
    python prepare_data.py --datasets cifar10 --force
"""
import argparse
import os

from common import (
    DATA_DIR,
    build_and_save_splits,
    dataset_names,
    get_datasets,
    num_classes,
    splits_dir,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=dataset_names(), choices=dataset_names())
    parser.add_argument("--force", action="store_true", help="Rewrite split files that already exist")
    args = parser.parse_args()

    print(f"Data directory: {DATA_DIR}")

    for name in args.datasets:
        print(f"\n=== {name} ({num_classes(name)} classes) ===")
        train_set, test_set = get_datasets(name, train_augment=False)
        print(f"  train: {len(train_set)} images, test: {len(test_set)} images")
        print(f"  classes: {train_set.classes[:5]}{' ...' if len(train_set.classes) > 5 else ''}")

        build_and_save_splits(name, train_set, test_set, force=args.force)
        out_dir = splits_dir(name)
        print(f"  wrote {len(os.listdir(out_dir))} split file(s) to {out_dir}")

    print("\nDone.")


if __name__ == "__main__":
    main()
