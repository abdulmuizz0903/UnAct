"""
unlearn_lib.io
Crash-safe, concurrency-safe persistence for results and checkpoints.

Why this exists
---------------
A writer that holds an exclusive flock and then does
`f.seek(0); f.truncate()` before rewriting the whole file is not enough. The flock
stops two GPUs from interleaving writes, but it does nothing about crashes: a
power loss or SIGKILL between the truncate and the final flush leaves an empty
or half-written file, destroying every previously accumulated row rather than
just the one being added.

Both helpers here write to a temporary file in the same directory and then
os.replace() it over the target. os.replace is atomic on POSIX within a single
filesystem, so a reader (or a crash) sees either the old file or the new one,
never a partial one.
"""
from __future__ import annotations

import csv
import fcntl
import os
import tempfile
from typing import Callable, Dict, Iterable, List, Optional, Sequence


def _atomic_write(path: str, write_fn: Callable[[object], None], newline: str = "") -> None:
    """Write via a same-directory temp file, then atomically rename over `path`."""
    directory = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(directory, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=directory, prefix=".tmp-", suffix=".csv")
    try:
        with os.fdopen(fd, "w", newline=newline) as f:
            write_fn(f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        # Leave the original file untouched on any failure.
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
        raise


def upsert_rows(
    rows: Iterable[Dict[str, object]],
    out_path: str,
    fieldnames: Sequence[str],
    key_fields: Sequence[str],
) -> int:
    """Merge `rows` into the CSV at `out_path`, keyed on `key_fields`.

    Holds an exclusive lock on a sidecar .lock file for the whole read-modify-write
    so two GPU workers can share one results file, and writes atomically so a crash
    cannot destroy rows that were already there.

    Returns the total number of rows in the file afterwards.
    """
    rows = list(rows)
    lock_path = out_path + ".lock"
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)

    def key_of(row: Dict[str, object]) -> tuple:
        return tuple(str(row.get(k, "")) for k in key_fields)

    with open(lock_path, "a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            existing: Dict[tuple, Dict[str, object]] = {}
            if os.path.exists(out_path):
                with open(out_path, newline="") as f:
                    reader = csv.DictReader(f)
                    if reader.fieldnames:
                        for r in reader:
                            existing[key_of(r)] = r
            for row in rows:
                existing[key_of(row)] = row

            def write(f):
                writer = csv.DictWriter(f, fieldnames=list(fieldnames), extrasaction="ignore")
                writer.writeheader()
                for k in sorted(existing):
                    writer.writerow(existing[k])

            _atomic_write(out_path, write)
            return len(existing)
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def existing_keys(out_path: str, key_fields: Sequence[str]) -> set:
    """Keys already present in a results CSV, for --skip-existing.

    Returns an empty set if the file does not exist yet. Read under a shared lock
    so a concurrent writer's atomic rename cannot be observed mid-flight.
    """
    if not os.path.exists(out_path):
        return set()
    lock_path = out_path + ".lock"
    with open(lock_path, "a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_SH)
        try:
            with open(out_path, newline="") as f:
                reader = csv.DictReader(f)
                if not reader.fieldnames:
                    return set()
                return {tuple(str(r.get(k, "")) for k in key_fields) for r in reader}
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def save_checkpoint_atomic(obj, path: str) -> str:
    """torch.save to a temp file in the same directory, then atomically rename.

    A crash during a plain torch.save leaves a truncated, unloadable file. For a
    60-minute ViT finetune that is the whole run.
    """
    import torch

    directory = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(directory, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=directory, prefix=".tmp-", suffix=".pt")
    os.close(fd)
    try:
        torch.save(obj, tmp)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
        raise
    return path
