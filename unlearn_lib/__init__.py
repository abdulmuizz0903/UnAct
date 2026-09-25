"""
unlearn_lib
Shared utilities for the UnAct experiment programme.

Lives at the repo root as a real package because the existing folders
("Baseline CIFAR Training", "Our_Method", "SSD", ...) contain spaces and cannot
be imported by name. Every runner adds the repo root to sys.path and imports
from here, so there is exactly one implementation of each helper.

Modules:
  io       atomic, flock-protected CSV upsert + checkpoint saving
  paths    MODELS_ROOT resolution and the checkpoint naming scheme
  manifest the checkpoint registry
  timing   wall-clock measurement that accounts for CUDA asynchrony
  splits   forget/retain index construction, incl. subsampling and multi-class
"""
