# UnAct: Gradient-Free Unlearning via Targeted Activation Intervention

Code for the ICLR 2027 submission. It contains UnAct, our SSD and LFSSD
implementations, the training scripts for the original and retrained models,
one script per experiment, every results file behind the paper, and the
pipeline that turns those results into the paper's tables and figures.

## Quick start: rebuild the paper's tables and figures

Every number in the paper is aggregated from `results/*.csv`. With pandas and
matplotlib installed, this runs on CPU in about 15 seconds:

```bash
pip install -r requirements.txt
python paper/make_all.py
```

Tables go to `paper/tables/` (`\input`-able LaTeX) and figures to
`paper/figures/` (PDF and PNG). `paper/tables/numbers.tex` defines one macro
per number quoted in the text, with its source file in a comment. Figure 1 is
drawn by hand and is not generated.

## Layout

| Path | Contents |
|---|---|
| `Our_Method/unlearn_cnn.py` | UnAct for ResNet-18 (`unlearn_blocks`; the paper uses `scope="layer4_last"`) |
| `Our_Method/unlearn_vit.py` | UnAct for ViT-B/16 (`unlearn_vit`; the paper uses `scope="ffn_dla_last4"`) |
| `SSD/ssd.py`, `SSD/lfssd.py`, `SSD/metrics.py` | SSD, LFSSD, and SSD's MIA/ZRF metrics |
| `unlearn_lib/` | Shared code: splits, loaders, evaluation, timing, crash-safe CSV writes, checkpoint paths |
| `Baseline CIFAR Training/` | Data preparation, original ResNet-18 training, retrain-from-scratch models |
| `Baseline ViT Training/` | ViT-B/16 fine-tuning (original and retrained) |
| `experiments/` | One script per experiment (table below), plus `analyse_*.py` selection scripts |
| `scripts/` | `run_all.sh` (the full programme), a detached job launcher, gold-model sweeps, selection helpers |
| `results/` | Every CSV the paper reads |
| `paper/` | `make_all.py` and the table and figure scripts |

## Using UnAct

UnAct needs only the forget set, and it never reads the labels:

```python
import sys; sys.path.insert(0, "Our_Method")
from unlearn_cnn import unlearn_blocks

# p = threshold_percentile, gamma = penalty_scale, k = iter_times
unlearn_blocks(model, forget_loader, device, scope="layer4_last",
               threshold_percentile=99, penalty_scale=0.01, iter_times=20)  # edits in place
```

Selected configurations (E6): CIFAR-10 p=99, γ=0.01, k=20; CIFAR-20 p=99, γ=0.1,
k=20; CIFAR-100 p=90, γ=0.3, k=1. Baselines run at one operating point per
dataset, as their papers do: SSD α 8/15/50 (E1) and LFSSD α 6/10/30 (E9).

## Reproducing the experiments

```bash
python "Baseline CIFAR Training/prepare_data.py"   # CIFAR download and split indices
scripts/launch.sh 0 run-all bash scripts/run_all.sh
scripts/status.sh                                  # progress
```

`scripts/run_all.sh` runs the whole programme in dependency order with the exact
arguments that produced each CSV: the original models, 27 retrained ResNet-18
and 5 retrained ViT-B/16 models, the tuning grids, then every experiment. Every step can be
re-run safely. Finished work is skipped, and an interrupted run resumes from
its last step, or from its last epoch for training. `launch.sh` detaches the
job, so it keeps running after you log out. Stop jobs with `scripts/stop.sh`.

Checkpoints are written under `$UNLEARN_MODELS_ROOT` (default `models/`). Set
`UNLEARN_PYTHON` to choose the interpreter (default `python`).

| Experiment | Script | Results | In the paper |
|---|---|---|---|
| E1 SSD grid | `e1_ssd_tune.py` | `e1_ssd_tune.csv` | SSD grid, sensitivity figures |
| E6 UnAct grid | `e6_unact_grid.py` | `e6_unact_grid.csv` | UnAct grid, coverage |
| E9a LFSSD grid | `e9_lfssd_tune.py` | `e9_lfssd_tune.csv` | LFSSD grid |
| E2 + E9 main table | `e2_main_table.py`, `e9_baselines.py` | `e2_main_table_l4.csv`, `e9_baselines.csv` | Main table (Δ, MIA, ZRF) |
| E3 forget-set size | `e3_forget_size.py`, `e3_lfssd_forget_size.py`, `analyse_e3_oracle.py` | `e3_*.csv` | Forget-set size figure, SSD per-n oracle |
| E4 multi-class / sequential | `e4_multiclass.py` | `e4_multiclass.csv`, `e4b_seq_orders.csv` | Sequential requests |
| E15 gentler CIFAR-100 sequential | `e15_seq_gentle.py` | `e15_seq_gentle.csv` | Sequential, CIFAR-100 |
| E5 ViT-B/16 | `e5_vit.py`, `e5_vit_ssd.py`, `e5c_vit_relearn.py`, `e5d_vit_forget_size.py` | `e5*.csv` | ViT table, variants, relearning, forget-set size |
| E7 relearning | `e7_relearn.py` | `e7_relearn.csv` | Relearning |
| E8 class selectivity | `e8_selectivity.py` | `e8_selectivity.csv` | Selectivity |
| E9 label-free check | `e9_labelfree_check.py` | `e9_labelfree_check.csv` | Label-free check |
| E11 classes SSD reports as hard | `e1`, `e6`, `e9_lfssd` with `--classes` | `e11_hard_classes_*.csv` | Hard classes |
| E12 output baselines, in/out ablation, probe | `e12_probe_ablation.py` | `e12_probe_ablation_*.csv` | What Δ measures |
| E13 retrain-free selection | `analyse_e13_selection.py` | `e13_selection.csv` | Selection |
| Cost | `measure_cost_breakdown.py` | `cost_breakdown_l4*.csv` | Cost vs. rounds k |

## Protocol notes

* **Δ** = |D_r − D_r^gold| + |D_f − D_f^gold| in points. It is computed for
  each forget class and then averaged. The retrained ("gold") model is the
  target for every method.
* **Forget classes, fixed before any results were seen:** CIFAR-10 {0,2,3,5,8}
  (also used for ViT-B/16); CIFAR-20 {3,4,10,14,19}; CIFAR-100 {3,20,51,69,85}.
  Seed 42 everywhere.
* **Evaluation device.** Models are evaluated on an NVIDIA L4 or A10 at a fixed
  evaluation batch size, and every row's `eval_device` column records which.
  On the same checkpoint, the two GPUs can disagree on one test sample in
  10,000 because of floating-point reduction order. The main table (E2) is
  evaluated on the L4. Accuracies in the checkpoint manifest are as-trained
  values, kept for provenance only.
* **Timings** all come from one L4, with runs serialised so that no other job
  shares the card. The warm-up repeat is discarded.
* **Determinism.** Retrained-model training is bit-reproducible on a given
  device (`CUBLAS_WORKSPACE_CONFIG` is set by `launch.sh`). The original
  ResNet-18 models are trained with bf16 autocast and cuDNN benchmark for speed,
  so re-training them gives slightly different models. The UnAct edit itself is
  device-invariant.
* Every results row records its method, configuration, seed, evaluation device
  and code version.

Model checkpoints are not included because of their size (ResNet-18 ≈ 45 MB,
ViT-B/16 ≈ 346 MB each). `run_all.sh` regenerates them.
