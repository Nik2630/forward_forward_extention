# Forward-Forward Extension

CPU-first PyTorch implementation of a research extension to the Forward-Forward (FF) algorithm.

This repository operationalizes the project plan in `project_proposal.md` and evaluates three core objectives:

1. Goodness function variants (`squared` vs `unsquared`).
2. Novel activation (Student-t negative log density).
3. Spatial/local goodness and local receptive-field connectivity (no weight sharing).

The stretch goal involving wake/sleep temporal decoupling is intentionally out of scope for this cycle.

## 1. Project Goals

The project studies whether FF performance and training behavior can be improved by changing:

1. The layer objective (goodness) function.
2. The hidden-layer activation function.
3. The spatial locality of constraints and connectivity.

Outputs are designed to be reproducible and report-ready:

1. Consolidated ablation tables.
2. Learning-curve and convergence figures.
3. Auto-generated final analysis markdown.

## 2. How This Implementation Works

### 2.1 Forward-Forward Training in This Repo

For each mini-batch, training performs:

1. Positive pass: true label is overlaid into input.
2. Negative pass: incorrect sampled label is overlaid.
3. Per-layer local objective: encourage high goodness for positive and low goodness for negative.

The local objective is implemented as BCE on logits:

1. `pos_logit = goodness(pos) - threshold`
2. `neg_logit = goodness(neg) - threshold`

### 2.2 Label Overlay Strategy

Images are flattened and the first 10 dimensions are replaced by one-hot label values.
This follows the supervised FF setup where classification is expressed through goodness by label-conditioned input.

### 2.3 Classification at Evaluation

For each test image, the model runs 10 forward passes (one per candidate label), accumulates total goodness, and picks the label with maximum goodness.

## 3. Core Components

### 3.1 Models

1. `src/models/ff_network.py`
  - `FFNetwork` stacks FF layers and computes local loss across layers.
2. `src/models/layers.py`
  - `FFLayer` for activation + normalization + detach-between-layers behavior.
  - `MaskedLinear` for optional fixed connectivity masks.
  - `build_local_receptive_mask` for non-weight-sharing local receptive fields.
3. `src/models/activations.py`
  - `relu` activation.
  - `student_t_nll` activation based on negative log density under Student-t.

### 3.2 Goodness Objectives

1. `src/goodness/base.py`
  - Abstract interface and BCE-based local loss helper.
2. `src/goodness/squared_sum.py`
  - Goodness = sum of squared pre-normalized activities.
3. `src/goodness/unsquared_sum.py`
  - Goodness = sum of pre-normalized activities.
4. `src/goodness/spatial.py`
  - Block-wise spatial goodness with configurable block size and stride.

### 3.3 Data

1. `src/data/loaders.py`
  - Seeded deterministic splits and dataloader generators.
  - MNIST jitter option.
  - Optional `train_subset` support for CPU-budget experiments.
2. `src/data/preprocessing.py`
  - Label overlay and negative-label sampling.

### 3.4 Training Utilities

1. `src/training/ff_trainer.py`
  - One FF training step.
  - Non-finite loss detection.
  - Gradient norm measurement and clipping.
2. `src/utils/logging.py`
  - Appends epoch diagnostics into CSV logs.
3. `src/utils/seed.py`
  - Reproducibility seed setup for random, NumPy, and PyTorch.

## 4. Experiment Scripts

1. `experiments/baseline_mnist.py`
  - Main configurable experiment runner for MNIST.
  - Supports goodness, activation, normalization, jitter, subset, and threshold options.
2. `experiments/goodness_variants.py`
  - Multi-seed paired runs for squared vs unsquared goodness.
3. `experiments/activation_experiments.py`
  - Activation-focused runs, delegated to baseline runner.
4. `experiments/spatial_local_goodness.py`
  - Local block-goodness diagnostics on MNIST or CIFAR-10.
5. `experiments/cifar10_reduced_ff.py`
  - Reduced CPU-budget CIFAR-10 FF training with local receptive first layer.
6. `experiments/ablation_summary.py`
  - Merges log files and extracts best row per `(run_name, seed)`.
7. `experiments/plot_learning_curves.py`
  - Plots metric-vs-epoch grouped by run identity.
8. `experiments/report_stats.py`
  - Produces run-level mean/std summary table.
9. `experiments/final_analysis.py`
  - Generates final markdown report and convergence-vs-error figure.

## 5. Environment Setup

### 5.1 Conda (recommended)

```bash
conda create -n ff-ext python=3.11 -y
conda activate ff-ext
pip install -r requirements.txt
```

### 5.2 Notes

1. Dependency versions in `requirements.txt` are pinned for reproducibility.
2. All scripts are CPU-compatible.

## 6. Quick Start

### 6.1 Smoke Baseline

```bash
PYTHONPATH=. python experiments/baseline_mnist.py \
  --epochs 1 \
  --batch-size 128 \
  --hidden-dim 256
```

### 6.2 Multi-seed Goodness Ablation

```bash
PYTHONPATH=. python experiments/goodness_variants.py \
  --epochs 3 \
  --batch-size 256 \
  --hidden-dim 256 \
  --layers 3 \
  --seeds 42,43,44 \
  --train-subset 12000
```

### 6.3 Activation Ablation (Student-t)

```bash
PYTHONPATH=. python experiments/activation_experiments.py \
  --activation student_t_nll \
  --goodness unsquared \
  --epochs 3 \
  --batch-size 256 \
  --hidden-dim 256 \
  --layers 3 \
  --seed 42 \
  --train-subset 12000 \
  --run-name activation_student_t
```

### 6.4 Spatial Diagnostics

```bash
PYTHONPATH=. python experiments/spatial_local_goodness.py \
  --dataset mnist \
  --steps 20 \
  --batch-size 128 \
  --block-size 4 \
  --stride 2 \
  --jitter-pixels 2 \
  --train-subset 2048 \
  --output results/logs/ablation_spatial_mnist.csv
```

### 6.5 Reduced CIFAR-10 FF (Local Receptive First Layer)

```bash
PYTHONPATH=. python experiments/cifar10_reduced_ff.py \
  --epochs 3 \
  --batch-size 128 \
  --hidden-dim 256 \
  --layers 3 \
  --train-subset 10000 \
  --grid-size 4 \
  --receptive-field 11 \
  --rf-channels 16 \
  --seed 42
```

## 7. Reporting Pipeline

Run these after experiments complete:

```bash
PYTHONPATH=. python experiments/ablation_summary.py \
  --logs-dir results/logs \
  --pattern 'ablation_*.csv' \
  --output results/ablations.csv \
  --best-output results/ablations_best.csv

PYTHONPATH=. python experiments/plot_learning_curves.py \
  --input results/ablations.csv \
  --metric test_error_pct \
  --hue run_name \
  --output results/figures/learning_curves.png

PYTHONPATH=. python experiments/report_stats.py \
  --input results/ablations_best.csv \
  --group-by run_name \
  --output results/summary_stats.csv

PYTHONPATH=. python experiments/final_analysis.py \
  --input results/ablations.csv \
  --summary-output results/summary_stats.csv \
  --figure-output results/figures/convergence_vs_error.png \
  --markdown-output results/final_analysis.md \
  --convergence-threshold 25
```

Generated report artifacts:

1. `results/ablations.csv`
2. `results/ablations_best.csv`
3. `results/summary_stats.csv`
4. `results/figures/learning_curves.png`
5. `results/figures/convergence_vs_error.png`
6. `results/final_analysis.md`

## 8. Tests

Run the test suite:

```bash
PYTHONPATH=. pytest -q
```

Current tests cover:

1. Goodness calculations.
2. Activation numerical sanity.
3. Objective-normalization coupling checks.
4. Training-step stability and outputs.
5. Spatial goodness shapes.
6. Local receptive mask correctness.

## 9. Repository Layout

```text
.
├── src/
│   ├── data/
│   ├── goodness/
│   ├── models/
│   ├── training/
│   └── utils/
├── experiments/
├── tests/
├── results/
├── project_proposal.md
├── REPRODUCIBILITY.md
└── README.md
```

## 10. Reproducibility Notes

1. Use fixed seeds for all comparative runs.
2. Keep train subsets and epoch budgets identical across variants.
3. Use the same log aggregation commands before generating conclusions.
4. See `REPRODUCIBILITY.md` for the full checklist.

## 11. Known Limitations

1. CIFAR-10 runs are intentionally reduced for CPU budget.
2. Current convergence thresholds are experiment-dependent and should be tuned per study.
3. Sleep/wake-phase FF decoupling is not implemented in this cycle.

## 12. Suggested Next Research Iterations

1. Extend epoch budget for top candidate runs to approach paper-level MNIST error.
2. Add a larger CIFAR-10 local-goodness study with controlled architecture sweeps.
3. Introduce explicit ablation scripts for wake/sleep scheduling as a separate phase.
