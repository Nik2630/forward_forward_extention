# Reproducibility Checklist

## Environment (conda)

1. Create environment:

   conda create -n ff-ext python=3.11 -y

2. Activate environment:

   conda activate ff-ext

3. Install dependencies:

   pip install -r requirements.txt

## Determinism settings

- All experiment scripts expose `--seed`.
- Data splits and shuffling use seeded generators.
- CUDA flags are set for deterministic behavior (safe on CPU as well).

## Canonical experiment commands

1. Baseline MNIST:

   PYTHONPATH=. python experiments/baseline_mnist.py --epochs 10 --batch-size 256 --hidden-dim 256 --layers 3 --seed 42 --run-name baseline_mnist --output results/logs/ablation_baseline_mnist.csv

2. Goodness ablation (3 seeds):

   PYTHONPATH=. python experiments/goodness_variants.py --epochs 10 --batch-size 256 --hidden-dim 256 --layers 3 --seeds 42,43,44 --train-subset 12000

3. Activation ablation (Student-t):

   PYTHONPATH=. python experiments/activation_experiments.py --activation student_t_nll --goodness unsquared --epochs 10 --batch-size 256 --hidden-dim 256 --layers 3 --seed 42 --train-subset 12000 --run-name activation_student_t --output results/logs/ablation_activation_student_t.csv

4. Spatial diagnostics (MNIST jitter):

   PYTHONPATH=. python experiments/spatial_local_goodness.py --dataset mnist --steps 50 --batch-size 128 --block-size 4 --stride 2 --jitter-pixels 2 --train-subset 4096 --output results/logs/ablation_spatial_mnist.csv

5. Reduced CIFAR-10 with local receptive mask:

   PYTHONPATH=. python experiments/cifar10_reduced_ff.py --epochs 5 --batch-size 128 --hidden-dim 256 --layers 3 --train-subset 10000 --grid-size 4 --receptive-field 11 --rf-channels 16 --seed 42 --output results/logs/ablation_cifar10_local_rf.csv

## Reporting pipeline

1. Merge ablation logs:

   PYTHONPATH=. python experiments/ablation_summary.py --logs-dir results/logs --pattern 'ablation_*.csv' --output results/ablations.csv --best-output results/ablations_best.csv

2. Create learning curves:

   PYTHONPATH=. python experiments/plot_learning_curves.py --input results/ablations.csv --metric test_error_pct --hue run_name --output results/figures/learning_curves.png

3. Create summary stats table:

   PYTHONPATH=. python experiments/report_stats.py --input results/ablations_best.csv --group-by run_name --output results/summary_stats.csv

4. Create final convergence/error analysis report:

   PYTHONPATH=. python experiments/final_analysis.py --input results/ablations.csv --summary-output results/summary_stats.csv --figure-output results/figures/convergence_vs_error.png --markdown-output results/final_analysis.md --convergence-threshold 25

## Expected artifacts

- results/logs/ablation_*.csv
- results/ablations.csv
- results/ablations_best.csv
- results/summary_stats.csv
- results/figures/learning_curves.png
- results/figures/convergence_vs_error.png
- results/final_analysis.md
