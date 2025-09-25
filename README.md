# Isometry Pursuit

A Python package for isometry pursuit experiments with both Jupyter notebook and command-line interfaces.

## Installation

```bash
git clone https://github.com/sjkoelle/isometry-pursuit.git
cd isometry-pursuit
pip install -e .
```

## Quick Start

The easiest way to run experiments:

```bash
# Load data
python -m isometry_pursuit.exec.load_iris_exec --output-dir data/iris
python -m isometry_pursuit.exec.load_wine_exec --output-dir data/wine_6 --D 6  # Sample 6 features per replicate

# Load ethanol data (requires specific file paths)
python -m isometry_pursuit.exec.load_ethanol_exec \
    --config-path /Users/samsonkoelle/montlake/experiments/configs/ethanol_full.json \
    --raw-data-path /Users/samsonkoelle/thesis_data/raw_data/ethanol.mat \
    --positions-path /Users/samsonkoelle/positions.npy \
    --indices-path /Users/samsonkoelle/indices.npy \
    --angles-path /Users/samsonkoelle/ethanolanglesfull_0524.npy \
    --output-dir data/ethanol

# Run experiments (using resampled data tensors)
# Note: D is automatically extracted from tensor shape
python -m isometry_pursuit.exec.run_experiment \
    --data-file data/iris/iris_standardized_resampled_tensor.npy \
    --dataset-name iris \
    --output-dir results

python -m isometry_pursuit.exec.run_experiment \
    --data-file data/wine_6/wine_standardized_resampled_tensor.npy \
    --dataset-name wine \
    --output-dir results


python -m isometry_pursuit.exec.run_experiment \
    --data-file data/ethanol/ethanol_tensor.npy \
    --dataset-name ethanol \
    --output-dir results

# # Run convergence analysis (separate from main experiments)
# Step 1: Generate convergence data
python -m isometry_pursuit.exec.run_convergence_analysis \
    --data-file data/iris/iris_standardized_resampled_tensor.npy \
    --dataset-name iris \
    --n-replicates 10 \
    --output-dir results

python -m isometry_pursuit.exec.run_convergence_analysis \
    --data-file data/wine_6/wine_standardized_resampled_tensor.npy \
    --dataset-name wine \
    --n-replicates 10 \
    --output-dir results


python -m isometry_pursuit.exec.run_convergence_analysis \
    --data-file data/ethanol/ethanol_tensor.npy \
    --dataset-name ethanol \
    --n-replicates 10 \
    --output-dir results

# Step 2: Generate convergence plots
python -m isometry_pursuit.exec.plot_convergence_analysis \
    --convergence-file results/iris_convergence_all_replicates.pkl \
    --dataset-name iris \
    --output-dir figures

python -m isometry_pursuit.exec.plot_convergence_analysis \
    --convergence-file results/wine_convergence_all_replicates.pkl \
    --dataset-name wine \
    --output-dir figures


python -m isometry_pursuit.exec.plot_convergence_analysis \
    --convergence-file results/ethanol_convergence_all_replicates.pkl \
    --dataset-name ethanol \
    --output-dir figures

# Compare methods across multiple datasets

# Greedy vs Two-Stage (Basis Pursuit → Brute Force)
python -m isometry_pursuit.exec.plot_difference_boxplot \
    --result-files results/iris_results.csv \
                   results/wine_results.csv \
                   results/ethanol_results.csv \
    --baseline-column "loss" \
    --comparison-column "two_stage_loss" \
    --output-file figures/cross_dataset_comparison.png \
    --title "Greedy vs Two-Stage Across Datasets"

# Greedy vs Two-Stage on Greedy (Isometry Pursuit → Greedy)
python -m isometry_pursuit.exec.plot_difference_boxplot \
    --result-files results/iris_results.csv \
                   results/wine_results.csv \
                   results/ethanol_results.csv \
    --baseline-column "loss" \
    --comparison-column "two_stage_greedy_loss" \
    --output-file figures/greedy_vs_twostage_greedy.png \
    --title "Greedy vs Two-Stage on Greedy Across Datasets"

# Two-Stage on Greedy vs Two-Stage (Basis Pursuit → Brute Force)
python -m isometry_pursuit.exec.plot_difference_boxplot \
    --result-files results/iris_results.csv \
                   results/wine_results.csv \
                   results/ethanol_results.csv \
    --baseline-column "two_stage_greedy_loss" \
    --comparison-column "two_stage_loss" \
    --output-file figures/twostage_greedy_vs_twostage.png \
    --title "Two-Stage on Greedy vs Two-Stage Across Datasets"


# Generate plots for each dataset
python -m isometry_pursuit.exec.plot_results \
    --results-file results/iris_results.csv \
    --dataset-name iris \
    --output-dir figures

python -m isometry_pursuit.exec.plot_results \
    --results-file results/wine_results.csv \
    --dataset-name wine \
    --output-dir figures


python -m isometry_pursuit.exec.plot_results \
    --results-file results/ethanol_results.csv \
    --dataset-name ethanol \
    --output-dir figures
```

## Data

### Dataset Parameters

- **Iris**: Uses all 4 features (no feature sampling)
- **Wine**: Samples D=6 features for each replicate from 13 available features
- **Ethanol**: Uses all available features (no feature sampling)

The ethanol data is available at `https://drive.google.com/file/d/1CvVOIXokjfTT0AtyeXSZMlojnfL3AcUL/view?usp=drive_link`.
