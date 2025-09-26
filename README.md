# Isometry Pursuit

A Python package for isometry pursuit experiments with both Jupyter notebook and command-line interfaces.

## Installation

```bash
git clone https://github.com/sjkoelle/isometry-pursuit.git
cd isometry-pursuit
pip install -e .
```

## Quick Start

### Generate Figure 1

```bash
# Generate Figure 1 plots (data visualization, selection, and normalization)
python -m isometry_pursuit.exec.generate_data_visualization --output-dir figures

# Generate Figure 1 plots (transformation and loss functions)
python -m isometry_pursuit.exec.generate_figure1 --output-dir figures
```

The easiest way to run experiments:

```bash
# Load data
python -m isometry_pursuit.exec.load_iris_exec --output-dir data/iris
python -m isometry_pursuit.exec.load_wine_exec --output-dir data/wine_6 --D 6  # Sample 6 features per replicate
python -m isometry_pursuit.exec.load_wine_exec --output-dir data/wine --R 1 # Use all 13 features

# Load ethanol data (requires specific file paths)
python -m isometry_pursuit.exec.load_ethanol_exec \
    --config-path /Users/samsonkoelle/montlake/experiments/configs/ethanol_full.json \
    --raw-data-path /Users/samsonkoelle/thesis_data/raw_data/ethanol.mat \
    --positions-path /Users/samsonkoelle/positions.npy \
    --indices-path /Users/samsonkoelle/indices.npy \
    --angles-path /Users/samsonkoelle/ethanolanglesfull_0524.npy \
    --output-dir data/ethanol

# Run isometry pursuit (basis pursuit) experiments (using resampled data tensors)
# Note: --D specifies target sparsity (number of features to select) and triggers line search
# Experiments now generate coefficient matrices for separate plotting analysis
python -m isometry_pursuit.exec.run_isometry_pursuit \
    --data-file data/iris/iris_standardized_resampled_tensor.npy \
    --dataset-name iris \
    --output-dir results

python -m isometry_pursuit.exec.run_isometry_pursuit \
    --data-file data/wine/wine_standardized_resampled_tensor.npy \
    --dataset-name wine \
    --output-dir results

python -m isometry_pursuit.exec.run_isometry_pursuit \
    --data-file data/ethanol/ethanol_tensor.npy \
    --dataset-name ethanol \
    --output-dir results

# Run isometry pursuit with target sparsity (triggers line search for optimal constraint violation)
python -m isometry_pursuit.exec.run_isometry_pursuit \
    --data-file data/wine/wine_standardized_resampled_tensor.npy \
    --dataset-name wine \
    --output-dir results

# Run greedy algorithm experiments
python -m isometry_pursuit.exec.run_greedy \
    --data-file data/wine/wine_standardized_resampled_tensor.npy \
    --dataset-name wine \
    --output-dir results


# # Run convergence analysis (separate from main experiments)
# Step 1: Generate convergence data
python -m isometry_pursuit.exec.run_convergence_analysis \
    --data-file data/iris/iris_standardized_resampled_tensor.npy \
    --dataset-name iris \
    --n-replicates 25 \
    --output-dir results

python -m isometry_pursuit.exec.run_convergence_analysis \
    --data-file data/wine_6/wine_standardized_resampled_tensor.npy \
    --dataset-name wine \
    --n-replicates 25 \
    --output-dir results


python -m isometry_pursuit.exec.run_convergence_analysis \
    --data-file data/ethanol/ethanol_tensor.npy \
    --dataset-name ethanol \
    --n-replicates 25 \
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

# Combined sparsity convergence across all datasets
python -m isometry_pursuit.exec.plot_combined_sparsity_convergence \
  --convergence-file results/iris_convergence_all_replicates.pkl \
  --convergence-file results/wine_convergence_all_replicates.pkl \
  --convergence-file results/ethanol_convergence_all_replicates.pkl \
  --output-file figures/combined_sparsity_convergence_boxplot.png \
  --title "Sparsity Convergence Across Datasets"

# Greedy vs Two-Stage (Basis Pursuit → Brute Force)
python -m isometry_pursuit.exec.plot_difference_boxplot \
    --result-files results/iris_results.csv \
                   results/wine_results.csv \
                   results/ethanol_results.csv \
    --baseline-column "loss" \
    --comparison-column "two_stage_loss" \
    --output-file figures/cross_dataset_comparison.png \
    --title "Greedy vs Two-Stage Across Datasets"
```

## Output Files

The `run_isometry_pursuit` command generates several output files:

### Standard Results

- **`{dataset_name}_results.csv`**: Main results with metrics for each replicate

### Coefficient Matrices (NEW)

- **`{dataset_name}_coefficient_matrices.pkl`**: Coefficient matrices for each replicate and constraint violation
  - Structure: `{replicate_idx: {constraint_violation: coefficient_matrix}}`
  - Dimensions: `(number_of_features, D)` for each coefficient matrix
  - Use this file to plot regularization paths and analyze coefficient convergence

### Convergence Analysis (if requested)

- **`{dataset_name}_standardized_{power}_{D}_convergence.pkl`**: Convergence logs for detailed analysis

The `run_greedy` command generates:

### Greedy Results

- **`{dataset_name}_greedy_results.csv`**: Greedy algorithm results with metrics for each replicate
- **`{dataset_name}_greedy_solutions.pkl`**: Greedy feature selection solutions for each replicate

## Data

### Dataset Parameters

- **Iris**: Uses all 4 features (no feature sampling)
- **Wine**: Samples D=6 features for each replicate from 13 available features
- **Ethanol**: Uses all available features (no feature sampling)

The ethanol data is available at `https://drive.google.com/file/d/1CvVOIXokjfTT0AtyeXSZMlojnfL3AcUL/view?usp=drive_link`.
