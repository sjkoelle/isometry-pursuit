# Isometry Pursuit

Convex recovery of orthonormal column-submatrices via multitask basis pursuit with symmetric normalization.

## Installation

```bash
pip install -e .
```

## Run experiments

### 1. Prepare data

**Wine and Iris:**

```bash
python -m isometry_pursuit.exec.load_wine_exec --output-dir data --R 25
python -m isometry_pursuit.exec.load_iris_exec --output-dir data --R 25
```

**Ethanol (requires download):**

```bash
mkdir -p data
gdown 1CvVOIXokjfTT0AtyeXSZMlojnfL3AcUL -O data/ethanol.mat
python -m isometry_pursuit.exec.load_ethanol_exec \
    --config-path data/ethanol_config.json \
    --raw-data-path data/ethanol.mat \
    --positions-path data/positions.npy \
    --indices-path data/indices.npy \
    --angles-path data/angles.npy \
    --output-dir data
```

### 2. Run experiments

```bash
# Wine (D=6)
python -m isometry_pursuit.exec.run_experiment \
    --data-file data/wine_standardized_resampled_tensor.npy \
    --output-path results/wine_results.csv \
    --D 6 --compute-brute

# Iris (D=4)
python -m isometry_pursuit.exec.run_experiment \
    --data-file data/iris_standardized_resampled_tensor.npy \
    --output-path results/iris_results.csv \
    --D 4 --compute-brute

# Ethanol
python -m isometry_pursuit.exec.run_experiment \
    --data-file data/ethanol_tensor.npy \
    --output-path results/ethanol_results.csv \
    --compute-brute
```

## Generate figures

```bash
# Figure 1: example (data, selection, normalization)
python -m isometry_pursuit.exec.generate_figure_1_example --output-dir figures

# Figure 1: method (transformation and loss functions)
python -m isometry_pursuit.exec.generate_figure_1_method --output-dir figures

# Figure 2: box plots (requires experiment results)
python -m isometry_pursuit.exec.generate_figure2 --results-dir results --output-dir figures
```
