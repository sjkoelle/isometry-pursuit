# Isometry Pursuit

Convex recovery of orthonormal column-submatrices via multitask basis pursuit with symmetric normalization.

## Setup

### 1. Create and activate a conda environment

```bash
conda create -n isometry_pursuit python=3.9
conda activate isometry_pursuit
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
pip install -e .
```

## Run experiments

### 1. Prepare data

**Wine and Iris:**

```bash
python -m isometry_pursuit.exec.load_wine_exec --output-dir data --R 1
python -m isometry_pursuit.exec.load_iris_exec --output-dir data --R 1
```

**Ethanol (requires download):**

```bash
mkdir -p data
gdown 1oaaDJWIPSJTRqSOkU8F3napGMPiNFQD1 -O data/ethanol.mat
python -m isometry_pursuit.exec.prepare_ethanol_exec \
    --raw-data-path data/ethanol.mat \
    --config-path configs/ethanol/ethanol_config.json \
    --output-dir data
python -m isometry_pursuit.exec.load_ethanol_exec \
    --config-path configs/ethanol/ethanol_config.json \
    --raw-data-path data/ethanol.mat \
    --positions-path data/positions.npy \
    --indices-path data/indices.npy \
    --angles-path data/angles.npy \
    --output-dir data \
    --R 5 
```

### 2. Run experiments

```bash
python -m isometry_pursuit.exec.run_experiment \
    --data-file data/wine_standardized_resampled_tensor.npy \
    --output-path results/wine_results.json

python -m isometry_pursuit.exec.run_experiment \
    --data-file data/iris_standardized_resampled_tensor.npy \
    --output-path results/iris_results.json

python -m isometry_pursuit.exec.run_experiment \
    --data-file data/ethanol_tensor.npy \
    --output-path results/ethanol_results.json
```

## Generate figures

```bash
# Figure 1: example (data, selection, normalization)
python -m isometry_pursuit.exec.generate_figure_1_example --output-dir figures

# Figure 1: method (transformation and loss functions)
python -m isometry_pursuit.exec.generate_figure_1_method --output-dir figures

# Isometry loss box plots (requires experiment results)
python -m isometry_pursuit.exec.plot_loss_comparison --results-file results/wine_results.json --output-dir figures
python -m isometry_pursuit.exec.plot_loss_comparison --results-file results/iris_results.json --output-dir figures
python -m isometry_pursuit.exec.plot_loss_comparison --results-file results/ethanol_results.json --output-dir figures

# Cardinality histograms (requires experiment results)
python -m isometry_pursuit.exec.plot_cardinality_histogram --results-file results/wine_results.json --output-dir figures
python -m isometry_pursuit.exec.plot_cardinality_histogram --results-file results/iris_results.json --output-dir figures
python -m isometry_pursuit.exec.plot_cardinality_histogram --results-file results/ethanol_results.json --output-dir figures --bin-width 25
```
