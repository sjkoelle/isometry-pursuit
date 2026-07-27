# Experiment Commands

## Setup

```bash
pip install -r requirements.txt
pip install -e .
```

## Prepare Data

Wine and Iris:

```bash
python -m isometry_pursuit.exec.load_wine_exec --output-dir data --R 25
python -m isometry_pursuit.exec.load_iris_exec --output-dir data --R 25
```

Ethanol:

```bash
mkdir -p data
gdown 1oaaDJWIPSJTRqSOkU8F3napGMPiNFQD1 -O data/ethanol.mat
python -m isometry_pursuit.exec.prepare_ethanol_exec --raw-data-path data/ethanol.mat --config-path configs/ethanol/ethanol_config.json --output-dir data
python -m isometry_pursuit.exec.load_ethanol_exec --config-path configs/ethanol/ethanol_config.json --raw-data-path data/ethanol.mat --positions-path data/positions.npy --indices-path data/indices.npy --angles-path data/angles.npy --output-dir data --R 25
```

## Run Main Experiments

```bash
python -m isometry_pursuit.exec.run_experiment --data-file data/wine_standardized_resampled_tensor.npy --output-path results/wine_results.json
python -m isometry_pursuit.exec.run_experiment --data-file data/iris_standardized_resampled_tensor.npy --output-path results/iris_results.json
python -m isometry_pursuit.exec.run_experiment --data-file data/ethanol_tensor.npy --output-path results/ethanol_results.json
```

## Generate Figures

```bash
python -m isometry_pursuit.exec.visualize_loss_landscapes --output-dir figures
python -m isometry_pursuit.exec.generate_figure_1_example --output-dir figures
python -m isometry_pursuit.exec.generate_figure_1_method --output-dir figures
python -m isometry_pursuit.exec.plot_loss_comparison --results-file results/wine_results.json --output-dir figures
python -m isometry_pursuit.exec.plot_cardinality_histogram --results-file results/wine_results.json --output-dir figures
```

## Convergence Analysis

```bash
python -m isometry_pursuit.exec.plot_convergence_analysis --convergence-file results/wine_convergence.pkl --dataset-name wine --output-dir figures
```

Use analogous `iris` and `ethanol` paths when those files exist.
