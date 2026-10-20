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
python -m isometry_pursuit.exec.load_wine_exec --output-dir data --R 25
python -m isometry_pursuit.exec.load_iris_exec --output-dir data --R 25
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
    --R 25 
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

Generate the manuscript experiment table rows from these outputs:

```bash
python -m isometry_pursuit.exec.generate_experiment_table \
    --results results/iris_results.json:Iris \
              results/wine_results.json:Wine \
              results/ethanol_results.json:Ethanol \
    --output archival/generated/experiment_table_rows.tex
```

**Synthetic identity-block experiment:**

```bash
python -m isometry_pursuit.exec.run_identity_experiment \
    --config configs/identity_experiment.yaml \
    --output-dir results/identity_experiment
```

This writes per-run selected/true indices, configuration metadata, and summary
tables to `results/identity_experiment`. Command-line options override values
in `configs/identity_experiment.yaml`.

A quick pilot sweep can be run with:

```bash
python -m isometry_pursuit.exec.run_identity_experiment \
    --config configs/identity_experiment_small.yaml \
    --output-dir results/identity_experiment_small
```

<!-- ```bash
python -m isometry_pursuit.exec.scs_convergence_restarts --no-adaptive-scale
python -m isometry_pursuit.exec.scs_convergence_trace --no-adaptive-scale
``` -->
## Generate figures

```bash
# Figure 2: loss function landscapes (6x2 grid comparing loss properties)
python -m isometry_pursuit.exec.visualize_loss_landscapes --output-dir figures

# Figure 1: example (data, selection, normalization)
python -m isometry_pursuit.exec.generate_figure_1_example --output-dir figures

# Figure 1: method (transformation and loss functions)
python -m isometry_pursuit.exec.generate_figure_1_method --output-dir figures

# Group-lasso loss box plots (requires experiment results)
python -m isometry_pursuit.exec.plot_loss_comparison --results-file results/wine_results.json --output-dir figures
python -m isometry_pursuit.exec.plot_loss_comparison --results-file results/iris_results.json --output-dir figures
python -m isometry_pursuit.exec.plot_loss_comparison --results-file results/ethanol_results.json --output-dir figures

# Cardinality histograms (requires experiment results)
python -m isometry_pursuit.exec.plot_cardinality_histogram --results-file results/wine_results.json --output-dir figures
python -m isometry_pursuit.exec.plot_cardinality_histogram --results-file results/iris_results.json --output-dir figures
python -m isometry_pursuit.exec.plot_cardinality_histogram --results-file results/ethanol_results.json --output-dir figures --bin-width 1

# Synthetic identity-block success rates (requires identity experiment results)
python -m isometry_pursuit.exec.plot_identity_experiment \
    --results-dir results/identity_experiment_small \
    --output-dir figures

# Compare BP with forward-backward and BP-seeded forward-backward
python -m isometry_pursuit.exec.plot_identity_experiment \
    --results-dir results/identity_experiment_small \
    --output-dir figures \
    --compare-methods
```

The identity plotter saves one success-rate curve per full dimension, e.g.
`figures/identity_experiment_strict_success_rate_D4.png`. With
`--compare-methods`, it saves method-comparison curves such as
`figures/identity_experiment_method_comparison_D4.png`.

## Analyze convergence (optional)

For detailed convergence analysis across solver iterations:

```bash
# Wine convergence
python -m isometry_pursuit.exec.plot_convergence_analysis \
    --convergence-file results/wine_convergence.pkl \
    --dataset-name wine \
    --output-dir figures

# Iris convergence
python -m isometry_pursuit.exec.plot_convergence_analysis \
    --convergence-file results/iris_convergence.pkl \
    --dataset-name iris \
    --output-dir figures

# Ethanol convergence
python -m isometry_pursuit.exec.plot_convergence_analysis \
    --convergence-file results/ethanol_convergence.pkl \
    --dataset-name ethanol \
    --output-dir figures
```

This generates:
- Coefficient convergence plots showing how feature weights evolve
- Sparsity convergence boxplots across replicates
- Support cardinality histograms

### Paper FoBa (smooth coefficient objectives)

`isometry_pursuit.foba` implements FoBa-obj and FoBa-gdt from Liu, Fujimaki,
and Ye ([ICML 2014, Algorithm 1](https://proceedings.mlr.press/v32/liub14.pdf)).
These start from zero, add one coordinate, refit the support, and repeatedly
prune using the paper's half-forward-gain criterion. Support size is adaptive.

```python
from isometry_pursuit.foba import foba_least_squares

# X: (samples, features); y: (samples,)
result = foba_least_squares(X, y, variant="obj", delta=1e-8)
# Alternatively: variant="gdt", epsilon=1e-8
# For our target-size adaptation (either variant):
# result = foba_least_squares(X, y, variant="obj", target_dimension=D)
beta, support = result["beta"], result["support"]
```

The loss is `0.5 * ||X @ beta - y||**2`, without sample normalization or an
intercept. `delta` is the FoBa-obj coordinate-gain threshold; `epsilon` is the
FoBa-gdt gradient threshold. With `target_dimension=D`, both variants ignore
those stopping thresholds and continue until D coordinates remain after backward
pruning, returning `status="target_dimension"`. The selected set can include
zero fitted coefficients. Pruning can prevent reaching D; the iteration cap
then returns `status="max_iterations"` and `converged=False`. Omitting the target
retains the paper's threshold stopping. `status` distinguishes threshold termination
from exhaustion of the safety iteration cap. `history` records additions,
deletions, support objectives, and deletion thresholds. For other smooth convex
losses, `foba` accepts objective, restricted-minimization, and coordinate or
gradient callbacks; see its docstring for the callback contracts.

The older `algorithm.forward_backward` function and existing experiment labels
refer to fixed-cardinality subset swap search. The new paper implementation
is separate: the pseudoinverse subset score used in those experiments is not
the smooth convex coefficient objective required by FoBa.

Run the targeted tests and reproducible 50-trial noiseless regression check:

```bash
python -m unittest discover -s tests -v
python -m isometry_pursuit.exec.run_foba_smoke
```

The latter saves per-trial outcomes and summary statistics to
`results/foba_smoke.json`.

### Four greedy + swap comparisons

The four methods are forward greedy + swap, backward greedy + swap,
IP-restricted forward greedy + swap, and IP-restricted backward greedy + swap.
Backward initialization deletes from the entire candidate dictionary down to D;
all methods then evaluate every one-column exchange and accept the best strict
improvement, up to the existing dictionary-size round cap. All use the same
pseudoinverse row-norm score. IP restriction applies to both stages.

`algorithm.greedy_swap(matrix, loss, initialization="forward" or "backward")`
runs either initialization followed by swaps. `experiment.compare_swap_initializations`
runs all four on an already transformed matrix and an IP support, saving the
initial and final supports/losses. The usual `run_experiment` command also saves
the two backward-initialized swap results alongside its existing forward results.
Legacy `forward_backward_*` result keys denote forward greedy + swap, not paper FoBa.

To compare all four methods using the exact IP supports from existing experiments:

```bash
python -m isometry_pursuit.exec.run_swap_comparison --ip-results results/wine_results.json --output-path results/wine_swap_comparison.json
python -m isometry_pursuit.exec.run_swap_comparison --ip-results results/iris_results.json --output-path results/iris_swap_comparison.json
python -m isometry_pursuit.exec.run_swap_comparison --ip-results results/ethanol_results.json --output-path results/ethanol_swap_comparison.json
```

Each command reruns all four searches with seed 1234, saves per-replicate JSON and
a summary CSV, and records the source IP run. Timings exclude the cached IP solve.
The tensor data must match the source IP run. For a synthetic identity-block pilot:

```bash
python -m isometry_pursuit.exec.run_identity_experiment --config configs/identity_experiment_small.yaml --output-dir results/identity_swap_comparison
python -m isometry_pursuit.exec.plot_identity_experiment --results-dir results/identity_swap_comparison --output-dir figures/identity_swap_comparison --compare-methods --methods foba,backward_swap,bp_foba,ip_backward_swap
```

The plotter retains the legacy `foba` and `bp_foba` option keys for the two
forward-initialized swap methods, but labels the curves as greedy + swap.
