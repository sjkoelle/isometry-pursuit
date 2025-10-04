# Isometry Pursuit

A Python package for isometry pursuit experiments with command-line interfaces.

## Installation

```bash
git clone https://github.com/sjkoelle/isometry-pursuit.git
cd isometry-pursuit
pip install -e .
```

## Quick Start

### Load Wine Data

```bash
# Load wine data
python -m isometry_pursuit.exec.load_wine_exec --output-dir data/wine --R 1
python -m isometry_pursuit.exec.generate_orthogonal_data \
    --output-dir data/orthogonal \
    --matrix-specs "[(2,1),(3,1)]" \
    --R 1 \
    --B 3
```

### Run Isometry Pursuit

```bash
# Run isometry pursuit with fixed constraint violation
python -m isometry_pursuit.exec.run_isometry_pursuit \
    --data-file data/wine/wine_standardized_resampled_tensor.npy \
    --output-dir results/wine_fixed \
    --constraint-violation 0.1

python -m isometry_pursuit.exec.run_isometry_pursuit \
    --data-file data/orthogonal/orthogonal_data_tensor.npy \
    --output-dir results/o_fixed \
    --constraint-violation 0.

```

### Run Line Search for Target Cardinality

```bash
# Find constraint violation that achieves target cardinality of 6
python -m isometry_pursuit.exec.run_line_search_for_cardinality \
    --data-file data/wine/wine_standardized_resampled_tensor.npy \
    --output-dir results/wine_line_search

python -m isometry_pursuit.exec.run_line_search_for_cardinality \
    --data-file data/orthogonal/orthogonal_data_tensor.npy \
    --output-dir results/orthogonal_line_search --target-cardinality 3

# Compute isometry loss of selected support
python -m isometry_pursuit.exec.compute_isometry_loss \
    --coefficient-matrix results/orthogonal_line_search/coefficient_matrix.npy \
    --selected-indices results/orthogonal_line_search/selected_indices.npy \
    --power 1.0 \
    --D 10 \
    --output-file results/orthogonal_line_search/isometry_loss.npy
```

## Output Files

### Isometry Pursuit Results

- **`coefficient_matrix.npy`**: The coefficient matrix from isometry pursuit
- **`selected_indices.npy`**: Indices of selected features

### Line Search Results

- **`coefficient_matrix.npy`**: The coefficient matrix from the optimal constraint violation
- **`selected_indices.npy`**: Indices of selected features
- **`line_search_result.pkl`**: Complete line search results including:
  - `best_violation`: Optimal constraint violation found
  - `best_cardinality`: Actual cardinality achieved
  - `target_cardinality`: Target cardinality requested
  - `result`: Full isometry pursuit result

## Wine Dataset

The wine dataset contains 13 features for experiments.
