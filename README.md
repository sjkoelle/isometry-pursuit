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
python -m isometry_pursuit.exec.load_iris_exec --output-dir data
python -m isometry_pursuit.exec.load_wine_exec --output-dir data --max-features 13
python -m isometry_pursuit.exec.load_word_embeddings --output-dir data --D 7

# Run experiments
python -m isometry_pursuit.exec.run_experiment \
    --data-file data/iris_standardized.csv \
    --dataset-name iris --D 4 --R 5 \
    --output-dir results

python -m isometry_pursuit.exec.run_experiment \
    --data-file data/wine_standardized_13.csv \
    --dataset-name wine --D 6 --R 5 \
    --output-dir results

python -m isometry_pursuit.exec.run_experiment \
    --data-file data/word_embeddings_7.csv \
    --dataset-name word_embeddings --D 7 --R 25 \
    --output-dir results

# Special ethanol experiment (requires data download first)
python -m isometry_pursuit.exec.ethanol_experiment \
    --data-file data/ethanol_standardized.csv \
    --D 2 --R 5 \
    --output-dir results

# Compare methods across multiple datasets
python -m isometry_pursuit.exec.plot_difference_boxplot \
    --result-files results/iris_results.csv \
                   results/wine_results.csv \
                   results/word_embeddings_results.csv \
                   results/ethanol_results.csv \
    --baseline-column "Losses" \
    --comparison-column "Two-Stage Losses" \
    --output-file figures/cross_dataset_comparison.png \
    --title "Greedy vs Two-Stage Across Datasets"


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
    --results-file results/word_embeddings_results.csv \
    --dataset-name word_embeddings \
    --output-dir figures

python -m isometry_pursuit.exec.plot_results \
    --results-file results/ethanol_results.csv \
    --dataset-name ethanol \
    --output-dir figures

# Or generate all figures at once
scripts/generate_figures.sh


```

## Advanced Usage

You can also customize parameters or run individual components:

```bash
# Load data with custom parameters
python -m isometry_pursuit.exec.load_wine_exec --output-dir data --max-features 6
python -m isometry_pursuit.exec.load_word_embeddings --output-dir data --D 5

# Run experiments with custom replicates/parameters
python -m isometry_pursuit.exec.run_experiment \
    --data-file data/iris_standardized.csv \
    --dataset-name iris --D 4 --R 100 \
    --compute-brute --log-convergence \
    --output-dir results

# Override defaults (power=1.0, frac=0.5) if needed
python -m isometry_pursuit.exec.run_experiment \
    --data-file data/iris_standardized.csv \
    --dataset-name iris --D 4 --power 2.0 --frac 0.8 --R 50 \
    --output-dir results

# Generate plots with convergence analysis (if available)
python -m isometry_pursuit.exec.plot_results \
    --results-file results/iris_standardized_1.0_4_results.csv \
    --dataset-name iris \
    --convergence-file results/iris_standardized_1.0_4_convergence.pkl \
    --output-dir figures

# Generate data visualization
python -m isometry_pursuit.exec.generate_data_visualization \
    --output-dir figures --num-vectors 30 --seed 2

# For ethanol data (requires manual download and preprocessing)
python -m isometry_pursuit.exec.load_ethanol_exec \
    --config-path configs/ethanol/ethanol_config.json \
    --raw-data-path data/ethanol.mat \
    --output-dir data
```

## Jupyter Notebooks (Legacy)

The original Jupyter notebooks are still available in the `experiments/` directory:

- `experiments/super_simple_example.ipynb` - Basic example
- `experiments/IrisWine.ipynb` - Iris and Wine dataset experiments
- `experiments/word_embeddings.ipynb` - Word embeddings experiments
- `experiments/Ethanol.ipynb` - Ethanol dataset experiments
- `experiments/Figure1.ipynb` - Figure 1 generation
- `experiments/Figure2.ipynb` - Figure 2 generation

**Note**: The executable scripts provide the same functionality as the notebooks and are recommended for reproducibility.

## Data

The ethanol data is available at `https://drive.google.com/file/d/1CvVOIXokjfTT0AtyeXSZMlojnfL3AcUL/view?usp=drive_link`.

## Output Structure

```
project_root/
├── data/                    # Processed datasets
│   ├── iris_standardized.csv
│   ├── wine_standardized_13.csv
│   ├── word_embeddings_standardized_10.csv
│   └── ethanol_standardized.csv
├── results/                 # Experiment results
│   ├── iris_standardized_1.0_4_results.csv
│   ├── wine_standardized_1.0_13_results.csv
│   ├── word_embeddings_standardized_1.0_10_results.csv
│   └── ethanol_standardized_1.0_2_results.csv
└── figures/                 # Generated plots
    ├── Figure_1b_bw.png
    ├── Figure_1c_bw.png
    ├── iris_isometry_losses_iso_box.png
    ├── wine_isometry_losses_ts_box.png
    └── ethanol_isometry_losses_box.png
```

## Documentation

For detailed usage instructions, see:

- `isometry_pursuit/` - Core package modules
- `scripts/` - Executable shell scripts
