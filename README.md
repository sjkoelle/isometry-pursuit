# Isometry Pursuit

A Python package for isometry pursuit experiments with both Jupyter notebook and command-line interfaces.

## Installation

```bash
git clone https://github.com/sjkoelle/isometry-pursuit.git
cd isometry-pursuit
pip install -e .
```

## Quick Start

The easiest way to run experiments is using the executable scripts:

```bash
# Run all experiments
scripts/run_all_experiments.st wenh

# Or run individual experiments
scripts/iris_experiment.sh
scripts/wine_experiment.sh
scripts/word_embeddings_experiment.sh
scripts/generate_figures.sh
```

## Individual Executables

You can also run individual components:

```bash
# Load data
python isometry_pursuit/exec/load_iris_exec.py --output-dir data
python isometry_pursuit/exec/load_wine_exec.py --output-dir data --max-features 13

# Run experiments
python isometry_pursuit/exec/run_experiment.py \
    --data-file data/iris_standardized.csv \
    --dataset-name iris \
    --D 4 \
    --power 1.0 \
    --frac 0.5 \
    --R 25 \
    --output-dir results

# Generate plots
python isometry_pursuit/exec/plot_results.py \
    --results-file results/iris_standardized_1.0_4_results.csv \
    --dataset-name iris \
    --D 4 \
    --power 1.0 \
    --output-dir figures
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
│   └── word_embeddings_standardized_10.csv
├── results/                 # Experiment results
│   ├── iris_standardized_1.0_4_results.csv
│   ├── wine_standardized_1.0_13_results.csv
│   └── word_embeddings_standardized_1.0_10_results.csv
└── figures/                 # Generated plots
    ├── Figure_1b_bw.png
    ├── Figure_1c_bw.png
    ├── iris_isometry_losses_iso_box.png
    └── wine_isometry_losses_ts_box.png
```

## Documentation

For detailed usage instructions, see:

- `isometry_pursuit/` - Core package modules
- `scripts/` - Executable shell scripts
