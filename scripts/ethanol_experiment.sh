#!/bin/bash

# Ethanol Dataset Experiment Pipeline
# This script orchestrates: data load → run experiment → plot results

set -e  # Exit on any error

# Configuration
DATASET_NAME="ethanol"
D=2  # Ethanol has target dimension 2
POWER=1.0
R=5  # Ethanol uses 5 resampling runs
COMPUTE_BRUTE=false
LOG_CONVERGENCE=false

# Data file paths (using the paths from Ethanol.ipynb)
CONFIG_PATH="/Users/samsonkoelle/montlake/experiments/configs/ethanol_full.json"
RAW_DATA_PATH="/Users/samsonkoelle/thesis_data/raw_data/ethanol.mat"
POSITIONS_PATH="/Users/samsonkoelle/positions.npy"
INDICES_PATH="/Users/samsonkoelle/indices.npy"
ANGLES_PATH="/Users/samsonkoelle/ethanolanglesfull_0524.npy"

# Directories
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
EXEC_DIR="$PROJECT_ROOT/isometry_pursuit/exec"
DATA_DIR="$PROJECT_ROOT/data"
RESULTS_DIR="$PROJECT_ROOT/results"
FIGURES_DIR="$PROJECT_ROOT/figures"

echo "=========================================="
echo "Ethanol Dataset Experiment Pipeline"
echo "=========================================="
echo "Dataset: $DATASET_NAME"
echo "Target dimension (D): $D"
echo "Power: $POWER"
echo "Number of data points: $R"
echo "Compute brute force: $COMPUTE_BRUTE"
echo "Log convergence: $LOG_CONVERGENCE"
echo ""

# Check if data files exist
if [ ! -f "$CONFIG_PATH" ]; then
    echo "Error: Configuration file not found at $CONFIG_PATH"
    echo "Please ensure the montlake repository is available and the config file exists."
    echo ""
fi

if [ ! -f "$RAW_DATA_PATH" ]; then
    echo "Error: Raw data file not found at $RAW_DATA_PATH"
    echo "Please ensure the ethanol.mat file is available."
    echo ""
fi

if [ ! -f "$POSITIONS_PATH" ]; then
    echo "Error: Positions file not found at $POSITIONS_PATH"
    echo "Please ensure the positions.npy file is available."
    echo ""
fi

if [ ! -f "$INDICES_PATH" ]; then
    echo "Error: Indices file not found at $INDICES_PATH"
    echo "Please ensure the indices.npy file is available."
    echo ""
fi

if [ ! -f "$ANGLES_PATH" ]; then
    echo "Error: Angles file not found at $ANGLES_PATH"
    echo "Please ensure the ethanolanglesfull_0524.npy file is available."
    echo ""
fi

# Step 1: Load and preprocess data
echo "Step 1: Loading and preprocessing Ethanol data..."
cd "$EXEC_DIR"
python load_ethanol_exec.py \
    --config-path "$CONFIG_PATH" \
    --raw-data-path "$RAW_DATA_PATH" \
    --positions-path "$POSITIONS_PATH" \
    --indices-path "$INDICES_PATH" \
    --angles-path "$ANGLES_PATH" \
    --output-dir "$DATA_DIR"

# Get the data file path
DATA_FILE="$DATA_DIR/ethanol_standardized.csv"
if [ ! -f "$DATA_FILE" ]; then
    echo "Error: Data file not found at $DATA_FILE"
fi
echo "Data loaded successfully: $DATA_FILE"
echo ""

# Step 2: Run experiment
echo "Step 2: Running isometry pursuit experiment..."
python ethanol_experiment.py \
    --data-file "$DATA_FILE" \
    --D "$D" \
    --power "$POWER" \
    --output-dir "$RESULTS_DIR"

if [ "$COMPUTE_BRUTE" = true ]; then
    python ethanol_experiment.py \
        --data-file "$DATA_FILE" \
        --D "$D" \
        --power "$POWER" \
        --compute-brute \
        --output-dir "$RESULTS_DIR"
fi

if [ "$LOG_CONVERGENCE" = true ]; then
    python ethanol_experiment.py \
        --data-file "$DATA_FILE" \
        --D "$D" \
        --power "$POWER" \
        --log-convergence \
        --output-dir "$RESULTS_DIR"
fi

echo "Experiment completed successfully!"
echo ""

echo "Experiment results saved to: $RESULTS_DIR"
echo "To generate plots, run: scripts/generate_figures.sh"

echo ""
echo "=========================================="
echo "Ethanol Experiment Pipeline Completed!"
echo "=========================================="
echo "Results saved to: $RESULTS_DIR"
echo "Plots saved to: $FIGURES_DIR"
echo "Data saved to: $DATA_DIR"
echo ""
echo "Note: The ethanol experiment uses individual data points as replicas,"
echo "unlike other experiments that use resampling across features."
