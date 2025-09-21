#!/bin/bash

# Iris Dataset Experiment Pipeline
# This script orchestrates: data load → run experiment → plot results

set -e  # Exit on any error

# Configuration
DATASET_NAME="iris"
D=4  # Iris has 4 features
POWER=1.0
FRAC=0.5
R=25
COMPUTE_BRUTE=false
LOG_CONVERGENCE=false

# Directories
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
EXEC_DIR="$PROJECT_ROOT/isometry_pursuit/exec"
DATA_DIR="$PROJECT_ROOT/data"
RESULTS_DIR="$PROJECT_ROOT/results"
FIGURES_DIR="$PROJECT_ROOT/figures"

echo "=========================================="
echo "Iris Dataset Experiment Pipeline"
echo "=========================================="
echo "Dataset: $DATASET_NAME"
echo "Target dimension (D): $D"
echo "Power: $POWER"
echo "Fraction: $FRAC"
echo "Resampling runs: $R"
echo "Compute brute force: $COMPUTE_BRUTE"
echo "Log convergence: $LOG_CONVERGENCE"
echo ""

# Step 1: Load and preprocess data
echo "Step 1: Loading and preprocessing Iris data..."
cd "$EXEC_DIR"
python load_iris_exec.py --output-dir "$DATA_DIR"

# Get the data file path
DATA_FILE="$DATA_DIR/iris_standardized.csv"
if [ ! -f "$DATA_FILE" ]; then
    echo "Error: Data file not found at $DATA_FILE"
    exit 1
fi
echo "Data loaded successfully: $DATA_FILE"
echo ""

# Step 2: Run experiment
echo "Step 2: Running isometry pursuit experiment..."
python run_experiment.py \
    --data-file "$DATA_FILE" \
    --dataset-name "$DATASET_NAME" \
    --D "$D" \
    --power "$POWER" \
    --frac "$FRAC" \
    --R "$R" \
    --output-dir "$RESULTS_DIR"

if [ "$COMPUTE_BRUTE" = true ]; then
    python run_experiment.py \
        --data-file "$DATA_FILE" \
        --dataset-name "$DATASET_NAME" \
        --D "$D" \
        --power "$POWER" \
        --frac "$FRAC" \
        --R "$R" \
        --compute-brute \
        --output-dir "$RESULTS_DIR"
fi

if [ "$LOG_CONVERGENCE" = true ]; then
    python run_experiment.py \
        --data-file "$DATA_FILE" \
        --dataset-name "$DATASET_NAME" \
        --D "$D" \
        --power "$POWER" \
        --frac "$FRAC" \
        --R "$R" \
        --log-convergence \
        --output-dir "$RESULTS_DIR"
fi

echo "Experiment completed successfully!"
echo ""

# Step 3: Generate plots
echo "Step 3: Generating plots..."
RESULTS_FILE="$RESULTS_DIR/${DATASET_NAME}_standardized_${POWER}_${D}_results.csv"
CONVERGENCE_FILE="$RESULTS_DIR/${DATASET_NAME}_standardized_${POWER}_${D}_convergence.pkl"

if [ -f "$RESULTS_FILE" ]; then
    python plot_results.py \
        --results-file "$RESULTS_FILE" \
        --dataset-name "$DATASET_NAME" \
        --D "$D" \
        --power "$POWER" \
        --output-dir "$FIGURES_DIR"
    
    if [ -f "$CONVERGENCE_FILE" ]; then
        python plot_results.py \
            --results-file "$RESULTS_FILE" \
            --dataset-name "$DATASET_NAME" \
            --D "$D" \
            --power "$POWER" \
            --convergence-file "$CONVERGENCE_FILE" \
            --output-dir "$FIGURES_DIR"
    fi
    
    echo "Plots generated successfully!"
else
    echo "Warning: Results file not found at $RESULTS_FILE"
fi

echo ""
echo "=========================================="
echo "Iris Experiment Pipeline Completed!"
echo "=========================================="
echo "Results saved to: $RESULTS_DIR"
echo "Plots saved to: $FIGURES_DIR"
echo "Data saved to: $DATA_DIR"
