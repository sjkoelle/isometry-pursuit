#!/bin/bash

# Iris Dataset Figure Generation
# This script generates all figures for the iris dataset

set -e  # Exit on any error

# Directories
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
EXEC_DIR="$PROJECT_ROOT/isometry_pursuit/exec"
FIGURES_DIR="$PROJECT_ROOT/figures"
RESULTS_DIR="$PROJECT_ROOT/results"

# Dataset configuration
DATASET_NAME="iris"
D=4
POWER=1.0

echo "=========================================="
echo "Iris Dataset Figure Generation"
echo "=========================================="
echo ""

# Step 1: Generate individual experiment result plots
echo "Generating experiment result plots for iris dataset..."
cd "$EXEC_DIR"

results_file="$RESULTS_DIR/${DATASET_NAME}_standardized_${POWER}_${D}_results.csv"
convergence_file="$RESULTS_DIR/${DATASET_NAME}_standardized_${POWER}_${D}_convergence.pkl"

if [ -f "$results_file" ]; then
    echo "  Generating plots for $DATASET_NAME dataset..."
    python plot_results.py \
        --results-file "$results_file" \
        --dataset-name "$DATASET_NAME" \
        --D "$D" \
        --power "$POWER" \
        --output-dir "$FIGURES_DIR"
    
    if [ -f "$convergence_file" ]; then
        echo "  Generating convergence plots for $DATASET_NAME dataset..."
        python plot_results.py \
            --results-file "$results_file" \
            --dataset-name "$DATASET_NAME" \
            --D "$D" \
            --power "$POWER" \
            --convergence-file "$convergence_file" \
            --output-dir "$FIGURES_DIR"
    fi
    
    echo "  Plots for $DATASET_NAME completed!"
else
    echo "  ERROR: No results file found at $results_file"
    exit 1
fi

echo ""

# List generated iris figures
echo "=========================================="
echo "Generated Iris Figures:"
echo "=========================================="

# List all iris-related PNG files in figures directory
find "$FIGURES_DIR" -name "iris*.png" -type f | sort | while read -r file; do
    # Get relative path from project root
    relative_path="${file#$PROJECT_ROOT/}"
    echo "  - $relative_path"
done

echo ""
echo "=========================================="
echo "Iris Figure Generation Completed!"
echo "=========================================="
echo "Figures saved to: $FIGURES_DIR"
