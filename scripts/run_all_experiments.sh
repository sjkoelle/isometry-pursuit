#!/bin/bash

# Master Script: Run All Experiments
# This script runs all available experiments in sequence

set -e  # Exit on any error

# Directories
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "=========================================="
echo "Running All Isometry Pursuit Experiments"
echo "=========================================="
echo ""

# Function to run experiment with error handling
run_experiment() {
    local experiment_name="$1"
    local script_path="$2"
    
    echo "=========================================="
    echo "Running: $experiment_name"
    echo "=========================================="
    
    if [ -f "$script_path" ]; then
        bash "$script_path"
        echo "$experiment_name completed successfully!"
    else
        echo "Warning: Script not found: $script_path"
    fi
    
    echo ""
}

# Run all experiments
echo "Starting experiment pipeline..."
echo ""

# 1. Iris experiment
run_experiment "Iris Dataset Experiment" "$SCRIPT_DIR/iris_experiment.sh"

# 2. Wine experiment
run_experiment "Wine Dataset Experiment" "$SCRIPT_DIR/wine_experiment.sh"

# 3. Word embeddings experiment
run_experiment "Word Embeddings Experiment" "$SCRIPT_DIR/word_embeddings_experiment.sh"

# 4. Generate figures
run_experiment "Figure Generation" "$SCRIPT_DIR/generate_figures.sh"

echo "=========================================="
echo "All Experiments Completed Successfully!"
echo "=========================================="
echo ""
echo "Summary of outputs:"
echo "- Data files: $PROJECT_ROOT/data/"
echo "- Results files: $PROJECT_ROOT/results/"
echo "- Figures: $PROJECT_ROOT/figures/"
echo ""
echo "You can now remove the notebook files and rely on these executables!"
