#!/bin/bash

# Figure Generation Pipeline
# This script generates Figure 1 and Figure 2 plots

set -e  # Exit on any error

# Directories
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
EXEC_DIR="$PROJECT_ROOT/isometry_pursuit/exec"
FIGURES_DIR="$PROJECT_ROOT/figures"
RESULTS_DIR="$PROJECT_ROOT/results"

echo "=========================================="
echo "Figure Generation Pipeline"
echo "=========================================="
echo ""

# Step 1: Generate Figure 1 (Transformation and Loss Functions)
echo "Step 1: Generating Figure 1 (Transformation and Loss Functions)..."
cd "$EXEC_DIR"
python generate_figure1.py --output-dir "$FIGURES_DIR"
echo "Figure 1 generated successfully!"
echo ""

# Step 2: Generate Figure 2 (Box plots from experiment results)
echo "Step 2: Generating Figure 2 (Box plots from experiment results)..."
python generate_figure2.py --results-dir "$RESULTS_DIR" --output-dir "$FIGURES_DIR"
echo "Figure 2 generated successfully!"
echo ""

# Step 3: Generate individual experiment result plots
echo "Step 3: Generating individual experiment result plots..."

# Function to generate plots for a dataset if results exist
generate_experiment_plots() {
    local dataset_name=$1
    local D=$2
    local power=$3
    
    local results_file="$RESULTS_DIR/${dataset_name}_standardized_${power}_${D}_results.csv"
    local convergence_file="$RESULTS_DIR/${dataset_name}_standardized_${power}_${D}_convergence.pkl"
    
    if [ -f "$results_file" ]; then
        echo "  Generating plots for $dataset_name dataset..."
        python plot_results.py \
            --results-file "$results_file" \
            --dataset-name "$dataset_name" \
            --D "$D" \
            --power "$power" \
            --output-dir "$FIGURES_DIR"
        
        if [ -f "$convergence_file" ]; then
            echo "  Generating convergence plots for $dataset_name dataset..."
            python plot_results.py \
                --results-file "$results_file" \
                --dataset-name "$dataset_name" \
                --D "$D" \
                --power "$power" \
                --convergence-file "$convergence_file" \
                --output-dir "$FIGURES_DIR"
        fi
        
        echo "  Plots for $dataset_name completed!"
    else
        echo "  Skipping $dataset_name - no results file found"
    fi
}

# Generate plots for each dataset
generate_experiment_plots "iris" 4 1.0
generate_experiment_plots "wine" 6 1.0
generate_experiment_plots "word_embeddings" 10 1.0
generate_experiment_plots "ethanol" 2 1.0

echo "Individual experiment plots completed!"
echo ""

# Step 4: Generate word co-occurrence graphs
echo "Step 4: Generating word co-occurrence network graphs..."
co_occurrence_file="$RESULTS_DIR/word_embeddings_standardized_1.0_10_co_occurrence.pkl"
if [ -f "$co_occurrence_file" ]; then
    echo "  Generating word co-occurrence graphs..."
    python generate_word_graphs.py \
        --co-occurrence-file "$co_occurrence_file" \
        --output-dir "$FIGURES_DIR"
    echo "  Word co-occurrence graphs completed!"
else
    echo "  Skipping word co-occurrence graphs - no co-occurrence data found"
    echo "  Expected file: $co_occurrence_file"
fi
echo ""

# List all generated figures
echo "=========================================="
echo "Generated Figures:"
echo "=========================================="

# List all PNG files in figures directory
find "$FIGURES_DIR" -name "*.png" -type f | sort | while read -r file; do
    # Get relative path from project root (macOS compatible)
    relative_path="${file#$PROJECT_ROOT/}"
    echo "  - $relative_path"
done

echo ""
echo "=========================================="
echo "Figure Generation Pipeline Completed!"
echo "=========================================="
echo "Figures saved to: $FIGURES_DIR"
