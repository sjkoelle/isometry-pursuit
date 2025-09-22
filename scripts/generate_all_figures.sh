#!/bin/bash

# Master Figure Generation Script
# This script generates Figure 1, Figure 2, and all dataset-specific plots

set -e  # Exit on any error

# Directories
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
EXEC_DIR="$PROJECT_ROOT/isometry_pursuit/exec"
FIGURES_DIR="$PROJECT_ROOT/figures"
RESULTS_DIR="$PROJECT_ROOT/results"

echo "=========================================="
echo "Master Figure Generation Pipeline"
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

# Step 3: Generate dataset-specific figures
echo "Step 3: Generating dataset-specific figures..."
cd "$SCRIPT_DIR"

# Run each dataset's figure generation script
echo "  Running iris figure generation..."
if [ -f "iris_figures.sh" ]; then
    chmod +x iris_figures.sh
    ./iris_figures.sh
else
    echo "  WARNING: iris_figures.sh not found"
fi

echo "  Running wine figure generation..."
if [ -f "wine_figures.sh" ]; then
    chmod +x wine_figures.sh
    ./wine_figures.sh
else
    echo "  WARNING: wine_figures.sh not found"
fi

echo "  Running word embeddings figure generation..."
if [ -f "word_embeddings_figures.sh" ]; then
    chmod +x word_embeddings_figures.sh
    ./word_embeddings_figures.sh
else
    echo "  WARNING: word_embeddings_figures.sh not found"
fi

echo "  Running ethanol figure generation..."
if [ -f "ethanol_figures.sh" ]; then
    chmod +x ethanol_figures.sh
    ./ethanol_figures.sh
else
    echo "  WARNING: ethanol_figures.sh not found"
fi

echo "Dataset-specific figures completed!"
echo ""

# List all generated figures
echo "=========================================="
echo "All Generated Figures:"
echo "=========================================="

# List all PNG files in figures directory
find "$FIGURES_DIR" -name "*.png" -type f | sort | while read -r file; do
    # Get relative path from project root
    relative_path="${file#$PROJECT_ROOT/}"
    echo "  - $relative_path"
done

echo ""
echo "=========================================="
echo "Master Figure Generation Pipeline Completed!"
echo "=========================================="
echo "Figures saved to: $FIGURES_DIR"
