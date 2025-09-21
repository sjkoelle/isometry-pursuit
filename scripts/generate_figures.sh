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
