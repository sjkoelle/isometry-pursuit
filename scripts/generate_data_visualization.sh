#!/bin/bash

# Data Visualization Generation Script
# This script generates the data.png visualization showing random vectors

set -e  # Exit on any error

# Directories
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
EXEC_DIR="$PROJECT_ROOT/isometry_pursuit/exec"
FIGURES_DIR="$PROJECT_ROOT/figures"

echo "=========================================="
echo "Data Visualization Generation"
echo "=========================================="
echo ""

# Generate data visualization
echo "Generating data visualization..."
cd "$EXEC_DIR"
python generate_data_visualization.py --output-dir "$FIGURES_DIR"

echo ""
echo "=========================================="
echo "Data Visualization Generation Completed!"
echo "=========================================="
echo "Data visualization saved to: $FIGURES_DIR/data.png"
