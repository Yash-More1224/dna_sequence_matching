#!/usr/bin/env fish
# Quick run script for Wagner-Fischer experiments
# Usage: ./run_experiments.fish

echo "=================================="
echo "Wagner-Fischer Algorithm Experiments"
echo "=================================="
echo ""

# Create necessary directories
echo "Setting up directories..."
mkdir -p data results/benchmarks results/accuracy results/plots

# Check if dependencies are installed
if not test -d venv
    echo "Virtual environment not found. Creating..."
    python -m venv venv
end

echo "Activating virtual environment..."
source venv/bin/activate.fish

echo "Installing dependencies..."
pip install -q -r requirements.txt

echo ""
echo "=================================="
echo "Running Experiments"
echo "=================================="
echo ""

# Run benchmarks
echo "1. Running performance benchmarks..."
python main.py benchmark --full

echo ""
echo "2. Running accuracy evaluation..."
python main.py accuracy --full

echo ""
echo "3. Generating visualizations..."
python main.py visualize \
    --benchmark-csv results/benchmarks/benchmark_results.csv \
    --accuracy-csv results/accuracy/accuracy_results.csv \
    --comparison

echo ""
echo "=================================="
echo "Experiments Complete!"
echo "=================================="
echo ""
echo "Results saved to:"
echo "  - Benchmarks: results/benchmarks/"
echo "  - Accuracy: results/accuracy/"
echo "  - Plots: results/plots/"
echo ""
