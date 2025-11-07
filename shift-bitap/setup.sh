#!/bin/bash

# Setup script for Shift-Or/Bitap DNA Sequence Matching Project
# =============================================================

echo "=========================================="
echo "Shift-Or/Bitap Project Setup"
echo "=========================================="
echo ""

# Check Python version
echo "Checking Python version..."
python3 --version

if [ $? -ne 0 ]; then
    echo "Error: Python 3 not found. Please install Python 3.8 or higher."
    exit 1
fi

echo ""
echo "Creating virtual environment..."
python3 -m venv venv

echo ""
echo "Activating virtual environment..."
source venv/bin/activate

echo ""
echo "Upgrading pip..."
pip install --upgrade pip

echo ""
echo "Installing dependencies..."
pip install -r requirements.txt

if [ $? -ne 0 ]; then
    echo "Error: Failed to install dependencies."
    exit 1
fi

echo ""
echo "Creating data cache directory..."
mkdir -p data/cache

echo ""
echo "Creating results directory..."
mkdir -p results/plots

echo ""
echo "Running tests..."
python main.py test

if [ $? -ne 0 ]; then
    echo "Warning: Some tests failed. Please review the output."
else
    echo "✓ All tests passed!"
fi

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "To get started:"
echo "  1. Activate the virtual environment: source venv/bin/activate"
echo "  2. Run the demo: python main.py demo"
echo "  3. Run experiments: python main.py experiments --full"
echo "  4. See README.md for more information"
echo ""
echo "Quick commands:"
echo "  python main.py search --pattern GATTACA --text-file genome.fasta"
echo "  python main.py benchmark --pattern ACGT --compare-regex"
echo "  python main.py experiments --pattern-scaling"
echo ""
