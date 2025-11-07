# Wagner-Fischer Algorithm for DNA Sequence Matching

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

A comprehensive implementation of the Wagner-Fischer algorithm (Levenshtein edit distance) for approximate DNA sequence matching. This project is part of the AAD (Advanced Algorithm Design) course, focusing on exact and approximate string matching in biological sequences.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Algorithm Details](#algorithm-details)
- [Usage Examples](#usage-examples)
- [Benchmarking](#benchmarking)
- [Project Structure](#project-structure)
- [Results](#results)
- [Contributing](#contributing)

## 🔬 Overview

The Wagner-Fischer algorithm computes the **Levenshtein edit distance** between two sequences using dynamic programming. This implementation is optimized for DNA sequence analysis and includes:

- ✅ Standard DP implementation with full matrix
- ✅ Space-optimized version (O(min(m,n)) space)
- ✅ Traceback for alignment reconstruction
- ✅ Sliding window pattern search
- ✅ Threshold-based early termination (Ukkonen's optimization)
- ✅ Comprehensive benchmarking suite
- ✅ Accuracy evaluation on synthetic and real data
- ✅ Visualization of results

## ✨ Features

### Core Algorithm
- **Edit distance computation**: Compute Levenshtein distance between sequences
- **Configurable costs**: Custom costs for substitutions, insertions, and deletions
- **Space optimization**: Memory-efficient variant using only two rows
- **Alignment traceback**: Reconstruct the sequence of edit operations

### Pattern Matching
- **Approximate search**: Find all matches within edit distance threshold
- **Multiple patterns**: Batch search for multiple patterns
- **Exact matching**: Optimized exact string matching
- **Motif finding**: Search with similarity thresholds

### Performance & Evaluation
- **Comprehensive benchmarks**: Latency, memory, throughput measurements
- **Accuracy metrics**: Precision, recall, F1 score
- **Comparison with regex**: Benchmark against Python's `re` module
- **Scalability tests**: Performance across varying sequence and pattern lengths

### Visualization
- **Performance plots**: Time vs pattern/text length
- **Memory usage graphs**: Peak memory consumption
- **Accuracy charts**: Precision/recall curves
- **Alignment visualization**: Visual representation of edit operations
- **Match heatmaps**: Density plots for match distribution

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup

1. **Clone the repository** (or navigate to the wagner-fischer directory):
```bash
cd wagner-fischer
```

2. **Create a virtual environment** (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Linux/Mac
# or
venv\Scripts\activate  # On Windows
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

### Dependencies
```
biopython>=1.83
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
pandas>=2.0.0
pyyaml>=6.0
memory_profiler>=0.61.0
psutil>=5.9.0
pytest>=7.4.0
```

## 🎯 Quick Start

### 1. Compute Edit Distance
```bash
python main.py distance ATCGATCG ATCGTTCG
# Output: Edit Distance: 2
```

### 2. Search for Pattern
```bash
python main.py search ATCG --text "GGATCGGGATCGAAA" --max-distance 2
# Output: Found 2 matches at positions...
```

### 3. Run Full Benchmark Suite
```bash
python main.py benchmark --full
```

### 4. Run Accuracy Evaluation
```bash
python main.py accuracy --full
```

### 5. Generate Visualizations
```bash
python main.py visualize --benchmark-csv results/benchmarks/benchmark_results.csv
```

## 📊 Algorithm Details

### Wagner-Fischer Dynamic Programming

The algorithm builds a matrix `dp[i][j]` where each cell represents the minimum edit distance between the first `i` characters of the source and the first `j` characters of the target.

**Recurrence relation:**
```
dp[i][j] = min(
    dp[i-1][j-1] + cost(substitute),  # Substitution
    dp[i-1][j] + cost(delete),        # Deletion
    dp[i][j-1] + cost(insert)         # Insertion
)
```

**Complexity:**
- **Time**: O(m × n) where m, n are sequence lengths
- **Space**: O(m × n) for full matrix, O(min(m,n)) for optimized version

### Optimizations Implemented

1. **Space Optimization**: Use only two rows instead of full matrix
2. **Ukkonen's Cutoff**: Early termination when distance exceeds threshold
3. **Diagonal Banding**: Restrict computation to k-diagonal band for approximate matching

## 💡 Usage Examples

### Python API

#### Basic Edit Distance
```python
from wf_core import WagnerFischer, levenshtein_distance

# Simple function
distance = levenshtein_distance("ATCG", "ATCG")
print(f"Distance: {distance}")  # 0

# With custom costs
wf = WagnerFischer(substitution_cost=2, insertion_cost=1, deletion_cost=1)
distance, _ = wf.compute_distance("ATCG", "TTCG")
print(f"Distance: {distance}")  # 2
```

#### Pattern Search
```python
from wf_search import PatternSearcher

searcher = PatternSearcher(max_distance=2)
matches = searcher.search("ATCG", "GGATCGGGATCGAAA")

for match in matches:
    print(f"Position: {match.position}, Distance: {match.edit_distance}")
```

#### With Alignment
```python
wf = WagnerFischer()
distance, operations = wf.compute_with_traceback("ATCG", "TTCG")

print(f"Distance: {distance}")
for op in operations:
    print(f"  {op}")
```

#### Loading FASTA Files
```python
from data_loader import FastaLoader

loader = FastaLoader()
sequences = loader.load("data/ecoli_k12.fna.gz")

for seq in sequences:
    print(f"{seq.id}: {len(seq.sequence)} bp")
```

### Command Line Interface

#### Compute Edit Distance with Alignment
```bash
python main.py distance ATCGATCG ATCGTTCG --show-alignment
```

#### Search in FASTA File
```bash
python main.py search TATAAT --text-file data/ecoli_k12.fna.gz --max-distance 2
```

#### Custom Benchmark
```bash
python main.py benchmark \
    --test-edit-distance \
    --pattern-lengths 10 20 50 100 \
    --text-length 10000 \
    --iterations 20
```

#### Download E. coli Genome
```bash
python main.py data --download-ecoli --data-dir data
```

#### Generate Synthetic Test Data
```bash
python main.py data --generate-synthetic --data-dir data
```

## 🧪 Benchmarking

The benchmarking suite evaluates:

### Performance Metrics
- **Latency**: Mean, median, std, min, max execution time
- **Memory**: Peak and current memory usage
- **Throughput**: Characters processed per second
- **Preprocessing time**: Time for algorithm setup

### Test Categories

1. **Edit Distance Scaling**: Performance vs pattern length
2. **Text Length Scaling**: Performance vs sequence length  
3. **Threshold Scaling**: Effect of edit distance threshold
4. **Regex Comparison**: Wagner-Fischer vs Python's `re` module

### Running Benchmarks

```bash
# Full suite
python main.py benchmark --full

# Individual tests
python main.py benchmark --test-edit-distance --pattern-lengths 10 50 100
python main.py benchmark --test-search --text-lengths 1000 10000 100000
python main.py benchmark --test-threshold --thresholds 0 1 2 5 10
python main.py benchmark --test-regex
```

### Results Location
- CSV: `results/benchmarks/benchmark_results.csv`
- JSON: `results/benchmarks/benchmark_results.json`
- Plots: `results/plots/`

## 📈 Accuracy Evaluation

### Metrics Computed
- **Precision**: TP / (TP + FP)
- **Recall**: TP / (TP + FN)
- **F1 Score**: 2 × (Precision × Recall) / (Precision + Recall)
- **Confusion Matrix**: TP, FP, TN, FN

### Test Types

1. **Exact Matching**: Verify 100% accuracy for exact matches
2. **Synthetic Mutations**: Test on sequences with controlled mutation rates
3. **Threshold Sensitivity**: Accuracy vs edit distance threshold

### Running Accuracy Tests

```bash
# Full evaluation
python main.py accuracy --full

# Custom tests
python main.py accuracy --test-exact --pattern-lengths 10 20 50
python main.py accuracy --test-mutations --mutation-rate 0.02 --max-distance 3
```

## 📁 Project Structure

```
wagner-fischer/
├── main.py                    # Main CLI entry point
├── wf_core.py                 # Core Wagner-Fischer implementation
├── wf_search.py               # Pattern search functionality
├── data_loader.py             # FASTA/FASTQ loading and data generation
├── benchmark.py               # Performance benchmarking
├── accuracy.py                # Accuracy evaluation
├── visualization.py           # Plotting and visualization
├── config.yaml                # Configuration file
├── requirements.txt           # Python dependencies
├── README.md                  # This file
│
├── tests/                     # Unit and integration tests
│   ├── __init__.py
│   ├── test_wf_core.py
│   ├── test_search.py
│   └── test_integration.py
│
├── data/                      # Datasets (gitignored)
│   ├── ecoli_k12.fna.gz
│   ├── synthetic_small.fasta
│   └── synthetic_medium.fasta
│
└── results/                   # Results and plots (gitignored)
    ├── benchmarks/
    │   ├── benchmark_results.csv
    │   └── benchmark_results.json
    ├── accuracy/
    │   ├── accuracy_results.csv
    │   └── confusion_matrix.csv
    └── plots/
        ├── benchmark_summary.png
        ├── accuracy_summary.png
        └── wf_vs_regex.png
```

## 🧪 Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_wf_core.py -v

# Run with coverage
pytest tests/ --cov=. --cov-report=html
```

## 📊 Results

Results from benchmarking and accuracy evaluation are automatically generated in the `results/` directory:

### Benchmark Results
- **CSV files**: Detailed metrics for all tests
- **Plots**: Visual summaries of performance

### Accuracy Results
- **Precision/Recall**: Accuracy metrics across thresholds
- **Confusion matrices**: Classification performance
- **F1 scores**: Overall accuracy

### Visualizations
All plots are saved as high-resolution PNG files (300 DPI) in `results/plots/`.

## 🎨 Configuration

Edit `config.yaml` to customize:

```yaml
algorithm:
  max_distance: 2
  substitution_cost: 1
  insertion_cost: 1
  deletion_cost: 1

benchmark:
  pattern_lengths: [10, 20, 50, 100]
  iterations: 10
  
visualization:
  dpi: 300
  figure_format: "png"
```

## 🤝 Contributing

This is an academic project for the AAD course. Contributions, suggestions, and feedback are welcome!

## 📝 License

This project is developed for academic purposes as part of the AAD course (SEM3).

## 👥 Authors

AAD Project Team - Semester 3, 2025

## 🙏 Acknowledgments

- E. coli genome data from NCBI
- Wagner-Fischer algorithm based on the classic dynamic programming approach
- Inspired by bioinformatics sequence alignment tools

## 📚 References

1. Wagner, R. A., & Fischer, M. J. (1974). "The String-to-String Correction Problem". Journal of the ACM.
2. Levenshtein, V. I. (1966). "Binary codes capable of correcting deletions, insertions, and reversals".
3. Ukkonen, E. (1985). "Algorithms for approximate string matching".

---

For questions or issues, please refer to the course documentation or contact the project team.
