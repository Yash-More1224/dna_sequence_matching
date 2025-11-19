# Wagner-Fischer Evaluation Scripts

This directory contains all scripts necessary to run comprehensive Wagner-Fischer algorithm evaluation on DNA sequences.

## Files

### Core Implementation
- **wf.py**: Wagner-Fischer algorithm implementation with multiple variants
  - Full matrix DP
  - Space-optimized (2-row) variant
  - Threshold-based early termination
  - Banded DP variant
  - Traceback for alignment reconstruction

### Evaluation Scripts
- **generate_ground_truth.py**: Generate synthetic patterns with controlled mutations
- **benchmark.py**: Run comprehensive benchmarks measuring performance, memory, and accuracy
- **generate_visualizations.py**: Create plots and text report from metrics
- **run_full_evaluation.py**: Master script to run complete evaluation pipeline

## Quick Start

### Option 1: Run Full Pipeline (Recommended)
```bash
cd /home/shrish-kadam/Documents/SEM3/AAD/dna_sequence_matching/wagner-fischer
python3 scripts/run_full_evaluation.py
```

This will automatically:
1. Generate ground truth from E. coli genome
2. Run all benchmarks (performance, accuracy, scalability, robustness)
3. Generate plots and comprehensive report

### Option 2: Run Steps Individually

#### Step 1: Generate Ground Truth
```bash
python3 scripts/generate_ground_truth.py \
  --fasta ../dataset/ecoli_k12_mg1655.fasta \
  --output results/ground_truth.json \
  --num-patterns 100 \
  --pattern-length 50 \
  --mutation-rates 0.0 0.005 0.01 0.02 0.05 0.1 \
  --seed 42
```

#### Step 2: Run Benchmarks
```bash
python3 scripts/benchmark.py \
  --fasta ../dataset/ecoli_k12_mg1655.fasta \
  --ground-truth results/ground_truth.json \
  --output-dir results \
  --threshold 5 \
  --pattern-counts 10 20 50 100
```

#### Step 3: Generate Visualizations
```bash
python3 scripts/generate_visualizations.py \
  --results-dir results \
  --metrics-file metrics.json
```

## Output Files

After running the evaluation, you'll find:

```
results/
├── ground_truth.json              # Ground truth dataset
├── metrics.json                   # Raw metrics (JSON)
├── wf_evaluation_report.txt       # Comprehensive text report
├── accuracy_vs_mutation_rate.png  # Precision/Recall/F1 plot
├── scalability.png                # Scalability analysis
├── robustness.png                 # Robustness analysis
└── performance_summary.png        # Performance overview
```

## Requirements

```bash
pip install numpy matplotlib psutil
```

## Configuration

Default parameters:
- **Pattern length**: 50 bp
- **Number of patterns**: 100
- **Mutation rates**: 0%, 0.5%, 1%, 2%, 5%, 10%
- **Edit distance threshold**: 5
- **Scalability test sizes**: 10, 20, 50, 100 patterns
- **Number of benchmark runs**: 5

Modify these in `run_full_evaluation.py` or pass as command-line arguments.

## Metrics Measured

### Performance
- Total runtime
- Mean/median/std/min/max latency
- Throughput (operations/second)
- Peak memory usage (tracemalloc and RSS)
- Preprocessing time (0 for WF)

### Accuracy
- Precision, Recall, F1 score
- True positives, False positives, False negatives
- Evaluated at each mutation rate

### Scalability
- Runtime vs pattern count
- Throughput vs pattern count
- Memory usage scaling

### Robustness
- F1 score vs mutation rate
- Latency vs mutation rate
- Performance degradation analysis

## Algorithm Variants

The implementation includes multiple WF variants:

1. **Full Matrix**: Standard O(m×n) time and space
2. **Space-Optimized**: O(m×n) time, O(min(m,n)) space
3. **Threshold-Based**: Early termination using Ukkonen's optimization
4. **Banded**: O(m×k) for sequences with small edit distance

## Notes

- All experiments use a fixed random seed (42) for reproducibility
- Ground truth uses non-overlapping pattern extraction
- Approximate matching uses 50% overlap threshold for TP detection
- Sliding window search accommodates insertions/deletions
