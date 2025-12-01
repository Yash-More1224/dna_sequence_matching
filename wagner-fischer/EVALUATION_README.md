# Wagner-Fischer Algorithm - Comprehensive Evaluation

This directory contains a comprehensive evaluation suite for the Wagner-Fischer algorithm, following the same evaluation criteria used for KMP and other algorithms.

## Overview

The Wagner-Fischer algorithm is evaluated across **6 key criteria**:

1. **Latency/Time** - Runtime performance and throughput
2. **Preprocessing** - Matrix initialization overhead
3. **Memory Usage** - Peak memory consumption (space-optimized)
4. **Accuracy** - Correctness validation
5. **Scalability** - Performance with varying input sizes
6. **Robustness** - Behavior across different edit distance thresholds

## Quick Start

### 1. Run Comprehensive Evaluation

```bash
python comprehensive_evaluation_full.py
```

This will:
- Test on all datasets (E. coli, Lambda Phage, Salmonella)
- Evaluate different pattern lengths (10, 20, 50, 100, 200 bp)
- Test various edit distance thresholds (0, 1, 2, 3, 5)
- Run multiple iterations for statistical reliability
- Save results to `results/benchmarks/`

**Note**: This may take 10-30 minutes depending on your system.

### 2. Generate Visualizations

```bash
python generate_visualizations.py
```

Creates plots in `results/plots/`:
- Latency vs edit distance
- Latency vs pattern length
- Preprocessing time
- Memory usage
- Accuracy metrics
- Scalability charts
- Robustness analysis
- Summary dashboard

### 3. Generate Report

```bash
python regenerate_report.py
```

Creates a comprehensive text report in `results/reports/` with:
- Executive summary
- Detailed performance tables
- Key findings for each criterion
- Comparison with exact matching algorithms

## Files Structure

```
wagner-fischer/
├── comprehensive_evaluation_full.py  # Main evaluation script
├── generate_visualizations.py        # Plot generation
├── regenerate_report.py              # Report generation
├── wf_core.py                         # Core algorithm
├── wf_search.py                       # Search implementation
├── data_loader.py                     # Dataset loading
└── results/
    ├── benchmarks/                    # Raw results (CSV & JSON)
    ├── plots/                         # Generated visualizations
    └── reports/                       # Comprehensive reports
```

## Evaluation Criteria Details

### 1. Latency/Time
- **Metrics**: Mean time, median time, standard deviation, throughput
- **Tests**: Various pattern lengths × edit distance thresholds × datasets
- **Expected**: O(n*m) time complexity

### 2. Preprocessing
- **Metrics**: Matrix initialization time, memory allocation
- **Tests**: Different pattern lengths
- **Expected**: O(m) preprocessing with space optimization

### 3. Memory Usage
- **Metrics**: Matrix memory, peak memory during operations
- **Tests**: Various pattern lengths
- **Expected**: O(m) space (2 rows instead of full n×m matrix)

### 4. Accuracy
- **Metrics**: Precision, recall, F1 score vs Python regex
- **Tests**: Exact matching (max_edit_distance=0)
- **Note**: Approximate matching validation requires different ground truth

### 5. Scalability
- **5A. Text Length**: Tests on increasing text sizes
- **5B. Pattern Count**: Tests with multiple patterns
- **Expected**: Linear scaling with each dimension

### 6. Robustness
- **Tests**: Different edit distance thresholds (0 to 5)
- **Metrics**: Performance consistency, match counts
- **Expected**: More matches with higher thresholds

## Key Differences from KMP

| Aspect | Wagner-Fischer | KMP |
|--------|----------------|-----|
| **Time Complexity** | O(n*m) | O(n+m) |
| **Space Complexity** | O(m) optimized | O(m) |
| **Matching Type** | Approximate | Exact |
| **Performance** | Slower | Faster |
| **Use Case** | Error-tolerant | Exact patterns |
| **Edit Distance** | Configurable | N/A |

## Performance Expectations

### Typical Results (E. coli ~4.6MB genome)

- **Throughput**: ~0.5-2 MB/s (decreases with edit distance)
- **Pattern 50bp, max_dist=2**: ~2-5 seconds
- **Memory**: < 10 KB (space-optimized)
- **Accuracy**: 100% for exact matching

### Comparison with Other Algorithms

Wagner-Fischer is **slower** than exact matching algorithms but provides:
- ✓ Approximate matching capability
- ✓ Error tolerance for sequencing errors
- ✓ Flexible edit distance threshold
- ✓ Handles insertions, deletions, substitutions

## Dependencies

```bash
pip install -r requirements.txt
```

Required packages:
- pandas
- matplotlib
- seaborn
- numpy

## Dataset Requirements

Datasets should be in `../dataset/`:
- `ecoli_k12_mg1655.fasta`
- `lambda_phage.fasta`
- `salmonella_typhimurium.fasta`

Download if needed:
```bash
cd ../dataset
python download_datasets.py
```

## Output Files

### Benchmarks (CSV & JSON)
- `latency_time_YYYYMMDD_HHMMSS.{csv,json}`
- `preprocessing_YYYYMMDD_HHMMSS.{csv,json}`
- `memory_YYYYMMDD_HHMMSS.{csv,json}`
- `accuracy_YYYYMMDD_HHMMSS.{csv,json}`
- `scalability_text_YYYYMMDD_HHMMSS.{csv,json}`
- `scalability_patterns_YYYYMMDD_HHMMSS.{csv,json}`
- `robustness_YYYYMMDD_HHMMSS.{csv,json}`

### Plots
- `latency_vs_edit_distance.png`
- `latency_vs_pattern_length.png`
- `preprocessing_time.png`
- `memory_usage.png`
- `accuracy_metrics.png`
- `scalability_text_length.png`
- `scalability_multiple_patterns.png`
- `robustness_edit_distances.png`
- `summary_dashboard.png`

### Reports
- `comprehensive_evaluation_YYYYMMDD_HHMMSS.txt`

## Customization

Edit parameters in `comprehensive_evaluation_full.py`:

```python
PATTERN_LENGTHS = [10, 20, 50, 100, 200]
EDIT_DISTANCES = [0, 1, 2, 3, 5]
NUM_RUNS = 10
TEXT_SCALE_SIZES = [1000, 5000, 10000, 50000, 100000, 500000]
PATTERN_COUNTS = [1, 5, 10, 20, 50]
```

## Troubleshooting

### "No datasets found"
```bash
cd ../dataset
python download_datasets.py
```

### "No results found"
Run the evaluation first:
```bash
python comprehensive_evaluation_full.py
```

### Memory issues
Reduce test sizes in the configuration:
- Decrease `TEXT_SCALE_SIZES`
- Reduce `PATTERN_LENGTHS`
- Lower `NUM_RUNS`

## Algorithm Use Cases

Wagner-Fischer is ideal for:
- **Error-tolerant pattern matching** in DNA sequences
- **Sequence alignment** with mismatches
- **Handling sequencing errors** (insertions, deletions, substitutions)
- **Finding similar sequences** (not just exact matches)
- **Mutation detection** with configurable thresholds

Not ideal for:
- Exact pattern matching (use KMP or Boyer-Moore instead)
- Real-time applications requiring high throughput
- Very long patterns or genomes (consider heuristic methods)

## Citation

Wagner, R. A., & Fischer, M. J. (1974). The string-to-string correction problem. Journal of the ACM (JACM), 21(1), 168-173.

## License

See project root for license information.
