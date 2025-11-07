# Experiment Results Directory

This directory contains all experimental results, benchmarks, and visualizations for the Shift-Or/Bitap algorithm analysis.

## Contents

### Data Files

- **experiment_results.json**: Complete experimental results in JSON format
- **pattern_length_scaling.csv**: Pattern length scaling experiment data
- **text_length_scaling.csv**: Text length scaling experiment data
- **mutation_rates.csv**: Mutation rate accuracy analysis
- **edit_distance_comparison.csv**: Edit distance performance data
- **vs_regex.csv**: Comparison with Python re module
- **motif_search.csv**: Motif finding results

### Visualizations (plots/)

- **latency_comparison.png**: Algorithm latency comparisons
- **throughput_comparison.png**: Throughput analysis
- **memory_comparison.png**: Memory usage comparison
- **scalability_text_length.png**: Text length scaling curves
- **scalability_pattern_length.png**: Pattern length scaling curves
- **accuracy_vs_errors.png**: Accuracy metrics vs edit distance
- **match_highlight_*.png**: Sequence match visualizations
- **density_heatmap_*.png**: Motif density heatmaps

## How to Generate Results

### Run Full Benchmark Suite

```bash
cd ..
python main.py experiments --full
```

This will:
1. Run all 6 experiments
2. Save results to JSON and CSV files
3. Generate visualization plots
4. Create summary statistics

### Run Individual Experiments

```bash
# Pattern length scaling
python main.py experiments --pattern-scaling

# Text length scaling
python main.py experiments --text-scaling

# Mutation rate analysis
python main.py experiments --mutation-rates

# Compare with regex
python main.py experiments --vs-regex
```

### Generate Specific Visualizations

```python
from visualization import PerformanceVisualizer, SequenceVisualizer
from benchmark import Benchmarker
from algorithm import ShiftOrBitap

# Example: Create latency comparison
benchmarker = Benchmarker()
pattern = "GATTACA"
text = "..." # your text here

results = benchmarker.compare_algorithms(pattern, text, num_runs=10)
PerformanceVisualizer.plot_latency_comparison(results, "plots/my_comparison.png")
```

## Interpreting Results

### Performance Metrics

**Latency (ms)**: Lower is better
- Preprocessing time: Time to build pattern masks
- Search time: Time to scan through text
- Total time: Preprocessing + search

**Throughput (MB/s)**: Higher is better
- Characters processed per second
- Indicates raw scanning speed

**Memory (KB)**: Lower is better
- Peak memory usage during search
- Includes pattern masks and state vectors

### Accuracy Metrics

**Precision**: TP / (TP + FP)
- Proportion of found matches that are correct
- High precision = few false positives

**Recall**: TP / (TP + FN)
- Proportion of actual matches that were found
- High recall = few false negatives

**F1 Score**: Harmonic mean of precision and recall
- Balanced accuracy measure
- Closer to 1.0 is better

### Scalability Analysis

**Linear Scaling**: Performance grows proportionally with input size
- Expected for the Shift-Or/Bitap algorithm
- O(n) time complexity

**Constant Throughput**: Throughput remains stable across input sizes
- Indicates good scalability
- No performance degradation

## Expected Results

Based on algorithm analysis, you should observe:

1. **Pattern Length Scaling**:
   - Near-constant performance for patterns ≤ 64 bp
   - Slight increase in preprocessing time with length
   - Search time relatively stable

2. **Text Length Scaling**:
   - Linear increase in search time
   - Constant throughput (MB/s)
   - Minimal memory increase

3. **Mutation Rates**:
   - High precision/recall at low mutation rates (≤10%)
   - Gradual decline at higher rates
   - Trade-off between sensitivity and specificity

4. **Edit Distance**:
   - Linear performance degradation with k
   - k=0 (exact) fastest
   - k=3 approximately 3-4x slower

5. **vs Python re**:
   - Competitive for short patterns (≤20 bp)
   - Python re may be faster for longer patterns
   - Shift-Or/Bitap advantages: approximate matching, DNA optimization

## Notes

- All experiments use random seed 42 for reproducibility
- Default number of runs: 5-10 (configurable)
- Text lengths: 1,000 to 1,000,000+ characters
- Pattern lengths: 5 to 50 base pairs

## Troubleshooting

**Missing results files**:
- Run experiments to generate them
- Check that output directory is writable

**Incomplete visualizations**:
- Ensure matplotlib and seaborn are installed
- Check for errors in experiment output

**Inconsistent results**:
- System load can affect timing measurements
- Run more iterations for statistical significance
- Use dedicated benchmarking environment

## Data Format

### JSON Structure

```json
{
  "metadata": {
    "timestamp": "2025-11-07T...",
    "algorithm": "Shift-Or/Bitap"
  },
  "experiments": [
    {
      "name": "experiment_name",
      "description": "...",
      "results": [
        {
          "metric1": value1,
          "metric2": value2,
          ...
        }
      ]
    }
  ]
}
```

### CSV Structure

Each CSV file contains experiment-specific columns:
- Common: pattern_length, text_length, num_runs
- Timing: search_time_ms, preprocessing_time_ms
- Performance: throughput_mbps, memory_kb
- Accuracy: precision, recall, f1_score

---

**Last Updated**: Auto-generated on experiment run  
**Format Version**: 1.0
