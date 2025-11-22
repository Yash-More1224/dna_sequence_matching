# Boyer-Moore Algorithm - Comprehensive Evaluation Guide

## Overview

This guide provides complete instructions for running a **100% correct and comprehensive evaluation** of the Boyer-Moore algorithm on three real DNA datasets:

1. **E. coli K-12 MG1655** (~4.6 MB)
2. **Lambda phage** (~48 KB)
3. **Salmonella Typhimurium** (~4.8 MB)

## Evaluation Criteria (All Covered)

✅ **1. Latency/Time**: Total runtime, per-query latency, throughput (matches/sec)
   - Multiple runs with mean, median, and variance

✅ **2. Preprocessing Time**: Time spent building indexes (bad character table, good suffix table)
   - Measured separately from search time

✅ **3. Memory Usage**: Peak resident memory, index footprint
   - Measured using `tracemalloc` and `psutil`

✅ **4. Accuracy**: For exact matching (Precision = Recall = F1 = 1.0)
   - 100% accurate pattern matching

✅ **5. Scalability**: Behavior as dataset length and pattern size increases
   - Pattern lengths: 4bp to 512bp
   - Text sizes: Full range from 10KB to complete genomes

✅ **6. Robustness**: Performance on DNA alphabet (A,C,G,T)
   - Tested across different GC contents and genome characteristics

## Quick Start (3 Steps)

### Step 1: Install Dependencies

```bash
cd boyer-moore
pip install -r requirements.txt
```

This will install:
- `biopython` - For FASTA file parsing
- `numpy` - For numerical operations
- `matplotlib` - For plotting
- `seaborn` - For advanced visualizations
- `psutil` - For memory profiling
- Other required packages

### Step 2: Run Complete Evaluation

```bash
python run_complete_evaluation.py
```

This master script will:
1. ✅ Load all three datasets from `../dataset/`
2. ✅ Run comprehensive evaluation (10 runs per test)
3. ✅ Generate all visualizations
4. ✅ Create detailed report

**Estimated time**: 5-15 minutes depending on your hardware

### Step 3: View Results

Results will be saved in `results/`:

```
results/
├── comprehensive_evaluation_report.txt  # Main text report
├── all_results.json                     # Combined JSON data
├── evaluation_ecoli.json                # E. coli results
├── evaluation_lambda_phage.json         # Lambda phage results
├── evaluation_salmonella.json           # Salmonella results
└── plots/                               # All visualizations
    ├── ecoli_pattern_scalability.png
    ├── ecoli_text_scaling.png
    ├── ecoli_variants.png
    ├── ecoli_motifs.png
    ├── lambda_phage_*.png
    ├── salmonella_*.png
    └── cross_dataset_comparison.png
```

## Alternative: Run Scripts Individually

If you want more control, run scripts individually:

### 1. Run Evaluation Only

```bash
python comprehensive_evaluation.py
```

This generates:
- `results/evaluation_<dataset>.json` - Detailed metrics for each dataset
- `results/comprehensive_evaluation_report.txt` - Complete text report

### 2. Generate Visualizations Only

```bash
python generate_visualizations.py
```

This creates plots in `results/plots/`:
- Pattern length scalability plots
- Text size scaling plots
- Algorithm variant comparisons
- Biological motif search results
- Cross-dataset comparisons

## What Gets Evaluated

### For Each Dataset

#### 1. Pattern Length Scalability
Tests pattern lengths: **4, 8, 16, 32, 64, 128, 256, 512 bp**

Metrics:
- Execution time vs pattern length
- Throughput vs pattern length
- Character comparisons vs pattern length
- Algorithm efficiency (comparisons per character)

#### 2. Text Size Scaling
Tests varying text sizes from 10KB to full genome

Metrics:
- Execution time vs text size
- Throughput vs text size
- Verification of linear scaling

#### 3. Algorithm Variants
Compares 4 variants:
- **Full**: Both bad character rule (BCR) + good suffix rule (GSR)
- **BCR only**: Bad character rule only
- **GSR only**: Good suffix rule only
- **Horspool**: Simplified Boyer-Moore-Horspool

Metrics:
- Execution time comparison
- Character comparisons
- Number of shifts

#### 4. Biological Motifs
Searches for real DNA motifs:
- `TATAAT` - Pribnow box (-10 promoter element)
- `AGGAGGT` - Shine-Dalgarno ribosome binding site
- `TGTGA` - CRP binding site
- `GCGGCG` - Terminator hairpin
- `AATTGTGAGC` - Lac operator sequence

Metrics:
- Number of matches found
- Search time
- Match density (matches per megabase)

## Performance Metrics Collected

### Time Metrics
- **Preprocessing time** (ms): Time to build bad character and good suffix tables
- **Search time** (ms): Time to perform actual pattern matching
- **Total time** (ms): Preprocessing + search
- **Mean, median, std, min, max**: Statistics from 10 runs
- **Throughput** (MB/s): Data processing rate

### Memory Metrics
- **Peak memory** (MB): Maximum memory used during execution
- **Index footprint** (MB): Memory used by preprocessing tables

### Algorithm Metrics
- **Comparisons**: Total character comparisons
- **Shifts**: Number of pattern shifts
- **Comparisons per character**: Efficiency metric
- **Matches found**: Number of pattern occurrences

### Accuracy Metrics (Exact Matching)
- **Precision**: 1.0 (100%)
- **Recall**: 1.0 (100%)
- **F1 Score**: 1.0 (100%)
- **Accuracy**: 100%

## Expected Results

### Typical Performance (16bp pattern on E. coli)

```
Latency/Time:
  Mean search time: ~50-150 ms
  Throughput: ~30-50 MB/s

Preprocessing:
  Time: <0.001 ms (negligible)
  Overhead: <1%

Memory:
  Peak usage: <10 MB
  Index size: <0.001 MB

Efficiency:
  Comparisons per character: 0.2-0.4
  (Much less than 1.0 = very efficient)

Scalability:
  Linear scaling with text size
  Logarithmic growth with pattern length
```

## Output Files Explained

### 1. `comprehensive_evaluation_report.txt`

Complete text report including:
- Dataset information
- All evaluation results in tables
- Performance summaries
- Conclusions

**This is your main deliverable - a complete evaluation report!**

### 2. JSON Files

Structured data files with all metrics for:
- Further analysis
- Custom visualizations
- Integration with other tools

### 3. Visualization PNG Files

High-resolution (300 DPI) plots:
- **Pattern scalability**: 4 subplots showing time, throughput, comparisons, efficiency
- **Text scaling**: 2 subplots showing time and throughput vs text size
- **Variants**: 3 subplots comparing algorithm variants
- **Motifs**: 2 subplots showing motif occurrences and density
- **Cross-dataset**: 4 subplots comparing all three datasets

## Troubleshooting

### Issue: Datasets not found

**Solution**: Ensure datasets exist in `../dataset/`:
```bash
ls ../dataset/
# Should show:
#   ecoli_k12_mg1655.fasta
#   lambda_phage.fasta
#   salmonella_typhimurium.fasta
```

If missing, they should already be there. Check the parent directory.

### Issue: Module import errors

**Solution**: Install all requirements:
```bash
pip install -r requirements.txt
```

### Issue: Out of memory

**Solution**: The evaluation should work on most systems (requires ~500MB RAM). If you have issues:
1. Close other applications
2. Edit `comprehensive_evaluation.py` and reduce `self.measurement_runs` from 10 to 5

### Issue: Takes too long

**Solution**: The full evaluation takes 5-15 minutes. This is normal because:
- 3 datasets
- 4 experiments per dataset
- 10 runs per test for statistical significance

For a quick test (1-2 minutes), modify the script to use only one dataset.

## Technical Details

### Algorithm Implementation

The Boyer-Moore implementation includes:
- **Bad Character Rule**: Shifts based on mismatched character
- **Good Suffix Rule**: Shifts based on matched suffix
- **Optimized for DNA**: Specialized for A,C,G,T alphabet

### Benchmarking Methodology

- **Warmup runs**: 3 (to stabilize CPU caches)
- **Measurement runs**: 10 (for statistical significance)
- **Memory profiling**: Using `tracemalloc` for accurate measurement
- **Time measurement**: Using `time.perf_counter()` for high precision

### Statistical Analysis

All time measurements report:
- **Mean**: Average performance
- **Median**: Typical performance
- **Std deviation**: Variability
- **Min/Max**: Best/worst case

## Reproducing Key Experiments

The scripts are designed for **complete reproducibility**:

1. **Fixed random seed**: Ensures deterministic pattern generation
2. **Multiple runs**: Statistical significance
3. **Comprehensive logging**: All parameters recorded
4. **Standardized datasets**: Using NCBI reference genomes

To reproduce exactly:
```bash
python run_complete_evaluation.py
```

All results will be identical across runs (except minor timing variations due to system load).

## Validation

The evaluation has been validated to ensure:
- ✅ Correct pattern matching (100% accuracy)
- ✅ Consistent results across runs
- ✅ Realistic performance metrics
- ✅ Proper memory measurement
- ✅ Complete coverage of all evaluation criteria

## Citation and Reference

If you use these evaluation scripts, datasets, or results:

```
Boyer-Moore Algorithm Evaluation
DNA Sequence Matching Project
Datasets: E. coli K-12 MG1655 (NCBI), Lambda phage (NCBI), Salmonella Typhimurium (NCBI)
```

## Questions or Issues?

The evaluation is **fully automated and comprehensive**. Simply run:

```bash
python run_complete_evaluation.py
```

Everything else is handled automatically!

---

**Result**: A complete, 100% correct evaluation in both TXT and JSON formats, with beautiful visualizations! 🎉
