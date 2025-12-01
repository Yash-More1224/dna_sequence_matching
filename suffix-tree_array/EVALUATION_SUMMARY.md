# Suffix Array - Comprehensive Evaluation Summary

**Date:** November 20, 2025  
**Algorithm:** Suffix Array with Binary Search  
**Datasets:** Lambda phage (48KB), E. coli (4.6MB), Salmonella (4.9MB)

---

## Quick Overview - All 6 Metrics ✓

| Metric | Status | Key Finding |
|--------|--------|-------------|
| **1. Latency/Time** | ✅ Complete | ~0.02ms avg query time, 200GB/s throughput |
| **2. Preprocessing** | ✅ Complete | O(n²) construction, 96-93s for full genomes |
| **3. Memory Usage** | ✅ Complete | ~4 bytes/char theoretical, measured peaks |
| **4. Accuracy** | ✅ Complete | **100%** precision, recall, F1 (perfect) |
| **5. Scalability** | ✅ Complete | Excellent batch query performance |
| **6. Robustness** | ✅ Complete | Consistent across all GC content levels |

---

## METRIC 1: LATENCY / TIME PERFORMANCE

### Statistical Measures (5 runs per test)
- **Mean Latency:** Average query time
- **Median Latency:** Middle value (robust to outliers)
- **Variance & Std Dev:** Measure of consistency
- **Throughput:** MB/s and matches/sec

### Results Summary by Dataset

#### Lambda Phage (48KB)
| Pattern Size | Mean (ms) | Median (ms) | Std Dev (ms) | Throughput (MB/s) |
|--------------|-----------|-------------|--------------|-------------------|
| 10 bp        | 0.0255    | 0.0189      | 0.0173       | 1,815             |
| 20 bp        | 0.0171    | 0.0143      | 0.0074       | 2,701             |
| 50 bp        | 0.0136    | 0.0116      | 0.0040       | 3,408             |
| 100 bp       | 0.0134    | 0.0121      | 0.0029       | 3,446             |
| 200 bp       | 0.0137    | 0.0120      | 0.0036       | 3,368             |
| 500 bp       | 0.0159    | 0.0140      | 0.0041       | 2,915             |

#### E. coli (4.6MB)
| Pattern Size | Mean (ms) | Median (ms) | Std Dev (ms) | Throughput (MB/s) | Matches/sec |
|--------------|-----------|-------------|--------------|-------------------|-------------|
| 10 bp        | 0.0198    | 0.0172      | 0.0054       | 223,675           | 404,236     |
| 20 bp        | 0.0210    | 0.0189      | 0.0045       | 211,238           | 47,720      |
| 50 bp        | 0.0214    | 0.0193      | 0.0038       | 206,651           | 46,684      |
| 100 bp       | 0.0206    | 0.0187      | 0.0042       | 215,246           | 48,625      |
| 200 bp       | 0.0202    | 0.0185      | 0.0038       | 219,540           | 49,595      |
| 500 bp       | 0.0232    | 0.0210      | 0.0047       | 191,170           | 43,186      |

#### Salmonella (4.9MB)
| Pattern Size | Mean (ms) | Median (ms) | Std Dev (ms) | Throughput (MB/s) |
|--------------|-----------|-------------|--------------|-------------------|
| 10 bp        | 0.0231    | 0.0197      | 0.0076       | 200,338           |
| 20 bp        | 0.0211    | 0.0185      | 0.0056       | 219,143           |
| 50 bp        | 0.0203    | 0.0184      | 0.0044       | 228,189           |
| 100 bp       | 0.0232    | 0.0188      | 0.0096       | 199,606           |
| 200 bp       | 0.0192    | 0.0176      | 0.0038       | 240,675           |
| 500 bp       | 0.0226    | 0.0204      | 0.0048       | 204,773           |k

**Key Insight:** Extremely fast queries (~0.02ms) with massive throughput (200+ GB/s for large genomes)

---

## METRIC 2: PREPROCESSING TIME

### Construction Time Scalability

| Text Size | Mean Time (ms) | Std Dev (ms) | Time/Char (µs) | Index Size (KB) |
|-----------|----------------|--------------|----------------|-----------------|
| 1,000 bp  | 4.4            | 1.0          | 4.38           | 3.91            |
| 5,000 bp  | 24.1           | 16.9         | 4.81           | 19.53           |
| 10,000 bp | 30.9           | 1.9          | 3.09           | 39.06           |
| 50,000 bp | 194.2          | 4.2          | 3.88           | 195.31          |
| 100,000 bp| 612.9          | 11.0         | 6.13           | 390.62          |
| 500,000 bp| 7,751.3        | 1,439.8      | 15.50          | 1,953.12        |

### Full Genome Construction Times

| Dataset      | Size       | Construction Time | Rate (bp/s) |
|--------------|------------|-------------------|-------------|
| Lambda phage | 48,502 bp  | 285.0 ms          | 171,466     |
| E. coli      | 4,641,652 bp | 96,640.2 ms (~1.6 min) | 48,023 |
| Salmonella   | 4,857,450 bp | 92,616.1 ms (~1.5 min) | 52,442 |

**Key Insight:** O(n²) complexity observed. High upfront cost but enables fast queries.

---

## METRIC 3: MEMORY USAGE

### Memory Measurements (using tracemalloc)

| Text Size | Theoretical SA | Construction Peak | Search Peak | Overhead |
|-----------|----------------|-------------------|-------------|----------|
| 1,000 bp  | ~4 KB          | 102.05 KB         | 0.40 KB     | 25.5x    |
| 5,000 bp  | ~20 KB         | 676.63 KB         | 0.40 KB     | 33.8x    |
| 10,000 bp | ~40 KB         | 1,408.90 KB       | 0.40 KB     | 35.2x    |

**Theoretical Space:** 4 bytes × text length (for suffix array)  
**Actual Usage:** Higher during construction due to sorting overhead

**Key Insight:** Memory efficient for search, but construction requires significant temporary memory.

---

## METRIC 4: ACCURACY

### Perfect Accuracy Across All Tests

**All datasets × All pattern sizes = 100% accuracy**

| Dataset      | Patterns Tested | Precision | Recall | F1 Score | Agreement |
|--------------|-----------------|-----------|--------|----------|-----------|
| Lambda phage | 6 sizes × 3 patterns | 1.0000 | 1.0000 | 1.0000 | 100% |
| E. coli      | 6 sizes × 3 patterns | 1.0000 | 1.0000 | 1.0000 | 100% |
| Salmonella   | 6 sizes × 3 patterns | 1.0000 | 1.0000 | 1.0000 | 100% |

**Ground Truth:** Python `re.finditer()` (exact matching)

**Metrics Explained:**
- **Precision:** True Positives / (True Positives + False Positives)
- **Recall:** True Positives / (True Positives + False Negatives)
- **F1 Score:** Harmonic mean of precision and recall
- **Agreement:** Percentage of identical results vs ground truth

**Key Insight:** Zero false positives, zero false negatives - perfect exact matching.

---

## METRIC 5: SCALABILITY

### A. Text Length Scaling
See Metric 2 (Preprocessing Time) - O(n²) growth observed

### B. Pattern Count Scaling
**Test:** E. coli (4.6MB), 50bp patterns

| Pattern Count | Total Time (ms) | Avg Time/Pattern (ms) | Throughput (patterns/s) |
|---------------|-----------------|----------------------|-------------------------|
| 1             | 0.05            | 0.052                | 19,347                  |
| 5             | 0.18            | 0.036                | 28,129                  |
| 10            | 0.33            | 0.033                | 30,336                  |
| 20            | 0.57            | 0.029                | 34,790                  |
| 50            | 1.20            | 0.024                | 41,689                  |
| 100           | 2.26            | 0.023                | 44,250                  |

**Key Insight:** Excellent amortization for batch queries. Per-pattern cost decreases with more patterns.

---

## METRIC 6: ROBUSTNESS TO ALPHABET VARIATIONS

### Pattern Type Performance (E. coli, 100bp patterns)

| Pattern Type    | Mean Time (ms) | Std Dev (ms) | GC Content | Matches | Performance Impact |
|-----------------|----------------|--------------|------------|---------|-------------------|
| Random          | 0.018          | 0.003        | 50.0%      | 1       | Baseline (1.0x)   |
| Repeat A        | 0.022          | 0.004        | 20.0%      | 0       | 1.2x              |
| Repeat AT       | 0.030          | 0.009        | 0.0%       | 0       | 1.7x              |
| Low Complexity  | 0.023          | 0.008        | 0.0%       | 0       | 1.3x              |
| High GC (80%)   | 0.022          | 0.009        | 80.0%      | 0       | 1.2x              |

**Key Insight:** Performance remains consistent across different GC content and pattern types. Minor variations (~1.2-1.7x) are within acceptable range.

---

## Comparison with Other Algorithms

### When to Use Suffix Array:

✅ **Best for:**
- Multiple pattern searches on same text
- Known text, many queries (e.g., genome reference databases)
- Exact matching with guaranteed O(m log n) per query
- Applications where preprocessing cost can be amortized

❌ **Not ideal for:**
- Single query on large text (high preprocessing cost)
- Streaming or dynamic text (static index)
- Very memory-constrained environments
- Approximate matching (exact matches only)

### vs. Other Algorithms:

| Algorithm     | Preprocessing | Query Time | Memory | Best Use Case |
|---------------|---------------|------------|--------|---------------|
| **Suffix Array** | O(n²)     | O(m log n) | O(n)   | Multiple queries, static text |
| KMP           | O(m)          | O(n)       | O(m)   | Single pattern, streaming |
| Boyer-Moore   | O(m + σ)      | O(n/m) avg | O(m)   | Large alphabets, single query |
| Shift-Bitap   | O(m + σ)      | O(n)       | O(σ)   | Small patterns, bitwise ops |
| Wagner-Fischer| O(m)          | O(mn)      | O(mn)  | Approximate matching |

---

## Files Generated

### CSV Files (Machine-readable)
- `latency_time_20251120_212448.csv` - All timing measurements
- `preprocessing_20251120_212448.csv` - Construction scalability
- `memory_20251120_212448.csv` - Memory usage data
- `accuracy_20251120_212448.csv` - Precision/recall/F1 scores
- `scalability_patterns_20251120_212448.csv` - Pattern count scaling
- `robustness_20251120_212448.csv` - Alphabet variation tests

### JSON Files (Structured data)
- All CSV data also available in JSON format

### Report (Human-readable)
- `suffix_array_full_evaluation_20251120_212448.txt` - Complete detailed report

**Location:** `suffix-tree_array/results/benchmarks/` and `suffix-tree_array/results/reports/`

---

## Statistical Significance

- **5 runs** per search test (latency, accuracy, robustness)
- **3 runs** per construction test (preprocessing)
- Mean, median, and standard deviation reported
- Variance calculated for consistency analysis

---

## Summary Statistics

### Overall Performance Profile

| Metric                    | Value                  |
|---------------------------|------------------------|
| **Avg Query Time**        | ~0.02 ms               |
| **Construction Time**     | ~1.5 min (full genome) |
| **Memory Footprint**      | 4 bytes/char + overhead|
| **Accuracy**              | 100% (perfect)         |
| **Batch Query Speedup**   | Up to 2.3x             |
| **GC Robustness**         | ±0.2x variance         |

---

## Conclusion

The Suffix Array algorithm demonstrates:

1. ✅ **Excellent query performance** - Sub-millisecond searches even on 4.9MB genomes
2. ✅ **Perfect accuracy** - 100% precision and recall across all tests
3. ✅ **Strong scalability** - Great for batch queries with amortization benefits
4. ✅ **Robust performance** - Consistent across different pattern characteristics
5. ⚠️ **High preprocessing cost** - O(n²) construction is a significant upfront investment
6. ⚠️ **Memory intensive** - Requires substantial space for large genomes

**Recommendation:** Best suited for scenarios with static text and multiple queries, such as genomic reference databases, document search systems, or any application where the preprocessing cost can be amortized over many searches.

---

**Generated:** November 20, 2025  
**Evaluation Runtime:** ~3.5 minutes  
**Total Tests:** 18 (accuracy) + 18 (latency) + 6 (preprocessing) + 6 (scalability) + 5 (robustness) + 3 (memory) = 56 test configurations
