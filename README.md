# DNA Sequence Pattern Matching: Comparative Analysis

A comprehensive implementation and evaluation of five distinct pattern matching algorithms for DNA sequence analysis.

## Project Overview

This project implements and experimentally compares classical exact matching algorithms (KMP, Boyer-Moore, Suffix Array) and approximate matching approaches (Shift-Or/Bitap, Wagner-Fischer) on real genomic datasets.

## Algorithms Implemented

| Algorithm | Type | Best Use Case |
|-----------|------|---------------|
| **Knuth-Morris-Pratt (KMP)** | Exact matching | Single queries, streaming data |
| **Boyer-Moore** | Exact matching | Long patterns, large alphabets |
| **Suffix Array** | Indexed search | Multiple queries on static genomes |
| **Shift-Or/Bitap** | Exact + Approximate | Short patterns, fuzzy matching |
| **Wagner-Fischer** | Approximate | Edit distance, sequence alignment |

## Datasets

- **E. coli K-12 MG1655**: 4.64 million base pairs
- **Lambda Phage**: 48.5 thousand base pairs
- **Salmonella typhimurium**: 4.86 million base pairs

## Key Results

- **Fastest Query**: Suffix Array (0.02 ms after indexing)
- **Most Consistent**: KMP (guaranteed O(n+m) performance)
- **Best for Mutations**: Shift-Or/Bitap (native k-error support)
- **Memory Efficient**: Wagner-Fischer / KMP (12-78 KB)

## Quick Start

### Download Datasets
```bash
python download_datasets.py
```

### Run Individual Algorithms

**KMP:**
```bash
cd kmp
python comprehensive_evaluation.py
```

**Boyer-Moore:**
```bash
cd boyer-moore
python run_complete_evaluation.py
```

**Shift-Or/Bitap:**
```bash
cd shift-bitap
python comprehensive_6criteria_evaluation.py
```

**Suffix Array:**
```bash
cd suffix-tree_array
python comprehensive_evaluation_balanced.py
```

**Wagner-Fischer:**
```bash
cd wagner-fischer
python comprehensive_evaluation.py
```

## Project Structure

```
dna_sequence_matching/
├── dataset/                                    # Genomic datasets (FASTA files)
├── boyer-moore/                                # Boyer-Moore implementation & results
├── kmp/                                        # KMP implementation & results
├── shift-bitap/                                # Shift-Or/Bitap implementation & results
├── suffix-tree_array/                          # Suffix Array implementation & results
├── wagner-fischer/                             # Wagner-Fischer implementation & results
├── report/                                     # LaTeX report and figures
├── COMPARATIVE_ANALYSIS_ALL_ALGORITHMS.txt     # Summary results
└── download_datasets.py                        # Dataset download script
```

## Evaluation Criteria

All algorithms were evaluated across 6 comprehensive criteria:

1. **Latency/Time Performance** - Query speed and throughput
2. **Preprocessing Time** - Index/table construction cost
3. **Memory Usage** - Peak memory and index size
4. **Accuracy** - Precision, recall, F1-score
5. **Scalability** - Performance with varying text/pattern sizes
6. **Robustness** - Handling mutations and sequence variations

## Requirements

**Python 3.10+** with the following packages:
- biopython
- numpy
- matplotlib
- seaborn
- memory_profiler

Install dependencies:
```bash
pip install -r <algorithm-folder>/requirements.txt
```

## Team

- Yash More
- Naman Singhal
- Shreyash Lohare
- Anagha Prajapati
- Shrish Kadam

## Key Findings

- **No universal best algorithm** - each excels in specific scenarios
- **Suffix Array** delivers exceptional query performance after expensive preprocessing
- **KMP** provides the best balance for single queries
- **Shift-Or/Bitap and Wagner-Fischer** are essential for handling mutations
- **DNA's 4-letter alphabet** significantly impacts algorithm performance characteristics

## References

See `report/references.bib` for complete bibliography including foundational papers by Knuth, Morris, Pratt, Boyer, Moore, and others.
