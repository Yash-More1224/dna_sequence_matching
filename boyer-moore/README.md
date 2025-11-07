# Boyer-Moore String Matching Algorithm for DNA Sequences

A comprehensive implementation and experimental analysis of the Boyer-Moore algorithm for DNA sequence matching, with a focus on the *E. coli* K-12 MG1655 genome.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Algorithm Details](#algorithm-details)
- [Project Structure](#project-structure)
- [Usage Examples](#usage-examples)
- [Running Experiments](#running-experiments)
- [Results](#results)
- [Testing](#testing)
- [Documentation](#documentation)
- [Requirements](#requirements)
- [Contributing](#contributing)

## 🔬 Overview

This project implements the **Boyer-Moore string matching algorithm** with both **Bad Character Rule (BCR)** and **Good Suffix Rule (GSR)** heuristics, specifically optimized for DNA sequence analysis. The implementation includes:

- Full Boyer-Moore algorithm with both heuristics
- Multiple algorithm variants (BCR-only, GSR-only, Horspool)
- Comprehensive benchmarking framework
- 8 detailed experiments analyzing performance characteristics
- Comparison with Python's built-in `re` module
- Real biological motif search capabilities
- Visualization and reporting tools

## ✨ Features

### Core Implementation
- ✅ Pure Python implementation (PEP 8 compliant)
- ✅ Bad Character Rule heuristic
- ✅ Good Suffix Rule heuristic
- ✅ Multiple algorithm variants
- ✅ Case-insensitive matching
- ✅ Find all occurrences
- ✅ Statistics tracking (comparisons, shifts)

### Data Handling
- ✅ Automatic E. coli genome download from NCBI
- ✅ FASTA/FASTQ file parsing (via Biopython)
- ✅ Synthetic DNA sequence generation
- ✅ Controlled mutation introduction

### Analysis & Benchmarking
- ✅ Time measurement (preprocessing + search)
- ✅ Memory profiling
- ✅ Throughput calculation
- ✅ Comparison with Python `re`
- ✅ 8 comprehensive experiments
- ✅ Statistical analysis

### Visualization
- ✅ Performance plots (time, memory, scaling)
- ✅ Comparison charts (variants, vs regex)
- ✅ Real motif search results
- ✅ High-quality PNG exports (300 DPI)

### Reporting
- ✅ Automated Markdown report generation
- ✅ Comprehensive analysis document
- ✅ Quick summary reports

## 📦 Installation

### Prerequisites

- Python 3.11 or higher
- pip package manager

### Setup

1. **Clone or navigate to the project directory:**
   ```bash
   cd boyer-moore
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Linux/Mac
   # or
   venv\Scripts\activate  # On Windows
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

### Dependencies

```
biopython>=1.81
numpy>=1.24
matplotlib>=3.7
seaborn>=0.12
pandas>=2.0
memory_profiler>=0.61
psutil>=5.9
pyyaml>=6.0
requests>=2.31
```

## 🚀 Quick Start

### Basic Usage

```python
from src.boyer_moore import BoyerMoore

# Create matcher
pattern = "TATAAT"  # Pribnow box motif
matcher = BoyerMoore(pattern)

# Search in sequence
text = "GCATCGTATAATAGAGAGTATACAGTACG"
matches = matcher.search(text)

print(f"Pattern found at positions: {matches}")
print(f"Statistics: {matcher.get_statistics()}")
```

### Load and Search E. coli Genome

```python
from src.data_loader import DatasetManager
from src.boyer_moore import BoyerMoore

# Load genome (downloads automatically if not present)
manager = DatasetManager()
genome = manager.load_ecoli_genome()

# Search for motif
pattern = "AGGAGGT"  # Shine-Dalgarno sequence
matcher = BoyerMoore(pattern)
matches = matcher.search(genome)

print(f"Found {len(matches)} occurrences of Shine-Dalgarno sequence")
```

### Run All Experiments

```bash
python run_experiments.py
```

This will:
1. Download E. coli genome (if needed)
2. Run all 8 experiments
3. Generate visualizations
4. Create comprehensive report

Results will be saved to `results/` directory.

## 🧬 Algorithm Details

### Boyer-Moore Algorithm

The Boyer-Moore algorithm is an efficient string-searching algorithm that preprocesses the pattern to enable skipping sections of the text.

#### Bad Character Rule (BCR)
When a mismatch occurs at position `i` in the pattern:
- Look up the mismatched text character in the pattern
- Shift the pattern to align its rightmost occurrence with the text position
- If the character doesn't appear, shift past it entirely

#### Good Suffix Rule (GSR)
When a mismatch occurs:
- Consider the suffix that matched before the mismatch
- Shift to align this suffix with its next occurrence in the pattern
- If no occurrence exists, shift based on the longest matching prefix

#### Time Complexity
- **Preprocessing:** O(m + |Σ|), where m = pattern length, |Σ| = alphabet size
- **Search:** 
  - Best case: O(n/m) - sublinear!
  - Worst case: O(nm)
  - Average case: O(n) for most practical inputs

#### Space Complexity
- O(m + |Σ|) for bad character and good suffix tables

### Variants Implemented

1. **Full Boyer-Moore:** Both BCR and GSR (optimal)
2. **BCR-only:** Simpler, good for large alphabets
3. **GSR-only:** Better for repetitive patterns
4. **Horspool:** Simplified BCR variant

## 📁 Project Structure

```
boyer-moore/
├── README.md                      # This file
├── requirements.txt               # Python dependencies
├── config.yaml                    # Configuration settings
├── run_experiments.py            # Main execution script
│
├── src/                          # Source code
│   ├── __init__.py
│   ├── boyer_moore.py           # Core algorithm
│   ├── boyer_moore_variants.py  # Algorithm variants
│   ├── data_loader.py           # Dataset management
│   ├── data_generator.py        # Synthetic data generation
│   └── utils.py                 # Utility functions
│
├── experiments/                  # Experiment scripts
│   ├── __init__.py
│   ├── benchmarks.py            # Benchmarking framework
│   └── experiments.py           # Experimental workflows
│
├── visualization/                # Plotting & reports
│   ├── __init__.py
│   ├── visualizations.py        # Plot generation
│   └── report_generator.py     # Markdown reports
│
├── tests/                        # Unit tests
│   ├── __init__.py
│   └── test_boyer_moore.py     # Test suite
│
├── datasets/                     # Data storage
│   └── ecoli_genome.fasta       # (Downloaded automatically)
│
└── results/                      # Output directory
    ├── plots/                   # Generated figures
    ├── tables/                  # JSON results
    └── reports/                 # Analysis reports
```

## 💻 Usage Examples

### Example 1: Simple Pattern Matching

```python
from src.boyer_moore import boyer_moore_search

text = "ACGTACGTACGT"
pattern = "ACGT"

matches = boyer_moore_search(text, pattern)
print(matches)  # [0, 4, 8]
```

### Example 2: Using Different Variants

```python
from src.boyer_moore_variants import get_variant

pattern = "GCAGAGAG"
text = "GCATCGCAGAGAGTATACAGTACG"

# Test each variant
for variant in ['full', 'bcr_only', 'gsr_only', 'horspool']:
    matcher = get_variant(pattern, variant)
    matches = matcher.search(text)
    stats = matcher.get_statistics()
    
    print(f"{variant}: {len(matches)} matches, "
          f"{stats['comparisons']} comparisons")
```

### Example 3: Generate Test Data

```python
from src.data_generator import DNAGenerator

gen = DNAGenerator(seed=42)

# Create test case with known match positions
text, pattern, positions = gen.generate_test_case(
    text_length=10000,
    pattern_length=15,
    num_occurrences=10
)

print(f"Pattern: {pattern}")
print(f"Expected positions: {positions}")
```

### Example 4: Memory Profiling

```python
from experiments.benchmarks import Benchmarker
from src.boyer_moore import BoyerMoore

benchmarker = Benchmarker()
matcher = BoyerMoore("TATAAT")

result = benchmarker.benchmark_boyer_moore(
    matcher, 
    text, 
    measure_memory=True
)

print(f"Peak memory: {result.peak_memory / 1024:.2f} KB")
print(f"Search time: {result.search_time * 1000:.3f} ms")
```

## 🧪 Running Experiments

### Run All Experiments

```bash
python run_experiments.py
```

### Run Specific Experiments

```bash
# Run experiments 1, 3, and 5
python run_experiments.py --experiments 1 3 5
```

### Skip Visualizations or Reports

```bash
# Run experiments only, skip plots
python run_experiments.py --skip-visualizations

# Skip final report generation
python run_experiments.py --skip-report
```

### Experiment List

1. **Pattern Length Analysis** - How latency varies with pattern length
2. **Text Size Scaling** - Scalability with increasing text size
3. **Alphabet Size Effect** - DNA vs larger alphabets
4. **Heuristic Contribution** - Comparing BCR, GSR, and combined
5. **Preprocessing Overhead** - Preprocessing vs search time
6. **Memory Footprint** - Memory usage analysis
7. **Real Motif Search** - Searching biological motifs in E. coli
8. **Comparison with Python re** - Benchmarking vs built-in regex

## 📊 Results

After running experiments, results are organized as:

```
results/
├── tables/
│   ├── exp1_pattern_length.json
│   ├── exp2_text_scaling.json
│   ├── exp3_alphabet_effect.json
│   ├── exp4_heuristic_contribution.json
│   ├── exp5_preprocessing_overhead.json
│   ├── exp6_memory_footprint.json
│   ├── exp7_real_motifs.json
│   └── exp8_compare_with_re.json
│
├── plots/
│   ├── pattern_length_vs_time.png
│   ├── text_scaling.png
│   ├── alphabet_effect.png
│   ├── heuristic_contribution.png
│   ├── preprocessing_overhead.png
│   ├── memory_footprint.png
│   ├── real_motifs.png
│   └── comparison_with_re.png
│
└── reports/
    ├── ANALYSIS_REPORT.md
    └── SUMMARY.md
```

View the comprehensive analysis in `results/reports/ANALYSIS_REPORT.md`.

## ✅ Testing

### Run Test Suite

```bash
cd tests
python test_boyer_moore.py
```

### Test Coverage

The test suite includes:
- ✅ Basic pattern matching
- ✅ Edge cases (empty, overlapping, case sensitivity)
- ✅ All algorithm variants
- ✅ Correctness validation
- ✅ Statistics tracking

### Example Test Output

```
test_simple_match (__main__.TestBoyerMooreBasic) ... ok
test_no_match (__main__.TestBoyerMooreBasic) ... ok
test_pattern_at_start (__main__.TestBoyerMooreBasic) ... ok
test_overlapping_matches (__main__.TestBoyerMooreBasic) ... ok
...

----------------------------------------------------------------------
Ran 25 tests in 0.123s

OK
```

## 📚 Documentation

### Configuration

Edit `config.yaml` to customize:
- Dataset URLs and paths
- Experiment parameters
- Visualization settings
- Output directories

### API Reference

#### BoyerMoore Class

```python
BoyerMoore(pattern: str, use_bcr: bool = True, use_gsr: bool = True)
```

**Methods:**
- `search(text: str) -> List[int]` - Find all occurrences
- `search_first(text: str) -> Optional[int]` - Find first occurrence
- `get_statistics() -> Dict[str, int]` - Get algorithm statistics

#### DatasetManager Class

```python
DatasetManager(data_dir: str = "datasets")
```

**Methods:**
- `download_ecoli_genome(force: bool = False) -> Path`
- `load_ecoli_genome(download_if_missing: bool = True) -> str`
- `load_fasta(filepath: str) -> SeqRecord`

## 📋 Requirements

- **Python:** 3.11+
- **OS:** Linux, macOS, Windows
- **Memory:** 8GB+ recommended for full genome analysis
- **Disk Space:** ~10MB for E. coli genome

## 🤝 Contributing

This is an academic project for the Advanced Algorithms and Data Structures course. For questions or issues:

1. Check existing documentation
2. Review test cases for usage examples
3. Examine experiment results

## 📝 License

Academic project - see course guidelines.

## 🙏 Acknowledgments

- **E. coli genome:** NCBI RefSeq (NC_000913.3)
- **Algorithm:** Boyer & Moore (1977)
- **Biopython:** Sequence I/O and parsing
- **Course:** Advanced Algorithms and Data Structures (AAD)

## 📧 Contact

For academic inquiries, contact your course instructor.

---

**Last Updated:** November 2025

**Version:** 1.0.0

**Status:** ✅ Complete Implementation
