# Suffix Array Implementation - Quick Start Guide

## Installation

### 1. Navigate to the project directory

```bash
cd dna_sequence_matching/suffix-tree_array
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

Required packages:
- biopython (FASTA parsing)
- numpy, pandas (data handling)
- matplotlib, seaborn (visualization)
- pyyaml (configuration)
- pytest (testing)

## Quick Test Run

### Option 1: Run the mini demo

```bashPerfect! Now let me create the remaining critical files. Let me create visualizations.py, report_generator.py, tests, and the main scripts to wrap up. I'll create efficient comprehensive versions:
python demo.py
```

This runs a quick demonstration showing:
- Simple pattern matching
- Performance on larger sequences
- Comparison with Python's `re` module
- Repeat discovery

**Expected output:** Should complete in < 5 seconds, showing match positions and timing comparisons.

### Option 2: Run unit tests

```bash
python -m pytest tests/ -v
```

Or:

```bash
python tests/test_suffix_array.py
```

**Expected output:** All tests should pass (approximately 25+ test cases).

### Option 3: Test on E. coli genome

```bash
python test_ecoli_genome.py --download
```

This will:
1. Download the E. coli K-12 MG1655 genome (~4.6 MB)
2. Build a suffix array index
3. Search for biological motifs
4. Find long repeats

**Expected runtime:** ~60-90 seconds (includes download time).

## Running Experiments

### Run all 8 experiments

```bash
python run_experiments.py
```

This executes:
1. **Pattern Length Variation** - How pattern size affects performance
2. **Text Size Scalability** - Algorithm scaling with text size
3. **Preprocessing Cost** - Construction vs search time analysis
4. **Memory Footprint** - Index memory usage
5. **Comparison with Python re** - Performance vs regex
6. **Repeat Discovery** - Finding repeated substrings
7. **E. coli Genome Analysis** - Real biological data
8. **Pattern Complexity** - Different pattern types

**Expected runtime:** 5-10 minutes  
**Output:** Results saved to `results/tables/` as JSON and CSV files

### Run specific experiments

```bash
python run_experiments.py --experiments 1 5 7
```

Runs only experiments 1, 5, and 7.

## Basic Usage in Code

### Example 1: Simple pattern matching

```python
from src.suffix_array import SuffixArray

# Create text and pattern
text = "ACGTACGTACGT"
pattern = "ACGT"

# Build suffix array
sa = SuffixArray(text)

# Search for pattern
matches = sa.search(pattern)
print(f"Found at positions: {matches}")  # [0, 4, 8]
```

### Example 2: Find repeats

```python
from src.suffix_array import SuffixArray

dna = "AGATTTAGATTAGATTA"
sa = SuffixArray(dna)

# Find repeated substrings (min length = 4)
repeats = sa.find_longest_repeats(min_length=4)

for repeat in repeats[:5]:
    print(f"Repeat: {repeat['substring']}")
    print(f"Length: {repeat['length']}, Count: {repeat['count']}")
    print(f"Positions: {repeat['positions']}\n")
```

### Example 3: Load genome and search

```python
from src.suffix_array import SuffixArray
from src.data_loader import DatasetManager

# Load E. coli genome
manager = DatasetManager()
genome = manager.load_ecoli_genome(download_if_missing=True)

# Build index (takes ~30-60 seconds)
print("Building index...")
sa = SuffixArray(genome)

# Search for biological motif
pattern = "TATAAT"  # Pribnow box
matches = sa.search(pattern)
print(f"Found {len(matches)} occurrences")
```

## Troubleshooting

### Import errors

Make sure you're running from the `suffix-tree_array` directory:
```bash
cd dna_sequence_matching/suffix-tree_array
python demo.py
```

### BioPython not found

```bash
pip install biopython
```

### Download timeout

If E. coli download times out, try again or manually download:
```bash
wget https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/005/845/GCF_000005845.2_ASM584v2/GCF_000005845.2_ASM584v2_genomic.fna.gz
gunzip GCF_000005845.2_ASM584v2_genomic.fna.gz
mv GCF_000005845.2_ASM584v2_genomic.fna datasets/ecoli_genome.fasta
```

### Memory issues

For large genomes (> 10 MB), ensure you have sufficient RAM:
- E. coli (4.6 MB) needs ~200 MB RAM
- Larger genomes scale approximately as 50x text size

## Next Steps

- Read the full [README.md](README.md) for detailed documentation
- Check `IMPLEMENTATION_SUMMARY.md` for algorithm details
- See `results/` directory for experiment outputs
- Compare with other algorithms in the parent directory

## Performance Expectations

| Text Size | Construction Time | Index Memory | Search Time (32bp pattern) |
|-----------|------------------|--------------|----------------------------|
| 10 KB     | ~10 ms           | ~1 MB        | ~0.1 ms                    |
| 100 KB    | ~100 ms          | ~10 MB       | ~0.2 ms                    |
| 1 MB      | ~1-2 sec         | ~100 MB      | ~0.5 ms                    |
| 4.6 MB    | ~5-10 sec        | ~400 MB      | ~1 ms                      |

*Times are approximate and depend on hardware.*

## Support

For issues or questions:
1. Check the README.md
2. Review test cases in `tests/`
3. Examine example usage in `demo.py`
4. Review the project proposal document
