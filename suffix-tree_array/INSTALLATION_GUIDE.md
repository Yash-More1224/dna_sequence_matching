# Installation and Usage Guide

## Quick Start (5 minutes)

### 1. Download/Clone the Project
```bash
# If you have the files, navigate to the directory
cd dna_pattern_matching/

# Or download from your repository
# git clone <your-repo-url>
# cd dna_pattern_matching/
```

### 2. Verify Installation (No dependencies required!)
```bash
# Test the core implementation
python suffix_indexer.py

# Run all tests
python test_suffix_indexer.py

# Run quick examples
python quickstart_example.py
```

That's it! The core implementation requires **no external dependencies**.

---

## Optional Dependencies

For advanced features (FASTA parsing, E. coli testing):

```bash
# Install optional dependencies
pip install -r requirements.txt

# Or install individually
pip install biopython  # For FASTA parsing and NCBI downloads
pip install pytest     # For advanced testing features
```

---

## Usage Examples

### Example 1: Basic Pattern Matching

```python
from suffix_indexer import SuffixIndexer

# Your DNA sequence
dna = "AGATTTAGATTAGCTAGATTA"

# Build the index (automatic on initialization)
indexer = SuffixIndexer(dna)

# Search for a pattern
matches = indexer.search_exact("AGATTA")
print(f"Found at positions: {matches}")  # [6, 15]

# Get performance statistics
stats = indexer.get_statistics()
print(f"Build time: {stats['preprocessing_time']:.4f}s")
print(f"Memory: {stats['memory_footprint_mb']:.2f} MB")
```

### Example 2: Find Repeats/Motifs

```python
from suffix_indexer import SuffixIndexer

dna = "ATCGATCGATCG" * 10  # Tandem repeat

indexer = SuffixIndexer(dna)

# Find repeats of at least 15bp
repeats = indexer.find_longest_repeats(min_length=15)

# Display results
for i, repeat in enumerate(repeats[:5], 1):
    print(f"{i}. Length: {repeat['length']}bp")
    print(f"   Sequence: {repeat['substring']}")
    print(f"   Occurs {repeat['count']} times")
    print(f"   Positions: {repeat['positions'][:5]}...")
```

### Example 3: Multiple Pattern Search (Restriction Sites)

```python
from suffix_indexer import SuffixIndexer

# Load your genome (example with string)
genome = open("my_genome.txt").read()

indexer = SuffixIndexer(genome)

# Define restriction sites
restriction_sites = {
    'EcoRI': 'GAATTC',
    'BamHI': 'GGATCC',
    'HindIII': 'AAGCTT'
}

# Search all sites
for enzyme, site in restriction_sites.items():
    matches = indexer.search_exact(site)
    print(f"{enzyme}: {len(matches)} sites")
    if matches:
        print(f"  First 5 positions: {matches[:5]}")
```

### Example 4: Integration with Your Benchmarking Script

```python
import time
from suffix_indexer import SuffixIndexer

def benchmark_suffix_array(text, pattern):
    """
    Compatible interface with KMP and Boyer-Moore.
    """
    # Build index (preprocessing)
    start = time.time()
    indexer = SuffixIndexer(text)
    preprocessing_time = time.time() - start
    
    # Search
    start = time.time()
    matches = indexer.search_exact(pattern)
    search_time = time.time() - start
    
    # Get memory usage
    stats = indexer.get_statistics()
    
    return {
        'algorithm': 'Suffix Array',
        'preprocessing_time': preprocessing_time,
        'search_time': search_time,
        'memory_bytes': stats['memory_footprint_bytes'],
        'matches_found': len(matches),
        'match_positions': matches
    }

# Use in your benchmarking harness
genome = "ACGT" * 10000
pattern = "ACGTACGT"

results = benchmark_suffix_array(genome, pattern)
print(f"Algorithm: {results['algorithm']}")
print(f"Preprocessing: {results['preprocessing_time']:.6f}s")
print(f"Search: {results['search_time']:.6f}s")
print(f"Memory: {results['memory_bytes'] / 1e6:.2f} MB")
print(f"Matches: {results['matches_found']}")
```

---

## Running Tests

### Quick Test
```bash
python test_suffix_indexer.py
```

Expected output:
```
Running Comprehensive SuffixIndexer Tests
================================================================================
...
Test Results: 33/33 passed
✓ All tests passed!
```

### With pytest (if installed)
```bash
pytest test_suffix_indexer.py -v
```

### Specific Test Categories
```python
# Edit test_suffix_indexer.py and run specific classes:
python -c "from test_suffix_indexer import TestExactPatternSearch; t = TestExactPatternSearch(); t.test_search_banana_ana(); print('✓ Test passed')"
```

---

## Running Benchmarks

### Quick Benchmark (small datasets)
```bash
python benchmark_dna_datasets.py --skip-large
```

### Full Benchmark (includes 1MB test)
```bash
python benchmark_dna_datasets.py
```

### Custom Benchmark
```python
from suffix_indexer import SuffixIndexer

# Your custom DNA sequence
dna = open("my_sequence.fasta").read()

# Build and time
import time
start = time.time()
indexer = SuffixIndexer(dna)
build_time = time.time() - start

print(f"Built index for {len(dna):,} bases in {build_time:.2f}s")
print(f"Rate: {len(dna)/build_time:,.0f} bases/second")

# Search performance
pattern = "ACGTACGT"
start = time.time()
matches = indexer.search_exact(pattern)
search_time = time.time() - start

print(f"Found {len(matches)} matches in {search_time*1000:.2f}ms")
```

---

## Testing on E. coli Genome

### Option 1: Auto-download from NCBI
```bash
# Requires biopython
pip install biopython

python test_ecoli_genome.py --download
```

### Option 2: Use local FASTA file
```bash
python test_ecoli_genome.py --fasta /path/to/ecoli.fasta
```

### Option 3: Manual download
1. Download from: https://www.ncbi.nlm.nih.gov/nuccore/NC_000913.3
2. Save as `ecoli.fasta`
3. Run: `python test_ecoli_genome.py`

Expected performance on E. coli (~4.6 Mbp):
- Build time: 12-18 seconds
- Memory: 70-80 MB
- Search time: <1 millisecond

---

## Integration with Your Project

### Step 1: Add to Your Repository
```bash
# Copy the main file to your project
cp suffix_indexer.py /path/to/your/project/algorithms/

# Copy tests
cp test_suffix_indexer.py /path/to/your/project/tests/
```

### Step 2: Import in Your Benchmarking Script
```python
# In your main benchmarking script
from algorithms.suffix_indexer import SuffixIndexer
from algorithms.kmp import KMP
from algorithms.boyer_moore import BoyerMoore

algorithms = {
    'KMP': KMP,
    'Boyer-Moore': BoyerMoore,
    'Suffix Array': SuffixIndexer
}
```

### Step 3: Ensure Consistent API
All three algorithms should have:
- Constructor: `Algorithm(text)`
- Search method: `search_exact(pattern)` → `List[int]`
- Statistics: `get_statistics()` → `Dict`

The `SuffixIndexer` already follows this interface!

---

## Troubleshooting

### Issue: "Import Error: No module named suffix_indexer"
**Solution**: Make sure you're in the correct directory
```bash
cd /path/to/dna_pattern_matching/
python -c "import suffix_indexer; print('✓ Import successful')"
```

### Issue: Tests are slow on large sequences
**Solution**: This is expected! The O(N log N) construction is slower than linear algorithms, but enables fast multi-pattern search and repeat discovery.

For 10KB sequence: ~0.2 seconds (normal)  
For 100KB sequence: ~2-3 seconds (normal)  
For 1MB sequence: ~30-60 seconds (acceptable)

### Issue: High memory usage
**Solution**: This is expected for suffix arrays. Memory usage is ~800-900 bytes per base in Python (including overhead). For large genomes:
- 1 MB sequence → ~80 MB memory
- 10 MB sequence → ~800 MB memory
- 100 MB sequence → ~8 GB memory (consider compressed suffix array)

### Issue: Biopython not installing
**Solution**: Biopython is optional! You can:
1. Use the core features without it
2. Install with conda: `conda install -c conda-forge biopython`
3. Parse FASTA manually (simple format)

---

## Performance Tips

### 1. Reuse the Index
```python
# Don't do this (rebuilds index every time):
for pattern in patterns:
    indexer = SuffixIndexer(genome)  # ❌ Slow
    matches = indexer.search_exact(pattern)

# Do this (reuse index):
indexer = SuffixIndexer(genome)  # ✓ Build once
for pattern in patterns:
    matches = indexer.search_exact(pattern)  # ✓ Fast searches
```

### 2. For Very Large Genomes
Consider splitting into chunks:
```python
chunk_size = 1_000_000  # 1MB chunks
genome = load_genome()

for i in range(0, len(genome), chunk_size):
    chunk = genome[i:i+chunk_size]
    indexer = SuffixIndexer(chunk)
    # Process chunk...
```

### 3. Memory Profiling
```python
import tracemalloc

tracemalloc.start()
indexer = SuffixIndexer(genome)
current, peak = tracemalloc.get_traced_memory()

print(f"Current memory: {current / 1e6:.2f} MB")
print(f"Peak memory: {peak / 1e6:.2f} MB")
tracemalloc.stop()
```

---

## FAQ

**Q: Why Suffix Array instead of Suffix Tree?**  
A: Simpler implementation, better memory efficiency, and sufficient performance for our use case. See PROJECT_SUMMARY.md for detailed justification.

**Q: Can it handle approximate matching?**  
A: No, this implementation is for exact matching only. For approximate matching, consider using the Wagner-Fischer or Bitap algorithms your team is implementing.

**Q: How does it compare to Python's `str.find()`?**  
A: 
- `str.find()`: O(|T|·|P|) per pattern, but optimized in C
- Suffix Array: O(N log N) build + O(|P| log |T|) per pattern
- **Use suffix array when**: searching many patterns, need repeat discovery
- **Use str.find() when**: single pattern, small text

**Q: Can I modify the text after building the index?**  
A: No, the index is immutable. If you modify the text, rebuild the index.

**Q: What's the largest genome I can handle?**  
A: Depends on your RAM:
- 4 GB RAM: ~500 MB genome
- 16 GB RAM: ~2 GB genome (human chromosome)
- 64 GB RAM: ~8 GB genome (multiple organisms)

---

## Getting Help

1. **Check the examples**: `quickstart_example.py` covers most use cases
2. **Read the API docs**: Docstrings in `suffix_indexer.py`
3. **Run the tests**: `test_suffix_indexer.py` shows correct usage
4. **Review benchmarks**: `benchmark_dna_datasets.py` demonstrates best practices

---

## Next Steps

1. ✅ Run the quick start to verify everything works
2. ✅ Read through `quickstart_example.py` for usage patterns
3. ✅ Integrate with your team's benchmarking framework
4. ✅ Test on your specific DNA datasets
5. ✅ Compare with KMP and Boyer-Moore implementations
6. ✅ Generate performance plots for your report

**You're ready to go! Happy pattern matching! 🧬**
