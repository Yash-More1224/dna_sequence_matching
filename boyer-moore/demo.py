#!/usr/bin/env python3
"""
Mini demo - Run one quick experiment
"""

import sys
sys.path.insert(0, '.')

from src.boyer_moore import BoyerMoore
from src.boyer_moore_variants import get_variant
from src.data_loader import DatasetManager
from experiments.benchmarks import Benchmarker
import time

print("=" * 70)
print("BOYER-MOORE MINI DEMO")
print("=" * 70)

# Load E. coli genome subset
print("\n1. Loading data...")
manager = DatasetManager()
genome = manager.load_ecoli_genome()
text = genome[:500000]  # First 500kb
print(f"   Using {len(text):,} bp of E. coli genome")

# Define test pattern
pattern = "TATAAT"  # Pribnow box motif
print(f"   Pattern: {pattern} (Pribnow box)")

# Test all variants
print("\n2. Testing all algorithm variants...")
print("-" * 70)
print(f"{'Variant':<15} {'Matches':<10} {'Time (ms)':<12} {'Comparisons':<15} {'Shifts'}")
print("-" * 70)

variants = ['full', 'bcr_only', 'gsr_only', 'horspool']
results = {}

for variant in variants:
    matcher = get_variant(pattern, variant)
    
    # Benchmark
    start = time.perf_counter()
    matches = matcher.search(text)
    elapsed = time.perf_counter() - start
    
    stats = matcher.get_statistics()
    
    results[variant] = {
        'matches': len(matches),
        'time': elapsed,
        'comparisons': stats.get('comparisons', 0),
        'shifts': stats.get('shifts', 0)
    }
    
    print(f"{variant:<15} {len(matches):<10} {elapsed*1000:<12.2f} "
          f"{stats.get('comparisons', 0):<15,} {stats.get('shifts', 0):,}")

# Compare with Python re
print("\n3. Comparing with Python's re module...")
print("-" * 70)

import re
start = time.perf_counter()
re_matches = [m.start() for m in re.finditer(pattern, text, re.IGNORECASE)]
re_time = time.perf_counter() - start

print(f"{'Python re':<15} {len(re_matches):<10} {re_time*1000:<12.2f} {'N/A':<15} N/A")

# Performance summary
print("\n4. Performance Summary")
print("-" * 70)

bm_time = results['full']['time']
speedup = re_time / bm_time if bm_time > 0 else 0

print(f"Boyer-Moore (full):  {bm_time*1000:.2f} ms")
print(f"Python re:           {re_time*1000:.2f} ms")
print(f"Speedup:             {speedup:.2f}x {'(BM faster)' if speedup > 1 else '(re faster)'}")
print(f"\nMatches found: {results['full']['matches']}")
print(f"Text size: {len(text):,} bp")
print(f"Throughput: {len(text)/bm_time/1_000_000:.2f} MB/s")

# Show first few match positions
print(f"\n5. First 10 match positions:")
print("-" * 70)
matcher = BoyerMoore(pattern)
matches = matcher.search(text)
print(matches[:10])

print("\n" + "=" * 70)
print("✓ MINI DEMO COMPLETE")
print("=" * 70)
print("\nFor full experiments with visualizations, run:")
print("  python run_experiments.py")
print("\nFor specific experiments only:")
print("  python run_experiments.py --experiments 1 4 8")
