#!/usr/bin/env python3
"""
Quick Demo - Run one quick experiment to verify installation
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.suffix_array import SuffixArray
from src.data_loader import DatasetManager
from src.data_generator import generate_random_sequence, generate_pattern
from src.utils import format_time, format_memory
import time
import re

print("=" * 70)
print("SUFFIX ARRAY MINI DEMO")
print("=" * 70)

# Example 1: Simple pattern matching
print("\n1. Simple Pattern Matching")
print("-" * 70)
text = "AGATTTAGATTAGCTAGATTA"
pattern = "AGATTA"

print(f"Text: {text}")
print(f"Pattern: {pattern}")

sa = SuffixArray(text, verbose=False)
matches = sa.search(pattern)

print(f"✓ Found {len(matches)} occurrences at positions: {matches}")

# Example 2: Larger sequence
print("\n2. Larger Sequence Test")
print("-" * 70)
text_size = 100000
pattern_len = 16

print(f"Generating {text_size:,} bp random sequence...")
text = generate_random_sequence(text_size, seed=42)
pattern = generate_pattern(pattern_len, seed=42)

print(f"Pattern: {pattern}")

print("\nBuilding suffix array...", end=" ")
start = time.perf_counter()
sa = SuffixArray(text, verbose=False)
build_time = time.perf_counter() - start
print(f"✓ {format_time(build_time)}")
print(f"Index memory: {format_memory(sa.memory_footprint)}")

print("\nSearching...", end=" ")
start = time.perf_counter()
matches = sa.search(pattern)
search_time = time.perf_counter() - start
print(f"✓ {format_time(search_time)}")
print(f"Found {len(matches)} matches")

# Example 3: Compare with Python re
print("\n3. Comparison with Python re")
print("-" * 70)

start = time.perf_counter()
re_matches = [m.start() for m in re.finditer(pattern, text)]
re_time = time.perf_counter() - start

print(f"Suffix Array search: {format_time(search_time)}")
print(f"Python re:           {format_time(re_time)}")
speedup = re_time / search_time if search_time > 0 else 0
print(f"Speedup:             {speedup:.2f}x {'(SA faster)' if speedup > 1 else '(re faster)'}")

# Example 4: Repeat discovery
print("\n4. Repeat Discovery")
print("-" * 70)
dna = "ATCGATCGATCG" * 5 + "GGGGCCCC" + "ATCGATCG" * 3
print(f"Sequence length: {len(dna)} bp")

sa = SuffixArray(dna, verbose=False)
repeats = sa.find_longest_repeats(min_length=8)

print(f"✓ Found {len(repeats)} repeats (min_length=8)")
print("\nTop 5 repeats:")
for i, repeat in enumerate(repeats[:5], 1):
    print(f"  {i}. '{repeat['substring'][:20]}...' (len={repeat['length']}, "
          f"count={repeat['count']})")

print("\n" + "=" * 70)
print("✓ MINI DEMO COMPLETE")
print("=" * 70)
print("\nFor full experiments with analysis, run:")
print("  python run_experiments.py")
print("\nFor specific experiments only:")
print("  python run_experiments.py --experiments 1 5 7")
print()
