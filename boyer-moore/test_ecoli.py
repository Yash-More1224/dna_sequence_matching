#!/usr/bin/env python3
"""
Test E. coli genome download and search
"""

import sys
sys.path.insert(0, 'src')

from data_loader import DatasetManager
from boyer_moore import BoyerMoore
import time

print("=" * 70)
print("E. COLI GENOME TEST")
print("=" * 70)

# Initialize dataset manager
manager = DatasetManager()

print("\n1. Downloading E. coli genome...")
print("-" * 70)
try:
    genome_path = manager.download_ecoli_genome()
    print(f"✓ Genome available at: {genome_path}")
except Exception as e:
    print(f"✗ Error downloading genome: {e}")
    sys.exit(1)

print("\n2. Loading genome sequence...")
print("-" * 70)
try:
    genome = manager.load_ecoli_genome()
    print(f"✓ Loaded genome: {len(genome):,} base pairs")
    print(f"  First 100 bp: {genome[:100]}")
except Exception as e:
    print(f"✗ Error loading genome: {e}")
    sys.exit(1)

# Search for biological motifs
print("\n3. Searching for biological motifs...")
print("-" * 70)

motifs = {
    'Pribnow box (TATAAT)': 'TATAAT',
    'Shine-Dalgarno (AGGAGGT)': 'AGGAGGT',
    'Start codon (ATG)': 'ATG',
    'Stop codon (TAA)': 'TAA',
}

for name, pattern in motifs.items():
    print(f"\n  Searching: {name}")
    matcher = BoyerMoore(pattern)
    
    # Search in first 100kb for speed
    text_subset = genome[:100000]
    
    start_time = time.perf_counter()
    matches = matcher.search(text_subset)
    search_time = time.perf_counter() - start_time
    
    stats = matcher.get_statistics()
    
    print(f"    Found: {len(matches)} occurrences in first 100kb")
    print(f"    Time: {search_time*1000:.2f} ms")
    print(f"    Comparisons: {stats['comparisons']:,}")
    print(f"    First 5 positions: {matches[:5] if len(matches) >= 5 else matches}")

# Performance test on larger region
print("\n4. Performance test on 1MB region...")
print("-" * 70)

pattern = "TATAAT"  # Pribnow box
text_1mb = genome[:1000000]

matcher = BoyerMoore(pattern)
start_time = time.perf_counter()
matches = matcher.search(text_1mb)
search_time = time.perf_counter() - start_time

throughput = len(text_1mb) / search_time / 1_000_000  # MB/s

print(f"  Pattern: {pattern}")
print(f"  Text size: {len(text_1mb):,} bp")
print(f"  Matches found: {len(matches):,}")
print(f"  Search time: {search_time*1000:.2f} ms")
print(f"  Throughput: {throughput:.2f} MB/s")

print("\n" + "=" * 70)
print("✓ E. COLI GENOME TEST COMPLETE")
print("=" * 70)
print("\nThe implementation is ready for full experiments!")
print("Run: python run_experiments.py")
