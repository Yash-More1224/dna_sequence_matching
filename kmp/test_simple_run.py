#!/usr/bin/env python3
"""Simple test to verify KMP works on downloaded datasets."""

import sys
from pathlib import Path

# Add kmp directory to path
sys.path.insert(0, str(Path(__file__).parent))

from kmp_algorithm import KMP
from data_loader import read_fasta

# Dataset directory
DATASET_DIR = Path(__file__).parent.parent / "dataset"

print("Testing KMP on downloaded datasets...")
print(f"Dataset directory: {DATASET_DIR}\n")

# Load E. coli
ecoli_path = DATASET_DIR / "ecoli_k12_mg1655.fasta"
print(f"Loading {ecoli_path.name}...")
records = read_fasta(ecoli_path)
ecoli_sequence = records[0].sequence
print(f"  Loaded: {len(ecoli_sequence):,} bp\n")

# Test pattern
pattern = "ATCGATCG"
print(f"Searching for pattern: {pattern}")

kmp = KMP(pattern)
matches = kmp.search(ecoli_sequence)

print(f"Found {len(matches)} matches")
if matches:
    print(f"First 10 match positions: {matches[:10]}")

print("\nTest complete!")
