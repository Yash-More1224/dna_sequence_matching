#!/usr/bin/env python3
"""
Quick test script to verify installation without requiring all dependencies
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

print("=" * 70)
print("SUFFIX ARRAY - QUICK VERIFICATION TEST")
print("=" * 70)
print()

try:
    from src.suffix_array import SuffixArray
    from src.data_generator import generate_random_sequence, generate_pattern
    from src.utils import format_time
    import time
    
    print("✓ Core modules imported successfully")
    print()
    
    # Test 1: Basic matching
    print("Test 1: Basic Pattern Matching")
    print("-" * 70)
    text = "ACGTACGTACGT"
    pattern = "ACGT"
    sa = SuffixArray(text, verbose=False)
    matches = sa.search(pattern)
    assert sorted(matches) == [0, 4, 8], "Basic matching failed"
    print(f"✓ Found {len(matches)} matches at positions {matches}")
    print()
    
    # Test 2: Larger sequence
    print("Test 2: Larger Sequence (10,000 bp)")
    print("-" * 70)
    text = generate_random_sequence(10000, seed=42)
    pattern = generate_pattern(16, seed=42)
    
    start = time.perf_counter()
    sa = SuffixArray(text, verbose=False)
    build_time = time.perf_counter() - start
    
    start = time.perf_counter()
    matches = sa.search(pattern)
    search_time = time.perf_counter() - start
    
    print(f"✓ Index built in {format_time(build_time)}")
    print(f"✓ Search completed in {format_time(search_time)}")
    print(f"✓ Found {len(matches)} matches")
    print()
    
    # Test 3: Repeat discovery
    print("Test 3: Repeat Discovery")
    print("-" * 70)
    text = "AGATTTAGATTAGATTA"
    sa = SuffixArray(text, verbose=False)
    repeats = sa.find_longest_repeats(min_length=4)
    print(f"✓ Found {len(repeats)} repeats (min_length=4)")
    if repeats:
        print(f"  Longest: '{repeats[0]['substring']}' ({repeats[0]['length']} bp)")
    print()
    
    # Test 4: Edge cases
    print("Test 4: Edge Cases")
    print("-" * 70)
    
    # Empty pattern
    sa = SuffixArray("ACGT", verbose=False)
    assert sa.search("") == [], "Empty pattern test failed"
    print("✓ Empty pattern handled correctly")
    
    # Pattern longer than text
    assert sa.search("ACGTACGTACGT") == [], "Long pattern test failed"
    print("✓ Long pattern handled correctly")
    
    # No match
    assert sa.search("ZZZZ") == [], "No match test failed"
    print("✓ No match handled correctly")
    print()
    
    print("=" * 70)
    print("✓ ALL TESTS PASSED")
    print("=" * 70)
    print()
    print("Installation verified successfully!")
    print()
    print("Next steps:")
    print("  1. Install BioPython: pip install biopython")
    print("  2. Run full demo: python demo.py")
    print("  3. Run experiments: python run_experiments.py")
    print()
    
except Exception as e:
    print(f"❌ TEST FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
