#!/usr/bin/env python3
"""
Quick Start Example for SuffixIndexer

This script demonstrates the basic usage of the SuffixIndexer
for DNA pattern matching tasks.

Run with: python quickstart_example.py
"""

from suffix_indexer import SuffixIndexer


def example_1_basic_search():
    """Example 1: Basic pattern matching."""
    print("=" * 70)
    print("EXAMPLE 1: Basic Pattern Matching")
    print("=" * 70)
    
    # Create a sample DNA sequence
    dna_sequence = "AGATTTAGATTAGCTAGATTA"
    print(f"\nDNA Sequence: {dna_sequence}")
    print(f"Length: {len(dna_sequence)} bases")
    
    # Build the index
    print("\nBuilding suffix array index...")
    indexer = SuffixIndexer(dna_sequence)
    
    # Search for patterns
    patterns = ["AGATTA", "GAT", "TAG", "MISSING"]
    
    print("\nSearching for patterns:")
    for pattern in patterns:
        matches = indexer.search_exact(pattern)
        if matches:
            print(f"  '{pattern}' → Found at positions: {matches}")
        else:
            print(f"  '{pattern}' → Not found")


def example_2_repeat_discovery():
    """Example 2: Finding repeats and motifs."""
    print("\n\n" + "=" * 70)
    print("EXAMPLE 2: Repeat and Motif Discovery")
    print("=" * 70)
    
    # Sequence with known repeats
    dna = "ATCGATCGATCG" + "GGGGCCCC" + "ATCGATCG" * 2
    print(f"\nDNA Sequence: {dna}")
    print(f"Length: {len(dna)} bases")
    
    indexer = SuffixIndexer(dna)
    
    # Find repeats of different minimum lengths
    for min_len in [4, 6, 8]:
        print(f"\nRepeats with minimum length {min_len}bp:")
        repeats = indexer.find_longest_repeats(min_length=min_len)
        
        for i, repeat in enumerate(repeats[:3], 1):
            print(f"  {i}. '{repeat['substring']}' "
                  f"(len={repeat['length']}, count={repeat['count']})")
            print(f"     Positions: {repeat['positions']}")


def example_3_restriction_sites():
    """Example 3: Finding restriction enzyme sites."""
    print("\n\n" + "=" * 70)
    print("EXAMPLE 3: Restriction Enzyme Site Analysis")
    print("=" * 70)
    
    # Simulated DNA sequence
    from random import seed, choice
    seed(42)
    bases = ['A', 'C', 'G', 'T']
    
    # Generate sequence with some known restriction sites
    dna = ''.join(choice(bases) for _ in range(500))
    # Insert EcoRI sites at known positions
    dna = list(dna)
    dna[100:106] = list('GAATTC')  # EcoRI at position 100
    dna[300:306] = list('GAATTC')  # EcoRI at position 300
    dna = ''.join(dna)
    
    print(f"\nDNA Sequence length: {len(dna)} bases")
    
    indexer = SuffixIndexer(dna)
    
    # Common restriction enzyme recognition sites
    enzymes = {
        'EcoRI': 'GAATTC',
        'BamHI': 'GGATCC',
        'HindIII': 'AAGCTT',
        'PstI': 'CTGCAG',
        'SmaI': 'CCCGGG'
    }
    
    print("\nRestriction enzyme analysis:")
    total_sites = 0
    
    for enzyme, site in enzymes.items():
        matches = indexer.search_exact(site)
        if matches:
            total_sites += len(matches)
            print(f"  {enzyme} ({site}): {len(matches)} site(s) at {matches}")
        else:
            print(f"  {enzyme} ({site}): No sites found")
    
    print(f"\nTotal restriction sites found: {total_sites}")


def example_4_performance_metrics():
    """Example 4: Performance analysis."""
    print("\n\n" + "=" * 70)
    print("EXAMPLE 4: Performance Metrics")
    print("=" * 70)
    
    import time
    
    # Test on sequences of different sizes
    sizes = [100, 500, 1000, 5000]
    
    print("\nPerformance across different sequence sizes:")
    print(f"{'Size (bp)':<12} {'Build Time':<15} {'Memory (KB)':<15} {'Search Time':<15}")
    print("-" * 70)
    
    for size in sizes:
        # Generate sequence
        from random import seed, choice
        seed(size)
        dna = ''.join(choice(['A', 'C', 'G', 'T']) for _ in range(size))
        
        # Build
        start = time.time()
        indexer = SuffixIndexer(dna)
        build_time = time.time() - start
        
        # Search
        pattern = "ACGT"
        start = time.time()
        matches = indexer.search_exact(pattern)
        search_time = time.time() - start
        
        stats = indexer.get_statistics()
        memory_kb = stats['memory_footprint_bytes'] / 1024
        
        print(f"{size:<12} {build_time:<15.4f} {memory_kb:<15.2f} {search_time:<15.6f}")


def example_5_biological_features():
    """Example 5: Biological feature detection."""
    print("\n\n" + "=" * 70)
    print("EXAMPLE 5: Biological Feature Detection")
    print("=" * 70)
    
    # Simulate a gene sequence
    # Promoter + Start codon + Gene + Stop codon
    promoter = "TATAAT" + "N" * 10 + "TTGACA"  # -10 and -35 boxes
    start = "ATG"
    gene = "GCTAGCTAGCTAGCTAGC" * 5  # 90bp coding sequence
    stop = "TAA"
    
    dna = promoter + "N" * 20 + start + gene + stop + "N" * 50
    dna = dna.replace('N', 'A')  # Replace N with A for simplicity
    
    print(f"\nSimulated gene sequence: {len(dna)} bases")
    
    indexer = SuffixIndexer(dna)
    
    # Search for biological features
    features = {
        'Pribnow box (-10)': 'TATAAT',
        '-35 box': 'TTGACA',
        'Start codon': 'ATG',
        'Stop codon (TAA)': 'TAA',
        'Stop codon (TAG)': 'TAG',
        'Stop codon (TGA)': 'TGA'
    }
    
    print("\nBiological features found:")
    for feature, sequence in features.items():
        matches = indexer.search_exact(sequence)
        if matches:
            print(f"  {feature}: {len(matches)} occurrence(s) at position(s) {matches}")
        else:
            print(f"  {feature}: Not found")


def main():
    """Run all examples."""
    print("\n" + "=" * 70)
    print("SUFFIX ARRAY DNA PATTERN MATCHING - QUICK START EXAMPLES")
    print("=" * 70)
    print("\nThis script demonstrates various use cases for the SuffixIndexer.")
    
    example_1_basic_search()
    example_2_repeat_discovery()
    example_3_restriction_sites()
    example_4_performance_metrics()
    example_5_biological_features()
    
    print("\n\n" + "=" * 70)
    print("Examples complete!")
    print("=" * 70)
    print("\nFor more information:")
    print("  - Read the README.md for full documentation")
    print("  - Run 'python test_suffix_indexer.py' for comprehensive tests")
    print("  - Run 'python benchmark_dna_datasets.py' for performance benchmarks")
    print("  - Check suffix_indexer.py for API documentation")
    print()


if __name__ == "__main__":
    main()
