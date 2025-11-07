#!/usr/bin/env python3
"""
Quick demo script for Wagner-Fischer algorithm.
Shows basic usage of the implementation.
"""

from wf_core import WagnerFischer, levenshtein_distance, similarity_ratio
from wf_search import PatternSearcher, find_motifs
from data_loader import SyntheticDataGenerator

def demo_edit_distance():
    """Demonstrate edit distance computation."""
    print("=" * 60)
    print("DEMO 1: Edit Distance Computation")
    print("=" * 60)
    
    wf = WagnerFischer()
    
    # Example 1: Identical sequences
    seq1 = "ATCGATCG"
    seq2 = "ATCGATCG"
    distance, _ = wf.compute_distance(seq1, seq2)
    print(f"\n1. Identical sequences:")
    print(f"   {seq1} vs {seq2}")
    print(f"   Distance: {distance}")
    
    # Example 2: Single substitution
    seq3 = "ATCGATCG"
    seq4 = "ATCGTTCG"
    distance, _ = wf.compute_distance(seq3, seq4)
    print(f"\n2. Single substitution:")
    print(f"   {seq3} vs {seq4}")
    print(f"   Distance: {distance}")
    
    # Example 3: With alignment
    distance, operations = wf.compute_with_traceback(seq3, seq4)
    print(f"\n3. With alignment traceback:")
    print(f"   Operations: {' → '.join(operations[:5])}...")


def demo_pattern_search():
    """Demonstrate pattern search."""
    print("\n" + "=" * 60)
    print("DEMO 2: Pattern Search")
    print("=" * 60)
    
    # DNA sequence with promoter-like motif
    text = "ATGCGCTATAATAGCTAGCTAGCGCGATATGATAAT"
    pattern = "TATAAT"  # Pribnow box (-10 box)
    
    print(f"\nSearching for pattern: {pattern}")
    print(f"In sequence: {text}")
    
    # Exact match
    searcher = PatternSearcher(max_distance=0)
    exact_matches = searcher.search(pattern, text)
    print(f"\nExact matches: {len(exact_matches)}")
    for match in exact_matches:
        print(f"  Position {match.position}: {match.matched_text}")
    
    # Approximate match (k=1)
    searcher = PatternSearcher(max_distance=1)
    approx_matches = searcher.search(pattern, text)
    print(f"\nApproximate matches (k=1): {len(approx_matches)}")
    for match in approx_matches[:3]:  # Show first 3
        print(f"  Position {match.position}: {match.matched_text} (distance={match.edit_distance})")


def demo_synthetic_data():
    """Demonstrate synthetic data generation."""
    print("\n" + "=" * 60)
    print("DEMO 3: Synthetic Data Generation")
    print("=" * 60)
    
    generator = SyntheticDataGenerator(seed=42)
    
    # Generate random sequence
    original = generator.generate_random_sequence(50)
    print(f"\nOriginal sequence:")
    print(f"  {original}")
    
    # Mutate it
    mutated, mutations = generator.mutate_sequence(
        original,
        substitution_rate=0.1,
        insertion_rate=0.05,
        deletion_rate=0.05
    )
    
    print(f"\nMutated sequence:")
    print(f"  {mutated}")
    print(f"\nMutations applied: {len(mutations)}")
    for mut in mutations[:5]:  # Show first 5
        print(f"  {mut}")
    
    # Compute distance
    distance = levenshtein_distance(original, mutated)
    similarity = similarity_ratio(original, mutated)
    print(f"\nEdit distance: {distance}")
    print(f"Similarity: {similarity:.2%}")


def demo_similarity():
    """Demonstrate similarity calculations."""
    print("\n" + "=" * 60)
    print("DEMO 4: Sequence Similarity")
    print("=" * 60)
    
    sequences = [
        ("ATCGATCG", "ATCGATCG"),  # Identical
        ("ATCGATCG", "ATCGTTCG"),  # 1 substitution
        ("ATCGATCG", "TTCGTTCG"),  # 2 substitutions
        ("AAAA", "TTTT"),           # Completely different
    ]
    
    print("\nSequence pairs and their similarities:")
    for seq1, seq2 in sequences:
        distance = levenshtein_distance(seq1, seq2)
        similarity = similarity_ratio(seq1, seq2)
        print(f"\n  {seq1} vs {seq2}")
        print(f"  Distance: {distance}, Similarity: {similarity:.2%}")


def main():
    """Run all demos."""
    print("\n" + "=" * 60)
    print("Wagner-Fischer Algorithm - Interactive Demo")
    print("=" * 60)
    
    demo_edit_distance()
    demo_pattern_search()
    demo_synthetic_data()
    demo_similarity()
    
    print("\n" + "=" * 60)
    print("Demo Complete!")
    print("=" * 60)
    print("\nFor more examples, see README.md")
    print("To run full experiments: python main.py benchmark --full")
    print()


if __name__ == "__main__":
    main()
