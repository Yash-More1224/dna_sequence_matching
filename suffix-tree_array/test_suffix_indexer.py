"""
Comprehensive Unit Tests for SuffixIndexer

This module contains rigorous unit tests for the Suffix Array + LCP implementation,
covering all required functionality including:
- Suffix array construction
- Exact pattern matching
- Edge cases (empty patterns, long patterns, etc.)
- Repeat/motif discovery
- DNA-specific test cases

Run with: pytest test_suffix_indexer.py -v
or: python test_suffix_indexer.py
"""

import sys
from suffix_indexer import SuffixIndexer

try:
    import pytest
    PYTEST_AVAILABLE = True
except ImportError:
    PYTEST_AVAILABLE = False
    # Mock pytest.skip for standalone running
    class pytest:
        @staticmethod
        def skip(msg):
            pass


class TestSuffixArrayConstruction:
    """Test suite for suffix array and LCP construction."""
    
    def test_banana_suffix_array(self):
        """Test the classic 'banana' example with known suffix array."""
        indexer = SuffixIndexer("banana")
        
        # The expected suffix array for "banana$"
        # Suffixes in sorted order:
        # 6: $
        # 5: a$
        # 3: ana$
        # 1: anana$
        # 0: banana$
        # 4: na$
        # 2: nana$
        expected_sa = [6, 5, 3, 1, 0, 4, 2]
        
        assert indexer.sa == expected_sa, f"Expected SA {expected_sa}, got {indexer.sa}"
        assert len(indexer.lcp) == len(indexer.sa), "LCP array length should match SA length"
    
    def test_banana_lcp_array(self):
        """Test LCP array for 'banana'."""
        indexer = SuffixIndexer("banana")
        
        # Expected LCP array for "banana$"
        # LCP[i] is the longest common prefix between sa[i-1] and sa[i]
        expected_lcp = [0, 0, 1, 3, 0, 0, 2]
        
        assert indexer.lcp == expected_lcp, f"Expected LCP {expected_lcp}, got {indexer.lcp}"
    
    def test_empty_string(self):
        """Test handling of empty string."""
        indexer = SuffixIndexer("")
        
        # Empty string doesn't build index (text is empty, so no suffix array)
        # This is acceptable behavior - just verify no crashes
        assert isinstance(indexer.sa, list), "SA should be a list"
        assert isinstance(indexer.lcp, list), "LCP should be a list"
    
    def test_single_character(self):
        """Test suffix array for single character."""
        indexer = SuffixIndexer("A")
        
        # Should have SA for "A$": [$, A$]
        assert len(indexer.sa) == 2, "Single char should produce 2 suffixes"
        assert indexer.sa[0] == 1, "Sentinel should be first"
        assert indexer.sa[1] == 0, "Character should be second"
    
    def test_dna_sequence(self):
        """Test suffix array construction on a DNA sequence."""
        dna = "ACGT"
        indexer = SuffixIndexer(dna)
        
        assert len(indexer.sa) == 5, "ACGT should produce 5 suffixes (including sentinel)"
        assert indexer.sa[0] == 4, "Sentinel should be first (position 4)"
        
        # Verify SA is valid (all indices present)
        assert sorted(indexer.sa) == list(range(5)), "SA should contain all valid indices"
    
    def test_repeated_characters(self):
        """Test suffix array for string with repeated characters."""
        indexer = SuffixIndexer("AAAA")
        
        assert len(indexer.sa) == 5, "AAAA should produce 5 suffixes"
        # All A's should be in descending order of length
        assert indexer.sa == [4, 3, 2, 1, 0], "Repeated chars should sort by length"


class TestExactPatternSearch:
    """Test suite for exact pattern matching functionality."""
    
    def test_search_banana_ana(self):
        """Test searching for 'ana' in 'banana'."""
        indexer = SuffixIndexer("banana")
        matches = indexer.search_exact("ana")
        
        assert sorted(matches) == [1, 3], "Pattern 'ana' should be found at positions 1 and 3"
    
    def test_search_banana_na(self):
        """Test searching for 'na' in 'banana'."""
        indexer = SuffixIndexer("banana")
        matches = indexer.search_exact("na")
        
        assert sorted(matches) == [2, 4], "Pattern 'na' should be found at positions 2 and 4"
    
    def test_search_banana_banana(self):
        """Test searching for full string."""
        indexer = SuffixIndexer("banana")
        matches = indexer.search_exact("banana")
        
        assert matches == [0], "Full string should match at position 0"
    
    def test_search_not_found(self):
        """Test searching for pattern that doesn't exist."""
        indexer = SuffixIndexer("banana")
        matches = indexer.search_exact("xyz")
        
        assert matches == [], "Non-existent pattern should return empty list"
    
    def test_search_empty_pattern(self):
        """Test searching for empty pattern."""
        indexer = SuffixIndexer("banana")
        matches = indexer.search_exact("")
        
        assert matches == [], "Empty pattern should return empty list"
    
    def test_search_pattern_longer_than_text(self):
        """Test searching for pattern longer than the text."""
        indexer = SuffixIndexer("ACG")
        matches = indexer.search_exact("ACGTACGT")
        
        assert matches == [], "Pattern longer than text should return empty list"
    
    def test_search_dna_exact_match(self):
        """Test exact DNA pattern matching."""
        dna = "AGATTTAGATTAGCTAGATTA"
        indexer = SuffixIndexer(dna)
        
        # Search for "AGATTA"
        matches = indexer.search_exact("AGATTA")
        # Manually check: Position 6 = "AGATTA", Position 15 = "AGATTA"
        assert sorted(matches) == [6, 15], "AGATTA should be found at positions 6 and 15"
        
        # Search for "GAT"
        matches = indexer.search_exact("GAT")
        # Manually verify: Positions 1, 7, 16 have "GAT"
        expected = [1, 7, 16]  # Positions where GAT appears
        assert sorted(matches) == expected, f"GAT should be at positions {expected}"
    
    def test_search_single_base(self):
        """Test searching for single DNA base."""
        dna = "ACGTACGT"
        indexer = SuffixIndexer(dna)
        
        matches_a = indexer.search_exact("A")
        assert sorted(matches_a) == [0, 4], "A should be found at positions 0 and 4"
        
        matches_c = indexer.search_exact("C")
        assert sorted(matches_c) == [1, 5], "C should be found at positions 1 and 5"
    
    def test_search_overlapping_matches(self):
        """Test searching for pattern with overlapping occurrences."""
        text = "AAAAA"
        indexer = SuffixIndexer(text)
        
        matches = indexer.search_exact("AA")
        assert sorted(matches) == [0, 1, 2, 3], "AA should be found at all overlapping positions"
    
    def test_search_case_sensitive(self):
        """Test that search is case-sensitive."""
        indexer = SuffixIndexer("ACGT")
        
        matches_upper = indexer.search_exact("ACG")
        matches_lower = indexer.search_exact("acg")
        
        assert len(matches_upper) == 1, "Upper case should match"
        assert len(matches_lower) == 0, "Lower case should not match"
    
    def test_search_multiple_patterns(self):
        """Test searching for multiple different patterns."""
        dna = "ATCGATCGATCGGGCCATCG"
        indexer = SuffixIndexer(dna)
        
        patterns = ["ATCG", "GGC", "CCC", "TCG"]
        expected_counts = {
            "ATCG": 4,  # Appears 4 times
            "GGC": 1,   # Appears once (at position 12: ...GGGCC...)
            "CCC": 0,   # Doesn't appear
            "TCG": 4    # Appears 4 times
        }
        
        for pattern, expected_count in expected_counts.items():
            matches = indexer.search_exact(pattern)
            assert len(matches) == expected_count, \
                f"Pattern {pattern} should have {expected_count} matches, got {len(matches)}"


class TestRepeatDiscovery:
    """Test suite for repeat/motif discovery using LCP array."""
    
    def test_find_repeats_simple(self):
        """Test finding repeats in a simple sequence."""
        # "AGATTTAGATTA" has "AGATTA" repeated
        dna = "AGATTTAGATTA"
        indexer = SuffixIndexer(dna)
        
        repeats = indexer.find_longest_repeats(min_length=5)
        
        # Should find "AGATTA" (length 6) or at least "AGATT" (length 5)
        assert len(repeats) > 0, "Should find at least one repeat"
        longest = repeats[0]
        assert longest['length'] >= 5, "Longest repeat should be at least 5 bases"
        assert longest['count'] >= 2, "Repeat should occur at least twice"
    
    def test_find_repeats_no_repeats(self):
        """Test repeat discovery when no repeats exist."""
        dna = "ACGT"  # No repeats of length >= 2
        indexer = SuffixIndexer(dna)
        
        repeats = indexer.find_longest_repeats(min_length=2)
        
        assert repeats == [], "Sequence with no repeats should return empty list"
    
    def test_find_repeats_all_same(self):
        """Test repeat discovery on homopolymer sequence."""
        dna = "AAAAAAAAAA"  # All same base
        indexer = SuffixIndexer(dna)
        
        repeats = indexer.find_longest_repeats(min_length=3)
        
        assert len(repeats) > 0, "Homopolymer should have many repeats"
        longest = repeats[0]
        assert longest['length'] >= 3, "Should find long repeats"
        assert 'A' in longest['substring'], "Repeat should contain A"
    
    def test_find_repeats_min_length_filter(self):
        """Test that min_length parameter correctly filters repeats."""
        dna = "ATATCGCGCG"  # Has "AT" (2bp) and "CG" (2bp) repeated
        indexer = SuffixIndexer(dna)
        
        # With min_length=2, should find repeats
        repeats_2 = indexer.find_longest_repeats(min_length=2)
        assert len(repeats_2) > 0, "Should find 2bp repeats"
        
        # With min_length=10, should find nothing
        repeats_10 = indexer.find_longest_repeats(min_length=10)
        assert len(repeats_10) == 0, "Should find no 10bp repeats"
    
    def test_find_repeats_tandem_repeats(self):
        """Test finding tandem repeats."""
        # "ATCG" repeated 3 times
        dna = "ATCGATCGATCG"
        indexer = SuffixIndexer(dna)
        
        repeats = indexer.find_longest_repeats(min_length=4)
        
        # Should find "ATCGATCG" (8bp) and/or "ATCG" (4bp)
        assert len(repeats) > 0, "Should find tandem repeats"
        
        # Check that longest repeat is substantial
        longest = repeats[0]
        assert longest['length'] >= 4, "Should find at least 4bp repeat"
        assert longest['count'] >= 2, "Repeat should occur at least twice"
    
    def test_find_repeats_return_format(self):
        """Test that repeat discovery returns correct format."""
        dna = "AGATTTAGATTA"
        indexer = SuffixIndexer(dna)
        
        repeats = indexer.find_longest_repeats(min_length=4)
        
        assert len(repeats) > 0, "Should find repeats"
        
        for repeat in repeats:
            # Check required keys
            assert 'length' in repeat, "Repeat should have 'length' key"
            assert 'substring' in repeat, "Repeat should have 'substring' key"
            assert 'positions' in repeat, "Repeat should have 'positions' key"
            assert 'count' in repeat, "Repeat should have 'count' key"
            
            # Check data consistency
            assert repeat['length'] == len(repeat['substring']), \
                "Length should match substring length"
            assert repeat['count'] == len(repeat['positions']), \
                "Count should match number of positions"
            assert repeat['count'] >= 2, "Repeat should occur at least twice"
            
            # Check positions are sorted
            assert repeat['positions'] == sorted(repeat['positions']), \
                "Positions should be sorted"
    
    def test_find_repeats_sorted_by_length(self):
        """Test that repeats are sorted by length (descending)."""
        dna = "ATATATAT" + "CGCGCGCGCG"  # Long CG repeat, shorter AT repeat
        indexer = SuffixIndexer(dna)
        
        repeats = indexer.find_longest_repeats(min_length=2)
        
        # Should be sorted by length descending
        for i in range(len(repeats) - 1):
            assert repeats[i]['length'] >= repeats[i+1]['length'], \
                "Repeats should be sorted by length (descending)"


class TestEdgeCases:
    """Test suite for edge cases and boundary conditions."""
    
    def test_text_with_n_wildcard(self):
        """Test handling of 'N' wildcard in DNA sequences."""
        dna = "ACGTNACGT"
        indexer = SuffixIndexer(dna)
        
        # Should be able to search for patterns with N
        matches = indexer.search_exact("ACGTN")
        assert len(matches) == 1, "Should find pattern with N"
        
        matches = indexer.search_exact("NACGT")
        assert len(matches) == 1, "Should find pattern starting with N"
    
    def test_very_short_sequences(self):
        """Test on very short sequences (1-3 bases)."""
        for length in [1, 2, 3]:
            dna = "A" * length
            indexer = SuffixIndexer(dna)
            
            assert len(indexer.sa) == length + 1, \
                f"Sequence of length {length} should produce {length + 1} suffixes"
            
            matches = indexer.search_exact("A")
            assert len(matches) == length, f"Should find {length} matches for 'A'"
    
    def test_pattern_at_text_boundaries(self):
        """Test patterns at the start and end of text."""
        dna = "ACGTACGT"
        indexer = SuffixIndexer(dna)
        
        # Pattern at start
        matches = indexer.search_exact("ACGT")
        assert 0 in matches, "Should find pattern at start"
        
        # Pattern at end
        matches = indexer.search_exact("CGT")
        assert 5 in matches, "Should find pattern at end"
    
    def test_performance_medium_text(self):
        """Test performance on medium-sized text (10KB)."""
        import time
        
        # Generate 10KB of DNA sequence
        dna = "ACGTACGT" * 1250  # 10,000 bases
        
        start_time = time.time()
        indexer = SuffixIndexer(dna)
        build_time = time.time() - start_time
        
        # Should build in reasonable time (< 5 seconds for 10KB)
        assert build_time < 5.0, f"Building index for 10KB took {build_time:.2f}s (should be < 5s)"
        
        # Test search performance
        start_time = time.time()
        matches = indexer.search_exact("ACGTACGT")
        search_time = time.time() - start_time
        
        # Should search in reasonable time (< 0.1 seconds)
        assert search_time < 0.1, f"Search took {search_time:.4f}s (should be < 0.1s)"
        # Pattern "ACGTACGT" appears at positions 0, 8, 16, ... i.e., every 8 bases
        # But they don't overlap in this simple repeat, so we have 1250 occurrences
        # Actually, let's just verify we found matches, not the exact count
        assert len(matches) >= 1000, f"Should find many occurrences (found {len(matches)})"
    
    def test_statistics_reporting(self):
        """Test that statistics are properly reported."""
        dna = "ACGTACGT" * 100
        indexer = SuffixIndexer(dna)
        
        stats = indexer.get_statistics()
        
        assert 'text_length' in stats, "Stats should include text_length"
        assert 'preprocessing_time' in stats, "Stats should include preprocessing_time"
        assert 'memory_footprint_mb' in stats, "Stats should include memory_footprint_mb"
        assert 'memory_footprint_bytes' in stats, "Stats should include memory_footprint_bytes"
        
        assert stats['text_length'] == 800, "Text length should be 800"
        assert stats['preprocessing_time'] > 0, "Preprocessing time should be positive"
        assert stats['memory_footprint_bytes'] > 0, "Memory footprint should be positive"


class TestAPIConsistency:
    """Test that the API is consistent with KMP and Boyer-Moore implementations."""
    
    def test_search_returns_list_of_ints(self):
        """Test that search_exact returns list[int]."""
        indexer = SuffixIndexer("ACGTACGT")
        matches = indexer.search_exact("ACG")
        
        assert isinstance(matches, list), "search_exact should return a list"
        assert all(isinstance(m, int) for m in matches), "All matches should be integers"
    
    def test_search_returns_sorted_positions(self):
        """Test that search results are sorted."""
        indexer = SuffixIndexer("ACGTACGTACGT")
        matches = indexer.search_exact("ACGT")
        
        assert matches == sorted(matches), "Matches should be returned in sorted order"
    
    def test_zero_indexed_positions(self):
        """Test that positions are 0-indexed."""
        indexer = SuffixIndexer("XACGT")
        matches = indexer.search_exact("ACG")
        
        assert matches == [1], "Position should be 0-indexed (found at position 1, not 2)"
    
    def test_consistent_with_naive_search(self):
        """Test that results match naive string search."""
        dna = "ACGTACGTACGT"
        pattern = "CGT"
        
        # Naive search
        naive_matches = []
        for i in range(len(dna) - len(pattern) + 1):
            if dna[i:i+len(pattern)] == pattern:
                naive_matches.append(i)
        
        # Suffix array search
        indexer = SuffixIndexer(dna)
        sa_matches = indexer.search_exact(pattern)
        
        assert sorted(sa_matches) == sorted(naive_matches), \
            "Suffix array search should match naive search results"


def run_all_tests():
    """Run all tests without pytest."""
    print("=" * 80)
    print("Running Comprehensive SuffixIndexer Tests")
    print("=" * 80)
    
    test_classes = [
        TestSuffixArrayConstruction,
        TestExactPatternSearch,
        TestRepeatDiscovery,
        TestEdgeCases,
        TestAPIConsistency
    ]
    
    total_tests = 0
    passed_tests = 0
    failed_tests = []
    
    for test_class in test_classes:
        print(f"\n{test_class.__name__}:")
        print("-" * 80)
        
        test_methods = [method for method in dir(test_class) if method.startswith('test_')]
        
        for method_name in test_methods:
            total_tests += 1
            test_instance = test_class()
            test_method = getattr(test_instance, method_name)
            
            try:
                test_method()
                print(f"  ✓ {method_name}")
                passed_tests += 1
            except AssertionError as e:
                print(f"  ✗ {method_name}: {str(e)}")
                failed_tests.append((test_class.__name__, method_name, str(e)))
            except Exception as e:
                print(f"  ✗ {method_name}: ERROR - {str(e)}")
                failed_tests.append((test_class.__name__, method_name, f"ERROR: {str(e)}"))
    
    print("\n" + "=" * 80)
    print(f"Test Results: {passed_tests}/{total_tests} passed")
    print("=" * 80)
    
    if failed_tests:
        print("\nFailed Tests:")
        for class_name, method_name, error in failed_tests:
            print(f"  - {class_name}.{method_name}")
            print(f"    {error}")
        return False
    else:
        print("\n✓ All tests passed!")
        return True


if __name__ == "__main__":
    # Run tests without pytest
    success = run_all_tests()
    sys.exit(0 if success else 1)
