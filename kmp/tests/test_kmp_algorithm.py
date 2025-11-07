"""
Unit tests for KMP algorithm implementation.

Tests cover LPS array construction, pattern search, edge cases,
and correctness validation.
"""

import pytest
from kmp.kmp_algorithm import (
    build_lps_array,
    kmp_search,
    KMP,
    kmp_search_multiple,
    kmp_count_matches,
    kmp_find_first,
    kmp_contains
)


class TestLPSArray:
    """Tests for LPS array construction."""
    
    def test_simple_pattern(self):
        """Test LPS array for simple pattern."""
        lps = build_lps_array("ABABC")
        assert lps == [0, 0, 1, 2, 0]
    
    def test_repeating_pattern(self):
        """Test LPS array for repeating pattern."""
        lps = build_lps_array("AAAA")
        assert lps == [0, 1, 2, 3]
    
    def test_no_prefix_suffix(self):
        """Test LPS array for pattern with no prefix-suffix."""
        lps = build_lps_array("ABCD")
        assert lps == [0, 0, 0, 0]
    
    def test_complex_pattern(self):
        """Test LPS array for complex pattern."""
        lps = build_lps_array("ABABCABAB")
        assert lps == [0, 0, 1, 2, 0, 1, 2, 3, 4]
    
    def test_single_character(self):
        """Test LPS array for single character."""
        lps = build_lps_array("A")
        assert lps == [0]
    
    def test_all_same(self):
        """Test LPS array for all same characters."""
        lps = build_lps_array("TTTTTT")
        assert lps == [0, 1, 2, 3, 4, 5]


class TestKMPSearch:
    """Tests for KMP search function."""
    
    def test_simple_match(self):
        """Test simple pattern match."""
        matches = kmp_search("ABABDABACDABABCABAB", "ABABC")
        assert matches == [10]
    
    def test_multiple_matches(self):
        """Test multiple overlapping matches."""
        matches = kmp_search("AAAAAAA", "AAA")
        assert matches == [0, 1, 2, 3, 4]
    
    def test_no_match(self):
        """Test when pattern not found."""
        matches = kmp_search("ATCG", "GGGG")
        assert matches == []
    
    def test_exact_match(self):
        """Test when text equals pattern."""
        matches = kmp_search("ATCG", "ATCG")
        assert matches == [0]
    
    def test_pattern_longer_than_text(self):
        """Test when pattern is longer than text."""
        matches = kmp_search("ATCG", "ATCGATCG")
        assert matches == []
    
    def test_empty_pattern(self):
        """Test with empty pattern."""
        matches = kmp_search("ATCG", "")
        assert matches == []
    
    def test_dna_sequence(self):
        """Test on DNA sequence."""
        text = "ATGCATGCATGC"
        pattern = "ATGC"
        matches = kmp_search(text, pattern)
        assert matches == [0, 4, 8]
    
    def test_overlapping_dna(self):
        """Test overlapping DNA patterns."""
        text = "CGCGCGCG"
        pattern = "GCGC"
        matches = kmp_search(text, pattern)
        assert matches == [1, 3, 5]


class TestKMPClass:
    """Tests for KMP class."""
    
    def test_initialization(self):
        """Test KMP initialization."""
        kmp = KMP("ATCG")
        assert kmp.pattern == "ATCG"
        assert len(kmp.lps) == 4
        assert kmp.preprocessing_time >= 0
    
    def test_empty_pattern_raises(self):
        """Test that empty pattern raises ValueError."""
        with pytest.raises(ValueError):
            KMP("")
    
    def test_search(self):
        """Test search method."""
        kmp = KMP("ATCG")
        matches = kmp.search("ATCGATCGATCG")
        assert matches == [0, 4, 8]
    
    def test_search_with_stats(self):
        """Test search_with_stats method."""
        kmp = KMP("ATCG")
        stats = kmp.search_with_stats("ATCGATCGATCG")
        
        assert 'matches' in stats
        assert 'num_matches' in stats
        assert 'search_time' in stats
        assert 'preprocessing_time' in stats
        assert 'total_time' in stats
        assert 'text_length' in stats
        assert 'pattern_length' in stats
        
        assert stats['num_matches'] == 3
        assert stats['text_length'] == 12
        assert stats['pattern_length'] == 4
        assert len(stats['matches']) == 3
    
    def test_get_lps_array(self):
        """Test get_lps_array method."""
        kmp = KMP("ABABC")
        lps = kmp.get_lps_array()
        assert lps == [0, 0, 1, 2, 0]
        
        # Test that it returns a copy
        lps[0] = 999
        assert kmp.lps[0] == 0
    
    def test_repr(self):
        """Test string representation."""
        kmp = KMP("ATCG")
        repr_str = repr(kmp)
        assert "KMP" in repr_str
        assert "ATCG" in repr_str


class TestHelperFunctions:
    """Tests for helper functions."""
    
    def test_kmp_search_multiple(self):
        """Test searching for multiple patterns."""
        text = "ABABCABAB"
        patterns = ["AB", "CAB"]
        results = kmp_search_multiple(text, patterns)
        
        assert "AB" in results
        assert "CAB" in results
        assert results["AB"] == [0, 2, 5, 7]
        assert results["CAB"] == [4]
    
    def test_kmp_count_matches(self):
        """Test counting matches."""
        count = kmp_count_matches("AAAAAAA", "AAA")
        assert count == 5
    
    def test_kmp_find_first(self):
        """Test finding first match."""
        pos = kmp_find_first("ABABCABAB", "CAB")
        assert pos == 4
        
        pos_none = kmp_find_first("ATCG", "GGG")
        assert pos_none is None
    
    def test_kmp_contains(self):
        """Test checking if pattern exists."""
        assert kmp_contains("ABABCABAB", "CAB") == True
        assert kmp_contains("ABABCABAB", "XYZ") == False


class TestEdgeCases:
    """Tests for edge cases and corner conditions."""
    
    def test_single_character_pattern(self):
        """Test with single character pattern."""
        matches = kmp_search("ATCGATCG", "A")
        assert matches == [0, 4]
    
    def test_single_character_text(self):
        """Test with single character text."""
        matches = kmp_search("A", "A")
        assert matches == [0]
        
        matches = kmp_search("A", "B")
        assert matches == []
    
    def test_very_long_pattern(self):
        """Test with very long pattern."""
        pattern = "ATCG" * 100
        text = "ATCG" * 1000
        kmp = KMP(pattern)
        matches = kmp.search(text)
        assert len(matches) > 0
    
    def test_case_sensitivity(self):
        """Test that search is case-sensitive."""
        matches = kmp_search("ATCG", "atcg")
        assert matches == []
        
        matches = kmp_search("ATCG", "ATCG")
        assert matches == [0]
    
    def test_special_patterns(self):
        """Test with special DNA patterns."""
        # All same base
        matches = kmp_search("AAAAAAAA", "AAAA")
        assert matches == [0, 1, 2, 3, 4]
        
        # Alternating bases
        matches = kmp_search("ATATATATATAT", "ATAT")
        assert matches == [0, 2, 4, 6, 8]


class TestCorrectness:
    """Tests to verify correctness against known results."""
    
    def test_against_naive_search(self):
        """Test KMP against naive string search."""
        text = "ATCGATCGATCGATCG"
        pattern = "ATCG"
        
        # Naive search
        naive_matches = []
        for i in range(len(text) - len(pattern) + 1):
            if text[i:i+len(pattern)] == pattern:
                naive_matches.append(i)
        
        # KMP search
        kmp_matches = kmp_search(text, pattern)
        
        assert kmp_matches == naive_matches
    
    def test_against_python_find(self):
        """Test KMP against Python's str.find."""
        text = "ATCGATCGATCGATCG"
        pattern = "ATCG"
        
        # Python find (find all)
        python_matches = []
        start = 0
        while True:
            pos = text.find(pattern, start)
            if pos == -1:
                break
            python_matches.append(pos)
            start = pos + 1
        
        # KMP search
        kmp_matches = kmp_search(text, pattern)
        
        assert kmp_matches == python_matches


class TestPerformance:
    """Performance-related tests."""
    
    def test_large_text(self):
        """Test on large text."""
        text = "ATCG" * 10000  # 40,000 characters
        pattern = "ATCGATCG"
        
        kmp = KMP(pattern)
        matches = kmp.search(text)
        
        # Should find many matches quickly
        assert len(matches) > 0
        assert kmp.preprocessing_time < 1.0  # Should be fast
    
    def test_no_match_performance(self):
        """Test performance when no match exists."""
        text = "ATCG" * 10000
        pattern = "GGGGGGGG"  # Unlikely to match
        
        kmp = KMP(pattern)
        matches = kmp.search(text)
        
        assert matches == []
        assert kmp.preprocessing_time < 1.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
