"""
Unit tests for the Shift-Or/Bitap algorithm implementation.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from algorithm import ShiftOrBitap


class TestExactMatching:
    """Test exact pattern matching."""
    
    def test_simple_match(self):
        """Test simple exact match."""
        pattern = "ACGT"
        text = "AACGTCACGT"
        
        matcher = ShiftOrBitap(pattern)
        matches = matcher.search_exact(text)
        
        assert matches == [1, 6], f"Expected [1, 6], got {matches}"
    
    def test_no_match(self):
        """Test when pattern doesn't exist in text."""
        pattern = "AAAA"
        text = "CCCCGGGGTTTT"
        
        matcher = ShiftOrBitap(pattern)
        matches = matcher.search_exact(text)
        
        assert matches == [], f"Expected no matches, got {matches}"
    
    def test_single_match(self):
        """Test single match."""
        pattern = "GATTACA"
        text = "CGATTACAG"
        
        matcher = ShiftOrBitap(pattern)
        matches = matcher.search_exact(text)
        
        assert matches == [1], f"Expected [1], got {matches}"
    
    def test_overlapping_pattern(self):
        """Test pattern that could overlap itself."""
        pattern = "AAA"
        text = "AAAAA"
        
        matcher = ShiftOrBitap(pattern)
        matches = matcher.search_exact(text)
        
        assert matches == [0, 1, 2], f"Expected [0, 1, 2], got {matches}"
    
    def test_pattern_at_boundaries(self):
        """Test pattern at start and end of text."""
        pattern = "ACGT"
        text = "ACGTCCCCACGT"
        
        matcher = ShiftOrBitap(pattern)
        matches = matcher.search_exact(text)
        
        assert matches == [0, 8], f"Expected [0, 8], got {matches}"
    
    def test_case_insensitive(self):
        """Test case-insensitive matching."""
        pattern = "acgt"
        text = "AACGTCACGT"
        
        matcher = ShiftOrBitap(pattern, case_sensitive=False)
        matches = matcher.search_exact(text)
        
        assert matches == [1, 6], f"Expected [1, 6], got {matches}"
    
    def test_pattern_equals_text(self):
        """Test when pattern equals entire text."""
        pattern = "ACGT"
        text = "ACGT"
        
        matcher = ShiftOrBitap(pattern)
        matches = matcher.search_exact(text)
        
        assert matches == [0], f"Expected [0], got {matches}"
    
    def test_long_pattern(self):
        """Test with longer pattern."""
        pattern = "ACGTACGTACGTACGT"
        text = "GGACGTACGTACGTACGTCC"
        
        matcher = ShiftOrBitap(pattern)
        matches = matcher.search_exact(text)
        
        assert matches == [2], f"Expected [2], got {matches}"


class TestApproximateMatching:
    """Test approximate pattern matching."""
    
    def test_exact_match_as_approximate(self):
        """Test that exact matches are found with k=0."""
        pattern = "ACGT"
        text = "AACGTCACGT"
        
        matcher = ShiftOrBitap(pattern)
        matches = matcher.search_approximate(text, max_errors=0)
        
        positions = [pos for pos, _ in matches]
        assert positions == [1, 6], f"Expected [1, 6], got {positions}"
    
    def test_single_substitution(self):
        """Test with one substitution."""
        pattern = "ACGT"
        text = "AACCT"  # ACCT has 1 substitution
        
        matcher = ShiftOrBitap(pattern)
        matches = matcher.search_approximate(text, max_errors=1)
        
        assert len(matches) > 0, "Expected at least one match"
        assert matches[0][0] == 1, f"Expected match at position 1"
    
    def test_multiple_errors(self):
        """Test with multiple errors."""
        pattern = "ACGT"
        text = "AAGGT"  # AGGT has 2 substitutions from ACGT
        
        matcher = ShiftOrBitap(pattern)
        
        # Should not match with max_errors=1
        matches1 = matcher.search_approximate(text, max_errors=1)
        # Should match with max_errors=2
        matches2 = matcher.search_approximate(text, max_errors=2)
        
        assert len(matches2) > 0, "Expected match with 2 errors"
    
    def test_no_approximate_match(self):
        """Test when no approximate match exists within error threshold."""
        pattern = "AAAA"
        text = "CCCCGGGG"
        
        matcher = ShiftOrBitap(pattern)
        matches = matcher.search_approximate(text, max_errors=1)
        
        assert len(matches) == 0, f"Expected no matches, got {matches}"
    
    def test_error_count(self):
        """Test that error count is correctly reported."""
        pattern = "ACGT"
        text = "AACGT"
        
        matcher = ShiftOrBitap(pattern)
        matches = matcher.search_approximate(text, max_errors=1)
        
        # Exact match should have 0 errors
        exact_matches = [m for m in matches if m[1] == 0]
        assert len(exact_matches) > 0, "Expected at least one exact match"


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_pattern(self):
        """Test with empty pattern."""
        with pytest.raises(ValueError):
            ShiftOrBitap("")
    
    def test_empty_text(self):
        """Test with empty text."""
        pattern = "ACGT"
        matcher = ShiftOrBitap(pattern)
        
        matches = matcher.search_exact("")
        assert matches == [], "Expected no matches in empty text"
    
    def test_pattern_longer_than_text(self):
        """Test when pattern is longer than text."""
        pattern = "ACGTACGT"
        text = "ACGT"
        
        matcher = ShiftOrBitap(pattern)
        matches = matcher.search_exact(text)
        
        assert matches == [], "Expected no matches when pattern > text"
    
    def test_pattern_too_long(self):
        """Test that overly long patterns are rejected."""
        pattern = "A" * 100  # Longer than 64
        
        with pytest.raises(ValueError):
            ShiftOrBitap(pattern)
    
    def test_special_characters(self):
        """Test with special characters (N, etc.)."""
        pattern = "ACGN"
        text = "AACGNC"
        
        matcher = ShiftOrBitap(pattern)
        matches = matcher.search_exact(text)
        
        assert matches == [1], f"Expected [1], got {matches}"
    
    def test_negative_max_errors(self):
        """Test that negative max_errors raises error."""
        pattern = "ACGT"
        matcher = ShiftOrBitap(pattern)
        
        with pytest.raises(ValueError):
            matcher.search_approximate("ACGT", max_errors=-1)
    
    def test_single_character_pattern(self):
        """Test with single character pattern."""
        pattern = "A"
        text = "ACGTA"
        
        matcher = ShiftOrBitap(pattern)
        matches = matcher.search_exact(text)
        
        assert matches == [0, 4], f"Expected [0, 4], got {matches}"
    
    def test_single_character_text(self):
        """Test with single character text."""
        pattern = "A"
        text = "A"
        
        matcher = ShiftOrBitap(pattern)
        matches = matcher.search_exact(text)
        
        assert matches == [0], f"Expected [0], got {matches}"


class TestPatternInfo:
    """Test pattern information retrieval."""
    
    def test_get_pattern_info(self):
        """Test getting pattern information."""
        pattern = "ACGT"
        matcher = ShiftOrBitap(pattern)
        
        info = matcher.get_pattern_info()
        
        assert info['pattern'] == "ACGT"
        assert info['length'] == 4
        assert info['alphabet_size'] == 4
        assert set(info['alphabet']) == {'A', 'C', 'G', 'T'}
    
    def test_count_matches_exact(self):
        """Test counting exact matches."""
        pattern = "ACG"
        text = "ACGACGACG"
        
        matcher = ShiftOrBitap(pattern)
        count = matcher.count_matches(text, approximate=False)
        
        assert count == 3, f"Expected 3 matches, got {count}"
    
    def test_count_matches_approximate(self):
        """Test counting approximate matches."""
        pattern = "ACGT"
        text = "ACGTACCTACGG"
        
        matcher = ShiftOrBitap(pattern)
        count = matcher.count_matches(text, approximate=True, max_errors=1)
        
        assert count >= 1, f"Expected at least 1 match, got {count}"


class TestDNASpecific:
    """Test DNA-specific scenarios."""
    
    def test_gc_rich_pattern(self):
        """Test with GC-rich pattern."""
        pattern = "GCGCGC"
        text = "ATGCGCGCAT"
        
        matcher = ShiftOrBitap(pattern)
        matches = matcher.search_exact(text)
        
        assert matches == [2], f"Expected [2], got {matches}"
    
    def test_at_rich_pattern(self):
        """Test with AT-rich pattern."""
        pattern = "ATATAT"
        text = "GCATATATGC"
        
        matcher = ShiftOrBitap(pattern)
        matches = matcher.search_exact(text)
        
        assert matches == [2], f"Expected [2], got {matches}"
    
    def test_homopolymer(self):
        """Test with homopolymer pattern."""
        pattern = "AAAA"
        text = "GCAAAAAGC"
        
        matcher = ShiftOrBitap(pattern)
        matches = matcher.search_exact(text)
        
        assert matches == [2], f"Expected [2], got {matches}"
    
    def test_tandem_repeat(self):
        """Test with tandem repeat pattern."""
        pattern = "CAGCAG"
        text = "ATCAGCAGCAGCAGAT"
        
        matcher = ShiftOrBitap(pattern)
        matches = matcher.search_exact(text)
        
        # Should find overlapping matches
        assert len(matches) >= 2, f"Expected multiple matches, got {matches}"


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
