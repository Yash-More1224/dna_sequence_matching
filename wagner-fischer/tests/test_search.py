"""
Tests for pattern search functionality.
"""

import pytest
from wf_search import PatternSearcher, Match, find_motifs


class TestPatternSearcher:
    """Test cases for PatternSearcher class."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.searcher = PatternSearcher(max_distance=2)
    
    def test_exact_match_found(self):
        """Test finding exact matches."""
        pattern = "ATCG"
        text = "GGGATCGAAA"
        
        matches = self.searcher.search(pattern, text)
        
        assert len(matches) >= 1
        assert any(m.edit_distance == 0 for m in matches)
    
    def test_no_match(self):
        """Test when pattern is not in text."""
        pattern = "AAAA"
        text = "TTTTTTTT"
        
        matches = self.searcher.search(pattern, text)
        
        # Should not find exact match
        exact_matches = [m for m in matches if m.edit_distance == 0]
        assert len(exact_matches) == 0
    
    def test_approximate_match(self):
        """Test finding approximate matches."""
        pattern = "ATCG"
        text = "GGGTTCGAAA"  # TTCG is 1 edit away from ATCG
        
        matches = self.searcher.search(pattern, text)
        
        # Should find approximate match
        assert len(matches) >= 1
        assert any(m.edit_distance <= 2 for m in matches)
    
    def test_multiple_matches(self):
        """Test finding multiple matches."""
        pattern = "AT"
        text = "ATATATATAT"
        
        matches = self.searcher.search(pattern, text)
        
        assert len(matches) >= 2
    
    def test_empty_pattern(self):
        """Test with empty pattern."""
        matches = self.searcher.search("", "ATCG")
        assert len(matches) == 0
    
    def test_empty_text(self):
        """Test with empty text."""
        matches = self.searcher.search("ATCG", "")
        assert len(matches) == 0
    
    def test_pattern_longer_than_text(self):
        """Test when pattern is longer than text."""
        pattern = "ATCGATCG"
        text = "AT"
        
        matches = self.searcher.search(pattern, text)
        
        # Might find some matches depending on threshold
        assert isinstance(matches, list)
    
    def test_exact_search_method(self):
        """Test search_exact method."""
        pattern = "ATCG"
        text = "GGATCGGGATCGAA"
        
        matches = self.searcher.search_exact(pattern, text)
        
        assert len(matches) == 2
        assert all(m.edit_distance == 0 for m in matches)
    
    def test_count_matches(self):
        """Test count_matches method."""
        pattern = "AT"
        text = "ATATATATAT"
        
        count = self.searcher.count_matches(pattern, text)
        
        assert count >= 0
        assert isinstance(count, int)
    
    def test_search_multiple_patterns(self):
        """Test searching for multiple patterns."""
        patterns = ["AT", "CG"]
        text = "ATCGATCG"
        
        results = self.searcher.search_multiple(patterns, text)
        
        assert len(results) == 2
        assert "AT" in results
        assert "CG" in results
    
    def test_match_properties(self):
        """Test that Match objects have correct properties."""
        pattern = "ATCG"
        text = "GGGATCGAAA"
        
        matches = self.searcher.search(pattern, text)
        
        if matches:
            match = matches[0]
            assert hasattr(match, 'position')
            assert hasattr(match, 'end_position')
            assert hasattr(match, 'matched_text')
            assert hasattr(match, 'edit_distance')
            assert match.end_position > match.position
    
    def test_alignment_return(self):
        """Test that alignment is returned when requested."""
        pattern = "ATCG"
        text = "ATCGATCG"
        
        matches = self.searcher.search(pattern, text, return_alignment=True)
        
        if matches:
            match = matches[0]
            if match.edit_distance == 0:
                assert match.alignment is not None


class TestFindMotifs:
    """Test the find_motifs convenience function."""
    
    def test_find_motifs_basic(self):
        """Test basic motif finding."""
        pattern = "ATCG"
        text = "GGATCGGGATCGAA"
        
        matches = find_motifs(pattern, text, max_distance=1)
        
        assert len(matches) >= 1
    
    def test_find_motifs_with_similarity(self):
        """Test motif finding with similarity threshold."""
        pattern = "ATCG"
        text = "GGATCGGGATCGAA"
        
        matches = find_motifs(pattern, text, max_distance=2, min_similarity=0.9)
        
        # Should filter by similarity
        assert all((1.0 - m.edit_distance / len(pattern)) >= 0.9 for m in matches)


class TestDNASequences:
    """Test with realistic DNA sequences."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.searcher = PatternSearcher(max_distance=2)
    
    def test_ecoli_motif(self):
        """Test searching for E. coli promoter-like sequence."""
        # TATAAT is part of -10 box
        pattern = "TATAAT"
        text = "ATGCTATAATAGCTAGC" * 10
        
        matches = self.searcher.search(pattern, text)
        
        assert len(matches) >= 1
    
    def test_restriction_site(self):
        """Test searching for restriction enzyme site."""
        # EcoRI site
        pattern = "GAATTC"
        text = "ATGCGAATTCATGC"
        
        matches = self.searcher.search_exact(pattern, text)
        
        assert len(matches) == 1
        assert matches[0].position == 4


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
