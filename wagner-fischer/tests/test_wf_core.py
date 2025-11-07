"""
Unit tests for Wagner-Fischer core algorithm.
"""

import pytest
import numpy as np
from wf_core import WagnerFischer, levenshtein_distance, similarity_ratio


class TestWagnerFischer:
    """Test cases for WagnerFischer class."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.wf = WagnerFischer()
    
    def test_identical_strings(self):
        """Test that identical strings have distance 0."""
        s1 = "ATCG"
        s2 = "ATCG"
        distance, _ = self.wf.compute_distance(s1, s2)
        assert distance == 0
    
    def test_empty_strings(self):
        """Test empty string cases."""
        # Both empty
        distance, _ = self.wf.compute_distance("", "")
        assert distance == 0
        
        # One empty
        distance, _ = self.wf.compute_distance("", "ATCG")
        assert distance == 4
        
        distance, _ = self.wf.compute_distance("ATCG", "")
        assert distance == 4
    
    def test_single_substitution(self):
        """Test single character substitution."""
        distance, _ = self.wf.compute_distance("ATCG", "ATCG")
        assert distance == 0
        
        distance, _ = self.wf.compute_distance("ATCG", "TTCG")
        assert distance == 1
    
    def test_single_insertion(self):
        """Test single character insertion."""
        distance, _ = self.wf.compute_distance("ATCG", "ATCGG")
        assert distance == 1
    
    def test_single_deletion(self):
        """Test single character deletion."""
        distance, _ = self.wf.compute_distance("ATCG", "TCG")
        assert distance == 1
    
    def test_dna_sequences(self):
        """Test on DNA sequences."""
        seq1 = "ACGTACGT"
        seq2 = "ACGTAGGT"  # One substitution
        distance, _ = self.wf.compute_distance(seq1, seq2)
        assert distance == 1
    
    def test_completely_different(self):
        """Test completely different strings."""
        s1 = "AAAA"
        s2 = "TTTT"
        distance, _ = self.wf.compute_distance(s1, s2)
        assert distance == 4
    
    def test_optimized_same_result(self):
        """Test that optimized version gives same result."""
        s1 = "ACGTACGTACGT"
        s2 = "ACGTAGGTACTT"
        
        dist1, _ = self.wf.compute_distance(s1, s2)
        dist2 = self.wf.compute_distance_optimized(s1, s2)
        
        assert dist1 == dist2
    
    def test_return_matrix(self):
        """Test that matrix is returned when requested."""
        distance, matrix = self.wf.compute_distance("AT", "AG", return_matrix=True)
        
        assert matrix is not None
        assert matrix.shape == (3, 3)
        assert matrix[0, 0] == 0
        assert matrix[2, 2] == distance
    
    def test_traceback(self):
        """Test alignment traceback."""
        distance, operations = self.wf.compute_with_traceback("ATCG", "ATCG")
        
        assert distance == 0
        assert all('match' in op for op in operations)
    
    def test_traceback_with_operations(self):
        """Test traceback with actual edit operations."""
        distance, operations = self.wf.compute_with_traceback("ATCG", "TTCG")
        
        assert distance == 1
        assert any('substitute' in op for op in operations)
    
    def test_threshold_within(self):
        """Test threshold optimization when within threshold."""
        distance, within = self.wf.compute_with_threshold("ATCG", "ATCG", threshold=2)
        
        assert within == True
        assert distance == 0
    
    def test_threshold_exceeded(self):
        """Test threshold when exceeded."""
        distance, within = self.wf.compute_with_threshold("AAAA", "TTTT", threshold=2)
        
        assert within == False
        assert distance >= 2
    
    def test_custom_costs(self):
        """Test with custom operation costs."""
        wf_custom = WagnerFischer(substitution_cost=2, insertion_cost=1, deletion_cost=1)
        
        # Substitution should cost more
        dist1, _ = wf_custom.compute_distance("A", "T")
        assert dist1 == 2
        
        # Insertion
        dist2, _ = wf_custom.compute_distance("A", "AT")
        assert dist2 == 1


class TestConvenienceFunctions:
    """Test convenience functions."""
    
    def test_levenshtein_distance(self):
        """Test levenshtein_distance function."""
        distance = levenshtein_distance("ATCG", "ATCG")
        assert distance == 0
        
        distance = levenshtein_distance("ATCG", "TTCG")
        assert distance == 1
    
    def test_similarity_ratio(self):
        """Test similarity_ratio function."""
        # Identical
        ratio = similarity_ratio("ATCG", "ATCG")
        assert ratio == 1.0
        
        # Completely different
        ratio = similarity_ratio("AAAA", "TTTT")
        assert ratio == 0.0
        
        # Partially similar
        ratio = similarity_ratio("ATCG", "TTCG")
        assert ratio == 0.75  # 1 difference in 4 characters
    
    def test_similarity_ratio_empty(self):
        """Test similarity ratio with empty strings."""
        ratio = similarity_ratio("", "")
        assert ratio == 1.0


class TestEdgeCases:
    """Test edge cases and special scenarios."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.wf = WagnerFischer()
    
    def test_long_sequences(self):
        """Test with long DNA sequences."""
        seq1 = "ATCG" * 100
        seq2 = "ATCG" * 100
        
        distance, _ = self.wf.compute_distance(seq1, seq2)
        assert distance == 0
    
    def test_reverse_sequences(self):
        """Test with reversed sequences."""
        seq1 = "ATCGATCG"
        seq2 = "GCTAGCTA"  # Reverse
        
        distance, _ = self.wf.compute_distance(seq1, seq2)
        assert distance > 0
    
    def test_repeated_characters(self):
        """Test with repeated characters."""
        distance, _ = self.wf.compute_distance("AAAA", "AAAA")
        assert distance == 0
        
        distance, _ = self.wf.compute_distance("AAAA", "AAAAA")
        assert distance == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
