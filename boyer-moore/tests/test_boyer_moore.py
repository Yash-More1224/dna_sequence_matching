"""
Test Suite for Boyer-Moore Implementation

Comprehensive unit tests for correctness validation.
"""

import unittest
import sys
sys.path.append('..')

from src.boyer_moore import BoyerMoore, boyer_moore_search
from src.boyer_moore_variants import get_variant
from src.data_generator import DNAGenerator


class TestBoyerMooreBasic(unittest.TestCase):
    """Test basic Boyer-Moore functionality."""
    
    def test_simple_match(self):
        """Test simple pattern matching."""
        text = "ACGTACGTACGT"
        pattern = "ACGT"
        
        bm = BoyerMoore(pattern)
        matches = bm.search(text)
        
        self.assertEqual(matches, [0, 4, 8])
    
    def test_no_match(self):
        """Test when pattern doesn't exist."""
        text = "AAAAAAAAAA"
        pattern = "CGCG"
        
        bm = BoyerMoore(pattern)
        matches = bm.search(text)
        
        self.assertEqual(matches, [])
    
    def test_single_match(self):
        """Test single occurrence."""
        text = "AAAACGTAAA"
        pattern = "CGT"
        
        bm = BoyerMoore(pattern)
        matches = bm.search(text)
        
        self.assertEqual(matches, [4])
    
    def test_pattern_at_start(self):
        """Test pattern at text start."""
        text = "ACGTACGT"
        pattern = "ACGT"
        
        bm = BoyerMoore(pattern)
        matches = bm.search(text)
        
        self.assertIn(0, matches)
    
    def test_pattern_at_end(self):
        """Test pattern at text end."""
        text = "AAAAAACGT"
        pattern = "ACGT"
        
        bm = BoyerMoore(pattern)
        matches = bm.search(text)
        
        self.assertEqual(matches, [5])
    
    def test_overlapping_matches(self):
        """Test overlapping pattern occurrences."""
        text = "AAAAAAA"
        pattern = "AAA"
        
        bm = BoyerMoore(pattern)
        matches = bm.search(text)
        
        # Should find non-overlapping matches
        self.assertTrue(len(matches) > 0)
    
    def test_case_insensitive(self):
        """Test case insensitivity."""
        text = "acgtACGT"
        pattern = "ACGT"
        
        bm = BoyerMoore(pattern)
        matches = bm.search(text)
        
        self.assertEqual(len(matches), 2)
    
    def test_search_first(self):
        """Test search_first method."""
        text = "AAAACGTAAACGTAAA"
        pattern = "CGT"
        
        bm = BoyerMoore(pattern)
        first_match = bm.search_first(text)
        
        self.assertEqual(first_match, 4)
    
    def test_search_first_no_match(self):
        """Test search_first with no match."""
        text = "AAAAAAA"
        pattern = "CGT"
        
        bm = BoyerMoore(pattern)
        first_match = bm.search_first(text)
        
        self.assertIsNone(first_match)


class TestBoyerMooreEdgeCases(unittest.TestCase):
    """Test edge cases."""
    
    def test_empty_pattern(self):
        """Test with empty pattern."""
        with self.assertRaises(ValueError):
            BoyerMoore("")
    
    def test_pattern_longer_than_text(self):
        """Test when pattern is longer than text."""
        text = "ACG"
        pattern = "ACGTACGT"
        
        bm = BoyerMoore(pattern)
        matches = bm.search(text)
        
        self.assertEqual(matches, [])
    
    def test_pattern_equals_text(self):
        """Test when pattern equals text."""
        text = "ACGTACGT"
        pattern = "ACGTACGT"
        
        bm = BoyerMoore(pattern)
        matches = bm.search(text)
        
        self.assertEqual(matches, [0])
    
    def test_single_character_pattern(self):
        """Test with single character pattern."""
        text = "ACGTACGT"
        pattern = "A"
        
        bm = BoyerMoore(pattern)
        matches = bm.search(text)
        
        self.assertEqual(matches, [0, 4])
    
    def test_repetitive_pattern(self):
        """Test with repetitive pattern."""
        text = "AAAAAAAAAA"
        pattern = "AAA"
        
        bm = BoyerMoore(pattern)
        matches = bm.search(text)
        
        self.assertTrue(len(matches) > 0)


class TestBoyerMooreVariants(unittest.TestCase):
    """Test different algorithm variants."""
    
    def setUp(self):
        """Set up test data."""
        self.text = "GCATCGCAGAGAGTATACAGTACG"
        self.pattern = "GCAGAGAG"
    
    def test_full_variant(self):
        """Test full Boyer-Moore."""
        matcher = get_variant(self.pattern, 'full')
        matches = matcher.search(self.text)
        self.assertIsInstance(matches, list)
    
    def test_bcr_only_variant(self):
        """Test BCR-only variant."""
        matcher = get_variant(self.pattern, 'bcr_only')
        matches = matcher.search(self.text)
        self.assertIsInstance(matches, list)
    
    def test_gsr_only_variant(self):
        """Test GSR-only variant."""
        matcher = get_variant(self.pattern, 'gsr_only')
        matches = matcher.search(self.text)
        self.assertIsInstance(matches, list)
    
    def test_horspool_variant(self):
        """Test Horspool variant."""
        matcher = get_variant(self.pattern, 'horspool')
        matches = matcher.search(self.text)
        self.assertIsInstance(matches, list)
    
    def test_variants_consistency(self):
        """Test that all variants find same matches."""
        variants = ['full', 'bcr_only', 'gsr_only', 'horspool']
        results = []
        
        for variant in variants:
            matcher = get_variant(self.pattern, variant)
            matches = matcher.search(self.text)
            results.append(set(matches))
        
        # All variants should find the same matches
        first_result = results[0]
        for result in results[1:]:
            self.assertEqual(first_result, result)


class TestCorrectness(unittest.TestCase):
    """Test correctness against known results."""
    
    def test_against_python_in(self):
        """Test against Python's 'in' operator."""
        gen = DNAGenerator(seed=42)
        text, pattern, expected_positions = gen.generate_test_case(
            text_length=1000,
            pattern_length=10,
            num_occurrences=5
        )
        
        bm = BoyerMoore(pattern)
        matches = bm.search(text)
        
        # Verify all expected matches are found
        for pos in expected_positions:
            self.assertIn(pos, matches)
            # Verify the match is correct
            self.assertEqual(text[pos:pos+len(pattern)], pattern)
    
    def test_random_patterns(self):
        """Test with random patterns."""
        gen = DNAGenerator(seed=123)
        
        for _ in range(10):
            text = gen.generate_random_sequence(500)
            pattern = gen.generate_pattern(10)
            
            bm = BoyerMoore(pattern)
            matches = bm.search(text)
            
            # Verify each match
            for pos in matches:
                found_pattern = text[pos:pos+len(pattern)]
                self.assertEqual(found_pattern.upper(), pattern.upper())


class TestStatistics(unittest.TestCase):
    """Test statistics tracking."""
    
    def test_statistics_returned(self):
        """Test that statistics are tracked."""
        text = "ACGTACGTACGT"
        pattern = "ACGT"
        
        bm = BoyerMoore(pattern)
        matches = bm.search(text)
        stats = bm.get_statistics()
        
        self.assertIn('comparisons', stats)
        self.assertIn('shifts', stats)
        self.assertIn('pattern_length', stats)
        self.assertGreater(stats['comparisons'], 0)
        self.assertGreater(stats['shifts'], 0)


def run_tests():
    """Run all tests."""
    unittest.main(argv=[''], verbosity=2, exit=False)


if __name__ == '__main__':
    run_tests()
