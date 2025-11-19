"""
Comprehensive Test Suite for Suffix Array Implementation

Tests correctness, edge cases, and functionality.
"""

import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.suffix_array import SuffixArray, suffix_array_search
from src.data_generator import DNAGenerator


class TestSuffixArrayBasic(unittest.TestCase):
    """Test basic Suffix Array functionality."""
    
    def test_simple_match(self):
        """Test simple pattern matching."""
        text = "ACGTACGTACGT"
        pattern = "ACGT"
        
        sa = SuffixArray(text, verbose=False)
        matches = sa.search(pattern)
        
        self.assertEqual(sorted(matches), [0, 4, 8])
    
    def test_no_match(self):
        """Test when pattern doesn't exist."""
        text = "AAAAAAAAAA"
        pattern = "CGCG"
        
        sa = SuffixArray(text, verbose=False)
        matches = sa.search(pattern)
        
        self.assertEqual(matches, [])
    
    def test_single_match(self):
        """Test single occurrence."""
        text = "AAAACGTAAA"
        pattern = "CGT"
        
        sa = SuffixArray(text, verbose=False)
        matches = sa.search(pattern)
        
        self.assertEqual(matches, [4])
    
    def test_pattern_at_start(self):
        """Test pattern at text start."""
        text = "ACGTACGT"
        pattern = "ACGT"
        
        sa = SuffixArray(text, verbose=False)
        matches = sa.search(pattern)
        
        self.assertIn(0, matches)
    
    def test_pattern_at_end(self):
        """Test pattern at text end."""
        text = "AAAAAACGT"
        pattern = "ACGT"
        
        sa = SuffixArray(text, verbose=False)
        matches = sa.search(pattern)
        
        self.assertEqual(matches, [5])
    
    def test_overlapping_matches(self):
        """Test overlapping pattern occurrences."""
        text = "AAAAAAA"
        pattern = "AAA"
        
        sa = SuffixArray(text, verbose=False)
        matches = sa.search(pattern)
        
        # Should find all overlapping matches
        self.assertTrue(len(matches) >= 3)
        self.assertIn(0, matches)
    
    def test_search_first(self):
        """Test search_first method."""
        text = "AAAACGTAAACGTAAA"
        pattern = "CGT"
        
        sa = SuffixArray(text, verbose=False)
        first_match = sa.search_first(pattern)
        
        self.assertEqual(first_match, 4)
    
    def test_search_first_no_match(self):
        """Test search_first when pattern doesn't exist."""
        text = "AAAAAAA"
        pattern = "CGT"
        
        sa = SuffixArray(text, verbose=False)
        first_match = sa.search_first(pattern)
        
        self.assertIsNone(first_match)
    
    def test_empty_pattern(self):
        """Test with empty pattern."""
        text = "ACGTACGT"
        pattern = ""
        
        sa = SuffixArray(text, verbose=False)
        matches = sa.search(pattern)
        
        self.assertEqual(matches, [])
    
    def test_pattern_longer_than_text(self):
        """Test when pattern is longer than text."""
        text = "ACGT"
        pattern = "ACGTACGTACGT"
        
        sa = SuffixArray(text, verbose=False)
        matches = sa.search(pattern)
        
        self.assertEqual(matches, [])


class TestSuffixArrayEdgeCases(unittest.TestCase):
    """Test edge cases and special scenarios."""
    
    def test_single_character_text(self):
        """Test with single character text."""
        text = "A"
        pattern = "A"
        
        sa = SuffixArray(text, verbose=False)
        matches = sa.search(pattern)
        
        self.assertEqual(matches, [0])
    
    def test_single_character_pattern(self):
        """Test with single character pattern."""
        text = "ACGTACGT"
        pattern = "A"
        
        sa = SuffixArray(text, verbose=False)
        matches = sa.search(pattern)
        
        self.assertEqual(sorted(matches), [0, 4])
    
    def test_entire_text_match(self):
        """Test when pattern is entire text."""
        text = "ACGTACGT"
        pattern = "ACGTACGT"
        
        sa = SuffixArray(text, verbose=False)
        matches = sa.search(pattern)
        
        self.assertEqual(matches, [0])
    
    def test_repeated_character(self):
        """Test text with repeated characters."""
        text = "AAAAAAA"
        pattern = "A"
        
        sa = SuffixArray(text, verbose=False)
        matches = sa.search(pattern)
        
        self.assertEqual(len(matches), 7)
    
    def test_all_different_characters(self):
        """Test text with no repeats."""
        text = "ACGT"
        pattern = "AC"
        
        sa = SuffixArray(text, verbose=False)
        matches = sa.search(pattern)
        
        self.assertEqual(matches, [0])


class TestSuffixArrayRepeats(unittest.TestCase):
    """Test repeat discovery functionality."""
    
    def test_find_repeats(self):
        """Test finding repeats."""
        text = "AGATTTAGATTAGATTA"
        
        sa = SuffixArray(text, verbose=False)
        repeats = sa.find_longest_repeats(min_length=4)
        
        self.assertTrue(len(repeats) > 0)
        # Check that repeats have required fields
        for repeat in repeats:
            self.assertIn('length', repeat)
            self.assertIn('substring', repeat)
            self.assertIn('positions', repeat)
            self.assertIn('count', repeat)
            self.assertGreaterEqual(repeat['length'], 4)
            self.assertGreaterEqual(repeat['count'], 2)
    
    def test_no_repeats(self):
        """Test text with no repeats."""
        text = "ACGT"
        
        sa = SuffixArray(text, verbose=False)
        repeats = sa.find_longest_repeats(min_length=2)
        
        self.assertEqual(repeats, [])
    
    def test_repeat_min_length(self):
        """Test repeat min_length parameter."""
        text = "AAAAAAAAAA"
        
        sa = SuffixArray(text, verbose=False)
        repeats = sa.find_longest_repeats(min_length=3)
        
        # All repeats should be at least length 3
        for repeat in repeats:
            self.assertGreaterEqual(repeat['length'], 3)


class TestSuffixArrayCorrectness(unittest.TestCase):
    """Test correctness against naive search."""
    
    def naive_search(self, text: str, pattern: str):
        """Naive pattern matching for comparison."""
        matches = []
        for i in range(len(text) - len(pattern) + 1):
            if text[i:i+len(pattern)] == pattern:
                matches.append(i)
        return matches
    
    def test_correctness_random(self):
        """Test correctness on random sequences."""
        generator = DNAGenerator(seed=42)
        
        for _ in range(10):
            text = generator.generate_random_sequence(1000)
            pattern = generator.generate_pattern(10)
            
            sa = SuffixArray(text, verbose=False)
            sa_matches = sorted(sa.search(pattern))
            naive_matches = sorted(self.naive_search(text, pattern))
            
            self.assertEqual(sa_matches, naive_matches,
                           f"Mismatch for pattern {pattern}")
    
    def test_correctness_various_lengths(self):
        """Test correctness for various pattern lengths."""
        generator = DNAGenerator(seed=42)
        text = generator.generate_random_sequence(5000)
        
        sa = SuffixArray(text, verbose=False)
        
        for pattern_len in [4, 8, 16, 32]:
            pattern = generator.generate_pattern(pattern_len)
            
            sa_matches = sorted(sa.search(pattern))
            naive_matches = sorted(self.naive_search(text, pattern))
            
            self.assertEqual(sa_matches, naive_matches,
                           f"Mismatch for pattern length {pattern_len}")


class TestSuffixArrayPerformance(unittest.TestCase):
    """Test performance characteristics."""
    
    def test_statistics(self):
        """Test that statistics are collected."""
        text = "ACGTACGTACGT"
        pattern = "ACGT"
        
        sa = SuffixArray(text, verbose=False)
        sa.search(pattern)
        
        stats = sa.get_statistics()
        
        self.assertIn('text_length', stats)
        self.assertIn('pattern_length', stats)
        self.assertIn('preprocessing_time', stats)
        self.assertIn('memory_footprint_bytes', stats)
        self.assertIn('comparisons', stats)
        
        self.assertEqual(stats['text_length'], len(text))
        self.assertEqual(stats['pattern_length'], len(pattern))
        self.assertGreater(stats['preprocessing_time'], 0)
        self.assertGreater(stats['memory_footprint_bytes'], 0)


class TestConvenienceFunction(unittest.TestCase):
    """Test convenience functions."""
    
    def test_suffix_array_search_function(self):
        """Test standalone search function."""
        text = "ACGTACGTACGT"
        pattern = "ACGT"
        
        matches = suffix_array_search(text, pattern, verbose=False)
        
        self.assertEqual(sorted(matches), [0, 4, 8])


def run_tests():
    """Run all tests."""
    unittest.main(argv=[''], verbosity=2, exit=False)


if __name__ == '__main__':
    print("=" * 70)
    print("RUNNING SUFFIX ARRAY TEST SUITE")
    print("=" * 70)
    print()
    
    unittest.main(verbosity=2)
