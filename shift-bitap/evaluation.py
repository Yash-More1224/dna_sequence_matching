"""
Evaluation Framework for Approximate Matching Accuracy
======================================================

This module provides tools for evaluating the accuracy of approximate
pattern matching algorithms, particularly for the Shift-Or/Bitap algorithm.

Metrics:
- Precision: TP / (TP + FP)
- Recall: TP / (TP + FN)
- F1 Score: 2 * (Precision * Recall) / (Precision + Recall)

Test scenarios:
- Synthetic mutations with known ground truth
- Varying edit distances
- Edge cases and corner cases

Author: DNA Sequence Matching Project
Date: November 2025
"""

from typing import List, Tuple, Dict, Set, Optional
from dataclasses import dataclass, asdict
import re


@dataclass
class AccuracyMetrics:
    """Container for accuracy evaluation metrics."""
    true_positives: int
    false_positives: int
    false_negatives: int
    true_negatives: int
    
    precision: float
    recall: float
    f1_score: float
    accuracy: float
    
    num_expected_matches: int
    num_found_matches: int
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)
    
    def __str__(self) -> str:
        """Human-readable representation."""
        lines = [
            "Accuracy Metrics",
            "=" * 50,
            f"True Positives:  {self.true_positives:6d}",
            f"False Positives: {self.false_positives:6d}",
            f"False Negatives: {self.false_negatives:6d}",
            f"True Negatives:  {self.true_negatives:6d}",
            "",
            f"Precision: {self.precision:.4f}",
            f"Recall:    {self.recall:.4f}",
            f"F1 Score:  {self.f1_score:.4f}",
            f"Accuracy:  {self.accuracy:.4f}",
            "",
            f"Expected matches: {self.num_expected_matches}",
            f"Found matches:    {self.num_found_matches}",
        ]
        return "\n".join(lines)


class ApproximateMatchEvaluator:
    """
    Evaluates the accuracy of approximate pattern matching.
    """
    
    @staticmethod
    def calculate_edit_distance(s1: str, s2: str) -> int:
        """
        Calculate Levenshtein edit distance between two strings.
        
        Args:
            s1: First string
            s2: Second string
            
        Returns:
            Edit distance
        """
        if len(s1) < len(s2):
            return ApproximateMatchEvaluator.calculate_edit_distance(s2, s1)
        
        if len(s2) == 0:
            return len(s1)
        
        # Use dynamic programming
        previous_row = range(len(s2) + 1)
        
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                # Cost of insertions, deletions, or substitutions
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]
    
    @staticmethod
    def is_approximate_match(pattern: str, text_segment: str, max_errors: int) -> bool:
        """
        Check if text segment is an approximate match of pattern.
        
        Args:
            pattern: Pattern string
            text_segment: Text segment to check
            max_errors: Maximum allowed edit distance
            
        Returns:
            True if segment matches within max_errors
        """
        distance = ApproximateMatchEvaluator.calculate_edit_distance(pattern, text_segment)
        return distance <= max_errors
    
    @staticmethod
    def find_ground_truth_matches(pattern: str, text: str, max_errors: int) -> List[Tuple[int, int]]:
        """
        Find all true approximate matches using brute force (ground truth).
        
        This is the gold standard for comparison, but slow for large texts.
        
        Args:
            pattern: Pattern to search for
            text: Text to search in
            max_errors: Maximum edit distance
            
        Returns:
            List of (position, actual_errors) tuples
        """
        matches = []
        pattern_len = len(pattern)
        
        # Check all possible positions
        for i in range(len(text) - pattern_len + max_errors + 1):
            # Check segments of varying lengths (due to possible insertions/deletions)
            for length in range(pattern_len - max_errors, pattern_len + max_errors + 1):
                if i + length <= len(text):
                    segment = text[i:i+length]
                    distance = ApproximateMatchEvaluator.calculate_edit_distance(pattern, segment)
                    
                    if distance <= max_errors:
                        matches.append((i, distance))
                        break  # Take first valid match at this position
        
        return matches
    
    @staticmethod
    def evaluate_matches(found_matches: List[Tuple[int, int]], 
                        ground_truth: List[Tuple[int, int]],
                        tolerance: int = 0) -> AccuracyMetrics:
        """
        Evaluate found matches against ground truth.
        
        Args:
            found_matches: List of (position, errors) from algorithm
            ground_truth: List of (position, errors) from ground truth
            tolerance: Position tolerance (matches within tolerance are considered same)
            
        Returns:
            AccuracyMetrics with precision, recall, F1, etc.
        """
        # Convert to sets of positions for easier comparison
        found_positions = {pos for pos, _ in found_matches}
        true_positions = {pos for pos, _ in ground_truth}
        
        # Account for tolerance
        def positions_match(pos1: int, pos2: int, tol: int) -> bool:
            return abs(pos1 - pos2) <= tol
        
        # Calculate TP, FP, FN
        true_positives = 0
        matched_true = set()
        matched_found = set()
        
        for found_pos in found_positions:
            for true_pos in true_positions:
                if true_pos not in matched_true and positions_match(found_pos, true_pos, tolerance):
                    true_positives += 1
                    matched_true.add(true_pos)
                    matched_found.add(found_pos)
                    break
        
        false_positives = len(found_positions) - true_positives
        false_negatives = len(true_positions) - true_positives
        true_negatives = 0  # Not applicable for pattern matching
        
        # Calculate metrics
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        total = true_positives + false_positives + false_negatives + true_negatives
        accuracy = (true_positives + true_negatives) / total if total > 0 else 0.0
        
        return AccuracyMetrics(
            true_positives=true_positives,
            false_positives=false_positives,
            false_negatives=false_negatives,
            true_negatives=true_negatives,
            precision=precision,
            recall=recall,
            f1_score=f1_score,
            accuracy=accuracy,
            num_expected_matches=len(ground_truth),
            num_found_matches=len(found_matches)
        )
    
    @staticmethod
    def evaluate_algorithm(algorithm_search_func, pattern: str, text: str, 
                          max_errors: int, tolerance: int = 0) -> AccuracyMetrics:
        """
        Evaluate an algorithm's approximate matching accuracy.
        
        Args:
            algorithm_search_func: Function that takes (text, max_errors) and returns matches
            pattern: Pattern being searched
            text: Text to search in
            max_errors: Maximum edit distance
            tolerance: Position tolerance for matching
            
        Returns:
            AccuracyMetrics
        """
        # Get algorithm results
        found_matches = algorithm_search_func(text, max_errors)
        
        # Get ground truth
        ground_truth = ApproximateMatchEvaluator.find_ground_truth_matches(
            pattern, text, max_errors
        )
        
        # Evaluate
        return ApproximateMatchEvaluator.evaluate_matches(
            found_matches, ground_truth, tolerance
        )


class TestCaseGenerator:
    """
    Generate test cases for evaluating approximate matching.
    """
    
    @staticmethod
    def create_exact_match_test(pattern: str, num_copies: int = 5) -> Tuple[str, List[int]]:
        """
        Create a test case with exact matches.
        
        Args:
            pattern: Pattern to embed
            num_copies: Number of times to embed pattern
            
        Returns:
            (text, list of expected positions)
        """
        from data_loader import SyntheticDataGenerator
        
        # Create background
        background = SyntheticDataGenerator.generate_random_sequence(1000, seed=42)
        text = list(background)
        positions = []
        
        # Embed exact copies
        for i in range(num_copies):
            pos = 100 + i * 150
            if pos + len(pattern) < len(text):
                for j, char in enumerate(pattern):
                    text[pos + j] = char
                positions.append(pos)
        
        return ''.join(text), positions
    
    @staticmethod
    def create_substitution_test(pattern: str, num_errors: int = 1, 
                                num_copies: int = 5) -> Tuple[str, List[Tuple[int, int]]]:
        """
        Create a test case with substitution mutations.
        
        Args:
            pattern: Pattern to mutate and embed
            num_errors: Number of substitutions per copy
            num_copies: Number of mutated copies
            
        Returns:
            (text, list of (position, errors) tuples)
        """
        from data_loader import SyntheticDataGenerator
        
        background = SyntheticDataGenerator.generate_random_sequence(1000, seed=42)
        text = list(background)
        positions = []
        
        for i in range(num_copies):
            pos = 100 + i * 150
            if pos + len(pattern) < len(text):
                # Create mutated pattern
                mutated = SyntheticDataGenerator.introduce_substitutions(
                    pattern, rate=num_errors/len(pattern), seed=42+i
                )
                
                # Embed it
                for j, char in enumerate(mutated):
                    text[pos + j] = char
                
                # Calculate actual errors
                actual_errors = sum(1 for a, b in zip(pattern, mutated) if a != b)
                positions.append((pos, actual_errors))
        
        return ''.join(text), positions
    
    @staticmethod
    def create_indel_test(pattern: str, num_copies: int = 5) -> Tuple[str, List[Tuple[int, int]]]:
        """
        Create a test case with insertion/deletion mutations.
        
        Args:
            pattern: Pattern to mutate
            num_copies: Number of copies
            
        Returns:
            (text, list of (position, errors) tuples)
        """
        from data_loader import SyntheticDataGenerator
        
        background = SyntheticDataGenerator.generate_random_sequence(1000, seed=42)
        text_parts = [background[:100]]
        positions = []
        
        for i in range(num_copies):
            # Create mutated pattern with indels
            mutated = SyntheticDataGenerator.introduce_indels(
                pattern, insertion_rate=0.05, deletion_rate=0.05, seed=42+i
            )
            
            positions.append((len(''.join(text_parts)), 
                            abs(len(mutated) - len(pattern))))
            
            text_parts.append(mutated)
            text_parts.append(
                SyntheticDataGenerator.generate_random_sequence(150, seed=100+i)
            )
        
        return ''.join(text_parts), positions


if __name__ == "__main__":
    print("Evaluation Framework Demo")
    print("=" * 80)
    
    from algorithm import ShiftOrBitap
    from data_loader import SyntheticDataGenerator
    
    # Test 1: Exact matches
    print("\nTest 1: Exact Matches")
    print("-" * 80)
    
    pattern = "GATTACA"
    text, expected_positions = TestCaseGenerator.create_exact_match_test(pattern, num_copies=5)
    
    print(f"Pattern: {pattern}")
    print(f"Text length: {len(text)}")
    print(f"Expected positions: {expected_positions}")
    
    matcher = ShiftOrBitap(pattern)
    found = matcher.search_exact(text)
    print(f"Found positions: {found}")
    print(f"Match: {set(found) == set(expected_positions)}")
    
    # Test 2: Approximate matches with substitutions
    print("\n\nTest 2: Approximate Matches (1 substitution)")
    print("-" * 80)
    
    text2, expected_approx = TestCaseGenerator.create_substitution_test(pattern, num_errors=1, num_copies=3)
    print(f"Pattern: {pattern}")
    print(f"Text length: {len(text2)}")
    print(f"Expected matches: {expected_approx}")
    
    found_approx = matcher.search_approximate(text2, max_errors=1)
    print(f"Found matches: {found_approx}")
    
    # Evaluate accuracy
    evaluator = ApproximateMatchEvaluator()
    metrics = evaluator.evaluate_matches(found_approx, expected_approx, tolerance=2)
    print(f"\n{metrics}")
    
    # Test 3: Ground truth comparison
    print("\n\nTest 3: Ground Truth Comparison (small text)")
    print("-" * 80)
    
    small_pattern = "ACGT"
    small_text = "AACGTXACGTYACGT"
    
    print(f"Pattern: {small_pattern}")
    print(f"Text: {small_text}")
    
    # Find ground truth
    ground_truth = evaluator.find_ground_truth_matches(small_pattern, small_text, max_errors=1)
    print(f"\nGround truth (max 1 error): {ground_truth}")
    
    # Algorithm results
    matcher3 = ShiftOrBitap(small_pattern)
    found3 = matcher3.search_approximate(small_text, max_errors=1)
    print(f"Algorithm found: {found3}")
    
    # Evaluate
    metrics3 = evaluator.evaluate_matches(found3, ground_truth)
    print(f"\n{metrics3}")
