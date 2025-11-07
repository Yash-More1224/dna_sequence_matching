"""
Evaluation utilities for comparing KMP with Python's re module.

This module provides functions to calculate accuracy metrics, compare
performance, and validate correctness of pattern matching algorithms.
"""

import re
from typing import List, Dict, Set, Tuple, Optional
from dataclasses import dataclass, asdict

from .kmp_algorithm import KMP, kmp_search
from .benchmarking import measure_time_once, measure_memory_usage


@dataclass
class AccuracyMetrics:
    """
    Container for accuracy metrics.
    
    Attributes:
        true_positives: Number of correct matches found
        false_positives: Number of incorrect matches found
        false_negatives: Number of missed matches
        precision: Precision score (TP / (TP + FP))
        recall: Recall score (TP / (TP + FN))
        f1_score: F1 score (harmonic mean of precision and recall)
        accuracy: Overall accuracy
    """
    true_positives: int
    false_positives: int
    false_negatives: int
    precision: float
    recall: float
    f1_score: float
    accuracy: float
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)
    
    def __repr__(self) -> str:
        """String representation."""
        return (f"AccuracyMetrics(P={self.precision:.4f}, "
                f"R={self.recall:.4f}, F1={self.f1_score:.4f})")


@dataclass
class ComparisonResult:
    """
    Container for algorithm comparison results.
    
    Attributes:
        kmp_time: KMP execution time
        re_time: Python re execution time
        speedup: Speedup factor (re_time / kmp_time)
        kmp_memory: KMP memory usage
        re_memory: Python re memory usage
        kmp_matches: Number of matches found by KMP
        re_matches: Number of matches found by re
        matches_agree: Whether both found the same matches
        accuracy: Accuracy metrics if ground truth available
    """
    kmp_time: float
    re_time: float
    speedup: float
    kmp_memory: int
    re_memory: int
    kmp_matches: int
    re_matches: int
    matches_agree: bool
    accuracy: Optional[AccuracyMetrics] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        result = asdict(self)
        if self.accuracy:
            result['accuracy'] = self.accuracy.to_dict()
        return result
    
    def __repr__(self) -> str:
        """String representation."""
        return (f"ComparisonResult(speedup={self.speedup:.2f}x, "
                f"matches_agree={self.matches_agree})")


def calculate_accuracy_metrics(predicted: List[int], 
                               ground_truth: List[int]) -> AccuracyMetrics:
    """
    Calculate accuracy metrics by comparing predicted matches to ground truth.
    
    Args:
        predicted: List of predicted match positions
        ground_truth: List of ground truth match positions
        
    Returns:
        AccuracyMetrics object
        
    Example:
        >>> predicted = [0, 2, 4]
        >>> ground_truth = [0, 2, 3, 4]
        >>> metrics = calculate_accuracy_metrics(predicted, ground_truth)
        >>> print(metrics.precision, metrics.recall)
    """
    pred_set = set(predicted)
    truth_set = set(ground_truth)
    
    # Calculate TP, FP, FN
    true_positives = len(pred_set & truth_set)
    false_positives = len(pred_set - truth_set)
    false_negatives = len(truth_set - pred_set)
    
    # Calculate metrics
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    total = len(truth_set)
    accuracy = true_positives / total if total > 0 else 0.0
    
    return AccuracyMetrics(
        true_positives=true_positives,
        false_positives=false_positives,
        false_negatives=false_negatives,
        precision=precision,
        recall=recall,
        f1_score=f1_score,
        accuracy=accuracy
    )


def validate_against_re(text: str, pattern: str, kmp_matches: List[int]) -> bool:
    """
    Validate KMP results against Python's re module.
    
    Args:
        text: The text searched
        pattern: The pattern searched for
        kmp_matches: List of match positions from KMP
        
    Returns:
        True if results match, False otherwise
    """
    re_matches = [m.start() for m in re.finditer(re.escape(pattern), text)]
    return kmp_matches == re_matches


def compare_with_re(text: str, 
                   pattern: str,
                   ground_truth: Optional[List[int]] = None) -> ComparisonResult:
    """
    Compare KMP algorithm with Python's re module.
    
    Args:
        text: Text to search in
        pattern: Pattern to search for
        ground_truth: Optional ground truth match positions for accuracy calculation
        
    Returns:
        ComparisonResult object with detailed comparison
        
    Example:
        >>> text = "ATCGATCG" * 1000
        >>> pattern = "ATCG"
        >>> result = compare_with_re(text, pattern)
        >>> print(f"KMP is {result.speedup:.2f}x vs re")
    """
    # Benchmark KMP
    kmp = KMP(pattern)
    kmp_matches, kmp_time = measure_time_once(kmp.search, text)
    _, _, kmp_memory = measure_memory_usage(kmp.search, text)
    
    # Benchmark re
    def re_search(t):
        return [m.start() for m in re.finditer(re.escape(pattern), t)]
    
    re_matches, re_time = measure_time_once(re_search, text)
    _, _, re_memory = measure_memory_usage(re_search, text)
    
    # Compare results
    matches_agree = sorted(kmp_matches) == sorted(re_matches)
    speedup = re_time / kmp_time if kmp_time > 0 else float('inf')
    
    # Calculate accuracy if ground truth provided
    accuracy = None
    if ground_truth is not None:
        accuracy = calculate_accuracy_metrics(kmp_matches, ground_truth)
    
    return ComparisonResult(
        kmp_time=kmp_time,
        re_time=re_time,
        speedup=speedup,
        kmp_memory=kmp_memory,
        re_memory=re_memory,
        kmp_matches=len(kmp_matches),
        re_matches=len(re_matches),
        matches_agree=matches_agree,
        accuracy=accuracy
    )


def test_correctness(test_cases: List[Tuple[str, str, List[int]]]) -> Dict[str, any]:
    """
    Test correctness of KMP implementation on multiple test cases.
    
    Args:
        test_cases: List of (text, pattern, expected_matches) tuples
        
    Returns:
        Dictionary with test results:
            - total: Total number of tests
            - passed: Number of passed tests
            - failed: Number of failed tests
            - success_rate: Percentage of passed tests
            - failures: List of failed test details
            
    Example:
        >>> from kmp.synthetic_data import generate_test_cases
        >>> test_cases = generate_test_cases()
        >>> results = test_correctness(test_cases)
        >>> print(f"Passed {results['passed']}/{results['total']} tests")
    """
    total = len(test_cases)
    passed = 0
    failures = []
    
    for i, (text, pattern, expected) in enumerate(test_cases):
        try:
            matches = kmp_search(text, pattern)
            
            if sorted(matches) == sorted(expected):
                passed += 1
            else:
                failures.append({
                    'test_id': i,
                    'text': text[:50] + '...' if len(text) > 50 else text,
                    'pattern': pattern,
                    'expected': expected,
                    'got': matches,
                    'reason': 'Matches do not match expected'
                })
        except Exception as e:
            failures.append({
                'test_id': i,
                'text': text[:50] + '...' if len(text) > 50 else text,
                'pattern': pattern,
                'expected': expected,
                'got': None,
                'reason': f'Exception: {str(e)}'
            })
    
    return {
        'total': total,
        'passed': passed,
        'failed': total - passed,
        'success_rate': (passed / total * 100) if total > 0 else 0.0,
        'failures': failures
    }


def test_against_re(text: str, patterns: List[str]) -> Dict[str, bool]:
    """
    Test KMP against re for multiple patterns.
    
    Args:
        text: Text to search in
        patterns: List of patterns to test
        
    Returns:
        Dictionary mapping each pattern to whether KMP matches re
    """
    results = {}
    
    for pattern in patterns:
        kmp_matches = kmp_search(text, pattern)
        re_matches = [m.start() for m in re.finditer(re.escape(pattern), text)]
        results[pattern] = (sorted(kmp_matches) == sorted(re_matches))
    
    return results


def benchmark_comparison(text_sizes: List[int],
                        pattern_length: int = 50,
                        seed: int = 42) -> List[ComparisonResult]:
    """
    Run comparison benchmarks across different text sizes.
    
    Args:
        text_sizes: List of text sizes to test
        pattern_length: Length of pattern to use
        seed: Random seed for reproducibility
        
    Returns:
        List of ComparisonResult objects for each text size
    """
    import random
    from .config import DNA_BASES
    
    results = []
    
    for size in text_sizes:
        # Generate random text and pattern
        random.seed(seed)
        text = ''.join(random.choice(DNA_BASES) for _ in range(size))
        pattern = ''.join(random.choice(DNA_BASES) for _ in range(pattern_length))
        
        # Compare
        result = compare_with_re(text, pattern)
        results.append(result)
        
        print(f"Text size {size}: KMP={result.kmp_time:.6f}s, re={result.re_time:.6f}s, "
              f"speedup={result.speedup:.2f}x")
    
    return results


def calculate_confusion_matrix(predicted: List[int], 
                               ground_truth: List[int],
                               text_length: int) -> Dict[str, int]:
    """
    Calculate confusion matrix for position-based matching.
    
    Args:
        predicted: Predicted match positions
        ground_truth: Ground truth match positions
        text_length: Length of the text
        
    Returns:
        Dictionary with TP, FP, TN, FN counts
    """
    pred_set = set(predicted)
    truth_set = set(ground_truth)
    
    tp = len(pred_set & truth_set)
    fp = len(pred_set - truth_set)
    fn = len(truth_set - pred_set)
    tn = text_length - tp - fp - fn  # Positions that are correctly not matches
    
    return {
        'true_positives': tp,
        'false_positives': fp,
        'true_negatives': tn,
        'false_negatives': fn
    }


def evaluate_sensitivity_to_alphabet(text: str, 
                                     pattern: str,
                                     modifications: List[str]) -> Dict[str, ComparisonResult]:
    """
    Evaluate how algorithm performance changes with different alphabets.
    
    Args:
        text: Base text
        pattern: Base pattern
        modifications: List of modification types to test
        
    Returns:
        Dictionary mapping modification type to ComparisonResult
    """
    results = {}
    
    # Test with original
    results['original'] = compare_with_re(text, pattern)
    
    # Test with modifications
    for mod in modifications:
        if mod == 'with_n':
            # Add N bases (unknown bases)
            mod_text = text.replace('A', 'N', len(text) // 20)
            results[mod] = compare_with_re(mod_text, pattern)
        elif mod == 'lowercase':
            # Test with lowercase
            results[mod] = compare_with_re(text.lower(), pattern.lower())
    
    return results


def print_comparison_summary(result: ComparisonResult) -> None:
    """
    Print a formatted summary of comparison results.
    
    Args:
        result: ComparisonResult object to summarize
    """
    print("\n" + "="*60)
    print("ALGORITHM COMPARISON SUMMARY")
    print("="*60)
    print(f"KMP Time:        {result.kmp_time:.6f} seconds")
    print(f"Python re Time:  {result.re_time:.6f} seconds")
    print(f"Speedup:         {result.speedup:.2f}x {'(KMP faster)' if result.speedup > 1 else '(re faster)'}")
    print(f"\nKMP Memory:      {result.kmp_memory / 1024:.2f} KB")
    print(f"Python re Memory:{result.re_memory / 1024:.2f} KB")
    print(f"\nKMP Matches:     {result.kmp_matches}")
    print(f"re Matches:      {result.re_matches}")
    print(f"Matches Agree:   {'✓ Yes' if result.matches_agree else '✗ No'}")
    
    if result.accuracy:
        print(f"\nAccuracy Metrics:")
        print(f"  Precision:     {result.accuracy.precision:.4f}")
        print(f"  Recall:        {result.accuracy.recall:.4f}")
        print(f"  F1 Score:      {result.accuracy.f1_score:.4f}")
    
    print("="*60 + "\n")
