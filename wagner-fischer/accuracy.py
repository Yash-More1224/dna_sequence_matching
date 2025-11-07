"""
Accuracy evaluation for Wagner-Fischer algorithm.
Measures precision, recall, F1 score on synthetic and real data.
"""

from typing import List, Tuple, Dict
from dataclasses import dataclass, asdict
import pandas as pd
import numpy as np
from pathlib import Path
import json

from wf_search import PatternSearcher, Match
from data_loader import SyntheticDataGenerator


@dataclass
class AccuracyMetrics:
    """Accuracy evaluation metrics."""
    test_name: str
    edit_distance_threshold: int
    
    true_positives: int
    false_positives: int
    false_negatives: int
    true_negatives: int
    
    precision: float
    recall: float
    f1_score: float
    accuracy: float
    
    total_patterns: int
    total_matches_found: int
    total_ground_truth: int


class AccuracyEvaluator:
    """
    Evaluate accuracy of Wagner-Fischer pattern matching.
    """
    
    def __init__(self, output_dir: str = "results/accuracy"):
        """
        Initialize evaluator.
        
        Args:
            output_dir: Directory to save results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results: List[AccuracyMetrics] = []
    
    def evaluate_synthetic_mutations(self,
                                    pattern_length: int = 50,
                                    text_length: int = 10000,
                                    num_patterns: int = 100,
                                    mutation_rate: float = 0.02,
                                    max_distance: int = 2) -> AccuracyMetrics:
        """
        Evaluate on synthetic data with known mutations.
        
        Args:
            pattern_length: Length of patterns
            text_length: Length of text
            num_patterns: Number of patterns to test
            mutation_rate: Mutation rate for generating mutated patterns
            max_distance: Maximum edit distance threshold
            
        Returns:
            Accuracy metrics
        """
        generator = SyntheticDataGenerator(seed=42)
        searcher = PatternSearcher(max_distance=max_distance)
        
        # Generate text
        text = generator.generate_random_sequence(text_length)
        
        true_positives = 0
        false_positives = 0
        false_negatives = 0
        
        ground_truth_positions = set()
        found_positions = set()
        
        print(f"Evaluating synthetic mutations (k={max_distance})...")
        
        for i in range(num_patterns):
            # Generate random pattern
            pattern = generator.generate_random_sequence(pattern_length)
            
            # Insert pattern at known position
            insert_pos = np.random.randint(0, text_length - pattern_length)
            text = text[:insert_pos] + pattern + text[insert_pos + pattern_length:]
            ground_truth_positions.add(insert_pos)
            
            # Mutate the pattern
            mutated_pattern, _ = generator.mutate_sequence(
                pattern,
                substitution_rate=mutation_rate,
                insertion_rate=mutation_rate / 2,
                deletion_rate=mutation_rate / 2
            )
            
            # Search for mutated pattern
            matches = searcher.search(mutated_pattern, text)
            
            for match in matches:
                found_positions.add(match.position)
        
        # Calculate metrics
        # A match is TP if it's close to a ground truth position (within pattern_length)
        for found_pos in found_positions:
            is_tp = any(abs(found_pos - gt_pos) <= pattern_length 
                       for gt_pos in ground_truth_positions)
            if is_tp:
                true_positives += 1
            else:
                false_positives += 1
        
        # Patterns we didn't find
        for gt_pos in ground_truth_positions:
            is_found = any(abs(found_pos - gt_pos) <= pattern_length 
                          for found_pos in found_positions)
            if not is_found:
                false_negatives += 1
        
        # Calculate metrics
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # True negatives are hard to define in pattern matching
        true_negatives = 0
        accuracy = (true_positives + true_negatives) / (true_positives + true_negatives + false_positives + false_negatives) if (true_positives + true_negatives + false_positives + false_negatives) > 0 else 0
        
        metrics = AccuracyMetrics(
            test_name="synthetic_mutations",
            edit_distance_threshold=max_distance,
            true_positives=true_positives,
            false_positives=false_positives,
            false_negatives=false_negatives,
            true_negatives=true_negatives,
            precision=precision,
            recall=recall,
            f1_score=f1,
            accuracy=accuracy,
            total_patterns=num_patterns,
            total_matches_found=len(found_positions),
            total_ground_truth=len(ground_truth_positions)
        )
        
        self.results.append(metrics)
        
        print(f"  Precision: {precision:.3f}")
        print(f"  Recall: {recall:.3f}")
        print(f"  F1: {f1:.3f}")
        
        return metrics
    
    def evaluate_exact_matching(self,
                               pattern_lengths: List[int] = None,
                               text_length: int = 10000,
                               num_tests: int = 50) -> List[AccuracyMetrics]:
        """
        Evaluate exact matching accuracy (should be 100% precision/recall).
        
        Args:
            pattern_lengths: List of pattern lengths to test
            text_length: Length of text
            num_tests: Number of tests per pattern length
            
        Returns:
            List of accuracy metrics
        """
        if pattern_lengths is None:
            pattern_lengths = [10, 20, 50, 100]
        
        generator = SyntheticDataGenerator(seed=42)
        searcher = PatternSearcher(max_distance=0)  # Exact match
        
        print("Evaluating exact matching accuracy...")
        
        results = []
        
        for plen in pattern_lengths:
            tp = fp = fn = 0
            
            for _ in range(num_tests):
                # Generate text and pattern
                text = generator.generate_random_sequence(text_length)
                pattern = generator.generate_random_sequence(plen)
                
                # Insert pattern at known position
                insert_pos = np.random.randint(0, text_length - plen)
                text = text[:insert_pos] + pattern + text[insert_pos + plen:]
                
                # Search
                matches = searcher.search(pattern, text)
                
                # Check if we found it
                found_at_correct_pos = any(m.position == insert_pos for m in matches)
                
                if found_at_correct_pos:
                    tp += 1
                else:
                    fn += 1
                
                # Count false positives (matches at wrong positions)
                fp += sum(1 for m in matches if m.position != insert_pos)
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            metrics = AccuracyMetrics(
                test_name=f"exact_matching_plen_{plen}",
                edit_distance_threshold=0,
                true_positives=tp,
                false_positives=fp,
                false_negatives=fn,
                true_negatives=0,
                precision=precision,
                recall=recall,
                f1_score=f1,
                accuracy=tp / num_tests,
                total_patterns=num_tests,
                total_matches_found=tp + fp,
                total_ground_truth=num_tests
            )
            
            self.results.append(metrics)
            results.append(metrics)
            
            print(f"  Pattern length {plen}: P={precision:.3f}, R={recall:.3f}, F1={f1:.3f}")
        
        return results
    
    def evaluate_threshold_sensitivity(self,
                                      pattern_length: int = 50,
                                      text_length: int = 10000,
                                      mutation_rate: float = 0.02,
                                      thresholds: List[int] = None) -> List[AccuracyMetrics]:
        """
        Evaluate how accuracy changes with edit distance threshold.
        
        Args:
            pattern_length: Length of patterns
            text_length: Length of text
            mutation_rate: Mutation rate
            thresholds: List of edit distance thresholds
            
        Returns:
            List of accuracy metrics
        """
        if thresholds is None:
            thresholds = [0, 1, 2, 3, 5, 7, 10]
        
        print("Evaluating threshold sensitivity...")
        
        results = []
        
        for threshold in thresholds:
            metrics = self.evaluate_synthetic_mutations(
                pattern_length=pattern_length,
                text_length=text_length,
                num_patterns=50,
                mutation_rate=mutation_rate,
                max_distance=threshold
            )
            results.append(metrics)
        
        return results
    
    def save_results(self, filename: str = "accuracy_results.csv"):
        """
        Save accuracy results.
        
        Args:
            filename: Output filename
        """
        if not self.results:
            print("No results to save")
            return
        
        df = pd.DataFrame([asdict(r) for r in self.results])
        output_path = self.output_dir / filename
        df.to_csv(output_path, index=False)
        print(f"Accuracy results saved to {output_path}")
        
        # Also save as JSON
        json_path = self.output_dir / filename.replace('.csv', '.json')
        df.to_json(json_path, orient='records', indent=2)
        print(f"Results also saved to {json_path}")
    
    def generate_confusion_matrix_data(self) -> pd.DataFrame:
        """
        Generate confusion matrix summary from results.
        
        Returns:
            DataFrame with confusion matrix data
        """
        data = []
        for result in self.results:
            data.append({
                'test_name': result.test_name,
                'threshold': result.edit_distance_threshold,
                'TP': result.true_positives,
                'FP': result.false_positives,
                'FN': result.false_negatives,
                'TN': result.true_negatives
            })
        
        return pd.DataFrame(data)
    
    def run_full_evaluation(self):
        """Run complete accuracy evaluation suite."""
        print("=" * 60)
        print("Running Wagner-Fischer Accuracy Evaluation")
        print("=" * 60)
        
        # Test 1: Exact matching
        self.evaluate_exact_matching(
            pattern_lengths=[10, 20, 50, 100],
            text_length=10000,
            num_tests=50
        )
        
        # Test 2: Threshold sensitivity
        self.evaluate_threshold_sensitivity(
            pattern_length=50,
            text_length=10000,
            mutation_rate=0.02,
            thresholds=[0, 1, 2, 3, 5]
        )
        
        # Save results
        self.save_results()
        
        # Save confusion matrix
        cm_df = self.generate_confusion_matrix_data()
        cm_path = self.output_dir / "confusion_matrix.csv"
        cm_df.to_csv(cm_path, index=False)
        print(f"Confusion matrix saved to {cm_path}")
        
        print("\n" + "=" * 60)
        print("Accuracy evaluation complete!")
        print("=" * 60)
