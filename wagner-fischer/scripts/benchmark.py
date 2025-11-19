"""
Comprehensive Benchmarking and Evaluation for Wagner-Fischer Algorithm
Measures performance, memory usage, accuracy, scalability, and robustness.
"""

import time
import json
import tracemalloc
import psutil
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict
import sys
import os

# Add parent directory to path to import wf module
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wf import WagnerFischer, EditDistanceResult


@dataclass
class PerformanceMetrics:
    """Performance measurement results."""
    total_runtime: float
    mean_latency: float
    median_latency: float
    std_latency: float
    min_latency: float
    max_latency: float
    throughput: float  # operations per second
    preprocessing_time: float
    peak_memory_mb: float
    peak_rss_mb: float


@dataclass
class AccuracyMetrics:
    """Accuracy measurement results."""
    true_positives: int
    false_positives: int
    false_negatives: int
    precision: float
    recall: float
    f1_score: float
    mutation_rate: float
    threshold: int


@dataclass
class Match:
    """Pattern match result."""
    position: int
    end_position: int
    edit_distance: int
    pattern_id: int


class WagnerFischerBenchmark:
    """
    Comprehensive benchmarking suite for Wagner-Fischer algorithm.
    """
    
    def __init__(self, output_dir: str = "results"):
        """
        Initialize benchmark suite.
        
        Args:
            output_dir: Directory to save results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.wf = WagnerFischer()
        self.process = psutil.Process()
    
    def measure_performance(self,
                          patterns: List[str],
                          target: str,
                          threshold: int,
                          num_runs: int = 5) -> PerformanceMetrics:
        """
        Measure performance metrics for pattern matching.
        
        Args:
            patterns: List of patterns to search
            target: Target sequence
            threshold: Edit distance threshold
            num_runs: Number of runs for statistics
            
        Returns:
            PerformanceMetrics object
        """
        latencies = []
        preprocessing_time = 0.0  # WF has no preprocessing
        
        # Start memory tracking
        tracemalloc.start()
        initial_rss = self.process.memory_info().rss / 1024 / 1024  # MB
        
        # Run benchmarks
        total_start = time.perf_counter()
        
        for run in range(num_runs):
            for pattern in patterns:
                start = time.perf_counter()
                
                # Perform sliding window search
                matches = self._sliding_window_search(pattern, target, threshold)
                
                latency = time.perf_counter() - start
                latencies.append(latency)
        
        total_runtime = time.perf_counter() - total_start
        
        # Get memory stats
        current, peak = tracemalloc.get_traced_memory()
        peak_memory_mb = peak / 1024 / 1024
        tracemalloc.stop()
        
        final_rss = self.process.memory_info().rss / 1024 / 1024
        peak_rss_mb = max(initial_rss, final_rss)
        
        # Calculate statistics
        latencies_array = np.array(latencies)
        throughput = len(latencies) / total_runtime if total_runtime > 0 else 0
        
        return PerformanceMetrics(
            total_runtime=total_runtime,
            mean_latency=float(np.mean(latencies_array)),
            median_latency=float(np.median(latencies_array)),
            std_latency=float(np.std(latencies_array)),
            min_latency=float(np.min(latencies_array)),
            max_latency=float(np.max(latencies_array)),
            throughput=throughput,
            preprocessing_time=preprocessing_time,
            peak_memory_mb=peak_memory_mb,
            peak_rss_mb=peak_rss_mb
        )
    
    def _sliding_window_search(self, 
                               pattern: str, 
                               target: str, 
                               threshold: int) -> List[Match]:
        """
        Optimized sliding window approximate pattern matching.
        
        Args:
            pattern: Pattern to search
            target: Target sequence
            threshold: Edit distance threshold
            
        Returns:
            List of Match objects
        """
        matches = []
        pattern_len = len(pattern)
        target_len = len(target)
        
        if pattern_len == 0 or target_len == 0:
            return matches
        
        # Use fixed window size for speed (pattern length)
        # This is a simplification but much faster
        window_size = pattern_len
        step_size = max(1, pattern_len // 4)  # Larger step for speed
        
        i = 0
        while i <= target_len - window_size:
            window = target[i:i + window_size]
            
            # Compute edit distance with threshold (faster banded approach)
            distance, within_threshold = self.wf.compute_with_threshold(
                pattern, window, threshold
            )
            
            if within_threshold:
                matches.append(Match(
                    position=i,
                    end_position=i + window_size,
                    edit_distance=int(distance),
                    pattern_id=-1
                ))
                i += window_size  # Skip overlapping matches
            else:
                i += step_size  # Use larger step when no match
        
        return matches
    
    def evaluate_accuracy(self,
                         ground_truth: Dict,
                         target: str,
                         threshold: int,
                         mutation_rate: float) -> AccuracyMetrics:
        """
        Evaluate accuracy against ground truth.
        
        Args:
            ground_truth: Ground truth data dictionary
            target: Target sequence
            threshold: Edit distance threshold
            mutation_rate: Mutation rate to evaluate
            
        Returns:
            AccuracyMetrics object
        """
        # Filter patterns by mutation rate
        test_patterns = [
            p for p in ground_truth['patterns']
            if abs(p['mutation_rate'] - mutation_rate) < 1e-6
        ]
        
        if not test_patterns:
            return AccuracyMetrics(0, 0, 0, 0.0, 0.0, 0.0, mutation_rate, threshold)
        
        # Build ground truth positions
        gt_positions = {}
        for p in test_patterns:
            pattern_id = p['pattern_id']
            gt_positions[pattern_id] = {
                'start': p['position_in_target'],
                'end': p['end_position_in_target'],
                'mutated': p['mutated']
            }
        
        # Search for each pattern
        true_positives = 0
        false_positives = 0
        false_negatives = 0
        matched_patterns = set()
        
        for pattern_id, gt_info in gt_positions.items():
            mutated_pattern = gt_info['mutated']
            gt_start = gt_info['start']
            gt_end = gt_info['end']
            
            # Search for pattern
            matches = self._sliding_window_search(mutated_pattern, target, threshold)
            
            # Check if any match overlaps with ground truth position
            found = False
            for match in matches:
                overlap = self._compute_overlap(
                    match.position, match.end_position,
                    gt_start, gt_end
                )
                
                if overlap >= 0.5:  # 50% overlap threshold
                    found = True
                    if pattern_id not in matched_patterns:
                        true_positives += 1
                        matched_patterns.add(pattern_id)
                    break
                else:
                    # This is a false positive
                    if not self._overlaps_any_ground_truth(
                        match.position, match.end_position, gt_positions
                    ):
                        false_positives += 1
            
            if not found:
                false_negatives += 1
        
        # Calculate metrics
        precision = true_positives / (true_positives + false_positives) \
            if (true_positives + false_positives) > 0 else 0.0
        recall = true_positives / (true_positives + false_negatives) \
            if (true_positives + false_negatives) > 0 else 0.0
        f1_score = 2 * precision * recall / (precision + recall) \
            if (precision + recall) > 0 else 0.0
        
        return AccuracyMetrics(
            true_positives=true_positives,
            false_positives=false_positives,
            false_negatives=false_negatives,
            precision=precision,
            recall=recall,
            f1_score=f1_score,
            mutation_rate=mutation_rate,
            threshold=threshold
        )
    
    def _compute_overlap(self, start1: int, end1: int, 
                        start2: int, end2: int) -> float:
        """
        Compute overlap ratio between two intervals.
        
        Returns:
            Overlap ratio (0.0 to 1.0)
        """
        overlap_start = max(start1, start2)
        overlap_end = min(end1, end2)
        
        if overlap_end <= overlap_start:
            return 0.0
        
        overlap_len = overlap_end - overlap_start
        shorter_interval = min(end1 - start1, end2 - start2)
        
        return overlap_len / shorter_interval if shorter_interval > 0 else 0.0
    
    def _overlaps_any_ground_truth(self, start: int, end: int, 
                                   gt_positions: Dict) -> bool:
        """
        Check if interval overlaps with any ground truth position.
        """
        for gt_info in gt_positions.values():
            if self._compute_overlap(start, end, 
                                    gt_info['start'], gt_info['end']) > 0:
                return True
        return False
    
    def run_scalability_experiment(self,
                                  target: str,
                                  base_patterns: List[str],
                                  threshold: int,
                                  pattern_counts: List[int]) -> List[Dict]:
        """
        Run scalability experiments with varying pattern counts.
        
        Args:
            target: Target sequence
            base_patterns: Base pattern set
            threshold: Edit distance threshold
            pattern_counts: List of pattern counts to test
            
        Returns:
            List of result dictionaries
        """
        results = []
        
        for count in pattern_counts:
            # Sample patterns
            patterns = base_patterns[:count] if count <= len(base_patterns) else base_patterns
            
            # Measure performance
            metrics = self.measure_performance(patterns, target, threshold, num_runs=3)
            
            results.append({
                'pattern_count': count,
                'total_runtime': metrics.total_runtime,
                'mean_latency': metrics.mean_latency,
                'throughput': metrics.throughput,
                'peak_memory_mb': metrics.peak_memory_mb
            })
            
            print(f"Scalability: {count} patterns -> {metrics.total_runtime:.3f}s, "
                  f"{metrics.throughput:.2f} ops/sec")
        
        return results
    
    def run_robustness_experiment(self,
                                 ground_truth: Dict,
                                 target: str,
                                 threshold: int) -> List[Dict]:
        """
        Run robustness experiments across mutation rates.
        
        Args:
            ground_truth: Ground truth data
            target: Target sequence
            threshold: Edit distance threshold
            
        Returns:
            List of result dictionaries
        """
        mutation_rates = ground_truth['mutation_rates']
        results = []
        
        for mutation_rate in mutation_rates:
            # Evaluate accuracy
            accuracy = self.evaluate_accuracy(ground_truth, target, threshold, mutation_rate)
            
            # Get patterns for this mutation rate
            patterns = [
                p['mutated'] for p in ground_truth['patterns']
                if abs(p['mutation_rate'] - mutation_rate) < 1e-6
            ]
            
            # Measure performance
            if patterns:
                perf = self.measure_performance(patterns[:20], target, threshold, num_runs=2)
                
                results.append({
                    'mutation_rate': mutation_rate,
                    'precision': accuracy.precision,
                    'recall': accuracy.recall,
                    'f1_score': accuracy.f1_score,
                    'mean_latency': perf.mean_latency,
                    'throughput': perf.throughput
                })
                
                print(f"Robustness: {mutation_rate:.1%} mutation -> "
                      f"F1={accuracy.f1_score:.3f}, latency={perf.mean_latency:.4f}s")
        
        return results
    
    def save_metrics(self, metrics: Dict, filename: str):
        """
        Save metrics to JSON file.
        
        Args:
            metrics: Metrics dictionary
            filename: Output filename
        """
        filepath = self.output_dir / filename
        with open(filepath, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"Metrics saved to {filepath}")


def load_fasta(filepath: str) -> Tuple[str, str]:
    """Load FASTA file."""
    seq_id = ""
    sequence = []
    
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                seq_id = line[1:].split()[0]
            else:
                sequence.append(line.upper())
    
    return seq_id, ''.join(sequence)


def load_ground_truth(filepath: str) -> Dict:
    """Load ground truth from JSON."""
    with open(filepath, 'r') as f:
        return json.load(f)


def main():
    """Run comprehensive Wagner-Fischer evaluation."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Wagner-Fischer Benchmark')
    parser.add_argument('--fasta', required=True, help='Path to FASTA file')
    parser.add_argument('--ground-truth', required=True, help='Path to ground truth JSON')
    parser.add_argument('--output-dir', default='results', help='Output directory')
    parser.add_argument('--threshold', type=int, default=5, 
                       help='Edit distance threshold')
    parser.add_argument('--pattern-counts', type=int, nargs='+',
                       default=[10, 20, 50, 100],
                       help='Pattern counts for scalability test')
    parser.add_argument('--target-length', type=int, default=10000,
                       help='Use first N bases of target (for speed)')
    
    args = parser.parse_args()
    
    # Load data
    print(f"Loading FASTA: {args.fasta}")
    target_id, target_full = load_fasta(args.fasta)
    target = target_full[:args.target_length]  # Use subset for faster evaluation
    print(f"Loaded: {target_id} ({len(target_full)} bp, using {len(target)} bp)")
    
    print(f"Loading ground truth: {args.ground_truth}")
    ground_truth = load_ground_truth(args.ground_truth)
    print(f"Loaded {len(ground_truth['patterns'])} patterns")
    
    # Initialize benchmark
    benchmark = WagnerFischerBenchmark(output_dir=args.output_dir)
    
    # Run experiments
    all_results = {
        'dataset': target_id,
        'target_length_full': len(target_full),
        'target_length_used': len(target),
        'threshold': args.threshold,
        'pattern_length': ground_truth['pattern_length']
    }
    
    # 1. Performance benchmark
    print("\n=== Performance Benchmark ===")
    test_patterns = [p['mutated'] for p in ground_truth['patterns'][:20]]  # Reduced for speed
    perf_metrics = benchmark.measure_performance(
        test_patterns, target, args.threshold, num_runs=3  # Reduced runs
    )
    all_results['performance'] = asdict(perf_metrics)
    print(f"Mean latency: {perf_metrics.mean_latency:.4f}s")
    print(f"Throughput: {perf_metrics.throughput:.2f} ops/sec")
    print(f"Peak memory: {perf_metrics.peak_memory_mb:.2f} MB")
    
    # 2. Accuracy evaluation
    print("\n=== Accuracy Evaluation ===")
    accuracy_results = []
    for mutation_rate in ground_truth['mutation_rates']:
        acc = benchmark.evaluate_accuracy(ground_truth, target, args.threshold, mutation_rate)
        accuracy_results.append(asdict(acc))
        print(f"Mutation {mutation_rate:.1%}: P={acc.precision:.3f}, "
              f"R={acc.recall:.3f}, F1={acc.f1_score:.3f}")
    all_results['accuracy'] = accuracy_results
    
    # 3. Scalability
    print("\n=== Scalability Experiment ===")
    scalability_results = benchmark.run_scalability_experiment(
        target, test_patterns, args.threshold, args.pattern_counts
    )
    all_results['scalability'] = scalability_results
    
    # 4. Robustness
    print("\n=== Robustness Experiment ===")
    robustness_results = benchmark.run_robustness_experiment(
        ground_truth, target, args.threshold
    )
    all_results['robustness'] = robustness_results
    
    # Save all metrics
    benchmark.save_metrics(all_results, 'metrics.json')
    
    print(f"\n✓ All experiments completed. Results saved to {args.output_dir}/")


if __name__ == '__main__':
    main()
