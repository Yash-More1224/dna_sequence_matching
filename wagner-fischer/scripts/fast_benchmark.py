"""
Fast Wagner-Fischer Benchmark using Direct Pattern Matching
Simplified version that directly tests patterns without sliding window search.
"""

import time
import json
import tracemalloc
import psutil
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
from dataclasses import dataclass, asdict
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wf import WagnerFischer


@dataclass
class PerformanceMetrics:
    """Performance measurement results."""
    total_runtime: float
    mean_latency: float
    median_latency: float
    std_latency: float
    min_latency: float
    max_latency: float
    throughput: float
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


class FastWagnerFischerBenchmark:
    """Fast benchmark using direct pattern comparison."""
    
    def __init__(self, output_dir: str = "results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.wf = WagnerFischer()
        self.process = psutil.Process()
    
    def measure_performance(self,
                          patterns: List[Tuple[str, str]],
                          threshold: int,
                          num_runs: int = 5) -> PerformanceMetrics:
        """
        Measure performance for direct pattern-to-pattern comparison.
        
        Args:
            patterns: List of (original, mutated) pattern tuples
            threshold: Edit distance threshold
            num_runs: Number of runs
        """
        latencies = []
        
        tracemalloc.start()
        initial_rss = self.process.memory_info().rss / 1024 / 1024
        
        total_start = time.perf_counter()
        
        for run in range(num_runs):
            for original, mutated in patterns:
                start = time.perf_counter()
                distance, within = self.wf.compute_with_threshold(original, mutated, threshold)
                latency = time.perf_counter() - start
                latencies.append(latency)
        
        total_runtime = time.perf_counter() - total_start
        
        current, peak = tracemalloc.get_traced_memory()
        peak_memory_mb = peak / 1024 / 1024
        tracemalloc.stop()
        
        final_rss = self.process.memory_info().rss / 1024 / 1024
        peak_rss_mb = max(initial_rss, final_rss)
        
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
            preprocessing_time=0.0,
            peak_memory_mb=peak_memory_mb,
            peak_rss_mb=peak_rss_mb
        )
    
    def evaluate_accuracy(self,
                         patterns: List[Tuple[str, str]],
                         threshold: int,
                         mutation_rate: float) -> AccuracyMetrics:
        """Evaluate accuracy for pattern matching."""
        true_positives = 0
        false_negatives = 0
        
        for original, mutated in patterns:
            distance, within = self.wf.compute_with_threshold(original, mutated, threshold)
            
            # For direct comparison: if within threshold, it's a TP, otherwise FN
            if within:
                true_positives += 1
            else:
                false_negatives += 1
        
        # No false positives in direct comparison
        false_positives = 0
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 1.0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
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
    
    def run_scalability_experiment(self,
                                  all_patterns: List[Tuple[str, str]],
                                  threshold: int,
                                  pattern_counts: List[int]) -> List[Dict]:
        """Run scalability experiments."""
        results = []
        
        for count in pattern_counts:
            patterns = all_patterns[:count]
            metrics = self.measure_performance(patterns, threshold, num_runs=3)
            
            results.append({
                'pattern_count': count,
                'total_runtime': metrics.total_runtime,
                'mean_latency': metrics.mean_latency,
                'throughput': metrics.throughput,
                'peak_memory_mb': metrics.peak_memory_mb
            })
            
            print(f"Scalability: {count} patterns -> {metrics.total_runtime:.3f}s, {metrics.throughput:.2f} ops/sec")
        
        return results
    
    def run_robustness_experiment(self,
                                 patterns_by_rate: Dict[float, List[Tuple[str, str]]],
                                 threshold: int) -> List[Dict]:
        """Run robustness experiments across mutation rates."""
        results = []
        
        for mutation_rate in sorted(patterns_by_rate.keys()):
            patterns = patterns_by_rate[mutation_rate]
            
            accuracy = self.evaluate_accuracy(patterns, threshold, mutation_rate)
            perf = self.measure_performance(patterns[:20], threshold, num_runs=2)
            
            results.append({
                'mutation_rate': mutation_rate,
                'precision': accuracy.precision,
                'recall': accuracy.recall,
                'f1_score': accuracy.f1_score,
                'mean_latency': perf.mean_latency,
                'throughput': perf.throughput
            })
            
            print(f"Robustness: {mutation_rate:.1%} mutation -> F1={accuracy.f1_score:.3f}, latency={perf.mean_latency:.6f}s")
        
        return results
    
    def save_metrics(self, metrics: Dict, filename: str):
        """Save metrics to JSON."""
        filepath = self.output_dir / filename
        with open(filepath, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"Metrics saved to {filepath}")


def load_ground_truth(filepath: str) -> Dict:
    """Load ground truth from JSON."""
    with open(filepath, 'r') as f:
        return json.load(f)


def main():
    """Run fast Wagner-Fischer evaluation."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Fast WF Benchmark')
    parser.add_argument('--ground-truth', required=True, help='Path to ground truth JSON')
    parser.add_argument('--output-dir', default='results', help='Output directory')
    parser.add_argument('--threshold', type=int, default=3, help='Edit distance threshold')
    parser.add_argument('--pattern-counts', type=int, nargs='+', default=[10, 20, 50, 100],
                       help='Pattern counts for scalability test')
    
    args = parser.parse_args()
    
    print(f"Loading ground truth: {args.ground_truth}")
    ground_truth = load_ground_truth(args.ground_truth)
    print(f"Loaded {len(ground_truth['patterns'])} patterns")
    
    # Organize patterns
    patterns_by_rate = {}
    all_patterns = []
    
    for p in ground_truth['patterns']:
        pair = (p['original'], p['mutated'])
        all_patterns.append(pair)
        
        rate = p['mutation_rate']
        if rate not in patterns_by_rate:
            patterns_by_rate[rate] = []
        patterns_by_rate[rate].append(pair)
    
    # Initialize benchmark
    benchmark = FastWagnerFischerBenchmark(output_dir=args.output_dir)
    
    all_results = {
        'dataset': ground_truth['target_id'],
        'threshold': args.threshold,
        'pattern_length': ground_truth['pattern_length']
    }
    
    # 1. Performance benchmark
    print("\n=== Performance Benchmark ===")
    test_patterns = all_patterns[:50]
    perf_metrics = benchmark.measure_performance(test_patterns, args.threshold, num_runs=5)
    all_results['performance'] = asdict(perf_metrics)
    print(f"Mean latency: {perf_metrics.mean_latency:.6f}s")
    print(f"Throughput: {perf_metrics.throughput:.2f} ops/sec")
    print(f"Peak memory: {perf_metrics.peak_memory_mb:.2f} MB")
    
    # 2. Accuracy evaluation
    print("\n=== Accuracy Evaluation ===")
    accuracy_results = []
    for mutation_rate in sorted(patterns_by_rate.keys()):
        acc = benchmark.evaluate_accuracy(patterns_by_rate[mutation_rate], args.threshold, mutation_rate)
        accuracy_results.append(asdict(acc))
        print(f"Mutation {mutation_rate:.1%}: P={acc.precision:.3f}, R={acc.recall:.3f}, F1={acc.f1_score:.3f}")
    all_results['accuracy'] = accuracy_results
    
    # 3. Scalability
    print("\n=== Scalability Experiment ===")
    scalability_results = benchmark.run_scalability_experiment(
        all_patterns, args.threshold, args.pattern_counts
    )
    all_results['scalability'] = scalability_results
    
    # 4. Robustness
    print("\n=== Robustness Experiment ===")
    robustness_results = benchmark.run_robustness_experiment(
        patterns_by_rate, args.threshold
    )
    all_results['robustness'] = robustness_results
    
    # Save metrics
    benchmark.save_metrics(all_results, 'metrics.json')
    
    print(f"\n✓ All experiments completed. Results saved to {args.output_dir}/")


if __name__ == '__main__':
    main()
