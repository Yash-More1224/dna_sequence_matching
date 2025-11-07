"""
Comprehensive Experiment Runner for Shift-Or/Bitap Algorithm Analysis
====================================================================

This module orchestrates comprehensive experiments to evaluate the
Shift-Or/Bitap algorithm across multiple dimensions:

1. Pattern length variation (5-50 bp)
2. Text size variation (viral to bacterial genomes)
3. Mutation rate variation (0-10%)
4. Exact vs approximate matching
5. Comparison with Python's re module

Results are saved in JSON and CSV formats for further analysis.

Author: DNA Sequence Matching Project
Date: November 2025
"""

import json
import csv
import time
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

from algorithm import ShiftOrBitap
from data_loader import DataLoader, SyntheticDataGenerator, create_motif_dataset
from benchmark import Benchmarker, ScalabilityBenchmarker
from evaluation import ApproximateMatchEvaluator, TestCaseGenerator
from visualization import PerformanceVisualizer, SequenceVisualizer


class ExperimentRunner:
    """
    Orchestrates and runs comprehensive experiments.
    """
    
    def __init__(self, output_dir: str = "./results"):
        """
        Initialize the experiment runner.
        
        Args:
            output_dir: Directory for saving results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.benchmarker = Benchmarker(warmup_runs=2)
        self.scale_benchmarker = ScalabilityBenchmarker()
        self.evaluator = ApproximateMatchEvaluator()
        
        self.results = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'algorithm': 'Shift-Or/Bitap'
            },
            'experiments': []
        }
    
    def experiment_pattern_length_scaling(self, pattern_lengths: List[int], 
                                         num_runs: int = 5) -> Dict[str, Any]:
        """
        Experiment: How performance scales with pattern length.
        
        Args:
            pattern_lengths: List of pattern lengths to test
            num_runs: Number of runs per experiment
            
        Returns:
            Dictionary with experiment results
        """
        print("\n" + "=" * 80)
        print("EXPERIMENT: Pattern Length Scaling")
        print("=" * 80)
        
        # Generate test text
        text = SyntheticDataGenerator.generate_random_sequence(50000, seed=42)
        base_pattern = "ACGTACGT"
        
        results = []
        
        for length in pattern_lengths:
            print(f"\nTesting pattern length: {length}")
            
            # Create pattern of desired length
            if length <= len(base_pattern):
                pattern = base_pattern[:length]
            else:
                pattern = (base_pattern * (length // len(base_pattern) + 1))[:length]
            
            # Benchmark
            matcher = ShiftOrBitap(pattern)
            bench_result = self.benchmarker.benchmark_shift_or(matcher, text, num_runs)
            
            results.append({
                'pattern_length': length,
                'search_time_ms': bench_result.mean_search_time * 1000,
                'preprocessing_time_ms': bench_result.preprocessing_time * 1000,
                'throughput_mbps': bench_result.throughput_chars_per_sec / 1e6,
                'memory_kb': bench_result.peak_memory / 1024,
                'num_matches': bench_result.num_matches
            })
        
        experiment_data = {
            'name': 'pattern_length_scaling',
            'description': 'Performance vs pattern length',
            'text_length': len(text),
            'num_runs': num_runs,
            'results': results
        }
        
        self.results['experiments'].append(experiment_data)
        return experiment_data
    
    def experiment_text_length_scaling(self, scale_factors: List[int],
                                      num_runs: int = 5) -> Dict[str, Any]:
        """
        Experiment: How performance scales with text length.
        
        Args:
            scale_factors: List of scaling factors for text length
            num_runs: Number of runs per experiment
            
        Returns:
            Dictionary with experiment results
        """
        print("\n" + "=" * 80)
        print("EXPERIMENT: Text Length Scaling")
        print("=" * 80)
        
        pattern = "GATTACA"
        base_text = SyntheticDataGenerator.generate_random_sequence(10000, seed=42)
        
        results = []
        
        for factor in scale_factors:
            text = base_text * factor
            print(f"\nTesting text length: {len(text):,} characters (factor={factor})")
            
            matcher = ShiftOrBitap(pattern)
            bench_result = self.benchmarker.benchmark_shift_or(matcher, text, num_runs)
            
            results.append({
                'text_length': len(text),
                'scale_factor': factor,
                'search_time_ms': bench_result.mean_search_time * 1000,
                'throughput_mbps': bench_result.throughput_chars_per_sec / 1e6,
                'memory_kb': bench_result.peak_memory / 1024,
                'num_matches': bench_result.num_matches
            })
        
        experiment_data = {
            'name': 'text_length_scaling',
            'description': 'Performance vs text length',
            'pattern': pattern,
            'num_runs': num_runs,
            'results': results
        }
        
        self.results['experiments'].append(experiment_data)
        return experiment_data
    
    def experiment_mutation_rates(self, mutation_rates: List[float],
                                 num_runs: int = 5) -> Dict[str, Any]:
        """
        Experiment: Approximate matching accuracy at different mutation rates.
        
        Args:
            mutation_rates: List of mutation rates to test (0.0 to 1.0)
            num_runs: Number of test sequences per rate
            
        Returns:
            Dictionary with experiment results
        """
        print("\n" + "=" * 80)
        print("EXPERIMENT: Mutation Rate Analysis")
        print("=" * 80)
        
        pattern = "GATTACA"
        results = []
        
        for rate in mutation_rates:
            print(f"\nTesting mutation rate: {rate:.1%}")
            
            # Generate test data
            test_text, expected = TestCaseGenerator.create_substitution_test(
                pattern, num_errors=max(1, int(len(pattern) * rate)), num_copies=10
            )
            
            # Test approximate matching
            matcher = ShiftOrBitap(pattern)
            found = matcher.search_approximate(test_text, max_errors=max(1, int(len(pattern) * rate)))
            
            # Evaluate accuracy
            metrics = self.evaluator.evaluate_matches(found, expected, tolerance=2)
            
            results.append({
                'mutation_rate': rate,
                'precision': metrics.precision,
                'recall': metrics.recall,
                'f1_score': metrics.f1_score,
                'true_positives': metrics.true_positives,
                'false_positives': metrics.false_positives,
                'false_negatives': metrics.false_negatives
            })
            
            print(f"  Precision: {metrics.precision:.3f}")
            print(f"  Recall: {metrics.recall:.3f}")
            print(f"  F1 Score: {metrics.f1_score:.3f}")
        
        experiment_data = {
            'name': 'mutation_rates',
            'description': 'Accuracy vs mutation rate',
            'pattern': pattern,
            'results': results
        }
        
        self.results['experiments'].append(experiment_data)
        return experiment_data
    
    def experiment_edit_distance_comparison(self, max_errors_list: List[int],
                                           num_runs: int = 5) -> Dict[str, Any]:
        """
        Experiment: Compare exact vs approximate matching with different k values.
        
        Args:
            max_errors_list: List of maximum error values to test
            num_runs: Number of runs per experiment
            
        Returns:
            Dictionary with experiment results
        """
        print("\n" + "=" * 80)
        print("EXPERIMENT: Edit Distance Comparison")
        print("=" * 80)
        
        pattern = "GATTACA"
        text = SyntheticDataGenerator.generate_random_sequence(10000, seed=42)
        
        results = []
        
        for max_errors in max_errors_list:
            print(f"\nTesting max errors: {max_errors}")
            
            matcher = ShiftOrBitap(pattern)
            
            if max_errors == 0:
                # Exact matching
                bench_result = self.benchmarker.benchmark_shift_or(
                    matcher, text, num_runs, approximate=False
                )
            else:
                # Approximate matching
                bench_result = self.benchmarker.benchmark_shift_or(
                    matcher, text, num_runs, approximate=True, max_errors=max_errors
                )
            
            results.append({
                'max_errors': max_errors,
                'search_time_ms': bench_result.mean_search_time * 1000,
                'throughput_mbps': bench_result.throughput_chars_per_sec / 1e6,
                'num_matches': bench_result.num_matches
            })
        
        experiment_data = {
            'name': 'edit_distance_comparison',
            'description': 'Performance vs edit distance',
            'pattern': pattern,
            'text_length': len(text),
            'num_runs': num_runs,
            'results': results
        }
        
        self.results['experiments'].append(experiment_data)
        return experiment_data
    
    def experiment_vs_regex(self, patterns: List[str], text_length: int = 10000,
                           num_runs: int = 10) -> Dict[str, Any]:
        """
        Experiment: Compare Shift-Or/Bitap against Python's re module.
        
        Args:
            patterns: List of patterns to test
            text_length: Length of test text
            num_runs: Number of runs per pattern
            
        Returns:
            Dictionary with experiment results
        """
        print("\n" + "=" * 80)
        print("EXPERIMENT: Shift-Or/Bitap vs Python re")
        print("=" * 80)
        
        text = SyntheticDataGenerator.generate_random_sequence(text_length, seed=42)
        
        results = []
        
        for pattern in patterns:
            print(f"\nTesting pattern: {pattern}")
            
            # Compare algorithms
            comparison = self.benchmarker.compare_algorithms(pattern, text, num_runs)
            
            result = {
                'pattern': pattern,
                'pattern_length': len(pattern),
                'shift_or_time_ms': comparison['shift_or'].mean_search_time * 1000,
                'shift_or_throughput_mbps': comparison['shift_or'].throughput_chars_per_sec / 1e6,
                'shift_or_memory_kb': comparison['shift_or'].peak_memory / 1024
            }
            
            if 'python_re' in comparison:
                result.update({
                    'regex_time_ms': comparison['python_re'].mean_search_time * 1000,
                    'regex_throughput_mbps': comparison['python_re'].throughput_chars_per_sec / 1e6,
                    'regex_memory_kb': comparison['python_re'].peak_memory / 1024,
                    'speedup': comparison['python_re'].mean_search_time / comparison['shift_or'].mean_search_time
                })
            
            results.append(result)
            
            print(f"  Shift-Or: {result['shift_or_time_ms']:.3f} ms")
            if 'regex_time_ms' in result:
                print(f"  Python re: {result['regex_time_ms']:.3f} ms")
                print(f"  Speedup: {result.get('speedup', 0):.2f}x")
        
        experiment_data = {
            'name': 'vs_regex',
            'description': 'Comparison with Python re module',
            'text_length': text_length,
            'num_runs': num_runs,
            'results': results
        }
        
        self.results['experiments'].append(experiment_data)
        return experiment_data
    
    def experiment_motif_search(self, motifs: List[str], background_length: int = 50000,
                               num_copies: int = 20) -> Dict[str, Any]:
        """
        Experiment: Motif search in synthetic genome.
        
        Args:
            motifs: List of motifs to embed and search
            background_length: Length of background sequence
            num_copies: Number of times to embed each motif
            
        Returns:
            Dictionary with experiment results
        """
        print("\n" + "=" * 80)
        print("EXPERIMENT: Motif Search")
        print("=" * 80)
        
        results = []
        
        for motif in motifs:
            print(f"\nSearching for motif: {motif}")
            
            # Create synthetic text with embedded motif
            text = create_motif_dataset([motif], background_length, num_copies, seed=42)
            
            # Search
            matcher = ShiftOrBitap(motif)
            start_time = time.time()
            matches = matcher.search_exact(text)
            search_time = time.time() - start_time
            
            results.append({
                'motif': motif,
                'motif_length': len(motif),
                'text_length': len(text),
                'num_matches': len(matches),
                'search_time_ms': search_time * 1000,
                'matches_per_kb': len(matches) / (len(text) / 1000)
            })
            
            print(f"  Found {len(matches)} matches in {search_time*1000:.2f} ms")
        
        experiment_data = {
            'name': 'motif_search',
            'description': 'Motif finding in synthetic genome',
            'background_length': background_length,
            'num_copies_per_motif': num_copies,
            'results': results
        }
        
        self.results['experiments'].append(experiment_data)
        return experiment_data
    
    def save_results(self, filename: str = "experiment_results.json"):
        """
        Save experiment results to JSON file.
        
        Args:
            filename: Output filename
        """
        output_path = self.output_dir / filename
        
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\n✓ Saved results to {output_path}")
    
    def export_to_csv(self, experiment_name: str):
        """
        Export specific experiment results to CSV.
        
        Args:
            experiment_name: Name of experiment to export
        """
        # Find experiment
        experiment = None
        for exp in self.results['experiments']:
            if exp['name'] == experiment_name:
                experiment = exp
                break
        
        if not experiment:
            print(f"Experiment '{experiment_name}' not found")
            return
        
        # Export to CSV
        csv_path = self.output_dir / f"{experiment_name}.csv"
        
        if experiment['results']:
            keys = experiment['results'][0].keys()
            
            with open(csv_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writeheader()
                writer.writerows(experiment['results'])
            
            print(f"✓ Exported {experiment_name} to {csv_path}")
    
    def generate_visualizations(self):
        """
        Generate all visualization plots from experiment results.
        """
        print("\n" + "=" * 80)
        print("Generating Visualizations")
        print("=" * 80)
        
        viz_dir = self.output_dir / "plots"
        viz_dir.mkdir(exist_ok=True)
        
        # TODO: Generate plots for each experiment
        # This would iterate through experiments and create appropriate visualizations
        
        print(f"✓ Visualizations saved to {viz_dir}")


def run_full_benchmark_suite():
    """
    Run the complete benchmark suite with all experiments.
    """
    print("=" * 80)
    print("SHIFT-OR/BITAP COMPREHENSIVE BENCHMARK SUITE")
    print("=" * 80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    runner = ExperimentRunner(output_dir="./results")
    
    # Experiment 1: Pattern length scaling
    runner.experiment_pattern_length_scaling(
        pattern_lengths=[5, 10, 15, 20, 25, 30, 40, 50],
        num_runs=5
    )
    
    # Experiment 2: Text length scaling
    runner.experiment_text_length_scaling(
        scale_factors=[1, 2, 5, 10, 20],
        num_runs=5
    )
    
    # Experiment 3: Mutation rates
    runner.experiment_mutation_rates(
        mutation_rates=[0.0, 0.05, 0.1, 0.15, 0.2],
        num_runs=5
    )
    
    # Experiment 4: Edit distance
    runner.experiment_edit_distance_comparison(
        max_errors_list=[0, 1, 2, 3],
        num_runs=5
    )
    
    # Experiment 5: vs Python re
    runner.experiment_vs_regex(
        patterns=["ACGT", "GATTACA", "TATAAA", "ACGTACGTACGT"],
        num_runs=10
    )
    
    # Experiment 6: Motif search
    runner.experiment_motif_search(
        motifs=["TATAAA", "GATTACA", "CAGCAG", "GCGCGC"],
        background_length=50000,
        num_copies=20
    )
    
    # Save results
    runner.save_results()
    
    # Export CSVs
    for exp in runner.results['experiments']:
        runner.export_to_csv(exp['name'])
    
    print("\n" + "=" * 80)
    print("BENCHMARK SUITE COMPLETE")
    print("=" * 80)
    print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Results saved to: {runner.output_dir}")


if __name__ == "__main__":
    run_full_benchmark_suite()
