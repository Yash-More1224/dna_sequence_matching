"""
Experiment orchestration for KMP algorithm analysis.

This module provides functions to run comprehensive benchmark experiments,
comparing KMP performance across different parameters and against Python's re module.
"""

import json
import time
from typing import List, Dict, Any, Optional
from pathlib import Path
from dataclasses import asdict

from .config import (EXPERIMENT_CONFIG, BENCHMARKS_DIR, REPORTS_DIR, 
                     DNA_BASES, RESULTS_DIR)
from .kmp_algorithm import KMP
from .synthetic_data import generate_random_sequence, generate_pattern
from .benchmarking import benchmark_kmp_search, BenchmarkResult
from .evaluation import compare_with_re, ComparisonResult, test_correctness
from .data_loader import load_dataset, SequenceRecord
from .utils import save_json, save_csv, format_time, format_memory
from .visualization import (plot_latency_vs_pattern_length, plot_latency_vs_text_size,
                           plot_memory_vs_text_size, plot_kmp_vs_re_comparison,
                           create_all_visualizations)


class ExperimentRunner:
    """
    Main class for running KMP algorithm experiments.
    
    Attributes:
        results_dir: Directory to save results
        seed: Random seed for reproducibility
    """
    
    def __init__(self, results_dir: Path = RESULTS_DIR, seed: int = None):
        """
        Initialize experiment runner.
        
        Args:
            results_dir: Directory to save results
            seed: Random seed for reproducibility
        """
        self.results_dir = results_dir
        self.seed = seed if seed is not None else EXPERIMENT_CONFIG['random_seed']
        
        # Create directories
        self.results_dir.mkdir(parents=True, exist_ok=True)
        BENCHMARKS_DIR.mkdir(parents=True, exist_ok=True)
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    
    def run_pattern_length_experiment(self, 
                                     text_length: int = 100000,
                                     pattern_lengths: Optional[List[int]] = None,
                                     save_results: bool = True) -> List[BenchmarkResult]:
        """
        Experiment 1: Vary pattern length and measure performance.
        
        Args:
            text_length: Length of text to use
            pattern_lengths: List of pattern lengths to test
            save_results: Whether to save results to file
            
        Returns:
            List of BenchmarkResult objects
        """
        if pattern_lengths is None:
            pattern_lengths = EXPERIMENT_CONFIG['pattern_lengths']
        
        print("\n" + "="*70)
        print("EXPERIMENT 1: Pattern Length Variation")
        print("="*70)
        print(f"Text length: {text_length:,} bp")
        print(f"Testing pattern lengths: {pattern_lengths}")
        print("-"*70)
        
        # Generate text once
        text = generate_random_sequence(text_length, seed=self.seed)
        
        results = []
        for pattern_len in pattern_lengths:
            print(f"\nTesting pattern length: {pattern_len} bp")
            
            # Generate pattern
            pattern = generate_pattern(pattern_len, seed=self.seed + pattern_len)
            
            # Create KMP instance and benchmark
            kmp = KMP(pattern)
            result = benchmark_kmp_search(kmp, text)
            results.append(result)
            
            print(f"  Preprocessing: {format_time(result.preprocessing_time)}")
            print(f"  Search time:   {format_time(result.search_time)}")
            print(f"  Matches found: {result.num_matches}")
        
        print("\n" + "="*70)
        
        # Save results
        if save_results:
            self._save_results(results, "pattern_length_experiment")
            plot_latency_vs_pattern_length(results, show=False)
        
        return results
    
    def run_text_size_experiment(self,
                                text_sizes: Optional[List[int]] = None,
                                pattern_length: int = 50,
                                save_results: bool = True) -> List[BenchmarkResult]:
        """
        Experiment 2: Vary text size and measure scalability.
        
        Args:
            text_sizes: List of text sizes (in bytes) to test
            pattern_length: Pattern length to use
            save_results: Whether to save results to file
            
        Returns:
            List of BenchmarkResult objects
        """
        if text_sizes is None:
            # Convert KB to bytes
            text_sizes = [size * 1024 for size in EXPERIMENT_CONFIG['text_sizes']]
        
        print("\n" + "="*70)
        print("EXPERIMENT 2: Text Size Scalability")
        print("="*70)
        print(f"Pattern length: {pattern_length} bp")
        print(f"Testing text sizes: {[s//1024 for s in text_sizes]} KB")
        print("-"*70)
        
        # Generate pattern once
        pattern = generate_pattern(pattern_length, seed=self.seed)
        kmp = KMP(pattern)
        
        results = []
        for text_size in text_sizes:
            print(f"\nTesting text size: {text_size // 1024} KB ({text_size:,} bp)")
            
            # Generate text
            text = generate_random_sequence(text_size, seed=self.seed + text_size)
            
            # Benchmark
            result = benchmark_kmp_search(kmp, text)
            results.append(result)
            
            throughput = text_size / result.search_time / 1_000_000  # MB/s
            print(f"  Search time:  {format_time(result.search_time)}")
            print(f"  Throughput:   {throughput:.2f} MB/s")
            print(f"  Memory:       {format_memory(result.memory_used)}")
        
        print("\n" + "="*70)
        
        # Save results
        if save_results:
            self._save_results(results, "text_size_experiment")
            plot_latency_vs_text_size(results, show=False)
            plot_memory_vs_text_size(results, show=False)
        
        return results
    
    def run_kmp_vs_re_experiment(self,
                                 text_sizes: Optional[List[int]] = None,
                                 pattern_length: int = 50,
                                 save_results: bool = True) -> List[ComparisonResult]:
        """
        Experiment 3: Compare KMP vs Python re module.
        
        Args:
            text_sizes: List of text sizes to test
            pattern_length: Pattern length to use
            save_results: Whether to save results to file
            
        Returns:
            List of ComparisonResult objects
        """
        if text_sizes is None:
            text_sizes = [size * 1024 for size in [1, 10, 50, 100, 500, 1000]]
        
        print("\n" + "="*70)
        print("EXPERIMENT 3: KMP vs Python re Comparison")
        print("="*70)
        print(f"Pattern length: {pattern_length} bp")
        print(f"Testing text sizes: {[s//1024 for s in text_sizes]} KB")
        print("-"*70)
        
        # Generate pattern once
        pattern = generate_pattern(pattern_length, seed=self.seed)
        
        results = []
        for text_size in text_sizes:
            print(f"\nTesting text size: {text_size // 1024} KB")
            
            # Generate text
            text = generate_random_sequence(text_size, seed=self.seed + text_size)
            
            # Compare
            result = compare_with_re(text, pattern)
            results.append(result)
            
            winner = "KMP" if result.speedup > 1 else "re"
            print(f"  KMP time:      {format_time(result.kmp_time)}")
            print(f"  re time:       {format_time(result.re_time)}")
            print(f"  Speedup:       {result.speedup:.2f}x ({winner} faster)")
            print(f"  Matches agree: {'✓' if result.matches_agree else '✗'}")
        
        print("\n" + "="*70)
        
        # Save results
        if save_results:
            self._save_comparison_results(results, text_sizes, "kmp_vs_re_experiment")
            plot_kmp_vs_re_comparison(results, text_sizes, show=False)
        
        return results
    
    def run_multiple_patterns_experiment(self,
                                        text_length: int = 100000,
                                        num_patterns_list: Optional[List[int]] = None,
                                        pattern_length: int = 50,
                                        save_results: bool = True) -> Dict[str, Any]:
        """
        Experiment 4: Test with multiple patterns.
        
        Args:
            text_length: Length of text to use
            num_patterns_list: List of pattern counts to test
            pattern_length: Length of each pattern
            save_results: Whether to save results to file
            
        Returns:
            Dictionary with results
        """
        if num_patterns_list is None:
            num_patterns_list = EXPERIMENT_CONFIG['num_patterns']
        
        print("\n" + "="*70)
        print("EXPERIMENT 4: Multiple Patterns")
        print("="*70)
        print(f"Text length: {text_length:,} bp")
        print(f"Pattern length: {pattern_length} bp")
        print(f"Testing pattern counts: {num_patterns_list}")
        print("-"*70)
        
        # Generate text once
        text = generate_random_sequence(text_length, seed=self.seed)
        
        results = []
        for num_patterns in num_patterns_list:
            print(f"\nTesting with {num_patterns} patterns")
            
            # Generate patterns
            patterns = [generate_pattern(pattern_length, seed=self.seed + i) 
                       for i in range(num_patterns)]
            
            # Search for each pattern
            start_time = time.perf_counter()
            total_matches = 0
            
            for pattern in patterns:
                kmp = KMP(pattern)
                matches = kmp.search(text)
                total_matches += len(matches)
            
            elapsed = time.perf_counter() - start_time
            
            result = {
                'num_patterns': num_patterns,
                'total_time': elapsed,
                'time_per_pattern': elapsed / num_patterns,
                'total_matches': total_matches
            }
            results.append(result)
            
            print(f"  Total time:    {format_time(elapsed)}")
            print(f"  Per pattern:   {format_time(elapsed / num_patterns)}")
            print(f"  Total matches: {total_matches}")
        
        print("\n" + "="*70)
        
        # Save results
        if save_results:
            save_path = BENCHMARKS_DIR / "multiple_patterns_experiment.json"
            save_json(results, save_path)
            print(f"Results saved to {save_path}")
        
        return {'results': results, 'text_length': text_length, 'pattern_length': pattern_length}
    
    def run_real_genome_experiment(self,
                                   dataset_name: str = 'ecoli',
                                   pattern_lengths: Optional[List[int]] = None,
                                   save_results: bool = True) -> Optional[List[BenchmarkResult]]:
        """
        Experiment 5: Test on real genomic data.
        
        Args:
            dataset_name: Name of dataset to load
            pattern_lengths: List of pattern lengths to test
            save_results: Whether to save results to file
            
        Returns:
            List of BenchmarkResult objects or None if dataset not available
        """
        if pattern_lengths is None:
            pattern_lengths = [20, 50, 100, 200]
        
        print("\n" + "="*70)
        print(f"EXPERIMENT 5: Real Genome Data ({dataset_name})")
        print("="*70)
        
        # Try to load dataset
        from .config import DATASETS_DIR
        genome = load_dataset(dataset_name, DATASETS_DIR)
        
        if genome is None:
            print(f"Dataset '{dataset_name}' not found. Please run download script first.")
            print("="*70)
            return None
        
        text = genome.sequence
        print(f"Genome loaded: {genome.id}")
        print(f"Length: {len(text):,} bp")
        print(f"Testing pattern lengths: {pattern_lengths}")
        print("-"*70)
        
        results = []
        for pattern_len in pattern_lengths:
            print(f"\nTesting pattern length: {pattern_len} bp")
            
            # Extract a real motif from the genome
            import random
            random.seed(self.seed + pattern_len)
            start_pos = random.randint(0, len(text) - pattern_len)
            pattern = text[start_pos:start_pos + pattern_len]
            
            # Benchmark
            kmp = KMP(pattern)
            result = benchmark_kmp_search(kmp, text)
            results.append(result)
            
            print(f"  Pattern: {pattern}")
            print(f"  Search time:   {format_time(result.search_time)}")
            print(f"  Matches found: {result.num_matches}")
        
        print("\n" + "="*70)
        
        # Save results
        if save_results:
            self._save_results(results, f"real_genome_{dataset_name}_experiment")
        
        return results
    
    def run_correctness_test(self) -> Dict[str, Any]:
        """
        Experiment 6: Correctness validation.
        
        Returns:
            Dictionary with test results
        """
        print("\n" + "="*70)
        print("EXPERIMENT 6: Correctness Validation")
        print("="*70)
        
        from .synthetic_data import generate_test_cases
        
        # Generate test cases
        test_cases = generate_test_cases()
        
        # Run tests
        results = test_correctness(test_cases)
        
        print(f"Total tests:   {results['total']}")
        print(f"Passed:        {results['passed']} ✓")
        print(f"Failed:        {results['failed']} ✗")
        print(f"Success rate:  {results['success_rate']:.1f}%")
        
        if results['failures']:
            print(f"\nFailures:")
            for failure in results['failures']:
                print(f"  Test {failure['test_id']}: {failure['reason']}")
        
        print("="*70)
        
        return results
    
    def run_all_experiments(self) -> Dict[str, Any]:
        """
        Run all experiments.
        
        Returns:
            Dictionary containing all experiment results
        """
        print("\n" + "#"*70)
        print("# RUNNING ALL KMP ALGORITHM EXPERIMENTS")
        print("#"*70)
        
        all_results = {}
        
        # Experiment 1: Pattern length
        print("\n[1/6] Running pattern length experiment...")
        all_results['pattern_length'] = self.run_pattern_length_experiment()
        
        # Experiment 2: Text size
        print("\n[2/6] Running text size experiment...")
        all_results['text_size'] = self.run_text_size_experiment()
        
        # Experiment 3: KMP vs re
        print("\n[3/6] Running KMP vs re comparison...")
        all_results['kmp_vs_re'] = self.run_kmp_vs_re_experiment()
        
        # Experiment 4: Multiple patterns
        print("\n[4/6] Running multiple patterns experiment...")
        all_results['multiple_patterns'] = self.run_multiple_patterns_experiment()
        
        # Experiment 5: Real genome (if available)
        print("\n[5/6] Running real genome experiment...")
        all_results['real_genome'] = self.run_real_genome_experiment()
        
        # Experiment 6: Correctness
        print("\n[6/6] Running correctness validation...")
        all_results['correctness'] = self.run_correctness_test()
        
        # Generate summary report
        self._generate_summary_report(all_results)
        
        print("\n" + "#"*70)
        print("# ALL EXPERIMENTS COMPLETED")
        print("#"*70)
        print(f"\nResults saved to: {self.results_dir}")
        
        return all_results
    
    def _save_results(self, results: List[BenchmarkResult], name: str) -> None:
        """Save benchmark results to JSON and CSV."""
        # Convert to dicts
        results_dicts = [r.to_dict() for r in results]
        
        # Save JSON
        json_path = BENCHMARKS_DIR / f"{name}.json"
        save_json(results_dicts, json_path)
        
        # Save CSV
        csv_path = BENCHMARKS_DIR / f"{name}.csv"
        save_csv(results_dicts, csv_path)
        
        print(f"Results saved to {json_path} and {csv_path}")
    
    def _save_comparison_results(self, results: List[ComparisonResult], 
                                text_sizes: List[int], name: str) -> None:
        """Save comparison results to JSON and CSV."""
        # Convert to dicts and add text sizes
        results_dicts = []
        for result, size in zip(results, text_sizes):
            d = result.to_dict()
            d['text_size'] = size
            results_dicts.append(d)
        
        # Save JSON
        json_path = BENCHMARKS_DIR / f"{name}.json"
        save_json(results_dicts, json_path)
        
        # Save CSV
        csv_path = BENCHMARKS_DIR / f"{name}.csv"
        save_csv(results_dicts, csv_path)
        
        print(f"Results saved to {json_path} and {csv_path}")
    
    def _generate_summary_report(self, all_results: Dict[str, Any]) -> None:
        """Generate a text summary report of all experiments."""
        report_path = REPORTS_DIR / "experiment_summary.txt"
        
        with open(report_path, 'w') as f:
            f.write("="*70 + "\n")
            f.write("KMP ALGORITHM - EXPERIMENT SUMMARY REPORT\n")
            f.write("="*70 + "\n\n")
            
            f.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Random seed: {self.seed}\n\n")
            
            # Pattern length experiment
            if 'pattern_length' in all_results and all_results['pattern_length']:
                f.write("-"*70 + "\n")
                f.write("EXPERIMENT 1: Pattern Length Variation\n")
                f.write("-"*70 + "\n")
                for r in all_results['pattern_length']:
                    f.write(f"Pattern: {r.pattern_length} bp | ")
                    f.write(f"Search: {format_time(r.search_time)} | ")
                    f.write(f"Matches: {r.num_matches}\n")
                f.write("\n")
            
            # Text size experiment
            if 'text_size' in all_results and all_results['text_size']:
                f.write("-"*70 + "\n")
                f.write("EXPERIMENT 2: Text Size Scalability\n")
                f.write("-"*70 + "\n")
                for r in all_results['text_size']:
                    throughput = r.text_length / r.search_time / 1_000_000
                    f.write(f"Text: {r.text_length//1024} KB | ")
                    f.write(f"Time: {format_time(r.search_time)} | ")
                    f.write(f"Throughput: {throughput:.2f} MB/s\n")
                f.write("\n")
            
            # KMP vs re
            if 'kmp_vs_re' in all_results and all_results['kmp_vs_re']:
                f.write("-"*70 + "\n")
                f.write("EXPERIMENT 3: KMP vs Python re\n")
                f.write("-"*70 + "\n")
                for r in all_results['kmp_vs_re']:
                    winner = "KMP" if r.speedup > 1 else "re"
                    f.write(f"Speedup: {r.speedup:.2f}x ({winner} faster) | ")
                    f.write(f"KMP: {format_time(r.kmp_time)} | ")
                    f.write(f"re: {format_time(r.re_time)}\n")
                f.write("\n")
            
            # Correctness
            if 'correctness' in all_results:
                f.write("-"*70 + "\n")
                f.write("EXPERIMENT 6: Correctness Validation\n")
                f.write("-"*70 + "\n")
                c = all_results['correctness']
                f.write(f"Total tests: {c['total']}\n")
                f.write(f"Passed: {c['passed']}\n")
                f.write(f"Failed: {c['failed']}\n")
                f.write(f"Success rate: {c['success_rate']:.1f}%\n\n")
            
            f.write("="*70 + "\n")
            f.write("END OF REPORT\n")
            f.write("="*70 + "\n")
        
        print(f"Summary report saved to {report_path}")


def run_quick_demo() -> None:
    """Run a quick demonstration of KMP algorithm."""
    print("\n" + "="*70)
    print("KMP ALGORITHM - QUICK DEMONSTRATION")
    print("="*70)
    
    # Example 1: Simple search
    text = "ABABDABACDABABCABAB"
    pattern = "ABABC"
    
    print(f"\nExample 1: Simple Search")
    print(f"Text:    {text}")
    print(f"Pattern: {pattern}")
    
    kmp = KMP(pattern)
    matches = kmp.search(text)
    
    print(f"LPS array: {kmp.get_lps_array()}")
    print(f"Matches found at positions: {matches}")
    
    # Example 2: DNA sequence
    from .synthetic_data import generate_random_sequence
    
    print(f"\n\nExample 2: DNA Sequence Search")
    text = generate_random_sequence(1000, seed=42)
    pattern = "ATCGATCG"
    
    print(f"Text length: {len(text)} bp")
    print(f"Pattern: {pattern}")
    
    kmp = KMP(pattern)
    stats = kmp.search_with_stats(text)
    
    print(f"Preprocessing time: {format_time(stats['preprocessing_time'])}")
    print(f"Search time: {format_time(stats['search_time'])}")
    print(f"Matches found: {stats['num_matches']}")
    print(f"Match positions: {stats['matches'][:10]}{'...' if len(stats['matches']) > 10 else ''}")
    
    print("\n" + "="*70)
