"""
Benchmarking Framework for Shift-Or/Bitap Algorithm
===================================================

This module provides comprehensive benchmarking utilities for measuring:
- Execution time (latency, throughput)
- Memory usage (peak, index footprint)
- Preprocessing time
- Comparison with Python's re module

Metrics collected:
- Mean, median, standard deviation of runtime
- Matches per second (throughput)
- Memory profiling data
- Preprocessing vs search time breakdown

Author: DNA Sequence Matching Project
Date: November 2025
"""

import time
import re
import tracemalloc
import statistics
from typing import List, Dict, Callable, Tuple, Optional, Any
from dataclasses import dataclass, asdict
import sys
import gc


@dataclass
class BenchmarkResult:
    """Container for benchmark results."""
    algorithm: str
    pattern_length: int
    text_length: int
    num_runs: int
    
    # Timing metrics (in seconds)
    preprocessing_time: float
    mean_search_time: float
    median_search_time: float
    std_search_time: float
    min_search_time: float
    max_search_time: float
    total_time: float
    
    # Throughput metrics
    throughput_chars_per_sec: float
    throughput_matches_per_sec: float
    
    # Memory metrics (in bytes)
    peak_memory: int
    memory_increment: int
    
    # Results
    num_matches: int
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)
    
    def __str__(self) -> str:
        """Human-readable string representation."""
        lines = [
            f"Benchmark Results for {self.algorithm}",
            f"{'=' * 60}",
            f"Pattern length: {self.pattern_length}",
            f"Text length: {self.text_length:,}",
            f"Number of runs: {self.num_runs}",
            f"",
            f"Timing:",
            f"  Preprocessing: {self.preprocessing_time*1000:.3f} ms",
            f"  Search (mean): {self.mean_search_time*1000:.3f} ms",
            f"  Search (median): {self.median_search_time*1000:.3f} ms",
            f"  Search (std): {self.std_search_time*1000:.3f} ms",
            f"  Search (min): {self.min_search_time*1000:.3f} ms",
            f"  Search (max): {self.max_search_time*1000:.3f} ms",
            f"  Total: {self.total_time*1000:.3f} ms",
            f"",
            f"Throughput:",
            f"  {self.throughput_chars_per_sec/1e6:.2f} MB/s",
            f"  {self.throughput_matches_per_sec:.2f} matches/s",
            f"",
            f"Memory:",
            f"  Peak: {self.peak_memory/1024:.2f} KB",
            f"  Increment: {self.memory_increment/1024:.2f} KB",
            f"",
            f"Results:",
            f"  Matches found: {self.num_matches}",
        ]
        return "\n".join(lines)


class Benchmarker:
    """
    Comprehensive benchmarking framework for pattern matching algorithms.
    """
    
    def __init__(self, warmup_runs: int = 2):
        """
        Initialize the benchmarker.
        
        Args:
            warmup_runs: Number of warmup runs before actual benchmarking
        """
        self.warmup_runs = warmup_runs
    
    def benchmark_function(self, func: Callable, *args, num_runs: int = 10, 
                          measure_memory: bool = True, **kwargs) -> Dict[str, Any]:
        """
        Benchmark a function with multiple runs.
        
        Args:
            func: Function to benchmark
            *args: Positional arguments for function
            num_runs: Number of benchmark runs
            measure_memory: Whether to measure memory usage
            **kwargs: Keyword arguments for function
            
        Returns:
            Dictionary with timing and memory metrics
        """
        # Warmup runs
        for _ in range(self.warmup_runs):
            _ = func(*args, **kwargs)
        
        # Actual benchmark runs
        times = []
        results = []
        
        for _ in range(num_runs):
            gc.collect()  # Clean up before each run
            
            start = time.perf_counter()
            result = func(*args, **kwargs)
            end = time.perf_counter()
            
            times.append(end - start)
            results.append(result)
        
        # Memory measurement (single run)
        peak_memory = 0
        memory_increment = 0
        
        if measure_memory:
            gc.collect()
            tracemalloc.start()
            
            _ = func(*args, **kwargs)
            
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            
            peak_memory = peak
            memory_increment = current
        
        return {
            'times': times,
            'mean_time': statistics.mean(times),
            'median_time': statistics.median(times),
            'std_time': statistics.stdev(times) if len(times) > 1 else 0.0,
            'min_time': min(times),
            'max_time': max(times),
            'peak_memory': peak_memory,
            'memory_increment': memory_increment,
            'results': results
        }
    
    def benchmark_shift_or(self, matcher, text: str, num_runs: int = 10,
                          approximate: bool = False, max_errors: int = 0) -> BenchmarkResult:
        """
        Benchmark the Shift-Or/Bitap algorithm.
        
        Args:
            matcher: ShiftOrBitap instance (already preprocessed)
            text: Text to search in
            num_runs: Number of benchmark runs
            approximate: Whether to use approximate matching
            max_errors: Maximum errors for approximate matching
            
        Returns:
            BenchmarkResult with all metrics
        """
        # Measure preprocessing time (create new matcher)
        pattern = matcher.pattern
        gc.collect()
        
        start = time.perf_counter()
        from algorithm import ShiftOrBitap
        test_matcher = ShiftOrBitap(pattern)
        preprocessing_time = time.perf_counter() - start
        
        # Benchmark search
        search_func = (matcher.search_approximate if approximate 
                      else matcher.search_exact)
        
        if approximate:
            benchmark_data = self.benchmark_function(
                search_func, text, max_errors, num_runs=num_runs
            )
        else:
            benchmark_data = self.benchmark_function(
                search_func, text, num_runs=num_runs
            )
        
        # Get match count
        matches = benchmark_data['results'][0]
        num_matches = len(matches)
        
        # Calculate throughput
        mean_time = benchmark_data['mean_time']
        throughput_chars = len(text) / mean_time if mean_time > 0 else 0
        throughput_matches = num_matches / mean_time if mean_time > 0 else 0
        
        # Create result
        result = BenchmarkResult(
            algorithm=f"Shift-Or/Bitap{'_approx' if approximate else '_exact'}",
            pattern_length=len(pattern),
            text_length=len(text),
            num_runs=num_runs,
            preprocessing_time=preprocessing_time,
            mean_search_time=benchmark_data['mean_time'],
            median_search_time=benchmark_data['median_time'],
            std_search_time=benchmark_data['std_time'],
            min_search_time=benchmark_data['min_time'],
            max_search_time=benchmark_data['max_time'],
            total_time=preprocessing_time + benchmark_data['mean_time'],
            throughput_chars_per_sec=throughput_chars,
            throughput_matches_per_sec=throughput_matches,
            peak_memory=benchmark_data['peak_memory'],
            memory_increment=benchmark_data['memory_increment'],
            num_matches=num_matches
        )
        
        return result
    
    def benchmark_regex(self, pattern: str, text: str, num_runs: int = 10,
                       flags: int = 0) -> BenchmarkResult:
        """
        Benchmark Python's re module for comparison.
        
        Args:
            pattern: Pattern to search for
            text: Text to search in
            num_runs: Number of benchmark runs
            flags: Regex flags (e.g., re.IGNORECASE)
            
        Returns:
            BenchmarkResult with all metrics
        """
        # Measure preprocessing time (compile regex)
        gc.collect()
        
        start = time.perf_counter()
        regex = re.compile(pattern, flags)
        preprocessing_time = time.perf_counter() - start
        
        # Benchmark search
        def search_func(text):
            return [m.start() for m in regex.finditer(text)]
        
        benchmark_data = self.benchmark_function(
            search_func, text, num_runs=num_runs
        )
        
        # Get match count
        matches = benchmark_data['results'][0]
        num_matches = len(matches)
        
        # Calculate throughput
        mean_time = benchmark_data['mean_time']
        throughput_chars = len(text) / mean_time if mean_time > 0 else 0
        throughput_matches = num_matches / mean_time if mean_time > 0 else 0
        
        # Create result
        result = BenchmarkResult(
            algorithm="Python_re",
            pattern_length=len(pattern),
            text_length=len(text),
            num_runs=num_runs,
            preprocessing_time=preprocessing_time,
            mean_search_time=benchmark_data['mean_time'],
            median_search_time=benchmark_data['median_time'],
            std_search_time=benchmark_data['std_time'],
            min_search_time=benchmark_data['min_time'],
            max_search_time=benchmark_data['max_time'],
            total_time=preprocessing_time + benchmark_data['mean_time'],
            throughput_chars_per_sec=throughput_chars,
            throughput_matches_per_sec=throughput_matches,
            peak_memory=benchmark_data['peak_memory'],
            memory_increment=benchmark_data['memory_increment'],
            num_matches=num_matches
        )
        
        return result
    
    def compare_algorithms(self, pattern: str, text: str, num_runs: int = 10,
                          approximate: bool = False, max_errors: int = 0) -> Dict[str, BenchmarkResult]:
        """
        Compare Shift-Or/Bitap against Python's re module.
        
        Args:
            pattern: Pattern to search for
            text: Text to search in
            num_runs: Number of runs for each algorithm
            approximate: Whether to test approximate matching
            max_errors: Maximum errors for approximate matching
            
        Returns:
            Dictionary mapping algorithm names to BenchmarkResults
        """
        from algorithm import ShiftOrBitap
        
        results = {}
        
        # Benchmark Shift-Or/Bitap
        matcher = ShiftOrBitap(pattern)
        results['shift_or'] = self.benchmark_shift_or(
            matcher, text, num_runs, approximate, max_errors
        )
        
        # Benchmark Python re (only for exact matching)
        if not approximate:
            results['python_re'] = self.benchmark_regex(pattern, text, num_runs)
        
        return results
    
    def print_comparison(self, results: Dict[str, BenchmarkResult]):
        """
        Print a comparison table of benchmark results.
        
        Args:
            results: Dictionary of algorithm results
        """
        print("\n" + "=" * 80)
        print("BENCHMARK COMPARISON")
        print("=" * 80)
        
        # Print each result
        for name, result in results.items():
            print(f"\n{result}")
        
        # Print speedup comparison
        if len(results) >= 2:
            algos = list(results.keys())
            base_time = results[algos[0]].mean_search_time
            
            print("\n" + "=" * 80)
            print("RELATIVE PERFORMANCE (Search Time)")
            print("=" * 80)
            
            for name, result in results.items():
                speedup = base_time / result.mean_search_time if result.mean_search_time > 0 else float('inf')
                print(f"{name:20s}: {speedup:6.2f}x (vs {algos[0]})")


class ScalabilityBenchmarker:
    """
    Specialized benchmarker for testing scalability.
    """
    
    def __init__(self):
        self.benchmarker = Benchmarker(warmup_runs=1)
    
    def benchmark_scaling_text_length(self, pattern: str, base_text: str,
                                     scale_factors: List[int], num_runs: int = 5) -> List[BenchmarkResult]:
        """
        Benchmark how performance scales with text length.
        
        Args:
            pattern: Pattern to search for
            base_text: Base text to scale
            scale_factors: List of scaling factors (e.g., [1, 2, 4, 8])
            num_runs: Number of runs per scale
            
        Returns:
            List of BenchmarkResults for each scale
        """
        from algorithm import ShiftOrBitap
        
        results = []
        matcher = ShiftOrBitap(pattern)
        
        for factor in scale_factors:
            # Scale the text
            scaled_text = base_text * factor
            
            print(f"Benchmarking text length: {len(scaled_text):,} chars (factor={factor})")
            
            result = self.benchmarker.benchmark_shift_or(matcher, scaled_text, num_runs)
            results.append(result)
        
        return results
    
    def benchmark_scaling_pattern_length(self, base_pattern: str, text: str,
                                        pattern_lengths: List[int], num_runs: int = 5) -> List[BenchmarkResult]:
        """
        Benchmark how performance scales with pattern length.
        
        Args:
            base_pattern: Base pattern to extend
            text: Text to search in
            pattern_lengths: List of pattern lengths to test
            num_runs: Number of runs per length
            
        Returns:
            List of BenchmarkResults for each pattern length
        """
        from algorithm import ShiftOrBitap
        
        results = []
        
        for length in pattern_lengths:
            # Create pattern of desired length
            if length <= len(base_pattern):
                pattern = base_pattern[:length]
            else:
                # Extend by repeating
                pattern = (base_pattern * (length // len(base_pattern) + 1))[:length]
            
            print(f"Benchmarking pattern length: {length}")
            
            matcher = ShiftOrBitap(pattern)
            result = self.benchmarker.benchmark_shift_or(matcher, text, num_runs)
            results.append(result)
        
        return results


if __name__ == "__main__":
    print("Benchmark Framework Demo")
    print("=" * 80)
    
    # Import algorithm
    from algorithm import ShiftOrBitap
    from data_loader import SyntheticDataGenerator
    
    # Generate test data
    pattern = "GATTACA"
    text = SyntheticDataGenerator.generate_random_sequence(10000, seed=42)
    
    # Embed pattern a few times
    for i in range(0, len(text) - len(pattern), 1000):
        text = text[:i] + pattern + text[i+len(pattern):]
    
    print(f"\nTest setup:")
    print(f"  Pattern: {pattern}")
    print(f"  Text length: {len(text):,} characters")
    
    # Run benchmarks
    benchmarker = Benchmarker()
    
    print("\n" + "=" * 80)
    print("Running benchmarks...")
    print("=" * 80)
    
    results = benchmarker.compare_algorithms(pattern, text, num_runs=10)
    benchmarker.print_comparison(results)
    
    # Scalability test
    print("\n" + "=" * 80)
    print("SCALABILITY TEST: Text Length")
    print("=" * 80)
    
    scale_bench = ScalabilityBenchmarker()
    short_text = SyntheticDataGenerator.generate_random_sequence(1000, seed=42)
    scale_results = scale_bench.benchmark_scaling_text_length(
        pattern, short_text, scale_factors=[1, 2, 5, 10], num_runs=3
    )
    
    print("\nScaling Results:")
    for result in scale_results:
        print(f"  Length {result.text_length:6,}: {result.mean_search_time*1000:7.3f} ms "
              f"({result.throughput_chars_per_sec/1e6:.2f} MB/s)")
