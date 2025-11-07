"""
Benchmarking Framework

Provides tools for measuring performance of Boyer-Moore algorithm:
- Time measurement
- Memory profiling
- Comparison with Python's re module
"""

import time
import re
import tracemalloc
import psutil
import statistics
from typing import List, Dict, Callable, Any, Tuple
from dataclasses import dataclass, field


@dataclass
class BenchmarkResult:
    """Store results from a single benchmark run."""
    
    algorithm: str
    pattern_length: int
    text_length: int
    num_matches: int
    
    # Time metrics (in seconds)
    preprocessing_time: float = 0.0
    search_time: float = 0.0
    total_time: float = 0.0
    
    # Memory metrics (in bytes)
    peak_memory: int = 0
    
    # Algorithm-specific metrics
    comparisons: int = 0
    shifts: int = 0
    
    # Metadata
    repetitions: int = 1
    warmup_runs: int = 0
    
    def __post_init__(self):
        """Calculate derived metrics."""
        if self.total_time == 0.0:
            self.total_time = self.preprocessing_time + self.search_time
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'algorithm': self.algorithm,
            'pattern_length': self.pattern_length,
            'text_length': self.text_length,
            'num_matches': self.num_matches,
            'preprocessing_time': self.preprocessing_time,
            'search_time': self.search_time,
            'total_time': self.total_time,
            'peak_memory': self.peak_memory,
            'comparisons': self.comparisons,
            'shifts': self.shifts,
            'repetitions': self.repetitions
        }


@dataclass
class AggregatedResult:
    """Store aggregated results from multiple runs."""
    
    algorithm: str
    pattern_length: int
    text_length: int
    
    # Time statistics (in seconds)
    mean_total_time: float = 0.0
    median_total_time: float = 0.0
    std_total_time: float = 0.0
    min_total_time: float = 0.0
    max_total_time: float = 0.0
    
    mean_search_time: float = 0.0
    mean_preprocessing_time: float = 0.0
    
    # Memory statistics
    mean_peak_memory: float = 0.0
    max_peak_memory: int = 0
    
    # Other metrics
    mean_comparisons: float = 0.0
    mean_shifts: float = 0.0
    mean_matches: float = 0.0
    
    num_runs: int = 0


class Benchmarker:
    """Benchmark Boyer-Moore implementations."""
    
    def __init__(self, warmup_runs: int = 3, min_runs: int = 5):
        """
        Initialize benchmarker.
        
        Args:
            warmup_runs: Number of warmup iterations
            min_runs: Minimum number of measurement runs
        """
        self.warmup_runs = warmup_runs
        self.min_runs = min_runs
    
    def benchmark_boyer_moore(self, matcher, text: str, 
                             measure_memory: bool = True) -> BenchmarkResult:
        """
        Benchmark a Boyer-Moore matcher instance.
        
        Args:
            matcher: Boyer-Moore matcher (already initialized with pattern)
            text: Text to search in
            measure_memory: Whether to measure memory usage
            
        Returns:
            Benchmark result
        """
        # Get preprocessing time from initialization
        # (We assume pattern is already preprocessed in matcher)
        preprocessing_time = 0.0  # Already done during __init__
        
        # Measure search time and memory
        if measure_memory:
            tracemalloc.start()
        
        start_time = time.perf_counter()
        matches = matcher.search(text)
        end_time = time.perf_counter()
        
        search_time = end_time - start_time
        
        peak_memory = 0
        if measure_memory:
            current, peak = tracemalloc.get_traced_memory()
            peak_memory = peak
            tracemalloc.stop()
        
        # Get statistics from matcher
        stats = matcher.get_statistics()
        
        return BenchmarkResult(
            algorithm=type(matcher).__name__,
            pattern_length=len(matcher.pattern),
            text_length=len(text),
            num_matches=len(matches),
            preprocessing_time=preprocessing_time,
            search_time=search_time,
            peak_memory=peak_memory,
            comparisons=stats.get('comparisons', 0),
            shifts=stats.get('shifts', 0),
            repetitions=1
        )
    
    def benchmark_python_re(self, pattern: str, text: str) -> BenchmarkResult:
        """
        Benchmark Python's built-in re module.
        
        Args:
            pattern: Pattern to search for
            text: Text to search in
            
        Returns:
            Benchmark result
        """
        # Measure compilation time
        start_time = time.perf_counter()
        compiled_pattern = re.compile(pattern, re.IGNORECASE)
        compile_time = time.perf_counter() - start_time
        
        # Measure search time
        tracemalloc.start()
        start_time = time.perf_counter()
        matches = [m.start() for m in compiled_pattern.finditer(text)]
        search_time = time.perf_counter() - start_time
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        return BenchmarkResult(
            algorithm="Python_re",
            pattern_length=len(pattern),
            text_length=len(text),
            num_matches=len(matches),
            preprocessing_time=compile_time,
            search_time=search_time,
            peak_memory=peak,
            comparisons=0,  # Not available for re
            shifts=0
        )
    
    def run_multiple(self, benchmark_func: Callable, 
                    num_runs: int = None) -> List[BenchmarkResult]:
        """
        Run benchmark multiple times.
        
        Args:
            benchmark_func: Function that returns BenchmarkResult
            num_runs: Number of runs (uses self.min_runs if None)
            
        Returns:
            List of benchmark results
        """
        if num_runs is None:
            num_runs = self.min_runs
        
        results = []
        
        # Warmup runs
        for _ in range(self.warmup_runs):
            benchmark_func()
        
        # Actual measurement runs
        for _ in range(num_runs):
            result = benchmark_func()
            results.append(result)
        
        return results
    
    def aggregate_results(self, results: List[BenchmarkResult]) -> AggregatedResult:
        """
        Aggregate multiple benchmark results.
        
        Args:
            results: List of benchmark results
            
        Returns:
            Aggregated result with statistics
        """
        if not results:
            raise ValueError("No results to aggregate")
        
        first = results[0]
        
        total_times = [r.total_time for r in results]
        search_times = [r.search_time for r in results]
        preprocessing_times = [r.preprocessing_time for r in results]
        peak_memories = [r.peak_memory for r in results]
        comparisons = [r.comparisons for r in results]
        shifts = [r.shifts for r in results]
        matches = [r.num_matches for r in results]
        
        return AggregatedResult(
            algorithm=first.algorithm,
            pattern_length=first.pattern_length,
            text_length=first.text_length,
            mean_total_time=statistics.mean(total_times),
            median_total_time=statistics.median(total_times),
            std_total_time=statistics.stdev(total_times) if len(total_times) > 1 else 0.0,
            min_total_time=min(total_times),
            max_total_time=max(total_times),
            mean_search_time=statistics.mean(search_times),
            mean_preprocessing_time=statistics.mean(preprocessing_times),
            mean_peak_memory=statistics.mean(peak_memories),
            max_peak_memory=max(peak_memories),
            mean_comparisons=statistics.mean(comparisons) if comparisons[0] > 0 else 0.0,
            mean_shifts=statistics.mean(shifts) if shifts[0] > 0 else 0.0,
            mean_matches=statistics.mean(matches),
            num_runs=len(results)
        )
    
    def compare_algorithms(self, text: str, pattern: str, 
                          matchers: Dict[str, Any]) -> Dict[str, AggregatedResult]:
        """
        Compare multiple algorithm implementations.
        
        Args:
            text: Text to search in
            pattern: Pattern to search for
            matchers: Dictionary mapping names to matcher classes
            
        Returns:
            Dictionary mapping algorithm names to aggregated results
        """
        comparison_results = {}
        
        for name, matcher_class in matchers.items():
            # Create matcher instance
            matcher = matcher_class(pattern)
            
            # Run benchmarks
            results = self.run_multiple(
                lambda: self.benchmark_boyer_moore(matcher, text, measure_memory=True)
            )
            
            # Aggregate
            agg_result = self.aggregate_results(results)
            comparison_results[name] = agg_result
        
        return comparison_results


if __name__ == "__main__":
    # Test benchmarking framework
    import sys
    sys.path.append('..')
    from src.boyer_moore import BoyerMoore
    from src.data_generator import DNAGenerator
    
    print("Benchmarking Framework Test")
    print("=" * 50)
    
    # Generate test data
    gen = DNAGenerator(seed=42)
    text, pattern, positions = gen.generate_test_case(
        text_length=10000,
        pattern_length=10,
        num_occurrences=5
    )
    
    print(f"Text length: {len(text):,}")
    print(f"Pattern: {pattern}")
    print(f"Expected matches: {len(positions)}")
    print()
    
    # Benchmark Boyer-Moore
    benchmarker = Benchmarker(warmup_runs=2, min_runs=5)
    
    matcher = BoyerMoore(pattern)
    results = benchmarker.run_multiple(
        lambda: benchmarker.benchmark_boyer_moore(matcher, text)
    )
    
    agg = benchmarker.aggregate_results(results)
    
    print("=== Boyer-Moore Results ===")
    print(f"Mean total time: {agg.mean_total_time*1000:.3f} ms")
    print(f"Std dev: {agg.std_total_time*1000:.3f} ms")
    print(f"Mean comparisons: {agg.mean_comparisons:.0f}")
    print(f"Mean shifts: {agg.mean_shifts:.0f}")
    print(f"Matches found: {agg.mean_matches:.0f}")
