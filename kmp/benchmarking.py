"""
Benchmarking utilities for measuring performance of KMP algorithm.

This module provides functions to measure time, memory usage, and other
performance metrics for pattern matching algorithms.
"""

import time
import tracemalloc
import statistics
import gc
from typing import Callable, List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, asdict
import sys

from .config import BENCHMARK_CONFIG


@dataclass
class BenchmarkResult:
    """
    Container for benchmark results.
    
    Attributes:
        algorithm: Name of the algorithm
        text_length: Length of text searched
        pattern_length: Length of pattern
        num_matches: Number of matches found
        preprocessing_time: Time for preprocessing (seconds)
        search_time: Time for search (seconds)
        total_time: Total time (seconds)
        memory_used: Peak memory usage (bytes)
        num_runs: Number of benchmark runs
        mean_time: Mean execution time
        median_time: Median execution time
        std_dev_time: Standard deviation of execution time
        min_time: Minimum execution time
        max_time: Maximum execution time
    """
    algorithm: str
    text_length: int
    pattern_length: int
    num_matches: int
    preprocessing_time: float
    search_time: float
    total_time: float
    memory_used: int
    num_runs: int
    mean_time: float
    median_time: float
    std_dev_time: float
    min_time: float
    max_time: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    def __repr__(self) -> str:
        """String representation."""
        return (f"BenchmarkResult(algorithm='{self.algorithm}', "
                f"text_len={self.text_length}, pattern_len={self.pattern_length}, "
                f"matches={self.num_matches}, time={self.mean_time:.6f}s)")


def measure_time_once(func: Callable, *args, **kwargs) -> Tuple[Any, float]:
    """
    Measure execution time of a function (single run).
    
    Args:
        func: Function to measure
        *args: Positional arguments for func
        **kwargs: Keyword arguments for func
        
    Returns:
        Tuple of (function_result, elapsed_time_seconds)
    """
    gc.collect()  # Run garbage collection before timing
    
    start = time.perf_counter()
    result = func(*args, **kwargs)
    end = time.perf_counter()
    
    return result, end - start


def measure_time_multiple(func: Callable, 
                         num_runs: int = 5,
                         warmup_runs: int = 2,
                         *args, **kwargs) -> Tuple[Any, List[float]]:
    """
    Measure execution time over multiple runs for statistical significance.
    
    Args:
        func: Function to measure
        num_runs: Number of timed runs
        warmup_runs: Number of warmup runs (not timed)
        *args: Positional arguments for func
        **kwargs: Keyword arguments for func
        
    Returns:
        Tuple of (function_result, list_of_times)
    """
    # Warmup runs
    for _ in range(warmup_runs):
        _ = func(*args, **kwargs)
    
    # Timed runs
    times = []
    result = None
    
    for _ in range(num_runs):
        result, elapsed = measure_time_once(func, *args, **kwargs)
        times.append(elapsed)
    
    return result, times


def measure_memory_usage(func: Callable, *args, **kwargs) -> Tuple[Any, int, int]:
    """
    Measure peak memory usage of a function.
    
    Args:
        func: Function to measure
        *args: Positional arguments for func
        **kwargs: Keyword arguments for func
        
    Returns:
        Tuple of (function_result, current_memory_bytes, peak_memory_bytes)
    """
    gc.collect()
    tracemalloc.start()
    
    result = func(*args, **kwargs)
    
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    return result, current, peak


def benchmark_kmp_search(kmp_instance, 
                        text: str,
                        num_runs: int = None,
                        warmup_runs: int = None,
                        measure_memory: bool = True) -> BenchmarkResult:
    """
    Comprehensive benchmark of KMP search.
    
    Args:
        kmp_instance: Instance of KMP class
        text: Text to search in
        num_runs: Number of benchmark runs (uses config default if None)
        warmup_runs: Number of warmup runs (uses config default if None)
        measure_memory: Whether to measure memory usage
        
    Returns:
        BenchmarkResult object with all metrics
    """
    if num_runs is None:
        num_runs = BENCHMARK_CONFIG['num_runs']
    if warmup_runs is None:
        warmup_runs = BENCHMARK_CONFIG['warmup_runs']
    
    # Get preprocessing time from KMP instance
    preprocessing_time = kmp_instance.preprocessing_time
    
    # Measure search time over multiple runs
    matches, times = measure_time_multiple(
        kmp_instance.search, 
        num_runs=num_runs,
        warmup_runs=warmup_runs,
        text=text
    )
    
    # Measure memory if requested
    memory_used = 0
    if measure_memory:
        _, _, peak_memory = measure_memory_usage(kmp_instance.search, text)
        memory_used = peak_memory
    
    # Calculate statistics
    mean_time = statistics.mean(times)
    median_time = statistics.median(times)
    std_dev = statistics.stdev(times) if len(times) > 1 else 0.0
    min_time = min(times)
    max_time = max(times)
    
    return BenchmarkResult(
        algorithm='KMP',
        text_length=len(text),
        pattern_length=len(kmp_instance.pattern),
        num_matches=len(matches),
        preprocessing_time=preprocessing_time,
        search_time=mean_time,
        total_time=preprocessing_time + mean_time,
        memory_used=memory_used,
        num_runs=num_runs,
        mean_time=mean_time,
        median_time=median_time,
        std_dev_time=std_dev,
        min_time=min_time,
        max_time=max_time
    )


def benchmark_function(func: Callable,
                      func_name: str,
                      num_runs: int = None,
                      warmup_runs: int = None,
                      measure_memory: bool = True,
                      *args, **kwargs) -> Dict[str, Any]:
    """
    Generic benchmark for any function.
    
    Args:
        func: Function to benchmark
        func_name: Name for identification
        num_runs: Number of benchmark runs
        warmup_runs: Number of warmup runs
        measure_memory: Whether to measure memory
        *args: Positional arguments for func
        **kwargs: Keyword arguments for func
        
    Returns:
        Dictionary with benchmark results
    """
    if num_runs is None:
        num_runs = BENCHMARK_CONFIG['num_runs']
    if warmup_runs is None:
        warmup_runs = BENCHMARK_CONFIG['warmup_runs']
    
    # Measure time
    result, times = measure_time_multiple(
        func, num_runs=num_runs, warmup_runs=warmup_runs, *args, **kwargs
    )
    
    # Measure memory
    memory_used = 0
    if measure_memory:
        _, _, peak_memory = measure_memory_usage(func, *args, **kwargs)
        memory_used = peak_memory
    
    return {
        'function': func_name,
        'num_runs': num_runs,
        'mean_time': statistics.mean(times),
        'median_time': statistics.median(times),
        'std_dev_time': statistics.stdev(times) if len(times) > 1 else 0.0,
        'min_time': min(times),
        'max_time': max(times),
        'memory_used': memory_used,
        'result': result
    }


def compare_algorithms(text: str,
                      pattern: str,
                      algorithms: Dict[str, Callable],
                      num_runs: int = None) -> Dict[str, BenchmarkResult]:
    """
    Compare multiple algorithms on the same input.
    
    Args:
        text: Text to search in
        pattern: Pattern to search for
        algorithms: Dictionary mapping algorithm names to search functions
        num_runs: Number of runs per algorithm
        
    Returns:
        Dictionary mapping algorithm names to BenchmarkResult objects
        
    Example:
        >>> from kmp.kmp_algorithm import KMP
        >>> import re
        >>> text = "ATCGATCG" * 1000
        >>> pattern = "ATCG"
        >>> kmp = KMP(pattern)
        >>> algorithms = {
        ...     'KMP': lambda t: kmp.search(t),
        ...     'Python re': lambda t: [m.start() for m in re.finditer(pattern, t)]
        ... }
        >>> results = compare_algorithms(text, pattern, algorithms)
    """
    results = {}
    
    for name, func in algorithms.items():
        print(f"Benchmarking {name}...")
        
        # Measure preprocessing time (if applicable)
        preprocessing_time = 0.0
        
        # For KMP, extract preprocessing time
        if hasattr(func, '__self__') and hasattr(func.__self__, 'preprocessing_time'):
            preprocessing_time = func.__self__.preprocessing_time
        
        # Benchmark the search
        result, times = measure_time_multiple(func, num_runs or BENCHMARK_CONFIG['num_runs'], text=text)
        _, _, memory = measure_memory_usage(func, text)
        
        # Calculate statistics
        mean_time = statistics.mean(times)
        
        results[name] = BenchmarkResult(
            algorithm=name,
            text_length=len(text),
            pattern_length=len(pattern),
            num_matches=len(result) if isinstance(result, list) else 0,
            preprocessing_time=preprocessing_time,
            search_time=mean_time,
            total_time=preprocessing_time + mean_time,
            memory_used=memory,
            num_runs=len(times),
            mean_time=mean_time,
            median_time=statistics.median(times),
            std_dev_time=statistics.stdev(times) if len(times) > 1 else 0.0,
            min_time=min(times),
            max_time=max(times)
        )
    
    return results


def calculate_throughput(num_operations: int, time_seconds: float) -> float:
    """
    Calculate throughput (operations per second).
    
    Args:
        num_operations: Number of operations performed
        time_seconds: Time taken in seconds
        
    Returns:
        Throughput (operations/second)
    """
    return num_operations / time_seconds if time_seconds > 0 else 0.0


def calculate_speedup(baseline_time: float, optimized_time: float) -> float:
    """
    Calculate speedup factor.
    
    Args:
        baseline_time: Baseline execution time
        optimized_time: Optimized execution time
        
    Returns:
        Speedup factor (baseline_time / optimized_time)
    """
    return baseline_time / optimized_time if optimized_time > 0 else float('inf')


def format_benchmark_results(results: List[BenchmarkResult]) -> str:
    """
    Format benchmark results as a readable table.
    
    Args:
        results: List of BenchmarkResult objects
        
    Returns:
        Formatted string table
    """
    if not results:
        return "No results to display"
    
    # Header
    header = (f"{'Algorithm':<15} {'Text Len':>10} {'Pattern':>8} "
             f"{'Matches':>8} {'Mean Time':>12} {'Memory':>10}")
    separator = "-" * len(header)
    
    lines = [separator, header, separator]
    
    # Data rows
    for r in results:
        line = (f"{r.algorithm:<15} {r.text_length:>10} {r.pattern_length:>8} "
               f"{r.num_matches:>8} {r.mean_time:>12.6f}s "
               f"{r.memory_used / 1024:>9.1f}KB")
        lines.append(line)
    
    lines.append(separator)
    
    return '\n'.join(lines)


class BenchmarkTimer:
    """
    Context manager for timing code blocks.
    
    Example:
        >>> with BenchmarkTimer("My operation") as timer:
        ...     # Do something
        ...     result = kmp.search(text)
        >>> print(f"Took {timer.elapsed:.4f} seconds")
    """
    
    def __init__(self, name: str = "Operation", verbose: bool = True):
        """
        Initialize timer.
        
        Args:
            name: Name of the operation being timed
            verbose: Whether to print timing automatically
        """
        self.name = name
        self.verbose = verbose
        self.start_time = None
        self.end_time = None
        self.elapsed = None
    
    def __enter__(self):
        """Start timing."""
        self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop timing and optionally print result."""
        self.end_time = time.perf_counter()
        self.elapsed = self.end_time - self.start_time
        
        if self.verbose:
            print(f"{self.name} took {self.elapsed:.6f} seconds")
        
        return False


def get_memory_usage() -> int:
    """
    Get current process memory usage.
    
    Returns:
        Memory usage in bytes
    """
    try:
        import psutil
        process = psutil.Process()
        return process.memory_info().rss
    except ImportError:
        # Fallback if psutil not available
        return tracemalloc.get_traced_memory()[0] if tracemalloc.is_tracing() else 0
