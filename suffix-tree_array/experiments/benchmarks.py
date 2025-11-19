"""
Benchmarking Framework for Suffix Array Implementation

Provides tools for measuring performance of Suffix Array algorithm:
- Time measurement (preprocessing + search)
- Memory profiling
- Comparison with Python's re module
- Statistics tracking
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
    index_memory: int = 0
    
    # Algorithm-specific metrics
    comparisons: int = 0
    
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
            'index_memory': self.index_memory,
            'comparisons': self.comparisons,
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
    mean_index_memory: float = 0.0
    
    # Other metrics
    mean_comparisons: float = 0.0
    mean_matches: float = 0.0
    throughput_mbps: float = 0.0
    
    num_runs: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'algorithm': self.algorithm,
            'pattern_length': self.pattern_length,
            'text_length': self.text_length,
            'mean_total_time': self.mean_total_time,
            'median_total_time': self.median_total_time,
            'std_total_time': self.std_total_time,
            'min_total_time': self.min_total_time,
            'max_total_time': self.max_total_time,
            'mean_search_time': self.mean_search_time,
            'mean_preprocessing_time': self.mean_preprocessing_time,
            'mean_peak_memory': self.mean_peak_memory,
            'max_peak_memory': self.max_peak_memory,
            'mean_index_memory': self.mean_index_memory,
            'mean_comparisons': self.mean_comparisons,
            'mean_matches': self.mean_matches,
            'throughput_mbps': self.throughput_mbps,
            'num_runs': self.num_runs
        }


class Benchmarker:
    """Benchmark Suffix Array implementations."""
    
    def __init__(self, warmup_runs: int = 3, min_runs: int = 5):
        """
        Initialize benchmarker.
        
        Args:
            warmup_runs: Number of warmup iterations
            min_runs: Minimum number of measurement runs
        """
        self.warmup_runs = warmup_runs
        self.min_runs = min_runs
    
    def benchmark_suffix_array(self, sa, text: str, pattern: str,
                               measure_memory: bool = True) -> BenchmarkResult:
        """
        Benchmark a Suffix Array instance.
        
        Args:
            sa: SuffixArray instance (already built)
            text: Text to search in
            pattern: Pattern to search for
            measure_memory: Whether to measure memory usage
            
        Returns:
            Benchmark result
        """
        # Get preprocessing time and memory from construction
        preprocessing_time = sa.preprocessing_time
        index_memory = sa.memory_footprint
        
        # Measure search time and memory
        if measure_memory:
            tracemalloc.start()
        
        start_time = time.perf_counter()
        matches = sa.search(pattern)
        end_time = time.perf_counter()
        
        search_time = end_time - start_time
        
        peak_memory = 0
        if measure_memory:
            current, peak = tracemalloc.get_traced_memory()
            peak_memory = peak
            tracemalloc.stop()
        
        # Get statistics
        stats = sa.get_statistics()
        
        return BenchmarkResult(
            algorithm="SuffixArray",
            pattern_length=len(pattern),
            text_length=len(text),
            num_matches=len(matches),
            preprocessing_time=preprocessing_time,
            search_time=search_time,
            peak_memory=peak_memory,
            index_memory=index_memory,
            comparisons=stats.get('comparisons', 0)
        )
    
    def benchmark_with_repetitions(self, sa, text: str, pattern: str,
                                   repetitions: int = 5) -> AggregatedResult:
        """
        Benchmark with multiple repetitions and aggregate results.
        
        Args:
            sa: SuffixArray instance
            text: Text to search in
            pattern: Pattern to search for
            repetitions: Number of repetitions
            
        Returns:
            Aggregated results
        """
        # Warmup runs
        for _ in range(self.warmup_runs):
            _ = sa.search(pattern)
        
        # Measurement runs
        results = []
        for _ in range(repetitions):
            result = self.benchmark_suffix_array(sa, text, pattern, measure_memory=False)
            results.append(result)
        
        # Aggregate statistics
        total_times = [r.total_time for r in results]
        search_times = [r.search_time for r in results]
        preprocessing_times = [r.preprocessing_time for r in results]
        
        aggregated = AggregatedResult(
            algorithm="SuffixArray",
            pattern_length=len(pattern),
            text_length=len(text),
            mean_total_time=statistics.mean(total_times),
            median_total_time=statistics.median(total_times),
            std_total_time=statistics.stdev(total_times) if len(total_times) > 1 else 0.0,
            min_total_time=min(total_times),
            max_total_time=max(total_times),
            mean_search_time=statistics.mean(search_times),
            mean_preprocessing_time=statistics.mean(preprocessing_times),
            mean_index_memory=results[0].index_memory,
            mean_comparisons=statistics.mean([r.comparisons for r in results]),
            mean_matches=statistics.mean([r.num_matches for r in results]),
            throughput_mbps=len(text) / statistics.mean(search_times) / 1_000_000,
            num_runs=repetitions
        )
        
        return aggregated
    
    def compare_with_re(self, text: str, pattern: str, 
                       repetitions: int = 5) -> Dict[str, Any]:
        """
        Benchmark Python's re module for comparison.
        
        Args:
            text: Text to search in
            pattern: Pattern to search for
            repetitions: Number of repetitions
            
        Returns:
            Dictionary with benchmark results
        """
        # Warmup
        for _ in range(self.warmup_runs):
            _ = [m.start() for m in re.finditer(pattern, text)]
        
        # Measurement
        times = []
        num_matches = 0
        
        for _ in range(repetitions):
            start_time = time.perf_counter()
            matches = [m.start() for m in re.finditer(pattern, text)]
            end_time = time.perf_counter()
            
            times.append(end_time - start_time)
            num_matches = len(matches)
        
        mean_time = statistics.mean(times)
        
        return {
            'algorithm': 'Python re',
            'pattern_length': len(pattern),
            'text_length': len(text),
            'num_matches': num_matches,
            'mean_time': mean_time,
            'median_time': statistics.median(times),
            'std_time': statistics.stdev(times) if len(times) > 1 else 0.0,
            'min_time': min(times),
            'max_time': max(times),
            'throughput_mbps': len(text) / mean_time / 1_000_000,
            'num_runs': repetitions
        }
