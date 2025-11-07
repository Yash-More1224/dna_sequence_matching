"""
Benchmarking utilities for Wagner-Fischer algorithm.
Measures latency, memory usage, and scalability.
"""

import time
import tracemalloc
import psutil
import gc
import re
from typing import List, Dict, Tuple, Callable, Any
from dataclasses import dataclass, asdict
import pandas as pd
import numpy as np
from pathlib import Path
import json

from wf_core import WagnerFischer
from wf_search import PatternSearcher
from data_loader import SyntheticDataGenerator


@dataclass
class BenchmarkResult:
    """Single benchmark result."""
    algorithm: str
    test_name: str
    pattern_length: int
    text_length: int
    edit_distance_threshold: int
    
    # Performance metrics
    time_mean: float
    time_median: float
    time_std: float
    time_min: float
    time_max: float
    
    # Memory metrics
    memory_peak_mb: float
    memory_current_mb: float
    
    # Result metrics
    matches_found: int
    
    # Additional info
    iterations: int = 1
    metadata: Dict = None


class PerformanceBenchmark:
    """
    Comprehensive benchmarking suite for Wagner-Fischer algorithm.
    """
    
    def __init__(self, output_dir: str = "results/benchmarks"):
        """
        Initialize benchmark suite.
        
        Args:
            output_dir: Directory to save results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results: List[BenchmarkResult] = []
    
    def measure_time(self, 
                     func: Callable, 
                     iterations: int = 10) -> Tuple[List[float], Any]:
        """
        Measure execution time over multiple iterations.
        
        Args:
            func: Function to benchmark
            iterations: Number of iterations
            
        Returns:
            Tuple of (times_list, last_result)
        """
        times = []
        result = None
        
        for _ in range(iterations):
            gc.collect()
            start = time.perf_counter()
            result = func()
            end = time.perf_counter()
            times.append(end - start)
        
        return times, result
    
    def measure_memory(self, func: Callable) -> Tuple[float, float, Any]:
        """
        Measure memory usage of function.
        
        Args:
            func: Function to benchmark
            
        Returns:
            Tuple of (peak_memory_mb, current_memory_mb, result)
        """
        gc.collect()
        tracemalloc.start()
        
        result = func()
        
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        current_mb = current / 1024 / 1024
        peak_mb = peak / 1024 / 1024
        
        return peak_mb, current_mb, result
    
    def benchmark_edit_distance(self,
                               pattern_lengths: List[int],
                               text_length: int = 10000,
                               iterations: int = 10):
        """
        Benchmark edit distance computation for varying pattern lengths.
        
        Args:
            pattern_lengths: List of pattern lengths to test
            text_length: Fixed text length
            iterations: Number of iterations per test
        """
        generator = SyntheticDataGenerator(seed=42)
        wf = WagnerFischer()
        
        print("Benchmarking edit distance computation...")
        
        for plen in pattern_lengths:
            pattern = generator.generate_random_sequence(plen)
            text = generator.generate_random_sequence(text_length)
            
            # Measure time
            times, _ = self.measure_time(
                lambda: wf.compute_distance(pattern, text),
                iterations=iterations
            )
            
            # Measure memory
            peak_mem, curr_mem, result = self.measure_memory(
                lambda: wf.compute_distance(pattern, text)
            )
            
            result_obj = BenchmarkResult(
                algorithm="wagner_fischer",
                test_name="edit_distance",
                pattern_length=plen,
                text_length=text_length,
                edit_distance_threshold=0,
                time_mean=np.mean(times),
                time_median=np.median(times),
                time_std=np.std(times),
                time_min=np.min(times),
                time_max=np.max(times),
                memory_peak_mb=peak_mem,
                memory_current_mb=curr_mem,
                matches_found=1,
                iterations=iterations
            )
            
            self.results.append(result_obj)
            print(f"  Pattern length {plen}: {result_obj.time_mean*1000:.2f}ms")
    
    def benchmark_pattern_search(self,
                                pattern_length: int,
                                text_lengths: List[int],
                                max_distance: int = 2,
                                iterations: int = 5):
        """
        Benchmark pattern search for varying text lengths.
        
        Args:
            pattern_length: Fixed pattern length
            text_lengths: List of text lengths to test
            max_distance: Maximum edit distance
            iterations: Number of iterations per test
        """
        generator = SyntheticDataGenerator(seed=42)
        searcher = PatternSearcher(max_distance=max_distance)
        
        print(f"Benchmarking pattern search (pattern={pattern_length}, k={max_distance})...")
        
        for tlen in text_lengths:
            pattern = generator.generate_random_sequence(pattern_length)
            text = generator.generate_random_sequence(tlen)
            
            # Measure time
            times, matches = self.measure_time(
                lambda: searcher.search(pattern, text),
                iterations=iterations
            )
            
            # Measure memory
            peak_mem, curr_mem, _ = self.measure_memory(
                lambda: searcher.search(pattern, text)
            )
            
            result_obj = BenchmarkResult(
                algorithm="wagner_fischer_search",
                test_name="pattern_search",
                pattern_length=pattern_length,
                text_length=tlen,
                edit_distance_threshold=max_distance,
                time_mean=np.mean(times),
                time_median=np.median(times),
                time_std=np.std(times),
                time_min=np.min(times),
                time_max=np.max(times),
                memory_peak_mb=peak_mem,
                memory_current_mb=curr_mem,
                matches_found=len(matches),
                iterations=iterations
            )
            
            self.results.append(result_obj)
            print(f"  Text length {tlen}: {result_obj.time_mean*1000:.2f}ms, {len(matches)} matches")
    
    def benchmark_threshold_scaling(self,
                                   pattern_length: int = 50,
                                   text_length: int = 10000,
                                   thresholds: List[int] = None,
                                   iterations: int = 5):
        """
        Benchmark effect of edit distance threshold on performance.
        
        Args:
            pattern_length: Pattern length
            text_length: Text length
            thresholds: List of edit distance thresholds
            iterations: Number of iterations per test
        """
        if thresholds is None:
            thresholds = [0, 1, 2, 3, 5, 7, 10]
        
        generator = SyntheticDataGenerator(seed=42)
        pattern = generator.generate_random_sequence(pattern_length)
        text = generator.generate_random_sequence(text_length)
        
        print("Benchmarking threshold scaling...")
        
        for threshold in thresholds:
            searcher = PatternSearcher(max_distance=threshold)
            
            times, matches = self.measure_time(
                lambda: searcher.search(pattern, text),
                iterations=iterations
            )
            
            peak_mem, curr_mem, _ = self.measure_memory(
                lambda: searcher.search(pattern, text)
            )
            
            result_obj = BenchmarkResult(
                algorithm="wagner_fischer_search",
                test_name="threshold_scaling",
                pattern_length=pattern_length,
                text_length=text_length,
                edit_distance_threshold=threshold,
                time_mean=np.mean(times),
                time_median=np.median(times),
                time_std=np.std(times),
                time_min=np.min(times),
                time_max=np.max(times),
                memory_peak_mb=peak_mem,
                memory_current_mb=curr_mem,
                matches_found=len(matches),
                iterations=iterations
            )
            
            self.results.append(result_obj)
            print(f"  Threshold {threshold}: {result_obj.time_mean*1000:.2f}ms, {len(matches)} matches")
    
    def benchmark_regex_comparison(self,
                                  pattern_length: int = 20,
                                  text_length: int = 10000,
                                  iterations: int = 10):
        """
        Compare Wagner-Fischer with Python's regex engine.
        
        Args:
            pattern_length: Pattern length
            text_length: Text length
            iterations: Number of iterations
        """
        generator = SyntheticDataGenerator(seed=42)
        pattern = generator.generate_random_sequence(pattern_length)
        text = generator.generate_random_sequence(text_length)
        
        print("Comparing with Python regex...")
        
        # Wagner-Fischer exact match
        searcher = PatternSearcher(max_distance=0)
        wf_times, wf_matches = self.measure_time(
            lambda: searcher.search(pattern, text),
            iterations=iterations
        )
        wf_peak_mem, wf_curr_mem, _ = self.measure_memory(
            lambda: searcher.search(pattern, text)
        )
        
        # Regex exact match
        regex_times, regex_matches = self.measure_time(
            lambda: [(m.start(), m.end()) for m in re.finditer(pattern, text)],
            iterations=iterations
        )
        regex_peak_mem, regex_curr_mem, _ = self.measure_memory(
            lambda: [(m.start(), m.end()) for m in re.finditer(pattern, text)]
        )
        
        # Store WF results
        self.results.append(BenchmarkResult(
            algorithm="wagner_fischer",
            test_name="regex_comparison",
            pattern_length=pattern_length,
            text_length=text_length,
            edit_distance_threshold=0,
            time_mean=np.mean(wf_times),
            time_median=np.median(wf_times),
            time_std=np.std(wf_times),
            time_min=np.min(wf_times),
            time_max=np.max(wf_times),
            memory_peak_mb=wf_peak_mem,
            memory_current_mb=wf_curr_mem,
            matches_found=len(wf_matches),
            iterations=iterations
        ))
        
        # Store regex results
        self.results.append(BenchmarkResult(
            algorithm="python_regex",
            test_name="regex_comparison",
            pattern_length=pattern_length,
            text_length=text_length,
            edit_distance_threshold=0,
            time_mean=np.mean(regex_times),
            time_median=np.median(regex_times),
            time_std=np.std(regex_times),
            time_min=np.min(regex_times),
            time_max=np.max(regex_times),
            memory_peak_mb=regex_peak_mem,
            memory_current_mb=regex_curr_mem,
            matches_found=len(regex_matches),
            iterations=iterations
        ))
        
        print(f"  Wagner-Fischer: {np.mean(wf_times)*1000:.2f}ms")
        print(f"  Python regex: {np.mean(regex_times)*1000:.2f}ms")
    
    def save_results(self, filename: str = "benchmark_results.csv"):
        """
        Save benchmark results to CSV.
        
        Args:
            filename: Output filename
        """
        if not self.results:
            print("No results to save")
            return
        
        df = pd.DataFrame([asdict(r) for r in self.results])
        output_path = self.output_dir / filename
        df.to_csv(output_path, index=False)
        print(f"Results saved to {output_path}")
        
        # Also save as JSON
        json_path = self.output_dir / filename.replace('.csv', '.json')
        df.to_json(json_path, orient='records', indent=2)
        print(f"Results also saved to {json_path}")
    
    def run_full_suite(self):
        """Run complete benchmark suite."""
        print("=" * 60)
        print("Running Wagner-Fischer Benchmark Suite")
        print("=" * 60)
        
        # Test 1: Edit distance scaling
        self.benchmark_edit_distance(
            pattern_lengths=[10, 20, 50, 100, 200, 500],
            text_length=1000,
            iterations=10
        )
        
        # Test 2: Text length scaling
        self.benchmark_pattern_search(
            pattern_length=30,
            text_lengths=[1000, 5000, 10000, 50000, 100000],
            max_distance=2,
            iterations=5
        )
        
        # Test 3: Threshold scaling
        self.benchmark_threshold_scaling(
            pattern_length=50,
            text_length=10000,
            thresholds=[0, 1, 2, 3, 5, 7, 10],
            iterations=5
        )
        
        # Test 4: Regex comparison
        self.benchmark_regex_comparison(
            pattern_length=20,
            text_length=10000,
            iterations=10
        )
        
        # Save all results
        self.save_results()
        
        print("\n" + "=" * 60)
        print("Benchmark suite complete!")
        print("=" * 60)
