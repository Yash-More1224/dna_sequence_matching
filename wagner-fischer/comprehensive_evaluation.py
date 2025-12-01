#!/usr/bin/env python3
"""
Comprehensive Wagner-Fischer Algorithm Evaluation

Evaluates Wagner-Fischer (Edit Distance) on all required criteria:
1. Latency / Time (runtime, per-query latency, throughput)
2. Preprocessing time (not applicable for WF, minimal overhead)
3. Memory usage (peak memory, DP matrix footprint)
4. Accuracy (precision, recall, F1 - fuzzy matching evaluation)
5. Scalability (text length, pattern set size, edit distance thresholds)
6. Robustness (alphabet size, mutation rates, pattern characteristics)
"""

import sys
import time
import json
import csv
import statistics
import tracemalloc
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple
import random
import re

# Add wagner-fischer directory to path
sys.path.insert(0, str(Path(__file__).parent))

from wf_core import WagnerFischer
from wf_search import PatternSearcher, Match
from data_loader import FastaLoader, Sequence

# Directories
DATASET_DIR = Path(__file__).parent.parent / "dataset"
RESULTS_DIR = Path(__file__).parent / "results"
BENCHMARKS_DIR = RESULTS_DIR / "benchmarks"
REPORTS_DIR = RESULTS_DIR / "reports"
PLOTS_DIR = RESULTS_DIR / "plots"

# Create directories
for d in [RESULTS_DIR, BENCHMARKS_DIR, REPORTS_DIR, PLOTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)


class ComprehensiveEvaluator:
    """Comprehensive evaluation of Wagner-Fischer algorithm."""
    
    def __init__(self):
        """Initialize evaluator."""
        self.datasets = {}
        self.results = {
            'latency_time': [],
            'preprocessing': [],
            'memory': [],
            'accuracy': [],
            'scalability_text': [],
            'scalability_patterns': [],
            'scalability_threshold': [],
            'robustness': []
        }
        self.loader = FastaLoader()
    
    def load_datasets(self) -> None:
        """Load all available datasets."""
        print("\n" + "="*80)
        print("LOADING DATASETS")
        print("="*80)
        
        dataset_files = {
            'ecoli': 'ecoli_k12_mg1655.fasta',
            'lambda_phage': 'lambda_phage.fasta',
            'salmonella': 'salmonella_typhimurium.fasta'
        }
        
        for name, filename in dataset_files.items():
            filepath = DATASET_DIR / filename
            if filepath.exists():
                print(f"\nLoading {name}...", end=' ')
                sequences = self.loader.load(str(filepath))
                if sequences:
                    sequence = sequences[0].sequence
                    # Calculate basic stats
                    gc_content = (sequence.count('G') + sequence.count('C')) / len(sequence)
                    self.datasets[name] = {
                        'sequence': sequence,
                        'gc_content': gc_content,
                        'length': len(sequence),
                        'name': name
                    }
                    print(f"✓ {len(sequence):,} bp (GC: {gc_content:.1%})")
        
        print(f"\n✓ Loaded {len(self.datasets)} dataset(s)")
    
    def mutate_sequence(self, sequence: str, mutation_rate: float) -> str:
        """Apply random mutations to a sequence."""
        sequence_list = list(sequence)
        bases = ['A', 'C', 'G', 'T']
        num_mutations = int(len(sequence) * mutation_rate)
        
        for _ in range(num_mutations):
            pos = random.randint(0, len(sequence) - 1)
            mutation_type = random.choice(['substitute', 'insert', 'delete'])
            
            if mutation_type == 'substitute':
                original = sequence_list[pos]
                new_base = random.choice([b for b in bases if b != original])
                sequence_list[pos] = new_base
            elif mutation_type == 'insert':
                sequence_list.insert(pos, random.choice(bases))
            elif mutation_type == 'delete' and len(sequence_list) > 1:
                sequence_list.pop(pos)
        
        return ''.join(sequence_list)
    
    def criterion_1_latency_time(self) -> None:
        """
        Criterion 1: Latency / Time
        - Total runtime for approximate matching
        - Per-query latency
        - Throughput (matches/sec and MB/sec)
        - Mean, median, variance over multiple runs
        """
        print("\n" + "="*80)
        print("CRITERION 1: LATENCY / TIME ANALYSIS")
        print("="*80)
        
        pattern_lengths = [10, 20, 50, 100, 200]
        edit_distances = [0, 1, 2, 3]
        num_runs = 5
        
        for dataset_name, data in self.datasets.items():
            sequence = data['sequence']
            # Use smaller subset for Wagner-Fischer due to computational intensity
            text_size = min(50000, len(sequence))
            text = sequence[:text_size]
            
            print(f"\n{dataset_name.upper()} ({text_size:,} bp)")
            print("-" * 80)
            
            for pattern_len in pattern_lengths:
                for max_dist in edit_distances:
                    # Extract pattern from sequence
                    start_pos = random.randint(0, len(text) - pattern_len)
                    pattern = text[start_pos:start_pos + pattern_len]
                    
                    # Create searcher
                    searcher = PatternSearcher(max_distance=max_dist)
                    
                    # Multiple runs for statistical significance
                    times = []
                    matches_list = []
                    
                    for run in range(num_runs):
                        start = time.perf_counter()
                        matches = searcher.search(pattern, text)
                        elapsed = time.perf_counter() - start
                        times.append(elapsed)
                        matches_list.append(len(matches))
                    
                    # Calculate statistics
                    mean_time = statistics.mean(times)
                    median_time = statistics.median(times)
                    std_dev = statistics.stdev(times) if len(times) > 1 else 0
                    min_time = min(times)
                    max_time = max(times)
                    
                    # Calculate throughput
                    text_size_mb = text_size / (1024 * 1024)
                    throughput_mbps = text_size_mb / mean_time if mean_time > 0 else 0
                    num_matches = matches_list[0]
                    matches_per_sec = num_matches / mean_time if num_matches > 0 and mean_time > 0 else 0
                    
                    result = {
                        'dataset': dataset_name,
                        'text_length': text_size,
                        'pattern_length': pattern_len,
                        'max_edit_distance': max_dist,
                        'num_runs': num_runs,
                        'mean_time_ms': mean_time * 1000,
                        'median_time_ms': median_time * 1000,
                        'std_dev_ms': std_dev * 1000,
                        'min_time_ms': min_time * 1000,
                        'max_time_ms': max_time * 1000,
                        'throughput_mbps': throughput_mbps,
                        'num_matches': num_matches,
                        'matches_per_sec': matches_per_sec
                    }
                    
                    self.results['latency_time'].append(result)
                    
                    print(f"  Pattern {pattern_len:3d}bp, dist={max_dist}: "
                          f"mean={mean_time*1000:8.2f}ms ± {std_dev*1000:6.2f}ms, "
                          f"throughput={throughput_mbps:6.3f}MB/s, "
                          f"matches={num_matches:4d}")
    
    def criterion_2_preprocessing(self) -> None:
        """
        Criterion 2: Preprocessing Time
        - Wagner-Fischer has minimal preprocessing (just initialization)
        - Measure overhead of creating WagnerFischer objects
        """
        print("\n" + "="*80)
        print("CRITERION 2: PREPROCESSING TIME (Initialization Overhead)")
        print("="*80)
        
        pattern_lengths = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000]
        num_runs = 50
        
        # Use one dataset for pattern generation
        dataset = list(self.datasets.values())[0]
        sequence = dataset['sequence']
        
        print(f"\nMeasuring initialization and setup time...")
        print("-" * 80)
        
        for pattern_len in pattern_lengths:
            if pattern_len > len(sequence):
                continue
            
            # Extract pattern
            start_pos = random.randint(0, len(sequence) - pattern_len)
            pattern = sequence[start_pos:start_pos + pattern_len]
            
            # Measure initialization time
            init_times = []
            for _ in range(num_runs):
                start = time.perf_counter()
                wf = WagnerFischer()
                elapsed = time.perf_counter() - start
                init_times.append(elapsed)
            
            # Measure single distance computation (preprocessing equivalent)
            comp_times = []
            wf = WagnerFischer()
            for _ in range(num_runs):
                target = sequence[start_pos:start_pos + pattern_len]
                start = time.perf_counter()
                distance, _ = wf.compute_distance(pattern, target)
                elapsed = time.perf_counter() - start
                comp_times.append(elapsed)
            
            mean_init = statistics.mean(init_times)
            mean_comp = statistics.mean(comp_times)
            std_dev_init = statistics.stdev(init_times) if len(init_times) > 1 else 0
            std_dev_comp = statistics.stdev(comp_times) if len(comp_times) > 1 else 0
            
            result = {
                'pattern_length': pattern_len,
                'num_runs': num_runs,
                'mean_init_time_us': mean_init * 1e6,
                'std_dev_init_us': std_dev_init * 1e6,
                'mean_computation_time_us': mean_comp * 1e6,
                'std_dev_comp_us': std_dev_comp * 1e6,
                'time_complexity_ratio': (mean_comp * 1e6) / (pattern_len * pattern_len)  # O(m^2) for same length
            }
            
            self.results['preprocessing'].append(result)
            
            print(f"  Pattern {pattern_len:5d}bp: "
                  f"init={mean_init*1e6:6.2f}µs, "
                  f"comp={mean_comp*1e6:9.2f}µs ± {std_dev_comp*1e6:7.2f}µs, "
                  f"ratio={result['time_complexity_ratio']:.3f}µs/bp²")
    
    def criterion_3_memory_usage(self) -> None:
        """
        Criterion 3: Memory Usage
        - Peak resident memory
        - DP matrix footprint
        - Space-optimized version comparison
        """
        print("\n" + "="*80)
        print("CRITERION 3: MEMORY USAGE")
        print("="*80)
        
        pattern_lengths = [10, 50, 100, 200, 500, 1000, 2000]
        
        # Use ecoli for testing
        dataset = self.datasets.get('ecoli') or list(self.datasets.values())[0]
        sequence = dataset['sequence']
        text_size = min(10000, len(sequence))
        text = sequence[:text_size]
        
        print(f"\nMeasuring memory usage on {text_size:,}bp text...")
        print("-" * 80)
        
        for pattern_len in pattern_lengths:
            if pattern_len > len(text):
                continue
            
            # Extract pattern
            start_pos = random.randint(0, len(text) - pattern_len)
            pattern = text[start_pos:start_pos + pattern_len]
            
            wf = WagnerFischer()
            
            # Measure memory for full matrix computation
            tracemalloc.start()
            distance, matrix = wf.compute_distance(pattern, pattern, return_matrix=True)
            current_full, peak_full = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            
            # Measure memory for optimized computation
            tracemalloc.start()
            distance_opt = wf.compute_distance_optimized(pattern, pattern)
            current_opt, peak_opt = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            
            # Calculate theoretical DP matrix size
            dp_matrix_memory = pattern_len * pattern_len * 4  # 4 bytes per int32
            dp_row_memory = pattern_len * 4 * 2  # 2 rows for optimized version
            
            # Measure PatternSearcher memory for actual search
            tracemalloc.start()
            searcher = PatternSearcher(max_distance=2)
            matches = searcher.search(pattern, text[:1000])  # Small text for memory test
            current_search, peak_search = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            
            result = {
                'pattern_length': pattern_len,
                'text_length': text_size,
                'theoretical_matrix_kb': dp_matrix_memory / 1024,
                'theoretical_optimized_kb': dp_row_memory / 1024,
                'full_matrix_peak_kb': peak_full / 1024,
                'optimized_peak_kb': peak_opt / 1024,
                'search_peak_kb': peak_search / 1024,
                'memory_reduction_ratio': peak_full / peak_opt if peak_opt > 0 else 0
            }
            
            self.results['memory'].append(result)
            
            print(f"  Pattern {pattern_len:4d}bp: "
                  f"Full={peak_full/1024:8.2f}KB, "
                  f"Opt={peak_opt/1024:7.2f}KB, "
                  f"Search={peak_search/1024:8.2f}KB, "
                  f"Reduction={result['memory_reduction_ratio']:.1f}x")
    
    def criterion_4_accuracy(self) -> None:
        """
        Criterion 4: Accuracy
        - Test approximate matching with known mutations
        - Calculate precision, recall, F1 score for fuzzy matching
        """
        print("\n" + "="*80)
        print("CRITERION 4: ACCURACY (Fuzzy Matching Evaluation)")
        print("="*80)
        
        pattern_lengths = [20, 50, 100, 200]
        mutation_rates = [0.05, 0.10, 0.15, 0.20]
        max_distances = [1, 2, 3, 5]
        num_tests_per_config = 10
        
        for dataset_name, data in self.datasets.items():
            sequence = data['sequence']
            text_size = min(10000, len(sequence))
            text = sequence[:text_size]
            
            print(f"\n{dataset_name.upper()}")
            print("-" * 80)
            
            for pattern_len in pattern_lengths:
                for max_dist in max_distances:
                    all_tp = 0
                    all_fp = 0
                    all_fn = 0
                    total_expected = 0
                    total_found = 0
                    
                    for _ in range(num_tests_per_config):
                        # Extract pattern
                        start_pos = random.randint(0, len(text) - pattern_len - 50)
                        pattern = text[start_pos:start_pos + pattern_len]
                        
                        # Create mutated version
                        mutated = self.mutate_sequence(pattern, mutation_rate=max_dist/pattern_len)
                        
                        # Insert mutated pattern at known position
                        insertion_pos = random.randint(pattern_len, len(text) - pattern_len - 50)
                        text_with_mutation = text[:insertion_pos] + mutated + text[insertion_pos + len(mutated):]
                        
                        # Search with Wagner-Fischer
                        searcher = PatternSearcher(max_distance=max_dist)
                        matches = searcher.search(pattern, text_with_mutation[:5000])
                        
                        # Check if we found the inserted mutation
                        found_insertion = False
                        for match in matches:
                            # Consider it found if within reasonable range of insertion
                            if abs(match.position - insertion_pos) <= max_dist + 5:
                                found_insertion = True
                                all_tp += 1
                                break
                        
                        if not found_insertion:
                            all_fn += 1
                        
                        # Count false positives (matches far from insertion point)
                        for match in matches:
                            if abs(match.position - insertion_pos) > max_dist + 5:
                                all_fp += 1
                        
                        total_expected += 1
                        total_found += len(matches)
                    
                    # Calculate metrics
                    precision = all_tp / (all_tp + all_fp) if (all_tp + all_fp) > 0 else 0
                    recall = all_tp / (all_tp + all_fn) if (all_tp + all_fn) > 0 else 0
                    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                    
                    result = {
                        'dataset': dataset_name,
                        'pattern_length': pattern_len,
                        'max_edit_distance': max_dist,
                        'num_tests': num_tests_per_config,
                        'true_positives': all_tp,
                        'false_positives': all_fp,
                        'false_negatives': all_fn,
                        'precision': precision,
                        'recall': recall,
                        'f1_score': f1,
                        'avg_matches_per_query': total_found / num_tests_per_config
                    }
                    
                    self.results['accuracy'].append(result)
                    
                    print(f"  Pattern {pattern_len:3d}bp, dist={max_dist}: "
                          f"P={precision:.3f}, R={recall:.3f}, F1={f1:.3f}, "
                          f"TP={all_tp}, FP={all_fp}, FN={all_fn}")
    
    def criterion_5_scalability_text_length(self) -> None:
        """
        Criterion 5a: Scalability - Text Length
        - How performance scales with increasing text size
        """
        print("\n" + "="*80)
        print("CRITERION 5a: SCALABILITY - Text Length")
        print("="*80)
        
        # Use largest dataset
        dataset = self.datasets.get('ecoli') or self.datasets.get('salmonella') or list(self.datasets.values())[0]
        full_sequence = dataset['sequence']
        
        # Test various text sizes (smaller than KMP due to computational intensity)
        text_sizes = [1000, 2000, 5000, 10000, 20000, 50000]
        pattern_len = 30
        max_dist = 2
        
        print(f"\nScaling text size (pattern={pattern_len}bp, max_dist={max_dist})...")
        print("-" * 80)
        
        for text_size in text_sizes:
            if text_size > len(full_sequence):
                continue
            
            # Use subsequence
            sequence = full_sequence[:text_size]
            
            # Extract pattern
            start_pos = random.randint(0, len(sequence) - pattern_len)
            pattern = sequence[start_pos:start_pos + pattern_len]
            
            # Create searcher
            searcher = PatternSearcher(max_distance=max_dist)
            
            # Measure search time
            times = []
            for _ in range(3):  # Fewer runs due to computational cost
                start = time.perf_counter()
                matches = searcher.search(pattern, sequence)
                elapsed = time.perf_counter() - start
                times.append(elapsed)
            
            mean_time = statistics.mean(times)
            throughput = (text_size / (1024*1024)) / mean_time if mean_time > 0 else 0
            
            result = {
                'text_length': text_size,
                'pattern_length': pattern_len,
                'max_edit_distance': max_dist,
                'mean_search_time_ms': mean_time * 1000,
                'throughput_mbps': throughput,
                'time_per_char_us': (mean_time * 1e6) / text_size
            }
            
            self.results['scalability_text'].append(result)
            
            print(f"  Text {text_size:6d}bp: "
                  f"{mean_time*1000:8.2f}ms, "
                  f"{throughput:6.4f}MB/s, "
                  f"{result['time_per_char_us']:.3f}µs/char")
    
    def criterion_5_scalability_pattern_count(self) -> None:
        """
        Criterion 5b: Scalability - Pattern Count
        - How performance scales with number of patterns
        """
        print("\n" + "="*80)
        print("CRITERION 5b: SCALABILITY - Multiple Patterns")
        print("="*80)
        
        dataset = list(self.datasets.values())[0]
        sequence = dataset['sequence']
        text_size = min(5000, len(sequence))
        text = sequence[:text_size]
        
        pattern_counts = [1, 5, 10, 20, 50]
        pattern_len = 30
        max_dist = 2
        
        print(f"\nSearching with multiple patterns (pattern={pattern_len}bp, max_dist={max_dist})...")
        print("-" * 80)
        
        for num_patterns in pattern_counts:
            # Generate patterns
            patterns = []
            for _ in range(num_patterns):
                start_pos = random.randint(0, len(sequence) - pattern_len)
                pattern = sequence[start_pos:start_pos + pattern_len]
                patterns.append(pattern)
            
            # Search with all patterns
            searcher = PatternSearcher(max_distance=max_dist)
            start = time.perf_counter()
            total_matches = 0
            for pattern in patterns:
                matches = searcher.search(pattern, text)
                total_matches += len(matches)
            elapsed = time.perf_counter() - start
            
            avg_time_per_pattern = elapsed / num_patterns
            
            result = {
                'num_patterns': num_patterns,
                'pattern_length': pattern_len,
                'text_length': text_size,
                'max_edit_distance': max_dist,
                'total_time_ms': elapsed * 1000,
                'avg_time_per_pattern_ms': avg_time_per_pattern * 1000,
                'total_matches': total_matches,
                'patterns_per_sec': num_patterns / elapsed if elapsed > 0 else 0
            }
            
            self.results['scalability_patterns'].append(result)
            
            print(f"  {num_patterns:2d} patterns: "
                  f"total={elapsed*1000:8.2f}ms, "
                  f"avg={avg_time_per_pattern*1000:7.2f}ms/pattern, "
                  f"throughput={result['patterns_per_sec']:.2f}patterns/s")
    
    def criterion_5_scalability_threshold(self) -> None:
        """
        Criterion 5c: Scalability - Edit Distance Threshold
        - How performance scales with increasing max edit distance
        """
        print("\n" + "="*80)
        print("CRITERION 5c: SCALABILITY - Edit Distance Threshold")
        print("="*80)
        
        dataset = list(self.datasets.values())[0]
        sequence = dataset['sequence']
        text_size = min(5000, len(sequence))
        text = sequence[:text_size]
        
        pattern_len = 50
        thresholds = [0, 1, 2, 3, 5, 7, 10]
        
        print(f"\nVarying edit distance threshold (pattern={pattern_len}bp)...")
        print("-" * 80)
        
        # Extract pattern
        start_pos = random.randint(0, len(text) - pattern_len)
        pattern = text[start_pos:start_pos + pattern_len]
        
        for max_dist in thresholds:
            searcher = PatternSearcher(max_distance=max_dist)
            
            # Measure search time
            times = []
            matches_count = []
            for _ in range(3):
                start = time.perf_counter()
                matches = searcher.search(pattern, text)
                elapsed = time.perf_counter() - start
                times.append(elapsed)
                matches_count.append(len(matches))
            
            mean_time = statistics.mean(times)
            mean_matches = statistics.mean(matches_count)
            
            result = {
                'pattern_length': pattern_len,
                'text_length': text_size,
                'max_edit_distance': max_dist,
                'mean_search_time_ms': mean_time * 1000,
                'mean_matches': mean_matches,
                'time_per_match_ms': (mean_time * 1000) / mean_matches if mean_matches > 0 else 0
            }
            
            self.results['scalability_threshold'].append(result)
            
            print(f"  Max dist={max_dist:2d}: "
                  f"{mean_time*1000:8.2f}ms, "
                  f"matches={mean_matches:6.1f}, "
                  f"{result['time_per_match_ms']:.3f}ms/match")
    
    def criterion_6_robustness(self) -> None:
        """
        Criterion 6: Robustness
        - Performance with DNA alphabet (A,C,G,T)
        - Effect of pattern characteristics (GC content, repeats, etc.)
        - Handling of mutations
        """
        print("\n" + "="*80)
        print("CRITERION 6: ROBUSTNESS (Pattern Characteristics & Mutations)")
        print("="*80)
        
        dataset = list(self.datasets.values())[0]
        sequence = dataset['sequence']
        text_size = min(5000, len(sequence))
        text = sequence[:text_size]
        pattern_len = 50
        max_dist = 2
        
        print(f"\nTesting different pattern types (max_dist={max_dist})...")
        print("-" * 80)
        
        pattern_types = [
            ('random', 'Random subsequence from genome'),
            ('high_GC', 'High GC content (70%)'),
            ('low_GC', 'Low GC content (30%)'),
            ('repeat_AT', 'Alternating AT pattern'),
            ('low_complexity', 'Low complexity (AAATAAATAAAT...)'),
            ('homopolymer', 'Homopolymer stretch (AAAA...)')
        ]
        
        for ptype, description in pattern_types:
            if ptype == 'random':
                start_pos = random.randint(0, len(sequence) - pattern_len)
                pattern = sequence[start_pos:start_pos + pattern_len]
            elif ptype == 'high_GC':
                gc_count = int(pattern_len * 0.7)
                pattern = 'G' * (gc_count // 2) + 'C' * (gc_count // 2) + 'A' * (pattern_len - gc_count)
            elif ptype == 'low_GC':
                at_count = int(pattern_len * 0.7)
                pattern = 'A' * (at_count // 2) + 'T' * (at_count // 2) + 'G' * (pattern_len - at_count)
            elif ptype == 'repeat_AT':
                pattern = 'AT' * (pattern_len // 2)
            elif ptype == 'low_complexity':
                pattern = 'AAAT' * (pattern_len // 4)
            elif ptype == 'homopolymer':
                pattern = 'A' * pattern_len
            
            searcher = PatternSearcher(max_distance=max_dist)
            
            # Measure performance
            times = []
            matches_list = []
            for _ in range(5):
                start = time.perf_counter()
                matches = searcher.search(pattern, text)
                elapsed = time.perf_counter() - start
                times.append(elapsed)
                matches_list.append(len(matches))
            
            mean_time = statistics.mean(times)
            std_dev = statistics.stdev(times) if len(times) > 1 else 0
            mean_matches = statistics.mean(matches_list)
            
            # Calculate pattern statistics
            gc_content = (pattern.count('G') + pattern.count('C')) / len(pattern)
            
            # Calculate complexity (unique k-mers)
            k = 4
            kmers = set()
            for i in range(len(pattern) - k + 1):
                kmers.add(pattern[i:i+k])
            complexity = len(kmers) / max(1, len(pattern) - k + 1)
            
            result = {
                'pattern_type': ptype,
                'description': description,
                'pattern_length': len(pattern),
                'gc_content': gc_content,
                'complexity_score': complexity,
                'max_edit_distance': max_dist,
                'mean_search_time_ms': mean_time * 1000,
                'std_dev_ms': std_dev * 1000,
                'mean_matches': mean_matches
            }
            
            self.results['robustness'].append(result)
            
            print(f"  {ptype:15s}: "
                  f"time={mean_time*1000:7.2f}ms ± {std_dev*1000:5.2f}ms, "
                  f"GC={gc_content:.1%}, "
                  f"complex={complexity:.2f}, "
                  f"matches={mean_matches:.0f}")
    
    def save_results(self) -> None:
        """Save all results to files."""
        print("\n" + "="*80)
        print("SAVING RESULTS")
        print("="*80)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save each criterion's results
        for criterion, data in self.results.items():
            if not data:
                continue
            
            # CSV
            csv_path = BENCHMARKS_DIR / f"{criterion}_{timestamp}.csv"
            with open(csv_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=data[0].keys())
                writer.writeheader()
                writer.writerows(data)
            print(f"✓ Saved {criterion}: {csv_path.name}")
            
            # JSON
            json_path = BENCHMARKS_DIR / f"{criterion}_{timestamp}.json"
            with open(json_path, 'w') as f:
                json.dump(data, f, indent=2)
        
        # Save combined results
        all_results_path = RESULTS_DIR / f"all_results_{timestamp}.json"
        with open(all_results_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"✓ Saved combined results: {all_results_path.name}")
    
    def generate_comprehensive_report(self) -> None:
        """Generate comprehensive evaluation report."""
        print("\n" + "="*80)
        print("GENERATING COMPREHENSIVE REPORT")
        print("="*80)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = REPORTS_DIR / f"comprehensive_evaluation_{timestamp}.txt"
        
        with open(report_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("WAGNER-FISCHER ALGORITHM - COMPREHENSIVE EVALUATION REPORT\n")
            f.write("="*80 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Executive Summary
            f.write("EXECUTIVE SUMMARY\n")
            f.write("-"*80 + "\n\n")
            
            f.write("This report presents a comprehensive evaluation of the Wagner-Fischer\n")
            f.write("algorithm for approximate pattern matching using edit distance in DNA\n")
            f.write("sequences, covering:\n\n")
            f.write("1. Latency/Time: Runtime characteristics for approximate matching\n")
            f.write("2. Preprocessing: Minimal initialization overhead analysis\n")
            f.write("3. Memory Usage: DP matrix footprint and space optimization\n")
            f.write("4. Accuracy: Fuzzy matching precision and recall\n")
            f.write("5. Scalability: Performance with varying parameters\n")
            f.write("6. Robustness: Behavior with different pattern types\n\n")
            
            # Datasets
            f.write("\nDATASETS EVALUATED\n")
            f.write("-"*80 + "\n\n")
            for name, data in self.datasets.items():
                f.write(f"{name}:\n")
                f.write(f"  Length: {data['length']:,} bp\n")
                f.write(f"  GC Content: {data['gc_content']:.2%}\n\n")
            
            # Algorithm Characteristics
            f.write("\nALGORITHM CHARACTERISTICS\n")
            f.write("-"*80 + "\n\n")
            f.write("Wagner-Fischer (Edit Distance / Levenshtein Distance):\n")
            f.write("  - Time Complexity: O(m*n) where m=pattern length, n=text length\n")
            f.write("  - Space Complexity: O(m*n) full matrix, O(min(m,n)) optimized\n")
            f.write("  - Type: Dynamic Programming, Approximate Matching\n")
            f.write("  - Supports: Substitutions, Insertions, Deletions\n")
            f.write("  - Use Case: Fuzzy matching with configurable edit distance\n\n")
            
            # Detailed Results
            for criterion in ['latency_time', 'preprocessing', 'memory', 'accuracy', 
                            'scalability_text', 'scalability_patterns', 'scalability_threshold', 
                            'robustness']:
                if not self.results[criterion]:
                    continue
                
                f.write(f"\n{criterion.upper().replace('_', ' ')}\n")
                f.write("-"*80 + "\n\n")
                
                if criterion == 'accuracy':
                    # Special handling for accuracy
                    f.write("Fuzzy Matching Performance:\n\n")
                    for r in self.results[criterion][:10]:  # First 10 results
                        f.write(f"  {r['dataset']} - Pattern {r['pattern_length']}bp, "
                               f"dist={r['max_edit_distance']}:\n")
                        f.write(f"    Precision: {r['precision']:.4f}\n")
                        f.write(f"    Recall: {r['recall']:.4f}\n")
                        f.write(f"    F1 Score: {r['f1_score']:.4f}\n\n")
                elif criterion == 'memory':
                    f.write("Memory Optimization Analysis:\n\n")
                    for r in self.results[criterion][:5]:
                        f.write(f"  Pattern {r['pattern_length']}bp:\n")
                        f.write(f"    Full Matrix: {r['full_matrix_peak_kb']:.2f} KB\n")
                        f.write(f"    Optimized: {r['optimized_peak_kb']:.2f} KB\n")
                        f.write(f"    Reduction: {r['memory_reduction_ratio']:.1f}x\n\n")
                else:
                    # Summary statistics
                    f.write("See detailed CSV/JSON files for complete data.\n")
                    f.write(f"Total measurements: {len(self.results[criterion])}\n\n")
            
            # Key Findings
            f.write("\nKEY FINDINGS\n")
            f.write("-"*80 + "\n\n")
            
            if self.results['accuracy']:
                valid_results = [r for r in self.results['accuracy'] 
                               if r['precision'] > 0 or r['recall'] > 0]
                if valid_results:
                    avg_precision = statistics.mean([r['precision'] for r in valid_results])
                    avg_recall = statistics.mean([r['recall'] for r in valid_results])
                    avg_f1 = statistics.mean([r['f1_score'] for r in valid_results])
                    f.write(f"1. Accuracy: Fuzzy matching performance (Precision={avg_precision:.3f}, ")
                    f.write(f"Recall={avg_recall:.3f}, F1={avg_f1:.3f})\n")
            
            if self.results['latency_time']:
                throughputs = [r['throughput_mbps'] for r in self.results['latency_time']]
                avg_throughput = statistics.mean(throughputs)
                f.write(f"2. Performance: Average throughput {avg_throughput:.4f} MB/s\n")
                f.write(f"   (Note: Significantly slower than exact matching algorithms)\n")
            
            if self.results['memory']:
                reductions = [r['memory_reduction_ratio'] for r in self.results['memory']]
                avg_reduction = statistics.mean(reductions)
                f.write(f"3. Memory: Space optimization achieves {avg_reduction:.1f}x reduction\n")
            
            if self.results['scalability_threshold']:
                f.write(f"4. Scalability: Performance degrades with increasing edit distance\n")
            
            if self.results['robustness']:
                f.write(f"5. Robustness: Performance varies with pattern complexity\n")
            
            # Comparison Note
            f.write("\nCOMPARISON WITH EXACT MATCHING ALGORITHMS\n")
            f.write("-"*80 + "\n\n")
            f.write("Wagner-Fischer provides approximate matching capabilities at the cost\n")
            f.write("of performance:\n")
            f.write("  - Much slower than KMP, Boyer-Moore for exact matching\n")
            f.write("  - Higher memory usage (O(m*n) vs O(m))\n")
            f.write("  - Enables fuzzy matching with mutations, insertions, deletions\n")
            f.write("  - Ideal for biological sequences with sequencing errors\n\n")
            
            f.write("\n" + "="*80 + "\n")
            f.write("END OF REPORT\n")
            f.write("="*80 + "\n")
        
        print(f"\n✓ Saved comprehensive report: {report_path.name}")
        
        # Print summary to console
        print("\n" + "="*80)
        print("EVALUATION SUMMARY")
        print("="*80)
        if self.results['accuracy']:
            valid_results = [r for r in self.results['accuracy'] 
                           if r['precision'] > 0 or r['recall'] > 0]
            if valid_results:
                avg_f1 = statistics.mean([r['f1_score'] for r in valid_results])
                print(f"Average F1 Score: {avg_f1:.3f}")
        
        if self.results['latency_time']:
            avg_time = statistics.mean([r['mean_time_ms'] for r in self.results['latency_time']])
            print(f"Average Search Time: {avg_time:.2f}ms")
        
        print("="*80)
    
    def run_all_evaluations(self) -> None:
        """Run all evaluation criteria."""
        print("\n" + "="*80)
        print("COMPREHENSIVE WAGNER-FISCHER EVALUATION")
        print("="*80)
        print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        self.load_datasets()
        
        if not self.datasets:
            print("\n✗ No datasets available!")
            return
        
        # Run all criteria
        self.criterion_1_latency_time()
        self.criterion_2_preprocessing()
        self.criterion_3_memory_usage()
        self.criterion_4_accuracy()
        self.criterion_5_scalability_text_length()
        self.criterion_5_scalability_pattern_count()
        self.criterion_5_scalability_threshold()
        self.criterion_6_robustness()
        
        # Save and report
        self.save_results()
        self.generate_comprehensive_report()
        
        print("\n" + "="*80)
        print("EVALUATION COMPLETE!")
        print("="*80)
        print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"\nResults saved to:")
        print(f"  Benchmarks: {BENCHMARKS_DIR}")
        print(f"  Reports: {REPORTS_DIR}")
        print("="*80 + "\n")


def main():
    """Main execution."""
    evaluator = ComprehensiveEvaluator()
    evaluator.run_all_evaluations()


if __name__ == "__main__":
    main()
