#!/usr/bin/env python3
"""
OPTIMIZED Comprehensive Suffix Array Evaluation

Key optimizations:
1. Use smaller text samples for most tests
2. Reduce number of runs
3. Skip expensive operations on large datasets
4. Focus on meaningful pattern lengths
"""

import sys
import time
import json
import csv
import statistics
import tracemalloc
from pathlib import Path
from datetime import datetime
from typing import List, Dict
import random
import re

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "kmp"))

from src.suffix_array import SuffixArray
from data_loader import read_fasta, sequence_stats

# Directories
DATASET_DIR = Path(__file__).parent.parent / "dataset"
RESULTS_DIR = Path(__file__).parent / "results"
BENCHMARKS_DIR = RESULTS_DIR / "benchmarks"
REPORTS_DIR = RESULTS_DIR / "reports"

for d in [RESULTS_DIR, BENCHMARKS_DIR, REPORTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# OPTIMIZED PARAMETERS
MAX_TEXT_SIZE = 100000  # 100KB max for construction tests
SAMPLE_SIZES = [1000, 5000, 10000, 50000, 100000]  # Smaller samples
PATTERN_LENGTHS = [10, 20, 50, 100]  # Fewer pattern lengths
NUM_RUNS = 3  # Reduced from 10
NUM_PATTERNS_PER_LENGTH = 2  # Reduced from 5

class OptimizedSuffixArrayEvaluator:
    """Optimized evaluator for suffix array."""
    
    def __init__(self):
        self.datasets = {}
        self.results = {
            'latency_time': [],
            'preprocessing': [],
            'memory': [],
            'accuracy': [],
            'scalability_text': [],
            'scalability_patterns': [],
            'robustness': []
        }
    
    def load_datasets(self) -> None:
        """Load datasets with size limits."""
        print("\n" + "="*80)
        print("LOADING DATASETS (with size limits for performance)")
        print("="*80)
        
        dataset_files = {
            'lambda_phage': 'lambda_phage.fasta',  # Small dataset first
            'ecoli': 'ecoli_k12_mg1655.fasta',
            'salmonella': 'salmonella_typhimurium.fasta'
        }
        
        for name, filename in dataset_files.items():
            filepath = DATASET_DIR / filename
            if filepath.exists():
                print(f"\nLoading {name}...", end=' ')
                records = read_fasta(filepath)
                if records:
                    sequence = records[0].sequence
                    
                    # Limit size for expensive operations
                    if len(sequence) > MAX_TEXT_SIZE and name != 'lambda_phage':
                        print(f"(using first {MAX_TEXT_SIZE:,} bp of {len(sequence):,} bp)", end=' ')
                        sequence = sequence[:MAX_TEXT_SIZE]
                    
                    stats = sequence_stats(sequence)
                    self.datasets[name] = {
                        'sequence': sequence,
                        'stats': stats,
                        'name': name
                    }
                    print(f"✓ {len(sequence):,} bp")
        
        print(f"\n✓ Loaded {len(self.datasets)} dataset(s)")
    
    def criterion_1_latency_time(self) -> None:
        """Criterion 1: Latency/Time - OPTIMIZED."""
        print("\n" + "="*80)
        print("CRITERION 1: LATENCY / TIME ANALYSIS (Optimized)")
        print("="*80)
        
        for dataset_name, data in self.datasets.items():
            sequence = data['sequence']
            print(f"\n{dataset_name.upper()} ({len(sequence):,} bp)")
            print("-" * 80)
            
            for pattern_len in PATTERN_LENGTHS:
                if pattern_len > len(sequence):
                    continue
                
                # Extract pattern
                start_pos = random.randint(0, len(sequence) - pattern_len)
                pattern = sequence[start_pos:start_pos + pattern_len]
                
                # Build suffix array once
                print(f"  Pattern {pattern_len:3d}bp: Building suffix array...", end=' ')
                construct_start = time.perf_counter()
                sa = SuffixArray(sequence)
                construct_time = time.perf_counter() - construct_start
                print(f"done ({construct_time*1000:.1f}ms)", end=' ')
                
                # Multiple search runs
                times = []
                for _ in range(NUM_RUNS):
                    start = time.perf_counter()
                    matches = sa.search(pattern)
                    elapsed = time.perf_counter() - start
                    times.append(elapsed)
                
                mean_time = statistics.mean(times)
                std_dev = statistics.stdev(times) if len(times) > 1 else 0
                throughput_mbps = (len(sequence) / (1024*1024)) / mean_time
                
                result = {
                    'dataset': dataset_name,
                    'text_length': len(sequence),
                    'pattern_length': pattern_len,
                    'num_runs': NUM_RUNS,
                    'construction_time_ms': construct_time * 1000,
                    'mean_search_time_ms': mean_time * 1000,
                    'std_dev_ms': std_dev * 1000,
                    'throughput_mbps': throughput_mbps,
                    'num_matches': len(matches),
                    'total_time_ms': (construct_time + mean_time) * 1000
                }
                
                self.results['latency_time'].append(result)
                print(f"search={mean_time*1000:.2f}ms, {len(matches)} matches")
    
    def criterion_2_preprocessing(self) -> None:
        """Criterion 2: Preprocessing Time - OPTIMIZED."""
        print("\n" + "="*80)
        print("CRITERION 2: PREPROCESSING TIME (Suffix Array Construction)")
        print("="*80)
        
        # Use lambda phage for most tests, small samples for others
        dataset = self.datasets.get('lambda_phage') or list(self.datasets.values())[0]
        full_sequence = dataset['sequence']
        
        print(f"\nMeasuring suffix array construction time...")
        print(f"Using samples from {len(full_sequence):,} bp sequence")
        print("-" * 80)
        
        for size in SAMPLE_SIZES:
            if size > len(full_sequence):
                continue
            
            sequence = full_sequence[:size]
            
            # Measure construction time
            times = []
            for run in range(NUM_RUNS):
                start = time.perf_counter()
                sa = SuffixArray(sequence)
                elapsed = time.perf_counter() - start
                times.append(elapsed)
                print(f"  Size {size:6d}bp: run {run+1}/{NUM_RUNS} = {elapsed*1000:.1f}ms", end='\r')
            
            mean_time = statistics.mean(times)
            std_dev = statistics.stdev(times) if len(times) > 1 else 0
            
            # Memory footprint
            sa_memory = size * 4  # Approximate: 4 bytes per suffix
            
            result = {
                'text_length': size,
                'num_runs': NUM_RUNS,
                'mean_construction_time_ms': mean_time * 1000,
                'std_dev_ms': std_dev * 1000,
                'sa_memory_bytes': sa_memory,
                'time_per_char_us': (mean_time * 1e6) / size
            }
            
            self.results['preprocessing'].append(result)
            print(f"  Size {size:6d}bp: {mean_time*1000:7.1f}ms ± {std_dev*1000:5.1f}ms, "
                  f"memory≈{sa_memory/1024:.1f}KB, {result['time_per_char_us']:.2f}µs/char")
    
    def criterion_3_memory_usage(self) -> None:
        """Criterion 3: Memory Usage - OPTIMIZED."""
        print("\n" + "="*80)
        print("CRITERION 3: MEMORY USAGE")
        print("="*80)
        
        test_sizes = [1000, 5000, 10000, 20000, 50000]
        dataset = self.datasets.get('lambda_phage') or list(self.datasets.values())[0]
        full_sequence = dataset['sequence']
        
        print(f"\nMeasuring memory usage...")
        print("-" * 80)
        
        for size in test_sizes:
            if size > len(full_sequence):
                continue
            
            sequence = full_sequence[:size]
            
            # Measure construction memory
            tracemalloc.start()
            sa = SuffixArray(sequence)
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            
            # Estimate SA size
            sa_size = len(sequence) * 4  # bytes
            
            result = {
                'text_length': size,
                'sa_memory_bytes': sa_size,
                'construction_peak_kb': peak / 1024,
                'current_kb': current / 1024
            }
            
            self.results['memory'].append(result)
            print(f"  Size {size:6d}bp: SA≈{sa_size/1024:6.1f}KB, "
                  f"Peak={peak/1024:7.1f}KB")
    
    def criterion_4_accuracy(self) -> None:
        """Criterion 4: Accuracy - OPTIMIZED."""
        print("\n" + "="*80)
        print("CRITERION 4: ACCURACY (vs Python re)")
        print("="*80)
        
        for dataset_name, data in self.datasets.items():
            sequence = data['sequence']
            print(f"\n{dataset_name.upper()}")
            print("-" * 80)
            
            for pattern_len in PATTERN_LENGTHS:
                all_tp = all_fp = all_fn = all_agree = 0
                
                # Build SA once for this dataset
                sa = SuffixArray(sequence)
                
                for _ in range(NUM_PATTERNS_PER_LENGTH):
                    start_pos = random.randint(0, len(sequence) - pattern_len)
                    pattern = sequence[start_pos:start_pos + pattern_len]
                    
                    # Suffix array search
                    sa_matches = set(sa.search(pattern))
                    
                    # re search
                    re_matches = set(m.start() for m in re.finditer(re.escape(pattern), sequence))
                    
                    tp = len(sa_matches & re_matches)
                    fp = len(sa_matches - re_matches)
                    fn = len(re_matches - sa_matches)
                    
                    all_tp += tp
                    all_fp += fp
                    all_fn += fn
                    if sa_matches == re_matches:
                        all_agree += 1
                
                precision = all_tp / (all_tp + all_fp) if (all_tp + all_fp) > 0 else 0
                recall = all_tp / (all_tp + all_fn) if (all_tp + all_fn) > 0 else 0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                
                result = {
                    'dataset': dataset_name,
                    'pattern_length': pattern_len,
                    'num_tests': NUM_PATTERNS_PER_LENGTH,
                    'true_positives': all_tp,
                    'false_positives': all_fp,
                    'false_negatives': all_fn,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'agreement_rate': all_agree / NUM_PATTERNS_PER_LENGTH
                }
                
                self.results['accuracy'].append(result)
                print(f"  Pattern {pattern_len:3d}bp: P={precision:.4f}, R={recall:.4f}, "
                      f"F1={f1:.4f}, Agreement={result['agreement_rate']:.1%}")
    
    def criterion_5_scalability_text(self) -> None:
        """Criterion 5a: Scalability - Text Length - OPTIMIZED."""
        print("\n" + "="*80)
        print("CRITERION 5a: SCALABILITY - Text Length")
        print("="*80)
        
        dataset = self.datasets.get('lambda_phage') or list(self.datasets.values())[0]
        full_sequence = dataset['sequence']
        pattern_len = 20
        
        print(f"\nScaling text size (pattern length={pattern_len}bp)...")
        print("-" * 80)
        
        for size in SAMPLE_SIZES:
            if size > len(full_sequence):
                continue
            
            sequence = full_sequence[:size]
            start_pos = random.randint(0, len(sequence) - pattern_len)
            pattern = sequence[start_pos:start_pos + pattern_len]
            
            # Build SA
            construct_start = time.perf_counter()
            sa = SuffixArray(sequence)
            construct_time = time.perf_counter() - construct_start
            
            # Search
            search_start = time.perf_counter()
            matches = sa.search(pattern)
            search_time = time.perf_counter() - search_start
            
            result = {
                'text_length': size,
                'pattern_length': pattern_len,
                'construction_time_ms': construct_time * 1000,
                'search_time_ms': search_time * 1000,
                'total_time_ms': (construct_time + search_time) * 1000,
                'num_matches': len(matches)
            }
            
            self.results['scalability_text'].append(result)
            print(f"  Size {size:6d}bp: construct={construct_time*1000:6.1f}ms, "
                  f"search={search_time*1000:5.2f}ms")
    
    def criterion_5_scalability_patterns(self) -> None:
        """Criterion 5b: Scalability - Pattern Count - OPTIMIZED."""
        print("\n" + "="*80)
        print("CRITERION 5b: SCALABILITY - Multiple Patterns")
        print("="*80)
        
        dataset = list(self.datasets.values())[0]
        sequence = dataset['sequence']
        pattern_len = 20
        pattern_counts = [1, 5, 10, 20, 50]
        
        print(f"\nSearching with multiple patterns (length={pattern_len}bp)...")
        print("-" * 80)
        
        # Build SA once
        print("  Building suffix array...", end=' ')
        construct_start = time.perf_counter()
        sa = SuffixArray(sequence)
        construct_time = time.perf_counter() - construct_start
        print(f"done ({construct_time*1000:.1f}ms)")
        
        for num_patterns in pattern_counts:
            patterns = []
            for _ in range(num_patterns):
                start_pos = random.randint(0, len(sequence) - pattern_len)
                pattern = sequence[start_pos:start_pos + pattern_len]
                patterns.append(pattern)
            
            start = time.perf_counter()
            total_matches = sum(len(sa.search(p)) for p in patterns)
            elapsed = time.perf_counter() - start
            
            result = {
                'num_patterns': num_patterns,
                'pattern_length': pattern_len,
                'text_length': len(sequence),
                'search_time_ms': elapsed * 1000,
                'avg_time_per_pattern_ms': (elapsed / num_patterns) * 1000,
                'total_matches': total_matches,
                'patterns_per_sec': num_patterns / elapsed
            }
            
            self.results['scalability_patterns'].append(result)
            print(f"  {num_patterns:3d} patterns: {elapsed*1000:6.1f}ms total, "
                  f"{result['avg_time_per_pattern_ms']:5.2f}ms/pattern")
    
    def criterion_6_robustness(self) -> None:
        """Criterion 6: Robustness - OPTIMIZED."""
        print("\n" + "="*80)
        print("CRITERION 6: ROBUSTNESS")
        print("="*80)
        
        dataset = list(self.datasets.values())[0]
        sequence = dataset['sequence']
        pattern_len = 50
        
        # Build SA once
        sa = SuffixArray(sequence)
        
        pattern_types = [
            ('random', 'Random from genome'),
            ('repeat_A', 'High A (80%)'),
            ('repeat_AT', 'Alternating AT')
        ]
        
        print(f"\nTesting pattern types...")
        print("-" * 80)
        
        for ptype, desc in pattern_types:
            if ptype == 'random':
                start_pos = random.randint(0, len(sequence) - pattern_len)
                pattern = sequence[start_pos:start_pos + pattern_len]
            elif ptype == 'repeat_A':
                pattern = 'A' * int(pattern_len * 0.8) + 'C' * int(pattern_len * 0.2)
            else:  # repeat_AT
                pattern = 'AT' * (pattern_len // 2)
            
            times = []
            for _ in range(NUM_RUNS):
                start = time.perf_counter()
                matches = sa.search(pattern)
                elapsed = time.perf_counter() - start
                times.append(elapsed)
            
            mean_time = statistics.mean(times)
            gc_content = (pattern.count('G') + pattern.count('C')) / len(pattern)
            
            result = {
                'pattern_type': ptype,
                'description': desc,
                'pattern_length': len(pattern),
                'gc_content': gc_content,
                'mean_search_time_ms': mean_time * 1000,
                'num_matches': len(matches)
            }
            
            self.results['robustness'].append(result)
            print(f"  {ptype:12s}: {mean_time*1000:6.2f}ms, "
                  f"GC={gc_content:.1%}, matches={len(matches)}")
    
    def save_results(self) -> None:
        """Save results to files."""
        print("\n" + "="*80)
        print("SAVING RESULTS")
        print("="*80)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for criterion, data in self.results.items():
            if not data:
                continue
            
            csv_path = BENCHMARKS_DIR / f"{criterion}_{timestamp}.csv"
            with open(csv_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=data[0].keys())
                writer.writeheader()
                writer.writerows(data)
            print(f"✓ Saved {criterion}: {csv_path.name}")
            
            json_path = BENCHMARKS_DIR / f"{criterion}_{timestamp}.json"
            with open(json_path, 'w') as f:
                json.dump(data, f, indent=2)
    
    def generate_report(self) -> None:
        """Generate evaluation report."""
        print("\n" + "="*80)
        print("GENERATING REPORT")
        print("="*80)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = REPORTS_DIR / f"suffix_array_evaluation_{timestamp}.txt"
        
        with open(report_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("SUFFIX ARRAY - COMPREHENSIVE EVALUATION REPORT\n")
            f.write("="*80 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("DATASETS\n")
            f.write("-"*80 + "\n\n")
            for name, data in self.datasets.items():
                stats = data['stats']
                f.write(f"{name}:\n")
                f.write(f"  Length: {stats['length']:,} bp\n")
                f.write(f"  GC Content: {stats['gc_content']:.2%}\n\n")
            
            if self.results['accuracy']:
                f.write("\nACCURACY RESULTS\n")
                f.write("-"*80 + "\n\n")
                for r in self.results['accuracy']:
                    f.write(f"{r['dataset']} - Pattern {r['pattern_length']}bp:\n")
                    f.write(f"  Precision: {r['precision']:.6f}\n")
                    f.write(f"  Recall: {r['recall']:.6f}\n")
                    f.write(f"  F1: {r['f1_score']:.6f}\n\n")
            
            f.write("\n" + "="*80 + "\n")
        
        print(f"✓ Saved report: {report_path.name}")
        
        with open(report_path, 'r') as f:
            print("\n" + f.read())
    
    def run_all(self) -> None:
        """Run all evaluations - OPTIMIZED."""
        print("\n" + "="*80)
        print("SUFFIX ARRAY - OPTIMIZED COMPREHENSIVE EVALUATION")
        print("="*80)
        print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("\nOptimizations:")
        print(f"  - Max text size: {MAX_TEXT_SIZE:,} bp")
        print(f"  - Runs per test: {NUM_RUNS}")
        print(f"  - Patterns per length: {NUM_PATTERNS_PER_LENGTH}")
        
        self.load_datasets()
        
        if not self.datasets:
            print("\n✗ No datasets loaded!")
            return
        
        self.criterion_1_latency_time()
        self.criterion_2_preprocessing()
        self.criterion_3_memory_usage()
        self.criterion_4_accuracy()
        self.criterion_5_scalability_text()
        self.criterion_5_scalability_patterns()
        self.criterion_6_robustness()
        
        self.save_results()
        self.generate_report()
        
        print("\n" + "="*80)
        print("EVALUATION COMPLETE!")
        print("="*80)
        print(f"End: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"\nResults: {BENCHMARKS_DIR}")
        print(f"Reports: {REPORTS_DIR}")
        print("="*80 + "\n")


def main():
    evaluator = OptimizedSuffixArrayEvaluator()
    evaluator.run_all()


if __name__ == "__main__":
    main()
