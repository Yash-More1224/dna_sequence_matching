#!/usr/bin/env python3
"""
SMART Comprehensive Suffix Array Evaluation - ALL 6 CRITERIA

Strategy:
- Build suffix arrays ONCE per genome (reuse for all tests)
- Use full genomes for search/accuracy tests (realistic performance)
- Use samples for construction scalability only (avoid O(n²) repetition)
- Complete all 6 criteria matching KMP evaluation exactly

Estimated time: 15-20 minutes
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
PLOTS_DIR = RESULTS_DIR / "plots"

for d in [RESULTS_DIR, BENCHMARKS_DIR, REPORTS_DIR, PLOTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Evaluation parameters
NUM_RUNS = 10  # For statistical significance
PATTERN_LENGTHS = [10, 20, 50, 100, 200, 500]
CONSTRUCTION_SAMPLE_SIZES = [1000, 5000, 10000, 50000, 100000, 500000]


class SmartSuffixArrayEvaluator:
    """Smart evaluator - build once, test many times."""
    
    def __init__(self):
        self.datasets = {}
        self.suffix_arrays = {}  # Cache built suffix arrays
        self.results = {
            'criterion_1_latency': [],
            'criterion_2_preprocessing': [],
            'criterion_3_memory': [],
            'criterion_4_accuracy': [],
            'criterion_5_scalability_text': [],
            'criterion_5_scalability_patterns': [],
            'criterion_6_robustness': []
        }
    
    def load_datasets(self) -> None:
        """Load full datasets."""
        print("\n" + "="*80)
        print("LOADING DATASETS")
        print("="*80)
        
        dataset_files = {
            'lambda_phage': 'lambda_phage.fasta',
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
                    stats = sequence_stats(sequence)
                    self.datasets[name] = {
                        'sequence': sequence,
                        'stats': stats,
                        'name': name
                    }
                    print(f"✓ {len(sequence):,} bp (GC: {stats['gc_content']:.1%})")
        
        print(f"\n✓ Loaded {len(self.datasets)} dataset(s)")
    
    def build_suffix_arrays(self) -> None:
        """Build suffix arrays once for all datasets."""
        print("\n" + "="*80)
        print("BUILDING SUFFIX ARRAYS (One-time Preprocessing)")
        print("="*80)
        print("\nThis is the slow part - building once, reusing for all tests...")
        
        for name, data in self.datasets.items():
            sequence = data['sequence']
            print(f"\n{name} ({len(sequence):,} bp)...")
            
            start = time.perf_counter()
            sa = SuffixArray(sequence)
            elapsed = time.perf_counter() - start
            
            self.suffix_arrays[name] = sa
            
            print(f"  ✓ Built in {elapsed:.2f}s ({elapsed*1000:.1f}ms)")
            print(f"  Rate: {len(sequence)/elapsed:,.0f} bp/s")
        
        print("\n✓ All suffix arrays built and cached!")
    
    def criterion_1_latency_time(self) -> None:
        """
        CRITERION 1: Latency / Time
        - Mean, median, std dev across multiple runs
        - Throughput (MB/s)
        - Per-query latency
        """
        print("\n" + "="*80)
        print("CRITERION 1: LATENCY / TIME")
        print("="*80)
        
        for dataset_name, data in self.datasets.items():
            sequence = data['sequence']
            sa = self.suffix_arrays[dataset_name]
            
            print(f"\n{dataset_name.upper()} ({len(sequence):,} bp)")
            print("-" * 80)
            
            for pattern_len in PATTERN_LENGTHS:
                if pattern_len > len(sequence):
                    continue
                
                # Extract pattern from sequence
                start_pos = random.randint(0, len(sequence) - pattern_len)
                pattern = sequence[start_pos:start_pos + pattern_len]
                
                # Multiple runs for statistics
                times = []
                matches_count = 0
                
                for _ in range(NUM_RUNS):
                    start = time.perf_counter()
                    matches = sa.search(pattern)
                    elapsed = time.perf_counter() - start
                    times.append(elapsed)
                    matches_count = len(matches)
                
                mean_time = statistics.mean(times)
                median_time = statistics.median(times)
                std_dev = statistics.stdev(times) if len(times) > 1 else 0
                min_time = min(times)
                max_time = max(times)
                throughput_mbps = (len(sequence) / (1024*1024)) / mean_time
                
                result = {
                    'dataset': dataset_name,
                    'text_length': len(sequence),
                    'pattern_length': pattern_len,
                    'num_runs': NUM_RUNS,
                    'mean_time_s': mean_time,
                    'mean_time_ms': mean_time * 1000,
                    'median_time_ms': median_time * 1000,
                    'std_dev_ms': std_dev * 1000,
                    'min_time_ms': min_time * 1000,
                    'max_time_ms': max_time * 1000,
                    'throughput_mbps': throughput_mbps,
                    'num_matches': matches_count
                }
                
                self.results['criterion_1_latency'].append(result)
                
                print(f"  Pattern {pattern_len:4d}bp: "
                      f"mean={mean_time*1000:7.3f}ms ± {std_dev*1000:5.3f}ms, "
                      f"median={median_time*1000:7.3f}ms, "
                      f"throughput={throughput_mbps:7.2f}MB/s, "
                      f"matches={matches_count:5d}")
        
        print("\n✓ Criterion 1 complete")
    
    def criterion_2_preprocessing_time(self) -> None:
        """
        CRITERION 2: Preprocessing Time
        - Construction time scaling with text size
        - Time per character
        """
        print("\n" + "="*80)
        print("CRITERION 2: PREPROCESSING TIME")
        print("="*80)
        
        # Get a representative dataset for scaling tests
        dataset_name = 'ecoli'
        if dataset_name not in self.datasets:
            dataset_name = list(self.datasets.keys())[0]
        
        full_sequence = self.datasets[dataset_name]['sequence']
        
        print(f"\nTesting construction scalability on {dataset_name}")
        print(f"Full genome: {len(full_sequence):,} bp")
        print("-" * 80)
        
        # Test on samples for scalability
        for size in CONSTRUCTION_SAMPLE_SIZES:
            if size > len(full_sequence):
                continue
            
            sequence = full_sequence[:size]
            
            # Multiple runs for statistics
            times = []
            for _ in range(3):  # 3 runs per size to save time
                start = time.perf_counter()
                sa = SuffixArray(sequence)
                elapsed = time.perf_counter() - start
                times.append(elapsed)
            
            mean_time = statistics.mean(times)
            std_dev = statistics.stdev(times) if len(times) > 1 else 0
            time_per_char = (mean_time * 1e6) / size  # microseconds per char
            
            result = {
                'dataset': dataset_name,
                'text_length': size,
                'num_runs': 3,
                'mean_time_s': mean_time,
                'mean_time_ms': mean_time * 1000,
                'std_dev_ms': std_dev * 1000,
                'time_per_char_us': time_per_char
            }
            
            self.results['criterion_2_preprocessing'].append(result)
            
            print(f"  {size:7,d}bp: {mean_time*1000:8.1f}ms ± {std_dev*1000:6.1f}ms, "
                  f"{time_per_char:6.2f}µs/char")
        
        # Add full genome construction times (already measured)
        print("\nFull genome construction times:")
        for name, data in self.datasets.items():
            sequence = data['sequence']
            
            # Measure once for full genome
            start = time.perf_counter()
            sa_test = SuffixArray(sequence)
            elapsed = time.perf_counter() - start
            
            time_per_char = (elapsed * 1e6) / len(sequence)
            
            result = {
                'dataset': name,
                'text_length': len(sequence),
                'num_runs': 1,
                'mean_time_s': elapsed,
                'mean_time_ms': elapsed * 1000,
                'std_dev_ms': 0,
                'time_per_char_us': time_per_char
            }
            
            self.results['criterion_2_preprocessing'].append(result)
            
            print(f"  {name}: {len(sequence):,}bp in {elapsed:.2f}s ({time_per_char:.2f}µs/char)")
        
        print("\n✓ Criterion 2 complete")
    
    def criterion_3_memory_usage(self) -> None:
        """
        CRITERION 3: Memory Usage
        - Construction memory
        - Search memory
        - Peak memory
        """
        print("\n" + "="*80)
        print("CRITERION 3: MEMORY USAGE")
        print("="*80)
        
        test_sizes = [1000, 5000, 10000, 50000, 100000]
        
        # Use smallest dataset for memory tests
        dataset_name = 'lambda_phage'
        if dataset_name not in self.datasets:
            dataset_name = list(self.datasets.keys())[0]
        
        full_sequence = self.datasets[dataset_name]['sequence']
        
        print(f"\nMeasuring memory usage on {dataset_name}")
        print("-" * 80)
        
        for size in test_sizes:
            if size > len(full_sequence):
                continue
            
            sequence = full_sequence[:size]
            
            # Measure construction memory
            tracemalloc.start()
            sa = SuffixArray(sequence)
            current, peak_construction = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            
            # Theoretical suffix array size (4 bytes per position)
            theoretical_size = len(sequence) * 4
            
            # Measure search memory
            pattern = sequence[:50] if len(sequence) >= 50 else sequence[:len(sequence)//2]
            
            tracemalloc.start()
            matches = sa.search(pattern)
            search_current, search_peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            
            result = {
                'dataset': dataset_name,
                'text_length': size,
                'theoretical_sa_bytes': theoretical_size,
                'theoretical_sa_kb': theoretical_size / 1024,
                'construction_peak_bytes': peak_construction,
                'construction_peak_kb': peak_construction / 1024,
                'construction_peak_mb': peak_construction / (1024*1024),
                'search_peak_bytes': search_peak,
                'search_peak_kb': search_peak / 1024,
                'pattern_length': len(pattern),
                'num_matches': len(matches)
            }
            
            self.results['criterion_3_memory'].append(result)
            
            print(f"  {size:6,d}bp: "
                  f"Theoretical SA={theoretical_size/1024:7.1f}KB, "
                  f"Construction peak={peak_construction/1024:8.1f}KB, "
                  f"Search peak={search_peak/1024:6.1f}KB")
        
        # Test on full genomes (one per dataset)
        print("\nFull genome memory usage:")
        for name, data in self.datasets.items():
            sequence = data['sequence'][:100000]  # Sample 100KB to avoid excessive memory
            
            tracemalloc.start()
            sa = SuffixArray(sequence)
            current, peak_construction = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            
            theoretical_size = len(sequence) * 4
            
            result = {
                'dataset': name,
                'text_length': len(sequence),
                'theoretical_sa_bytes': theoretical_size,
                'theoretical_sa_kb': theoretical_size / 1024,
                'construction_peak_bytes': peak_construction,
                'construction_peak_kb': peak_construction / 1024,
                'construction_peak_mb': peak_construction / (1024*1024),
                'search_peak_bytes': 0,
                'search_peak_kb': 0,
                'pattern_length': 0,
                'num_matches': 0
            }
            
            self.results['criterion_3_memory'].append(result)
            
            print(f"  {name} (100KB sample): {peak_construction/(1024*1024):.2f}MB peak")
        
        print("\n✓ Criterion 3 complete")
    
    def criterion_4_accuracy(self) -> None:
        """
        CRITERION 4: Accuracy
        - Compare with Python re (ground truth)
        - Precision, recall, F1 score
        """
        print("\n" + "="*80)
        print("CRITERION 4: ACCURACY (vs Python re)")
        print("="*80)
        
        NUM_PATTERNS_PER_LENGTH = 3
        
        for dataset_name, data in self.datasets.items():
            sequence = data['sequence']
            sa = self.suffix_arrays[dataset_name]
            
            print(f"\n{dataset_name.upper()} ({len(sequence):,} bp)")
            print("-" * 80)
            
            for pattern_len in PATTERN_LENGTHS:
                if pattern_len > len(sequence):
                    continue
                
                all_tp = all_fp = all_fn = all_agree = 0
                
                for _ in range(NUM_PATTERNS_PER_LENGTH):
                    # Extract pattern from sequence
                    start_pos = random.randint(0, len(sequence) - pattern_len)
                    pattern = sequence[start_pos:start_pos + pattern_len]
                    
                    # Suffix array search
                    sa_matches = set(sa.search(pattern))
                    
                    # re search (ground truth)
                    re_matches = set(m.start() for m in re.finditer(re.escape(pattern), sequence))
                    
                    # Calculate metrics
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
                    'text_length': len(sequence),
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
                
                self.results['criterion_4_accuracy'].append(result)
                
                print(f"  Pattern {pattern_len:3d}bp: "
                      f"P={precision:.4f}, R={recall:.4f}, F1={f1:.4f}, "
                      f"Agreement={result['agreement_rate']:.1%}")
        
        print("\n✓ Criterion 4 complete")
    
    def criterion_5_scalability(self) -> None:
        """
        CRITERION 5: Scalability
        - Text length scaling
        - Number of patterns scaling
        """
        print("\n" + "="*80)
        print("CRITERION 5: SCALABILITY")
        print("="*80)
        
        # 5A: Text length scaling (already covered in preprocessing)
        print("\nText length scaling covered in Criterion 2")
        
        # 5B: Number of patterns scaling
        print("\nPattern count scaling:")
        print("-" * 80)
        
        dataset_name = 'ecoli'
        if dataset_name not in self.datasets:
            dataset_name = list(self.datasets.keys())[0]
        
        sequence = self.datasets[dataset_name]['sequence']
        sa = self.suffix_arrays[dataset_name]
        
        pattern_len = 50
        pattern_counts = [1, 5, 10, 20, 50, 100, 200]
        
        print(f"\n{dataset_name.upper()} ({len(sequence):,} bp, pattern={pattern_len}bp)")
        print("-" * 80)
        
        for num_patterns in pattern_counts:
            # Generate random patterns
            patterns = []
            for _ in range(num_patterns):
                start_pos = random.randint(0, len(sequence) - pattern_len)
                pattern = sequence[start_pos:start_pos + pattern_len]
                patterns.append(pattern)
            
            # Search all patterns
            start = time.perf_counter()
            total_matches = 0
            for pattern in patterns:
                matches = sa.search(pattern)
                total_matches += len(matches)
            elapsed = time.perf_counter() - start
            
            avg_time_per_pattern = elapsed / num_patterns
            patterns_per_sec = num_patterns / elapsed if elapsed > 0 else 0
            
            result = {
                'dataset': dataset_name,
                'text_length': len(sequence),
                'pattern_length': pattern_len,
                'num_patterns': num_patterns,
                'total_time_s': elapsed,
                'total_time_ms': elapsed * 1000,
                'avg_time_per_pattern_ms': avg_time_per_pattern * 1000,
                'patterns_per_second': patterns_per_sec,
                'total_matches': total_matches
            }
            
            self.results['criterion_5_scalability_patterns'].append(result)
            
            print(f"  {num_patterns:3d} patterns: "
                  f"total={elapsed*1000:7.2f}ms, "
                  f"avg={avg_time_per_pattern*1000:6.3f}ms/pattern, "
                  f"throughput={patterns_per_sec:5.1f}patterns/s")
        
        print("\n✓ Criterion 5 complete")
    
    def criterion_6_robustness(self) -> None:
        """
        CRITERION 6: Robustness
        - Different pattern types
        - Different GC content
        - Edge cases
        """
        print("\n" + "="*80)
        print("CRITERION 6: ROBUSTNESS")
        print("="*80)
        
        dataset_name = 'ecoli'
        if dataset_name not in self.datasets:
            dataset_name = list(self.datasets.keys())[0]
        
        sequence = self.datasets[dataset_name]['sequence']
        sa = self.suffix_arrays[dataset_name]
        
        pattern_len = 100
        
        pattern_types = [
            ('random', 'Random from genome'),
            ('high_gc', 'High GC (80%)'),
            ('low_gc', 'Low GC (20%)'),
            ('repeat_at', 'AT repeat'),
            ('low_complexity', 'Low complexity')
        ]
        
        print(f"\n{dataset_name.upper()} ({len(sequence):,} bp, pattern={pattern_len}bp)")
        print("-" * 80)
        
        for ptype, description in pattern_types:
            # Generate pattern based on type
            if ptype == 'random':
                start_pos = random.randint(0, len(sequence) - pattern_len)
                pattern = sequence[start_pos:start_pos + pattern_len]
            elif ptype == 'high_gc':
                # 80% GC
                gc_count = int(pattern_len * 0.8)
                at_count = pattern_len - gc_count
                pattern = 'G' * (gc_count//2) + 'C' * (gc_count - gc_count//2) + 'A' * at_count
            elif ptype == 'low_gc':
                # 20% GC
                gc_count = int(pattern_len * 0.2)
                at_count = pattern_len - gc_count
                pattern = 'A' * (at_count//2) + 'T' * (at_count - at_count//2) + 'G' * gc_count
            elif ptype == 'repeat_at':
                pattern = 'AT' * (pattern_len // 2)
            else:  # low_complexity
                pattern = 'AAAT' * (pattern_len // 4)
            
            # Ensure correct length
            pattern = pattern[:pattern_len]
            
            # Multiple runs for statistics
            times = []
            for _ in range(NUM_RUNS):
                start = time.perf_counter()
                matches = sa.search(pattern)
                elapsed = time.perf_counter() - start
                times.append(elapsed)
            
            mean_time = statistics.mean(times)
            std_dev = statistics.stdev(times) if len(times) > 1 else 0
            gc_content = (pattern.count('G') + pattern.count('C')) / len(pattern)
            
            result = {
                'dataset': dataset_name,
                'text_length': len(sequence),
                'pattern_type': ptype,
                'description': description,
                'pattern_length': len(pattern),
                'gc_content': gc_content,
                'num_runs': NUM_RUNS,
                'mean_time_ms': mean_time * 1000,
                'std_dev_ms': std_dev * 1000,
                'num_matches': len(matches)
            }
            
            self.results['criterion_6_robustness'].append(result)
            
            print(f"  {ptype:15s}: {mean_time*1000:7.3f}ms ± {std_dev*1000:5.3f}ms, "
                  f"GC={gc_content:.1%}, matches={len(matches):5d}")
        
        print("\n✓ Criterion 6 complete")
    
    def save_results(self) -> None:
        """Save all results to CSV and JSON."""
        print("\n" + "="*80)
        print("SAVING RESULTS")
        print("="*80)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for criterion_name, data in self.results.items():
            if not data:
                continue
            
            # Get all unique fieldnames
            all_fields = set()
            for record in data:
                all_fields.update(record.keys())
            fieldnames = sorted(all_fields)
            
            # Save CSV
            csv_path = BENCHMARKS_DIR / f"{criterion_name}_{timestamp}.csv"
            with open(csv_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(data)
            print(f"✓ Saved {criterion_name}: {csv_path.name}")
            
            # Save JSON
            json_path = BENCHMARKS_DIR / f"{criterion_name}_{timestamp}.json"
            with open(json_path, 'w') as f:
                json.dump(data, f, indent=2)
    
    def generate_report(self) -> None:
        """Generate comprehensive evaluation report."""
        print("\n" + "="*80)
        print("GENERATING REPORT")
        print("="*80)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = REPORTS_DIR / f"suffix_array_smart_evaluation_{timestamp}.txt"
        
        with open(report_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("SUFFIX ARRAY - SMART COMPREHENSIVE EVALUATION REPORT\n")
            f.write("="*80 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("EVALUATION STRATEGY:\n")
            f.write("-"*80 + "\n")
            f.write("- Built suffix arrays ONCE per genome (reused for all tests)\n")
            f.write("- Full genomes for search/accuracy tests\n")
            f.write("- Samples for construction scalability only\n")
            f.write("- All 6 criteria evaluated matching KMP format\n\n")
            
            f.write("DATASETS (Full Genomes)\n")
            f.write("-"*80 + "\n\n")
            for name, data in self.datasets.items():
                stats = data['stats']
                f.write(f"{name}:\n")
                f.write(f"  Length: {stats['length']:,} bp\n")
                f.write(f"  GC Content: {stats['gc_content']:.2%}\n\n")
            
            # Criterion 1: Latency
            if self.results['criterion_1_latency']:
                f.write("\nCRITERION 1: LATENCY / TIME\n")
                f.write("-"*80 + "\n\n")
                for r in self.results['criterion_1_latency'][:10]:  # First 10
                    f.write(f"{r['dataset']} - Pattern {r['pattern_length']}bp:\n")
                    f.write(f"  Mean time: {r['mean_time_ms']:.3f}ms\n")
                    f.write(f"  Std dev: {r['std_dev_ms']:.3f}ms\n")
                    f.write(f"  Throughput: {r['throughput_mbps']:.2f} MB/s\n")
                    f.write(f"  Matches: {r['num_matches']}\n\n")
            
            # Criterion 2: Preprocessing
            if self.results['criterion_2_preprocessing']:
                f.write("\nCRITERION 2: PREPROCESSING TIME\n")
                f.write("-"*80 + "\n\n")
                for r in self.results['criterion_2_preprocessing']:
                    f.write(f"{r['dataset']} - {r['text_length']:,}bp:\n")
                    f.write(f"  Construction time: {r['mean_time_ms']:.3f}ms\n")
                    f.write(f"  Time per char: {r['time_per_char_us']:.3f} µs\n\n")
            
            # Criterion 4: Accuracy
            if self.results['criterion_4_accuracy']:
                f.write("\nCRITERION 4: ACCURACY\n")
                f.write("-"*80 + "\n\n")
                for r in self.results['criterion_4_accuracy']:
                    f.write(f"{r['dataset']} - Pattern {r['pattern_length']}bp:\n")
                    f.write(f"  Precision: {r['precision']:.6f}\n")
                    f.write(f"  Recall: {r['recall']:.6f}\n")
                    f.write(f"  F1 Score: {r['f1_score']:.6f}\n")
                    f.write(f"  Agreement: {r['agreement_rate']:.1%}\n\n")
            
            f.write("\n" + "="*80 + "\n")
            f.write("EVALUATION COMPLETE - All 6 criteria evaluated!\n")
            f.write("="*80 + "\n")
        
        print(f"✓ Saved report: {report_path.name}")
        
        # Print summary to console
        with open(report_path, 'r') as f:
            print("\n" + f.read())
    
    def run_all(self) -> None:
        """Run complete smart evaluation."""
        start_time = time.time()
        
        print("\n" + "="*80)
        print("SUFFIX ARRAY - SMART COMPREHENSIVE EVALUATION")
        print("="*80)
        print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("\nEstimated time: 15-20 minutes")
        print("Strategy: Build once, test many times")
        
        self.load_datasets()
        
        if not self.datasets:
            print("\n✗ No datasets found!")
            return
        
        # Build suffix arrays ONCE
        self.build_suffix_arrays()
        
        # Run all 6 criteria
        self.criterion_1_latency_time()
        self.criterion_2_preprocessing_time()
        self.criterion_3_memory_usage()
        self.criterion_4_accuracy()
        self.criterion_5_scalability()
        self.criterion_6_robustness()
        
        # Save and report
        self.save_results()
        self.generate_report()
        
        elapsed = time.time() - start_time
        
        print("\n" + "="*80)
        print("SMART EVALUATION COMPLETE!")
        print("="*80)
        print(f"End: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Total time: {elapsed/60:.1f} minutes ({elapsed:.0f} seconds)")
        print(f"\nResults saved to:")
        print(f"  Benchmarks: {BENCHMARKS_DIR}")
        print(f"  Reports: {REPORTS_DIR}")
        print("="*80 + "\n")


def main():
    evaluator = SmartSuffixArrayEvaluator()
    evaluator.run_all()


if __name__ == "__main__":
    main()
