#!/usr/bin/env python3
"""
Balanced Comprehensive Suffix Array Evaluation

Strategy:
- Build suffix array ONCE per dataset (reuse it)
- Use full genomes for search tests
- Use samples only for construction scalability
- This gives realistic performance while staying efficient
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

# BALANCED PARAMETERS
NUM_RUNS = 5  # For search timing
NUM_PATTERNS_PER_LENGTH = 3
PATTERN_LENGTHS = [10, 20, 50, 100, 200, 500]
CONSTRUCTION_SAMPLE_SIZES = [1000, 5000, 10000, 50000, 100000, 500000]  # For scalability tests


class BalancedSuffixArrayEvaluator:
    """Balanced evaluator - build once, search many times."""
    
    def __init__(self):
        self.datasets = {}
        self.suffix_arrays = {}  # Cache built suffix arrays
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
        """Load full datasets."""
        print("\n" + "="*80)
        print("LOADING FULL DATASETS")
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
        print("BUILDING SUFFIX ARRAYS (one-time preprocessing)")
        print("="*80)
        
        for name, data in self.datasets.items():
            sequence = data['sequence']
            print(f"\n{name} ({len(sequence):,} bp)...", end=' ')
            
            start = time.perf_counter()
            sa = SuffixArray(sequence)
            elapsed = time.perf_counter() - start
            
            self.suffix_arrays[name] = sa
            
            print(f"✓ Built in {elapsed:.2f}s ({elapsed*1000:.1f}ms)")
            
            # Store construction time
            self.results['preprocessing'].append({
                'dataset': name,
                'text_length': len(sequence),
                'construction_time_s': elapsed,
                'construction_time_ms': elapsed * 1000
            })
    
    def criterion_1_latency_time(self) -> None:
        """Criterion 1: Search latency on FULL genomes."""
        print("\n" + "="*80)
        print("CRITERION 1: LATENCY / TIME (Search on Full Genomes)")
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
                
                # Multiple search runs
                times = []
                matches_count = 0
                
                for _ in range(NUM_RUNS):
                    start = time.perf_counter()
                    matches = sa.search(pattern)
                    elapsed = time.perf_counter() - start
                    times.append(elapsed)
                    matches_count = len(matches)
                
                mean_time = statistics.mean(times)
                std_dev = statistics.stdev(times) if len(times) > 1 else 0
                throughput_mbps = (len(sequence) / (1024*1024)) / mean_time
                
                result = {
                    'dataset': dataset_name,
                    'text_length': len(sequence),
                    'pattern_length': pattern_len,
                    'num_runs': NUM_RUNS,
                    'mean_search_time_ms': mean_time * 1000,
                    'median_time_ms': statistics.median(times) * 1000,
                    'std_dev_ms': std_dev * 1000,
                    'min_time_ms': min(times) * 1000,
                    'max_time_ms': max(times) * 1000,
                    'throughput_mbps': throughput_mbps,
                    'num_matches': matches_count
                }
                
                self.results['latency_time'].append(result)
                
                print(f"  Pattern {pattern_len:4d}bp: "
                      f"mean={mean_time*1000:7.3f}ms ± {std_dev*1000:5.3f}ms, "
                      f"throughput={throughput_mbps:7.2f}MB/s, "
                      f"matches={matches_count:5d}")
    
    def criterion_2_preprocessing_scalability(self) -> None:
        """Criterion 2: Construction time scalability."""
        print("\n" + "="*80)
        print("CRITERION 2: PREPROCESSING SCALABILITY (Construction Time)")
        print("="*80)
        
        # Use E. coli for samples
        dataset = self.datasets.get('ecoli') or list(self.datasets.values())[0]
        full_sequence = dataset['sequence']
        
        print(f"\nTesting construction time on samples from {len(full_sequence):,} bp genome")
        print("-" * 80)
        
        for size in CONSTRUCTION_SAMPLE_SIZES:
            if size > len(full_sequence):
                continue
            
            sequence = full_sequence[:size]
            
            # Measure construction
            times = []
            for run in range(3):
                start = time.perf_counter()
                sa = SuffixArray(sequence)
                elapsed = time.perf_counter() - start
                times.append(elapsed)
            
            mean_time = statistics.mean(times)
            std_dev = statistics.stdev(times) if len(times) > 1 else 0
            
            result = {
                'text_length': size,
                'mean_construction_time_ms': mean_time * 1000,
                'std_dev_ms': std_dev * 1000,
                'time_per_char_us': (mean_time * 1e6) / size,
                'sa_memory_bytes': size * 4
            }
            
            self.results['preprocessing'].append(result)
            
            print(f"  {size:7,d}bp: {mean_time*1000:8.1f}ms ± {std_dev*1000:6.1f}ms, "
                  f"{result['time_per_char_us']:6.2f}µs/char")
    
    def criterion_3_memory_usage(self) -> None:
        """Criterion 3: Memory footprint."""
        print("\n" + "="*80)
        print("CRITERION 3: MEMORY USAGE")
        print("="*80)
        
        test_sizes = [1000, 5000, 10000, 50000, 100000]
        dataset = list(self.datasets.values())[0]
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
            
            # Measure search memory
            pattern = sequence[:50]
            tracemalloc.start()
            matches = sa.search(pattern)
            search_current, search_peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            
            result = {
                'text_length': size,
                'sa_memory_bytes': size * 4,
                'construction_peak_kb': peak / 1024,
                'search_peak_kb': search_peak / 1024,
                'total_peak_kb': (peak + search_peak) / 1024
            }
            
            self.results['memory'].append(result)
            
            print(f"  {size:6,d}bp: SA≈{size*4/1024:7.1f}KB, "
                  f"Construct peak={peak/1024:8.1f}KB, "
                  f"Search peak={search_peak/1024:6.1f}KB")
    
    def criterion_4_accuracy(self) -> None:
        """Criterion 4: Accuracy on full genomes."""
        print("\n" + "="*80)
        print("CRITERION 4: ACCURACY (vs Python re on Full Genomes)")
        print("="*80)
        
        for dataset_name, data in self.datasets.items():
            sequence = data['sequence']
            sa = self.suffix_arrays[dataset_name]
            
            print(f"\n{dataset_name.upper()} ({len(sequence):,} bp)")
            print("-" * 80)
            
            for pattern_len in PATTERN_LENGTHS:
                all_tp = all_fp = all_fn = all_agree = 0
                
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
                
                self.results['accuracy'].append(result)
                
                print(f"  Pattern {pattern_len:3d}bp: "
                      f"P={precision:.4f}, R={recall:.4f}, F1={f1:.4f}, "
                      f"Agreement={result['agreement_rate']:.1%}")
    
    def criterion_5_scalability_patterns(self) -> None:
        """Criterion 5: Multiple pattern search."""
        print("\n" + "="*80)
        print("CRITERION 5: SCALABILITY - Multiple Patterns")
        print("="*80)
        
        dataset_name = 'ecoli'
        data = self.datasets[dataset_name]
        sequence = data['sequence']
        sa = self.suffix_arrays[dataset_name]
        
        pattern_len = 50
        pattern_counts = [1, 5, 10, 20, 50, 100]
        
        print(f"\n{dataset_name.upper()} ({len(sequence):,} bp, pattern length={pattern_len}bp)")
        print("-" * 80)
        
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
                'dataset': dataset_name,
                'num_patterns': num_patterns,
                'pattern_length': pattern_len,
                'text_length': len(sequence),
                'search_time_ms': elapsed * 1000,
                'avg_time_per_pattern_ms': (elapsed / num_patterns) * 1000,
                'total_matches': total_matches,
                'patterns_per_sec': num_patterns / elapsed
            }
            
            self.results['scalability_patterns'].append(result)
            
            print(f"  {num_patterns:3d} patterns: "
                  f"total={elapsed*1000:7.2f}ms, "
                  f"avg={result['avg_time_per_pattern_ms']:6.3f}ms/pattern, "
                  f"throughput={result['patterns_per_sec']:5.1f}patterns/s")
    
    def criterion_6_robustness(self) -> None:
        """Criterion 6: Robustness across pattern types."""
        print("\n" + "="*80)
        print("CRITERION 6: ROBUSTNESS (Pattern Types on Full Genome)")
        print("="*80)
        
        dataset_name = 'ecoli'
        data = self.datasets[dataset_name]
        sequence = data['sequence']
        sa = self.suffix_arrays[dataset_name]
        
        pattern_len = 100
        
        pattern_types = [
            ('random', 'Random from genome'),
            ('repeat_A', 'High A content (80%)'),
            ('repeat_AT', 'Alternating AT'),
            ('low_complexity', 'Low complexity (AAATAAAT...)'),
            ('high_GC', 'High GC content (80%)')
        ]
        
        print(f"\n{dataset_name.upper()} ({len(sequence):,} bp, pattern length={pattern_len}bp)")
        print("-" * 80)
        
        for ptype, desc in pattern_types:
            if ptype == 'random':
                start_pos = random.randint(0, len(sequence) - pattern_len)
                pattern = sequence[start_pos:start_pos + pattern_len]
            elif ptype == 'repeat_A':
                pattern = 'A' * int(pattern_len * 0.8) + 'C' * int(pattern_len * 0.1) + 'G' * (pattern_len - int(pattern_len * 0.9))
            elif ptype == 'repeat_AT':
                pattern = 'AT' * (pattern_len // 2)
            elif ptype == 'low_complexity':
                pattern = 'AAAT' * (pattern_len // 4)
            else:  # high_GC
                pattern = 'G' * int(pattern_len * 0.4) + 'C' * int(pattern_len * 0.4) + 'A' * (pattern_len - int(pattern_len * 0.8))
            
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
                'pattern_type': ptype,
                'description': desc,
                'pattern_length': len(pattern),
                'gc_content': gc_content,
                'mean_search_time_ms': mean_time * 1000,
                'std_dev_ms': std_dev * 1000,
                'num_matches': len(matches)
            }
            
            self.results['robustness'].append(result)
            
            print(f"  {ptype:15s}: {mean_time*1000:7.3f}ms ± {std_dev*1000:5.3f}ms, "
                  f"GC={gc_content:.1%}, matches={len(matches):5d}")
    
    def save_results(self) -> None:
        """Save all results."""
        print("\n" + "="*80)
        print("SAVING RESULTS")
        print("="*80)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for criterion, data in self.results.items():
            if not data:
                continue
            
            # Get all unique fieldnames from all records
            all_fields = set()
            for record in data:
                all_fields.update(record.keys())
            fieldnames = sorted(all_fields)
            
            csv_path = BENCHMARKS_DIR / f"{criterion}_{timestamp}.csv"
            with open(csv_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(data)
            print(f"✓ Saved {criterion}: {csv_path.name}")
            
            json_path = BENCHMARKS_DIR / f"{criterion}_{timestamp}.json"
            with open(json_path, 'w') as f:
                json.dump(data, f, indent=2)
    
    def generate_report(self) -> None:
        """Generate comprehensive report."""
        print("\n" + "="*80)
        print("GENERATING REPORT")
        print("="*80)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = REPORTS_DIR / f"suffix_array_comprehensive_{timestamp}.txt"
        
        with open(report_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("SUFFIX ARRAY - COMPREHENSIVE EVALUATION REPORT\n")
            f.write("="*80 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("DATASETS (Full Genomes)\n")
            f.write("-"*80 + "\n\n")
            for name, data in self.datasets.items():
                stats = data['stats']
                f.write(f"{name}:\n")
                f.write(f"  Length: {stats['length']:,} bp\n")
                f.write(f"  GC Content: {stats['gc_content']:.2%}\n\n")
            
            if self.results['accuracy']:
                f.write("\nACCURACY RESULTS (Full Genomes)\n")
                f.write("-"*80 + "\n\n")
                for r in self.results['accuracy']:
                    f.write(f"{r['dataset']} ({r['text_length']:,}bp) - Pattern {r['pattern_length']}bp:\n")
                    f.write(f"  Precision: {r['precision']:.6f}\n")
                    f.write(f"  Recall: {r['recall']:.6f}\n")
                    f.write(f"  F1: {r['f1_score']:.6f}\n")
                    f.write(f"  Agreement: {r['agreement_rate']:.1%}\n\n")
            
            if self.results['latency_time']:
                f.write("\nSEARCH PERFORMANCE (Full Genomes)\n")
                f.write("-"*80 + "\n\n")
                for r in self.results['latency_time'][:5]:  # Show first few
                    f.write(f"{r['dataset']} - Pattern {r['pattern_length']}bp:\n")
                    f.write(f"  Mean search time: {r['mean_search_time_ms']:.3f}ms\n")
                    f.write(f"  Throughput: {r['throughput_mbps']:.2f} MB/s\n")
                    f.write(f"  Matches: {r['num_matches']}\n\n")
            
            f.write("\n" + "="*80 + "\n")
        
        print(f"✓ Saved report: {report_path.name}")
        
        with open(report_path, 'r') as f:
            print("\n" + f.read())
    
    def run_all(self) -> None:
        """Run complete evaluation."""
        print("\n" + "="*80)
        print("SUFFIX ARRAY - COMPREHENSIVE EVALUATION (Full Genomes)")
        print("="*80)
        print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        self.load_datasets()
        
        if not self.datasets:
            print("\n✗ No datasets!")
            return
        
        # Build suffix arrays ONCE
        self.build_suffix_arrays()
        
        # Run all evaluations using pre-built suffix arrays
        self.criterion_1_latency_time()
        self.criterion_2_preprocessing_scalability()
        self.criterion_3_memory_usage()
        self.criterion_4_accuracy()
        self.criterion_5_scalability_patterns()
        self.criterion_6_robustness()
        
        self.save_results()
        self.generate_report()
        
        print("\n" + "="*80)
        print("EVALUATION COMPLETE!")
        print("="*80)
        print(f"End: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"\nResults: {BENCHMARKS_DIR}")
        print("="*80 + "\n")


def main():
    evaluator = BalancedSuffixArrayEvaluator()
    evaluator.run_all()


if __name__ == "__main__":
    main()
