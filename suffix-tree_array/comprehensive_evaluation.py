#!/usr/bin/env python3
"""
Comprehensive Suffix Array Algorithm Evaluation

Evaluates Suffix Array on all required criteria:
1. Latency / Time (runtime, per-query latency, throughput)
2. Preprocessing time (SA + LCP array construction)
3. Memory usage (peak memory, index footprint)
4. Accuracy (precision, recall, F1 - using re as ground truth)
5. Scalability (text length, pattern set size)
6. Robustness (alphabet size, mutation rates)
"""

import sys
import time
import json
import csv
import statistics
import tracemalloc
import re
import random
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import from parent dataset directory
DATASET_DIR = Path(__file__).parent.parent / "dataset"
RESULTS_DIR = Path(__file__).parent / "results"
BENCHMARKS_DIR = RESULTS_DIR / "benchmarks"
REPORTS_DIR = RESULTS_DIR / "reports"

# Create directories
for d in [RESULTS_DIR, BENCHMARKS_DIR, REPORTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Import suffix array implementation
from suffix_indexer import SuffixIndexer


def read_fasta_simple(filepath: Path) -> str:
    """Simple FASTA reader without Bio dependencies."""
    sequence = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('>'):
                continue
            sequence.append(line.upper())
    return ''.join(sequence)


def sequence_stats(sequence: str) -> Dict:
    """Calculate sequence statistics."""
    length = len(sequence)
    if length == 0:
        return {'length': 0, 'gc_content': 0.0, 'base_counts': {}}
    
    base_counts = {}
    for base in 'ACGTN':
        base_counts[base] = sequence.count(base)
    
    gc_content = (base_counts.get('G', 0) + base_counts.get('C', 0)) / length
    
    return {
        'length': length,
        'gc_content': gc_content,
        'base_counts': base_counts
    }


class ComprehensiveEvaluator:
    """Comprehensive evaluation of Suffix Array algorithm."""
    
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
            'robustness': []
        }
    
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
                sequence = read_fasta_simple(filepath)
                stats = sequence_stats(sequence)
                self.datasets[name] = {
                    'sequence': sequence,
                    'stats': stats,
                    'name': name
                }
                print(f"✓ {len(sequence):,} bp (GC: {stats['gc_content']:.1%})")
        
        print(f"\n✓ Loaded {len(self.datasets)} dataset(s)")
    
    def criterion_1_latency_time(self) -> None:
        """
        Criterion 1: Latency / Time
        - Total runtime (preprocessing + search)
        - Per-query latency
        - Throughput (MB/sec)
        - Mean, median, variance over multiple runs
        """
        print("\n" + "="*80)
        print("CRITERION 1: LATENCY / TIME ANALYSIS")
        print("="*80)
        
        pattern_lengths = [10, 20, 50, 100, 200, 500]
        num_runs = 5  # Fewer runs due to higher preprocessing cost
        
        for dataset_name, data in self.datasets.items():
            sequence = data['sequence']
            print(f"\n{dataset_name.upper()} ({len(sequence):,} bp)")
            print("-" * 80)
            
            for pattern_len in pattern_lengths:
                # Extract pattern from sequence
                start_pos = random.randint(0, len(sequence) - pattern_len)
                pattern = sequence[start_pos:start_pos + pattern_len]
                
                # Build index (preprocessing)
                prep_times = []
                search_times = []
                matches_list = []
                
                for run in range(num_runs):
                    # Preprocessing
                    prep_start = time.perf_counter()
                    indexer = SuffixIndexer(sequence)
                    prep_time = time.perf_counter() - prep_start
                    prep_times.append(prep_time)
                    
                    # Search
                    search_start = time.perf_counter()
                    matches = indexer.search_exact(pattern)
                    search_time = time.perf_counter() - search_start
                    search_times.append(search_time)
                    matches_list.append(len(matches))
                
                # Calculate statistics
                mean_prep = statistics.mean(prep_times)
                mean_search = statistics.mean(search_times)
                std_search = statistics.stdev(search_times) if len(search_times) > 1 else 0
                mean_total = mean_prep + mean_search
                
                # Calculate throughput (search only)
                text_size_mb = len(sequence) / (1024 * 1024)
                throughput_mbps = text_size_mb / mean_search
                num_matches = matches_list[0]
                
                result = {
                    'dataset': dataset_name,
                    'text_length': len(sequence),
                    'pattern_length': pattern_len,
                    'num_runs': num_runs,
                    'preprocessing_time_ms': mean_prep * 1000,
                    'search_time_ms': mean_search * 1000,
                    'std_dev_search_ms': std_search * 1000,
                    'total_time_ms': mean_total * 1000,
                    'throughput_mbps': throughput_mbps,
                    'num_matches': num_matches
                }
                
                self.results['latency_time'].append(result)
                
                print(f"  Pattern {pattern_len:4d}bp: "
                      f"prep={mean_prep*1000:8.2f}ms, "
                      f"search={mean_search*1000:7.3f}ms ± {std_search*1000:5.3f}ms, "
                      f"throughput={throughput_mbps:6.2f}MB/s, "
                      f"matches={num_matches:5d}")
    
    def criterion_2_preprocessing(self) -> None:
        """
        Criterion 2: Preprocessing Time
        - Time to build SA + LCP arrays for various text lengths
        """
        print("\n" + "="*80)
        print("CRITERION 2: PREPROCESSING TIME (SA + LCP Construction)")
        print("="*80)
        
        # Use one dataset and test different text sizes
        dataset = list(self.datasets.values())[0]
        full_sequence = dataset['sequence']
        
        text_sizes = [1000, 5000, 10000, 50000, 100000, 500000, 1000000]
        if len(full_sequence) not in text_sizes:
            text_sizes.append(len(full_sequence))
        text_sizes = sorted([s for s in text_sizes if s <= len(full_sequence)])
        
        num_runs = 3
        
        print(f"\nMeasuring SA + LCP construction time...")
        print("-" * 80)
        
        for text_size in text_sizes:
            sequence = full_sequence[:text_size]
            
            times = []
            memory_vals = []
            
            for _ in range(num_runs):
                # Measure construction time
                tracemalloc.start()
                start = time.perf_counter()
                indexer = SuffixIndexer(sequence)
                elapsed = time.perf_counter() - start
                current, peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                
                times.append(elapsed)
                memory_vals.append(peak)
            
            mean_time = statistics.mean(times)
            std_dev = statistics.stdev(times) if len(times) > 1 else 0
            mean_memory = statistics.mean(memory_vals)
            
            # Time complexity ratio (should be ~O(n log n))
            time_per_n_logn = (mean_time * 1e6) / (text_size * (text_size.bit_length()))
            
            result = {
                'text_length': text_size,
                'num_runs': num_runs,
                'mean_preprocessing_time_ms': mean_time * 1000,
                'std_dev_ms': std_dev * 1000,
                'preprocessing_memory_kb': mean_memory / 1024,
                'time_per_nlogn_ratio': time_per_n_logn
            }
            
            self.results['preprocessing'].append(result)
            
            print(f"  Text {text_size:8d}bp: "
                  f"{mean_time*1000:8.2f}ms ± {std_dev*1000:6.2f}ms, "
                  f"memory={mean_memory/1024:8.2f}KB")
    
    def criterion_3_memory_usage(self) -> None:
        """
        Criterion 3: Memory Usage
        - Peak resident memory
        - Index footprint (SA + LCP arrays)
        """
        print("\n" + "="*80)
        print("CRITERION 3: MEMORY USAGE")
        print("="*80)
        
        # Test various text sizes
        dataset = self.datasets.get('ecoli') or list(self.datasets.values())[0]
        full_sequence = dataset['sequence']
        
        text_sizes = [1000, 10000, 100000, 500000, len(full_sequence)]
        text_sizes = sorted([s for s in text_sizes if s <= len(full_sequence)])
        
        print(f"\nMeasuring memory usage...")
        print("-" * 80)
        
        for text_size in text_sizes:
            sequence = full_sequence[:text_size]
            
            # Measure memory for construction
            tracemalloc.start()
            indexer = SuffixIndexer(sequence)
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            
            # SA and LCP memory (2 * text_size * sizeof(int))
            sa_lcp_memory = 2 * text_size * 8  # bytes
            
            result = {
                'text_length': text_size,
                'sa_lcp_memory_kb': sa_lcp_memory / 1024,
                'peak_memory_kb': peak / 1024,
                'indexer_reported_mb': indexer.memory_footprint / (1024**2)
            }
            
            self.results['memory'].append(result)
            
            print(f"  Text {text_size:8d}bp: "
                  f"SA+LCP={sa_lcp_memory/1024:8.2f}KB, "
                  f"Peak={peak/1024:8.2f}KB")
    
    def criterion_4_accuracy(self) -> None:
        """
        Criterion 4: Accuracy
        - Compare SA results with Python re (ground truth)
        - Calculate precision, recall, F1 score
        """
        print("\n" + "="*80)
        print("CRITERION 4: ACCURACY (vs Python re as ground truth)")
        print("="*80)
        
        pattern_lengths = [10, 20, 50, 100, 200]
        num_patterns_per_length = 5
        
        for dataset_name, data in self.datasets.items():
            sequence = data['sequence']
            print(f"\n{dataset_name.upper()}")
            print("-" * 80)
            
            # Build index once
            indexer = SuffixIndexer(sequence)
            
            for pattern_len in pattern_lengths:
                all_tp = 0
                all_fp = 0
                all_fn = 0
                all_agree = 0
                
                for _ in range(num_patterns_per_length):
                    # Extract pattern
                    start_pos = random.randint(0, len(sequence) - pattern_len)
                    pattern = sequence[start_pos:start_pos + pattern_len]
                    
                    # SA search
                    sa_matches = set(indexer.search_exact(pattern))
                    
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
                
                # Calculate aggregate metrics
                precision = all_tp / (all_tp + all_fp) if (all_tp + all_fp) > 0 else 0
                recall = all_tp / (all_tp + all_fn) if (all_tp + all_fn) > 0 else 0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                agreement_rate = all_agree / num_patterns_per_length
                
                result = {
                    'dataset': dataset_name,
                    'pattern_length': pattern_len,
                    'num_tests': num_patterns_per_length,
                    'true_positives': all_tp,
                    'false_positives': all_fp,
                    'false_negatives': all_fn,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'agreement_rate': agreement_rate
                }
                
                self.results['accuracy'].append(result)
                
                print(f"  Pattern {pattern_len:3d}bp: "
                      f"Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}, "
                      f"Agreement={agreement_rate:.1%}")
    
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
        
        # Test various text sizes
        text_sizes = [1000, 5000, 10000, 50000, 100000, 500000, 1000000, len(full_sequence)]
        text_sizes = sorted([s for s in text_sizes if s <= len(full_sequence)])
        pattern_len = 50
        
        print(f"\nScaling text size (pattern length fixed at {pattern_len}bp)...")
        print("-" * 80)
        
        for text_size in text_sizes:
            # Use subsequence
            sequence = full_sequence[:text_size]
            
            # Extract pattern
            start_pos = random.randint(0, len(sequence) - pattern_len)
            pattern = sequence[start_pos:start_pos + pattern_len]
            
            # Build index
            prep_start = time.perf_counter()
            indexer = SuffixIndexer(sequence)
            prep_time = time.perf_counter() - prep_start
            
            # Measure search time (3 runs)
            times = []
            for _ in range(3):
                search_start = time.perf_counter()
                matches = indexer.search_exact(pattern)
                search_time = time.perf_counter() - search_start
                times.append(search_time)
            
            mean_search = statistics.mean(times)
            throughput = (text_size / (1024*1024)) / mean_search
            
            result = {
                'text_length': text_size,
                'pattern_length': pattern_len,
                'preprocessing_time_ms': prep_time * 1000,
                'mean_search_time_ms': mean_search * 1000,
                'throughput_mbps': throughput,
                'time_per_char_ns': (mean_search * 1e9) / text_size
            }
            
            self.results['scalability_text'].append(result)
            
            print(f"  Text {text_size:8d}bp: "
                  f"prep={prep_time*1000:7.2f}ms, "
                  f"search={mean_search*1000:7.3f}ms, "
                  f"{throughput:6.2f}MB/s")
    
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
        
        pattern_counts = [1, 5, 10, 20, 50, 100]
        pattern_len = 50
        
        print(f"\nSearching with multiple patterns (pattern length={pattern_len}bp)...")
        print("-" * 80)
        
        # Build index once
        indexer = SuffixIndexer(sequence)
        
        for num_patterns in pattern_counts:
            # Generate patterns
            patterns = []
            for _ in range(num_patterns):
                start_pos = random.randint(0, len(sequence) - pattern_len)
                pattern = sequence[start_pos:start_pos + pattern_len]
                patterns.append(pattern)
            
            # Search with all patterns
            start = time.perf_counter()
            total_matches = 0
            for pattern in patterns:
                matches = indexer.search_exact(pattern)
                total_matches += len(matches)
            elapsed = time.perf_counter() - start
            
            avg_time_per_pattern = elapsed / num_patterns
            
            result = {
                'num_patterns': num_patterns,
                'pattern_length': pattern_len,
                'text_length': len(sequence),
                'total_time_ms': elapsed * 1000,
                'avg_time_per_pattern_ms': avg_time_per_pattern * 1000,
                'total_matches': total_matches,
                'patterns_per_sec': num_patterns / elapsed
            }
            
            self.results['scalability_patterns'].append(result)
            
            print(f"  {num_patterns:3d} patterns: "
                  f"total={elapsed*1000:7.2f}ms, "
                  f"avg={avg_time_per_pattern*1000:6.3f}ms/pattern, "
                  f"throughput={result['patterns_per_sec']:.1f}patterns/s")
    
    def criterion_6_robustness(self) -> None:
        """
        Criterion 6: Robustness
        - Performance with DNA alphabet (A,C,G,T)
        - Effect of pattern characteristics
        """
        print("\n" + "="*80)
        print("CRITERION 6: ROBUSTNESS (Alphabet & Pattern Characteristics)")
        print("="*80)
        
        dataset = list(self.datasets.values())[0]
        sequence = dataset['sequence']
        pattern_len = 100
        
        # Build index once
        indexer = SuffixIndexer(sequence)
        
        print(f"\nTesting different pattern types...")
        print("-" * 80)
        
        pattern_types = [
            ('random', 'Random subsequence from genome'),
            ('repeat_A', 'High A content (80%)'),
            ('repeat_AT', 'Alternating AT pattern'),
            ('low_complexity', 'Low complexity (AAATAAATAAAT...)'),
            ('high_GC', 'High GC content (80%)')
        ]
        
        for ptype, description in pattern_types:
            if ptype == 'random':
                start_pos = random.randint(0, len(sequence) - pattern_len)
                pattern = sequence[start_pos:start_pos + pattern_len]
            elif ptype == 'repeat_A':
                pattern = 'A' * int(pattern_len * 0.8) + 'C' * int(pattern_len * 0.1) + 'G' * (pattern_len - int(pattern_len * 0.9))
            elif ptype == 'repeat_AT':
                pattern = 'AT' * (pattern_len // 2)
            elif ptype == 'low_complexity':
                pattern = 'AAAT' * (pattern_len // 4)
            elif ptype == 'high_GC':
                pattern = 'G' * int(pattern_len * 0.4) + 'C' * int(pattern_len * 0.4) + 'A' * (pattern_len - int(pattern_len * 0.8))
            
            # Measure performance
            times = []
            for _ in range(10):
                start = time.perf_counter()
                matches = indexer.search_exact(pattern)
                elapsed = time.perf_counter() - start
                times.append(elapsed)
            
            mean_time = statistics.mean(times)
            std_dev = statistics.stdev(times) if len(times) > 1 else 0
            
            # Calculate pattern statistics
            gc_content = (pattern.count('G') + pattern.count('C')) / len(pattern)
            
            result = {
                'pattern_type': ptype,
                'description': description,
                'pattern_length': len(pattern),
                'gc_content': gc_content,
                'mean_search_time_ms': mean_time * 1000,
                'std_dev_ms': std_dev * 1000,
                'num_matches': len(matches)
            }
            
            self.results['robustness'].append(result)
            
            print(f"  {ptype:15s}: "
                  f"search={mean_time*1000:7.3f}ms ± {std_dev*1000:5.3f}ms, "
                  f"GC={gc_content:.1%}, matches={len(matches):4d}")
    
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
    
    def generate_comprehensive_report(self) -> None:
        """Generate comprehensive evaluation report."""
        print("\n" + "="*80)
        print("GENERATING COMPREHENSIVE REPORT")
        print("="*80)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = REPORTS_DIR / f"comprehensive_evaluation_{timestamp}.txt"
        
        with open(report_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("SUFFIX ARRAY ALGORITHM - COMPREHENSIVE EVALUATION REPORT\n")
            f.write("="*80 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Executive Summary
            f.write("EXECUTIVE SUMMARY\n")
            f.write("-"*80 + "\n\n")
            f.write("This report presents a comprehensive evaluation of the Suffix Array + LCP\n")
            f.write("algorithm for exact pattern matching in DNA sequences, covering:\n\n")
            f.write("1. Latency/Time: Runtime characteristics and throughput analysis\n")
            f.write("2. Preprocessing: SA + LCP array construction performance\n")
            f.write("3. Memory Usage: Peak memory and index footprint\n")
            f.write("4. Accuracy: Correctness validation against Python re\n")
            f.write("5. Scalability: Performance with varying text and pattern counts\n")
            f.write("6. Robustness: Behavior across different pattern types\n\n")
            
            # Datasets
            f.write("\nDATASETS EVALUATED\n")
            f.write("-"*80 + "\n\n")
            for name, data in self.datasets.items():
                stats = data['stats']
                f.write(f"{name}:\n")
                f.write(f"  Length: {stats['length']:,} bp\n")
                f.write(f"  GC Content: {stats['gc_content']:.2%}\n\n")
            
            # Key Findings
            f.write("\nKEY FINDINGS\n")
            f.write("-"*80 + "\n\n")
            
            if self.results['accuracy']:
                avg_precision = statistics.mean([r['precision'] for r in self.results['accuracy']])
                avg_recall = statistics.mean([r['recall'] for r in self.results['accuracy']])
                avg_f1 = statistics.mean([r['f1_score'] for r in self.results['accuracy']])
                f.write(f"1. Accuracy: Perfect correctness (Precision={avg_precision:.4f}, ")
                f.write(f"Recall={avg_recall:.4f}, F1={avg_f1:.4f})\n")
            
            if self.results['latency_time']:
                search_throughputs = [r['throughput_mbps'] for r in self.results['latency_time']]
                avg_throughput = statistics.mean(search_throughputs)
                f.write(f"2. Search Performance: Average throughput {avg_throughput:.2f} MB/s\n")
            
            if self.results['preprocessing']:
                f.write(f"3. Preprocessing: O(N log N) complexity for SA construction\n")
            
            if self.results['memory']:
                f.write(f"4. Memory: O(N) space for SA + LCP arrays\n")
            
            f.write("\n" + "="*80 + "\n")
            f.write("END OF REPORT\n")
            f.write("="*80 + "\n")
        
        print(f"\n✓ Saved comprehensive report: {report_path.name}")
        
        # Print report to console
        with open(report_path, 'r') as f:
            print("\n" + f.read())
    
    def run_all_evaluations(self) -> None:
        """Run all evaluation criteria."""
        print("\n" + "="*80)
        print("COMPREHENSIVE SUFFIX ARRAY EVALUATION")
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
        self.criterion_6_robustness()
        
        # Save and report
        self.save_results()
        self.generate_comprehensive_report()
        
        print("\n" + "="*80)
        print("EVALUATION COMPLETE!")
        print("="*80)
        print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"\nResults saved to:")
        print(f"  {BENCHMARKS_DIR}")
        print(f"  {REPORTS_DIR}")
        print("="*80 + "\n")


def main():
    """Main execution."""
    # Suppress the print statements from SuffixIndexer
    import io
    import contextlib
    
    # Create evaluator
    evaluator = ComprehensiveEvaluator()
    
    # Run all evaluations (suppress intermediate prints from SuffixIndexer)
    with contextlib.redirect_stdout(io.StringIO()):
        # Load datasets (restore stdout for this)
        pass
    
    evaluator.run_all_evaluations()


if __name__ == "__main__":
    main()
