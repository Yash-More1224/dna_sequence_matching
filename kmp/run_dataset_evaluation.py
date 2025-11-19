#!/usr/bin/env python3
"""
Run KMP algorithm on downloaded datasets and perform comprehensive evaluation.

This script:
1. Loads downloaded DNA sequence datasets
2. Runs KMP pattern matching with various pattern lengths
3. Evaluates performance metrics (time, memory, throughput)
4. Compares with Python's re module
5. Generates detailed reports and visualizations
"""

import sys
import os
from pathlib import Path
import time
import json
import csv
from datetime import datetime
from typing import List, Dict, Tuple
import random

# Add kmp directory to path
kmp_dir = Path(__file__).parent
if str(kmp_dir) not in sys.path:
    sys.path.insert(0, str(kmp_dir))

from kmp_algorithm import KMP, kmp_search
from data_loader import read_fasta, sequence_stats
from benchmarking import benchmark_kmp_search, BenchmarkResult
from evaluation import compare_with_re, print_comparison_summary
from config import DNA_BASES

# Dataset configuration
DATASET_DIR = Path(__file__).parent.parent / "dataset"
RESULTS_DIR = Path(__file__).parent / "results"
PLOTS_DIR = RESULTS_DIR / "plots"
BENCHMARKS_DIR = RESULTS_DIR / "benchmarks"
REPORTS_DIR = RESULTS_DIR / "reports"

# Create directories
for d in [RESULTS_DIR, PLOTS_DIR, BENCHMARKS_DIR, REPORTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)


class DatasetEvaluator:
    """Evaluate KMP algorithm on DNA sequence datasets."""
    
    def __init__(self):
        """Initialize evaluator."""
        self.datasets = {}
        self.results = []
        self.comparison_results = []
    
    def load_datasets(self) -> None:
        """Load all available datasets."""
        print("="*70)
        print("Loading Datasets")
        print("="*70)
        
        dataset_files = {
            'ecoli': 'ecoli_k12_mg1655.fasta',
            'lambda_phage': 'lambda_phage.fasta',
            'salmonella': 'salmonella_typhimurium.fasta'
        }
        
        for name, filename in dataset_files.items():
            filepath = DATASET_DIR / filename
            if filepath.exists():
                print(f"\nLoading {name}...")
                records = read_fasta(filepath)
                if records:
                    sequence = records[0].sequence
                    stats = sequence_stats(sequence)
                    
                    self.datasets[name] = {
                        'sequence': sequence,
                        'filename': filename,
                        'stats': stats,
                        'record': records[0]
                    }
                    
                    print(f"  ✓ Loaded: {len(sequence):,} bp")
                    print(f"  GC content: {stats['gc_content']:.2%}")
                else:
                    print(f"  ✗ Failed to load {filepath}")
            else:
                print(f"  ✗ File not found: {filepath}")
        
        print(f"\n✓ Loaded {len(self.datasets)} dataset(s)")
    
    def generate_patterns(self, sequence: str, lengths: List[int], num_per_length: int = 3) -> List[Tuple[str, int]]:
        """
        Generate patterns from sequence.
        
        Args:
            sequence: Source sequence
            lengths: List of pattern lengths to generate
            num_per_length: Number of patterns per length
            
        Returns:
            List of (pattern, length) tuples
        """
        patterns = []
        random.seed(42)  # For reproducibility
        
        for length in lengths:
            for _ in range(num_per_length):
                if length <= len(sequence):
                    # Extract random subsequence
                    start = random.randint(0, len(sequence) - length)
                    pattern = sequence[start:start+length]
                    patterns.append((pattern, length))
        
        return patterns
    
    def run_kmp_benchmarks(self, dataset_name: str, pattern_lengths: List[int] = None) -> None:
        """
        Run KMP benchmarks on a dataset.
        
        Args:
            dataset_name: Name of dataset to benchmark
            pattern_lengths: List of pattern lengths to test
        """
        if dataset_name not in self.datasets:
            print(f"Dataset {dataset_name} not loaded!")
            return
        
        if pattern_lengths is None:
            pattern_lengths = [10, 20, 50, 100, 200, 500]
        
        dataset = self.datasets[dataset_name]
        sequence = dataset['sequence']
        
        print(f"\n{'='*70}")
        print(f"Benchmarking KMP on {dataset_name}")
        print(f"{'='*70}")
        print(f"Sequence length: {len(sequence):,} bp")
        
        # Generate patterns
        patterns = self.generate_patterns(sequence, pattern_lengths, num_per_length=3)
        
        print(f"\nTesting {len(patterns)} patterns across {len(pattern_lengths)} length categories")
        print(f"Pattern lengths: {pattern_lengths}")
        
        # Run benchmarks
        for i, (pattern, length) in enumerate(patterns, 1):
            print(f"\n[{i}/{len(patterns)}] Pattern length: {length} bp")
            
            # Create KMP instance
            kmp = KMP(pattern)
            
            # Run benchmark
            result = benchmark_kmp_search(kmp, sequence, num_runs=5, measure_memory=True)
            
            # Add metadata
            result_dict = result.to_dict()
            result_dict['dataset'] = dataset_name
            result_dict['pattern'] = pattern[:50] + '...' if len(pattern) > 50 else pattern
            result_dict['timestamp'] = datetime.now().isoformat()
            
            self.results.append(result_dict)
            
            # Print summary
            print(f"  Matches found: {result.num_matches:,}")
            print(f"  Preprocessing: {result.preprocessing_time*1000:.4f} ms")
            print(f"  Search time: {result.search_time*1000:.4f} ms")
            print(f"  Total time: {result.total_time*1000:.4f} ms")
            print(f"  Memory: {result.memory_used/1024:.2f} KB")
            
            # Calculate throughput
            throughput_mbps = (len(sequence) / (1024*1024)) / result.search_time
            print(f"  Throughput: {throughput_mbps:.2f} MB/s")
    
    def compare_with_re(self, dataset_name: str, pattern_lengths: List[int] = None) -> None:
        """
        Compare KMP with Python re module.
        
        Args:
            dataset_name: Name of dataset
            pattern_lengths: List of pattern lengths to test
        """
        if dataset_name not in self.datasets:
            print(f"Dataset {dataset_name} not loaded!")
            return
        
        if pattern_lengths is None:
            pattern_lengths = [10, 20, 50, 100]
        
        dataset = self.datasets[dataset_name]
        sequence = dataset['sequence']
        
        print(f"\n{'='*70}")
        print(f"Comparing KMP vs Python re on {dataset_name}")
        print(f"{'='*70}")
        
        # Generate patterns
        patterns = self.generate_patterns(sequence, pattern_lengths, num_per_length=2)
        
        print(f"\nTesting {len(patterns)} patterns")
        
        for i, (pattern, length) in enumerate(patterns, 1):
            print(f"\n[{i}/{len(patterns)}] Pattern length: {length} bp")
            
            # Compare
            result = compare_with_re(sequence, pattern)
            
            # Store result
            result_dict = result.to_dict()
            result_dict['dataset'] = dataset_name
            result_dict['pattern_length'] = length
            result_dict['timestamp'] = datetime.now().isoformat()
            
            self.comparison_results.append(result_dict)
            
            # Print summary
            print_comparison_summary(result)
    
    def save_results(self) -> None:
        """Save all results to files."""
        print(f"\n{'='*70}")
        print("Saving Results")
        print(f"{'='*70}")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save benchmark results
        if self.results:
            # Save as CSV
            csv_path = BENCHMARKS_DIR / f"kmp_benchmarks_{timestamp}.csv"
            with open(csv_path, 'w', newline='') as f:
                if self.results:
                    writer = csv.DictWriter(f, fieldnames=self.results[0].keys())
                    writer.writeheader()
                    writer.writerows(self.results)
            print(f"✓ Saved benchmark results: {csv_path}")
            
            # Save as JSON
            json_path = BENCHMARKS_DIR / f"kmp_benchmarks_{timestamp}.json"
            with open(json_path, 'w') as f:
                json.dump(self.results, f, indent=2)
            print(f"✓ Saved benchmark results: {json_path}")
        
        # Save comparison results
        if self.comparison_results:
            csv_path = BENCHMARKS_DIR / f"kmp_vs_re_{timestamp}.csv"
            with open(csv_path, 'w', newline='') as f:
                if self.comparison_results:
                    # Flatten nested accuracy dict if present
                    flattened = []
                    for r in self.comparison_results:
                        row = r.copy()
                        if row.get('accuracy'):
                            acc = row.pop('accuracy')
                            for k, v in acc.items():
                                row[f'accuracy_{k}'] = v
                        flattened.append(row)
                    
                    writer = csv.DictWriter(f, fieldnames=flattened[0].keys())
                    writer.writeheader()
                    writer.writerows(flattened)
            print(f"✓ Saved comparison results: {csv_path}")
            
            json_path = BENCHMARKS_DIR / f"kmp_vs_re_{timestamp}.json"
            with open(json_path, 'w') as f:
                json.dump(self.comparison_results, f, indent=2)
            print(f"✓ Saved comparison results: {json_path}")
    
    def generate_report(self) -> None:
        """Generate comprehensive evaluation report."""
        print(f"\n{'='*70}")
        print("Generating Evaluation Report")
        print(f"{'='*70}")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = REPORTS_DIR / f"kmp_evaluation_report_{timestamp}.txt"
        
        with open(report_path, 'w') as f:
            f.write("="*70 + "\n")
            f.write("KMP Algorithm Evaluation Report\n")
            f.write("="*70 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("\n")
            
            # Dataset summary
            f.write("DATASETS\n")
            f.write("-"*70 + "\n")
            for name, data in self.datasets.items():
                stats = data['stats']
                f.write(f"\n{name}:\n")
                f.write(f"  Length: {stats['length']:,} bp\n")
                f.write(f"  GC Content: {stats['gc_content']:.2%}\n")
                f.write(f"  Base counts: A={stats['base_counts']['A']:,}, "
                       f"C={stats['base_counts']['C']:,}, "
                       f"G={stats['base_counts']['G']:,}, "
                       f"T={stats['base_counts']['T']:,}\n")
            
            # Benchmark summary
            if self.results:
                f.write("\n\nBENCHMARK RESULTS\n")
                f.write("-"*70 + "\n")
                
                # Group by dataset
                datasets = {}
                for r in self.results:
                    ds = r['dataset']
                    if ds not in datasets:
                        datasets[ds] = []
                    datasets[ds].append(r)
                
                for ds, results in datasets.items():
                    f.write(f"\n{ds}:\n")
                    f.write(f"  Total patterns tested: {len(results)}\n")
                    
                    avg_search_time = sum(r['search_time'] for r in results) / len(results)
                    avg_memory = sum(r['memory_used'] for r in results) / len(results)
                    total_matches = sum(r['num_matches'] for r in results)
                    
                    f.write(f"  Average search time: {avg_search_time*1000:.4f} ms\n")
                    f.write(f"  Average memory: {avg_memory/1024:.2f} KB\n")
                    f.write(f"  Total matches found: {total_matches:,}\n")
                    
                    # Pattern length breakdown
                    pattern_lengths = {}
                    for r in results:
                        pl = r['pattern_length']
                        if pl not in pattern_lengths:
                            pattern_lengths[pl] = []
                        pattern_lengths[pl].append(r)
                    
                    f.write(f"\n  Pattern Length Breakdown:\n")
                    for pl in sorted(pattern_lengths.keys()):
                        prs = pattern_lengths[pl]
                        avg_time = sum(pr['search_time'] for pr in prs) / len(prs)
                        f.write(f"    {pl} bp: {avg_time*1000:.4f} ms (avg), {len(prs)} patterns\n")
            
            # Comparison summary
            if self.comparison_results:
                f.write("\n\nKMP vs PYTHON RE COMPARISON\n")
                f.write("-"*70 + "\n")
                
                for r in self.comparison_results:
                    f.write(f"\nDataset: {r['dataset']}, Pattern length: {r['pattern_length']} bp\n")
                    f.write(f"  KMP time: {r['kmp_time']*1000:.4f} ms\n")
                    f.write(f"  re time: {r['re_time']*1000:.4f} ms\n")
                    f.write(f"  Speedup: {r['speedup']:.2f}x\n")
                    f.write(f"  Matches agree: {r['matches_agree']}\n")
            
            f.write("\n" + "="*70 + "\n")
            f.write("End of Report\n")
            f.write("="*70 + "\n")
        
        print(f"✓ Saved evaluation report: {report_path}")
        
        # Also print to console
        with open(report_path, 'r') as f:
            print(f"\n{f.read()}")
    
    def generate_visualizations(self) -> None:
        """Generate visualization plots."""
        if not self.results:
            print("No results to visualize")
            return
        
        print(f"\n{'='*70}")
        print("Generating Visualizations")
        print(f"{'='*70}")
        
        try:
            import matplotlib.pyplot as plt
            import pandas as pd
            
            # Convert results to DataFrame
            df = pd.DataFrame(self.results)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Plot 1: Latency vs Pattern Length
            fig, ax = plt.subplots(figsize=(10, 6))
            for dataset in df['dataset'].unique():
                data = df[df['dataset'] == dataset]
                grouped = data.groupby('pattern_length')['search_time'].mean()
                ax.plot(grouped.index, grouped.values * 1000, marker='o', label=dataset)
            
            ax.set_xlabel('Pattern Length (bp)')
            ax.set_ylabel('Search Time (ms)')
            ax.set_title('KMP Search Time vs Pattern Length')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plot_path = PLOTS_DIR / f"latency_vs_pattern_length_{timestamp}.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"✓ Saved plot: {plot_path}")
            
            # Plot 2: Memory Usage
            fig, ax = plt.subplots(figsize=(10, 6))
            for dataset in df['dataset'].unique():
                data = df[df['dataset'] == dataset]
                grouped = data.groupby('pattern_length')['memory_used'].mean()
                ax.plot(grouped.index, grouped.values / 1024, marker='s', label=dataset)
            
            ax.set_xlabel('Pattern Length (bp)')
            ax.set_ylabel('Memory Usage (KB)')
            ax.set_title('KMP Memory Usage vs Pattern Length')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plot_path = PLOTS_DIR / f"memory_usage_{timestamp}.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"✓ Saved plot: {plot_path}")
            
            # Plot 3: Throughput
            df['throughput_mbps'] = (df['text_length'] / (1024*1024)) / df['search_time']
            
            fig, ax = plt.subplots(figsize=(10, 6))
            for dataset in df['dataset'].unique():
                data = df[df['dataset'] == dataset]
                grouped = data.groupby('pattern_length')['throughput_mbps'].mean()
                ax.plot(grouped.index, grouped.values, marker='^', label=dataset)
            
            ax.set_xlabel('Pattern Length (bp)')
            ax.set_ylabel('Throughput (MB/s)')
            ax.set_title('KMP Throughput vs Pattern Length')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plot_path = PLOTS_DIR / f"throughput_{timestamp}.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"✓ Saved plot: {plot_path}")
            
            # Plot 4: KMP vs re Comparison
            if self.comparison_results:
                comp_df = pd.DataFrame(self.comparison_results)
                
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
                
                # Time comparison
                for dataset in comp_df['dataset'].unique():
                    data = comp_df[comp_df['dataset'] == dataset]
                    grouped_kmp = data.groupby('pattern_length')['kmp_time'].mean()
                    grouped_re = data.groupby('pattern_length')['re_time'].mean()
                    
                    ax1.plot(grouped_kmp.index, grouped_kmp.values * 1000, 
                            marker='o', label=f'{dataset} (KMP)')
                    ax1.plot(grouped_re.index, grouped_re.values * 1000, 
                            marker='s', linestyle='--', label=f'{dataset} (re)')
                
                ax1.set_xlabel('Pattern Length (bp)')
                ax1.set_ylabel('Time (ms)')
                ax1.set_title('KMP vs Python re: Time Comparison')
                ax1.legend()
                ax1.grid(True, alpha=0.3)
                
                # Speedup
                for dataset in comp_df['dataset'].unique():
                    data = comp_df[comp_df['dataset'] == dataset]
                    grouped = data.groupby('pattern_length')['speedup'].mean()
                    ax2.plot(grouped.index, grouped.values, marker='o', label=dataset)
                
                ax2.axhline(y=1.0, color='r', linestyle='--', label='Parity')
                ax2.set_xlabel('Pattern Length (bp)')
                ax2.set_ylabel('Speedup (re_time / kmp_time)')
                ax2.set_title('KMP Speedup over Python re')
                ax2.legend()
                ax2.grid(True, alpha=0.3)
                
                plot_path = PLOTS_DIR / f"kmp_vs_re_{timestamp}.png"
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                plt.close()
                print(f"✓ Saved plot: {plot_path}")
                
        except ImportError:
            print("⚠ matplotlib not available, skipping visualizations")
        except Exception as e:
            print(f"✗ Error generating visualizations: {e}")


def main():
    """Main execution function."""
    print("\n" + "="*70)
    print("KMP Algorithm Dataset Evaluation")
    print("="*70)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Initialize evaluator
    evaluator = DatasetEvaluator()
    
    # Load datasets
    evaluator.load_datasets()
    
    if not evaluator.datasets:
        print("\n✗ No datasets loaded. Please run download_datasets.py first.")
        return
    
    # Define test parameters
    pattern_lengths_full = [10, 20, 50, 100, 200, 500]
    pattern_lengths_comparison = [10, 20, 50, 100]
    
    # Run benchmarks on all datasets
    for dataset_name in evaluator.datasets.keys():
        evaluator.run_kmp_benchmarks(dataset_name, pattern_lengths_full)
    
    # Run comparisons with re
    for dataset_name in evaluator.datasets.keys():
        evaluator.compare_with_re(dataset_name, pattern_lengths_comparison)
    
    # Save all results
    evaluator.save_results()
    
    # Generate report
    evaluator.generate_report()
    
    # Generate visualizations
    evaluator.generate_visualizations()
    
    print("\n" + "="*70)
    print("Evaluation Complete!")
    print("="*70)
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nResults saved to:")
    print(f"  Benchmarks: {BENCHMARKS_DIR}")
    print(f"  Plots: {PLOTS_DIR}")
    print(f"  Reports: {REPORTS_DIR}")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
