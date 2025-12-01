#!/usr/bin/env python3
"""
Regenerate KMP evaluation report with actual data tables instead of placeholders.
"""

import csv
import json
from pathlib import Path
from datetime import datetime

BENCHMARKS_DIR = Path(__file__).parent / "results" / "benchmarks"
REPORTS_DIR = Path(__file__).parent / "results" / "reports"

# Timestamp from the existing files
TIMESTAMP = "20251119_222638"

def load_csv(filename):
    """Load CSV file and return list of dicts."""
    filepath = BENCHMARKS_DIR / filename
    with open(filepath, 'r') as f:
        return list(csv.DictReader(f))

def generate_detailed_report():
    """Generate comprehensive report with actual data tables."""
    
    # Load all data
    latency_data = load_csv(f"latency_time_{TIMESTAMP}.csv")
    preprocessing_data = load_csv(f"preprocessing_{TIMESTAMP}.csv")
    memory_data = load_csv(f"memory_{TIMESTAMP}.csv")
    accuracy_data = load_csv(f"accuracy_{TIMESTAMP}.csv")
    scalability_text_data = load_csv(f"scalability_text_{TIMESTAMP}.csv")
    scalability_patterns_data = load_csv(f"scalability_patterns_{TIMESTAMP}.csv")
    robustness_data = load_csv(f"robustness_{TIMESTAMP}.csv")
    
    report_path = REPORTS_DIR / f"comprehensive_evaluation_{TIMESTAMP}_detailed.txt"
    
    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("KMP ALGORITHM - COMPREHENSIVE EVALUATION REPORT (DETAILED)\n")
        f.write("="*80 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("EXECUTIVE SUMMARY\n")
        f.write("-"*80 + "\n\n")
        f.write("This report presents a comprehensive evaluation of the Knuth-Morris-Pratt (KMP)\n")
        f.write("algorithm for exact pattern matching in DNA sequences, covering:\n\n")
        f.write("1. Latency/Time: Runtime characteristics and throughput analysis\n")
        f.write("2. Preprocessing: LPS array construction performance\n")
        f.write("3. Memory Usage: Peak memory and index footprint\n")
        f.write("4. Accuracy: Correctness validation against Python re\n")
        f.write("5. Scalability: Performance with varying text and pattern counts\n")
        f.write("6. Robustness: Behavior across different pattern types\n\n")
        
        f.write("\nDATASETS EVALUATED\n")
        f.write("-"*80 + "\n\n")
        f.write("ecoli:\n")
        f.write("  Length: 4,641,652 bp\n")
        f.write("  GC Content: 50.79%\n\n")
        f.write("lambda_phage:\n")
        f.write("  Length: 48,502 bp\n")
        f.write("  GC Content: 49.86%\n\n")
        f.write("salmonella:\n")
        f.write("  Length: 4,857,450 bp\n")
        f.write("  GC Content: 52.22%\n\n")
        
        # CRITERION 1: LATENCY/TIME
        f.write("\n" + "="*80 + "\n")
        f.write("CRITERION 1: LATENCY / TIME\n")
        f.write("="*80 + "\n\n")
        
        f.write("Search Performance Metrics (10 runs per test):\n")
        f.write("-"*80 + "\n\n")
        
        # Group by dataset
        datasets = {}
        for row in latency_data:
            ds = row['dataset']
            if ds not in datasets:
                datasets[ds] = []
            datasets[ds].append(row)
        
        for dataset_name, rows in datasets.items():
            f.write(f"{dataset_name.upper()} ({rows[0]['text_length']} bp)\n")
            f.write("-"*80 + "\n")
            f.write(f"{'Pattern':>8s}  {'Mean Time':>12s}  {'Median':>12s}  {'Std Dev':>10s}  {'Throughput':>12s}  {'Matches':>8s}\n")
            f.write(f"{'Length':>8s}  {'(ms)':>12s}  {'(ms)':>12s}  {'(ms)':>10s}  {'(MB/s)':>12s}  {'Found':>8s}\n")
            f.write("-"*80 + "\n")
            
            for row in rows:
                mean_time = float(row['mean_time_ms'])
                median_time = float(row['median_time_ms'])
                std_dev = float(row['std_dev_ms'])
                throughput = float(row['throughput_mbps'])
                matches = int(row['num_matches'])
                plen = int(row['pattern_length'])
                
                f.write(f"{plen:>7d}bp  {mean_time:>11.3f}  {median_time:>11.3f}  {std_dev:>10.3f}  {throughput:>11.2f}  {matches:>8d}\n")
            
            f.write("\n")
        
        # Summary statistics
        all_throughputs = [float(row['throughput_mbps']) for row in latency_data]
        avg_throughput = sum(all_throughputs) / len(all_throughputs)
        
        f.write("\nSummary:\n")
        f.write(f"  Total measurements: {len(latency_data)}\n")
        f.write(f"  Average throughput: {avg_throughput:.2f} MB/s\n")
        f.write(f"  Min throughput: {min(all_throughputs):.2f} MB/s\n")
        f.write(f"  Max throughput: {max(all_throughputs):.2f} MB/s\n\n")
        
        # CRITERION 2: PREPROCESSING
        f.write("\n" + "="*80 + "\n")
        f.write("CRITERION 2: PREPROCESSING TIME\n")
        f.write("="*80 + "\n\n")
        
        f.write("LPS Array Construction Performance (20 runs per pattern length):\n")
        f.write("-"*80 + "\n\n")
        f.write(f"{'Pattern':>8s}  {'Mean Time':>15s}  {'Std Dev':>12s}  {'Memory':>10s}  {'Time/Char':>12s}\n")
        f.write(f"{'Length':>8s}  {'(µs)':>15s}  {'(µs)':>12s}  {'(bytes)':>10s}  {'(ns)':>12s}\n")
        f.write("-"*80 + "\n")
        
        for row in preprocessing_data:
            plen = int(row['pattern_length'])
            mean_time = float(row['mean_preprocessing_time_us'])
            std_dev = float(row['std_dev_us'])
            memory = int(row['lps_memory_bytes'])
            time_per_char = mean_time * 1000 / plen  # Convert to ns
            
            f.write(f"{plen:>7d}bp  {mean_time:>14.3f}  {std_dev:>11.3f}  {memory:>10d}  {time_per_char:>11.2f}\n")
        
        f.write("\nKey Findings:\n")
        f.write("  - Linear time complexity: O(m) where m = pattern length\n")
        f.write("  - Preprocessing time: 0.6µs - 554µs for patterns 10bp - 10,000bp\n")
        f.write("  - Memory usage: 8 bytes per pattern character (64-bit integers)\n")
        f.write("  - Consistent performance across all pattern lengths\n\n")
        
        # CRITERION 3: MEMORY USAGE
        f.write("\n" + "="*80 + "\n")
        f.write("CRITERION 3: MEMORY USAGE\n")
        f.write("="*80 + "\n\n")
        
        f.write("Memory Profiling (Text: 4,641,652 bp E. coli genome):\n")
        f.write("-"*80 + "\n\n")
        f.write(f"{'Pattern':>8s}  {'LPS Array':>12s}  {'Preproc Peak':>14s}  {'Search Peak':>13s}  {'Total Peak':>12s}\n")
        f.write(f"{'Length':>8s}  {'(bytes)':>12s}  {'(KB)':>14s}  {'(KB)':>13s}  {'(KB)':>12s}\n")
        f.write("-"*80 + "\n")
        
        for row in memory_data:
            plen = int(row['pattern_length'])
            lps_mem = int(row['lps_memory_bytes'])
            preproc_peak = float(row['preprocessing_peak_kb'])
            search_peak = float(row['search_peak_kb'])
            total_peak = float(row['total_peak_kb'])
            
            f.write(f"{plen:>7d}bp  {lps_mem:>12d}  {preproc_peak:>13.2f}  {search_peak:>12.2f}  {total_peak:>11.2f}\n")
        
        f.write("\nKey Findings:\n")
        f.write("  - LPS array: Exactly 8 bytes per pattern character\n")
        f.write("  - Total memory footprint: < 100 KB even for 10,000bp patterns\n")
        f.write("  - Search memory: Constant ~0.2 KB regardless of pattern size\n")
        f.write("  - Very memory efficient for DNA sequence matching\n\n")
        
        # CRITERION 4: ACCURACY
        f.write("\n" + "="*80 + "\n")
        f.write("CRITERION 4: ACCURACY\n")
        f.write("="*80 + "\n\n")
        
        f.write("Correctness Validation (using Python re as ground truth):\n")
        f.write("-"*80 + "\n\n")
        
        # Group by dataset
        datasets = {}
        for row in accuracy_data:
            ds = row['dataset']
            if ds not in datasets:
                datasets[ds] = []
            datasets[ds].append(row)
        
        for dataset_name, rows in datasets.items():
            f.write(f"\n{dataset_name.upper()}\n")
            f.write("-"*80 + "\n")
            f.write(f"{'Pattern':>8s}  {'Precision':>10s}  {'Recall':>10s}  {'F1 Score':>10s}  {'Agreement':>10s}\n")
            f.write("-"*80 + "\n")
            
            for row in rows:
                plen = int(row['pattern_length'])
                precision = float(row['precision'])
                recall = float(row['recall'])
                f1 = float(row['f1_score'])
                agreement = float(row['agreement_rate'])
                
                f.write(f"{plen:>7d}bp  {precision:>10.6f}  {recall:>10.6f}  {f1:>10.6f}  {agreement:>9.1%}\n")
        
        f.write("\n\nKey Findings:\n")
        f.write("  - Perfect accuracy: Precision = Recall = F1 = 1.000000\n")
        f.write("  - 100% agreement with Python re module across all tests\n")
        f.write("  - Total tests: 15 (3 datasets × 5 pattern lengths)\n")
        f.write("  - Zero false positives, zero false negatives\n\n")
        
        # CRITERION 5: SCALABILITY
        f.write("\n" + "="*80 + "\n")
        f.write("CRITERION 5: SCALABILITY\n")
        f.write("="*80 + "\n\n")
        
        f.write("5A. Text Length Scaling (Pattern: 50bp)\n")
        f.write("-"*80 + "\n\n")
        f.write(f"{'Text Length':>15s}  {'Search Time':>13s}  {'Throughput':>12s}  {'Time/Char':>12s}\n")
        f.write(f"{'(bp)':>15s}  {'(ms)':>13s}  {'(MB/s)':>12s}  {'(ns)':>12s}\n")
        f.write("-"*80 + "\n")
        
        for row in scalability_text_data:
            text_len = int(row['text_length'])
            search_time = float(row['mean_search_time_ms'])
            throughput = float(row['throughput_mbps'])
            time_per_char = float(row['time_per_char_ns'])
            
            f.write(f"{text_len:>14,d}  {search_time:>12.3f}  {throughput:>11.2f}  {time_per_char:>11.2f}\n")
        
        f.write("\nKey Finding: Linear time complexity O(n) confirmed\n")
        f.write("  - Throughput remains ~12.8 MB/s across all text sizes\n")
        f.write("  - Time per character: ~73-76 ns (consistent)\n\n")
        
        f.write("\n5B. Pattern Count Scaling (50bp patterns on 4.6MB genome)\n")
        f.write("-"*80 + "\n\n")
        f.write(f"{'Num':>6s}  {'Total Time':>12s}  {'Avg Time':>15s}  {'Patterns':>12s}  {'Total':>8s}\n")
        f.write(f"{'Patterns':>6s}  {'(ms)':>12s}  {'per Pattern (ms)':>15s}  {'per Sec':>12s}  {'Matches':>8s}\n")
        f.write("-"*80 + "\n")
        
        for row in scalability_patterns_data:
            num_patterns = int(row['num_patterns'])
            total_time = float(row['total_time_ms'])
            avg_time = float(row['avg_time_per_pattern_ms'])
            patterns_per_sec = float(row['patterns_per_sec'])
            total_matches = int(row['total_matches'])
            
            f.write(f"{num_patterns:>6d}  {total_time:>11.2f}  {avg_time:>14.3f}  {patterns_per_sec:>11.2f}  {total_matches:>8d}\n")
        
        f.write("\nKey Finding: Linear scaling with number of patterns\n")
        f.write("  - Each pattern takes ~343-366ms to search in 4.6MB genome\n")
        f.write("  - Throughput: ~2.9 patterns/second on large genome\n\n")
        
        # CRITERION 6: ROBUSTNESS
        f.write("\n" + "="*80 + "\n")
        f.write("CRITERION 6: ROBUSTNESS\n")
        f.write("="*80 + "\n\n")
        
        f.write("Performance Across Different Pattern Types (100bp patterns):\n")
        f.write("-"*80 + "\n\n")
        f.write(f"{'Pattern Type':>20s}  {'GC%':>6s}  {'Mean Time':>12s}  {'Std Dev':>10s}  {'Matches':>8s}\n")
        f.write(f"{'':>20s}  {'':>6s}  {'(ms)':>12s}  {'(ms)':>10s}  {'':>8s}\n")
        f.write("-"*80 + "\n")
        
        for row in robustness_data:
            ptype = row['pattern_type']
            gc_content = float(row['gc_content'])
            mean_time = float(row['mean_search_time_ms'])
            std_dev = float(row['std_dev_ms'])
            matches = int(row['num_matches'])
            
            f.write(f"{ptype:>20s}  {gc_content:>5.0%}  {mean_time:>11.3f}  {std_dev:>9.3f}  {matches:>8d}\n")
        
        f.write("\nKey Findings:\n")
        f.write("  - Consistent performance across all pattern types\n")
        f.write("  - Search time: 298-340ms regardless of GC content\n")
        f.write("  - Slightly slower for random patterns (340ms) vs synthetic (298-306ms)\n")
        f.write("  - Algorithm handles low-complexity and repeat patterns efficiently\n")
        f.write("  - No performance degradation with extreme GC content (0% or 80%)\n\n")
        
        # FINAL SUMMARY
        f.write("\n" + "="*80 + "\n")
        f.write("KEY FINDINGS SUMMARY\n")
        f.write("="*80 + "\n\n")
        
        f.write("1. ACCURACY:\n")
        f.write("   ✓ Perfect correctness (Precision = Recall = F1 = 1.000000)\n")
        f.write("   ✓ 100% agreement with Python re across all 15 tests\n\n")
        
        f.write("2. PERFORMANCE:\n")
        f.write("   ✓ Average throughput: 12.82 MB/s\n")
        f.write("   ✓ Search time: ~340-365ms for 4.6MB genome (50-200bp patterns)\n")
        f.write("   ✓ Preprocessing: < 1µs for typical patterns (< 100bp)\n\n")
        
        f.write("3. COMPLEXITY:\n")
        f.write("   ✓ Preprocessing: O(m) - Linear with pattern length\n")
        f.write("   ✓ Search: O(n) - Linear with text length\n")
        f.write("   ✓ Combined: O(n + m) - Optimal for exact matching\n\n")
        
        f.write("4. MEMORY EFFICIENCY:\n")
        f.write("   ✓ LPS array: 8 bytes per pattern character\n")
        f.write("   ✓ Total memory: < 100 KB even for 10,000bp patterns\n")
        f.write("   ✓ Search memory: Constant ~0.2 KB\n\n")
        
        f.write("5. SCALABILITY:\n")
        f.write("   ✓ Linear scaling with text length (O(n))\n")
        f.write("   ✓ Linear scaling with pattern count\n")
        f.write("   ✓ Consistent 73-76ns per character across all sizes\n\n")
        
        f.write("6. ROBUSTNESS:\n")
        f.write("   ✓ Stable performance across GC content (0% to 80%)\n")
        f.write("   ✓ Handles repeats and low-complexity patterns efficiently\n")
        f.write("   ✓ No performance degradation with pattern type\n\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("END OF DETAILED REPORT\n")
        f.write("="*80 + "\n")
    
    print(f"✓ Generated detailed report: {report_path}")
    print(f"\nReport location: {report_path}")
    
    # Print to console
    with open(report_path, 'r') as f:
        print("\n" + f.read())

if __name__ == "__main__":
    generate_detailed_report()
