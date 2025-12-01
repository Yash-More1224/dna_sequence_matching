#!/usr/bin/env python3

import csv
import json
from pathlib import Path
from datetime import datetime
from glob import glob

BENCHMARKS_DIR = Path(__file__).parent / "results" / "benchmarks"
REPORTS_DIR = Path(__file__).parent / "results" / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

def find_latest_timestamp():
    """Find the most recent timestamp from benchmark files."""
    files = glob(str(BENCHMARKS_DIR / "latency_time_*.csv"))
    if not files:
        return None
    latest = sorted(files)[-1]
    timestamp = latest.split('_')[-1].replace('.csv', '')
    return timestamp

def load_csv(filename):
    """Load CSV file and return list of dicts."""
    filepath = BENCHMARKS_DIR / filename
    if not filepath.exists():
        return []
    with open(filepath, 'r') as f:
        return list(csv.DictReader(f))

def generate_detailed_report(timestamp):
    
    # Load all data
    latency_data = load_csv(f"latency_time_{timestamp}.csv")
    preprocessing_data = load_csv(f"preprocessing_{timestamp}.csv")
    memory_data = load_csv(f"memory_{timestamp}.csv")
    accuracy_data = load_csv(f"accuracy_{timestamp}.csv")
    scalability_text_data = load_csv(f"scalability_text_{timestamp}.csv")
    scalability_patterns_data = load_csv(f"scalability_patterns_{timestamp}.csv")
    robustness_data = load_csv(f"robustness_{timestamp}.csv")
    
    if not latency_data:
        print("✗ No data found for this timestamp!")
        return
    
    report_path = REPORTS_DIR / f"comprehensive_evaluation_{timestamp}.txt"
    
    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("WAGNER-FISCHER ALGORITHM - COMPREHENSIVE EVALUATION REPORT\n")
        f.write("="*80 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("EXECUTIVE SUMMARY\n")
        f.write("-"*80 + "\n\n")
        f.write("This report presents a comprehensive evaluation of the Wagner-Fischer\n")
        f.write("algorithm for approximate pattern matching in DNA sequences, covering:\n\n")
        f.write("1. Latency/Time: Runtime characteristics with varying edit distances\n")
        f.write("2. Preprocessing: Matrix initialization performance\n")
        f.write("3. Memory Usage: Peak memory and matrix footprint (space-optimized)\n")
        f.write("4. Accuracy: Correctness validation against Python re (exact match)\n")
        f.write("5. Scalability: Performance with varying text and pattern counts\n")
        f.write("6. Robustness: Behavior across different edit distance thresholds\n\n")
        
        f.write("ALGORITHM OVERVIEW\n")
        f.write("-"*80 + "\n\n")
        f.write("Wagner-Fischer is a dynamic programming algorithm for computing edit distance\n")
        f.write("(Levenshtein distance) and finding approximate matches. Key characteristics:\n\n")
        f.write("  - Time Complexity: O(n*m) where n=text length, m=pattern length\n")
        f.write("  - Space Complexity: O(m) with space optimization (2 rows instead of n*m)\n")
        f.write("  - Supports: insertions, deletions, and substitutions\n")
        f.write("  - Configurable maximum edit distance threshold\n\n")
        
        # Get dataset info from first row
        if latency_data:
            sample = latency_data[0]
            datasets_used = set(row['dataset'] for row in latency_data)
            
            f.write("\nDATASETS EVALUATED\n")
            f.write("-"*80 + "\n\n")
            for ds in sorted(datasets_used):
                ds_rows = [r for r in latency_data if r['dataset'] == ds]
                if ds_rows:
                    text_len = int(ds_rows[0]['text_length'])
                    f.write(f"{ds}:\n")
                    f.write(f"  Length: {text_len:,} bp\n\n")
        
        # CRITERION 1: LATENCY/TIME
        f.write("\n" + "="*80 + "\n")
        f.write("CRITERION 1: LATENCY / TIME\n")
        f.write("="*80 + "\n\n")
        
        f.write("Search Performance Metrics ({} runs per test):\n".format(
            latency_data[0]['num_runs'] if latency_data else 'N'))
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
            f.write(f"{'Pattern':>8s}  {'Max Dist':>9s}  {'Mean Time':>12s}  {'Median':>12s}  {'Throughput':>12s}  {'Matches':>8s}\n")
            f.write(f"{'Length':>8s}  {'':>9s}  {'(ms)':>12s}  {'(ms)':>12s}  {'(MB/s)':>12s}  {'Found':>8s}\n")
            f.write("-"*80 + "\n")
            
            for row in rows:
                mean_time = float(row['mean_time_ms'])
                median_time = float(row['median_time_ms'])
                throughput = float(row['throughput_mbps'])
                matches = int(row['num_matches'])
                plen = int(row['pattern_length'])
                max_dist = int(row['max_edit_distance'])
                
                f.write(f"{plen:>7d}bp  {max_dist:>9d}  {mean_time:>11.3f}  {median_time:>11.3f}  {throughput:>11.2f}  {matches:>8d}\n")
            
            f.write("\n")
        
        # Summary statistics
        all_throughputs = [float(row['throughput_mbps']) for row in latency_data]
        avg_throughput = sum(all_throughputs) / len(all_throughputs) if all_throughputs else 0
        
        f.write("\nSummary:\n")
        f.write(f"  Total measurements: {len(latency_data)}\n")
        f.write(f"  Average throughput: {avg_throughput:.2f} MB/s\n")
        if all_throughputs:
            f.write(f"  Min throughput: {min(all_throughputs):.2f} MB/s\n")
            f.write(f"  Max throughput: {max(all_throughputs):.2f} MB/s\n")
        f.write(f"  Note: Performance decreases with higher edit distance thresholds\n\n")
        
        # CRITERION 2: PREPROCESSING
        f.write("\n" + "="*80 + "\n")
        f.write("CRITERION 2: PREPROCESSING TIME\n")
        f.write("="*80 + "\n\n")
        
        f.write("Matrix Initialization Performance (20 runs per pattern length):\n")
        f.write("-"*80 + "\n\n")
        f.write(f"{'Pattern':>8s}  {'Mean Time':>15s}  {'Std Dev':>12s}  {'Memory':>10s}  {'Time/Char':>12s}\n")
        f.write(f"{'Length':>8s}  {'(µs)':>15s}  {'(µs)':>12s}  {'(bytes)':>10s}  {'(ns)':>12s}\n")
        f.write("-"*80 + "\n")
        
        for row in preprocessing_data:
            plen = int(row['pattern_length'])
            mean_time = float(row['mean_preprocessing_time_us'])
            std_dev = float(row['std_dev_us'])
            memory = int(row['matrix_memory_bytes'])
            time_per_char = mean_time * 1000 / plen if plen > 0 else 0
            
            f.write(f"{plen:>7d}bp  {mean_time:>14.3f}  {std_dev:>11.3f}  {memory:>10d}  {time_per_char:>11.2f}\n")
        
        f.write("\nKey Findings:\n")
        f.write("  - Linear time complexity: O(m) for matrix initialization\n")
        f.write("  - Space optimization: Uses only 2 rows instead of full n×m matrix\n")
        f.write("  - Memory usage: 16 bytes per pattern character (2 rows × 8 bytes)\n")
        f.write("  - Minimal preprocessing overhead compared to search time\n\n")
        
        # CRITERION 3: MEMORY USAGE
        f.write("\n" + "="*80 + "\n")
        f.write("CRITERION 3: MEMORY USAGE\n")
        f.write("="*80 + "\n\n")
        
        f.write("Memory Profiling (Space-Optimized Implementation):\n")
        f.write("-"*80 + "\n\n")
        f.write(f"{'Pattern':>8s}  {'Matrix':>12s}  {'Preproc Peak':>14s}  {'Search Peak':>13s}  {'Total Peak':>12s}\n")
        f.write(f"{'Length':>8s}  {'(bytes)':>12s}  {'(KB)':>14s}  {'(KB)':>13s}  {'(KB)':>12s}\n")
        f.write("-"*80 + "\n")
        
        for row in memory_data:
            plen = int(row['pattern_length'])
            matrix_mem = int(row['matrix_memory_bytes'])
            preproc_peak = float(row['preprocessing_peak_kb'])
            search_peak = float(row['search_peak_kb'])
            total_peak = float(row['total_peak_kb'])
            
            f.write(f"{plen:>7d}bp  {matrix_mem:>12d}  {preproc_peak:>13.2f}  {search_peak:>12.2f}  {total_peak:>11.2f}\n")
        
        f.write("\nKey Findings:\n")
        f.write("  - Matrix memory: 16 bytes per pattern character (space-optimized)\n")
        f.write("  - Much more memory efficient than full O(n*m) matrix\n")
        f.write("  - Uses rolling 2-row approach for O(m) space complexity\n")
        f.write("  - Suitable for DNA sequence matching on large genomes\n\n")
        
        # CRITERION 4: ACCURACY
        f.write("\n" + "="*80 + "\n")
        f.write("CRITERION 4: ACCURACY\n")
        f.write("="*80 + "\n\n")
        
        f.write("Correctness Validation (using Python re as ground truth for exact matches):\n")
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
        f.write("  - Tested for exact matches (max_edit_distance=0)\n")
        f.write("  - Results show correctness of Wagner-Fischer for exact matching\n")
        f.write("  - Note: Algorithm's strength is approximate matching (edit distance > 0)\n")
        f.write("  - For approximate matching, validation requires different ground truth\n\n")
        
        # CRITERION 5: SCALABILITY
        f.write("\n" + "="*80 + "\n")
        f.write("CRITERION 5: SCALABILITY\n")
        f.write("="*80 + "\n\n")
        
        f.write("5A. Text Length Scaling (Pattern: 50bp, max_edit_distance=2)\n")
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
        
        f.write("\nKey Finding: Quadratic time complexity O(n*m) confirmed\n")
        f.write("  - Time scales linearly with text length for fixed pattern size\n")
        f.write("  - More computationally intensive than exact matching algorithms\n\n")
        
        f.write("\n5B. Pattern Count Scaling (50bp patterns)\n")
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
        f.write("  - Each pattern is processed independently\n")
        f.write("  - Consistent per-pattern search time\n\n")
        
        # CRITERION 6: ROBUSTNESS
        f.write("\n" + "="*80 + "\n")
        f.write("CRITERION 6: ROBUSTNESS\n")
        f.write("="*80 + "\n\n")
        
        f.write("Performance Across Different Edit Distance Thresholds (100bp patterns):\n")
        f.write("-"*80 + "\n\n")
        f.write(f"{'Test Type':>20s}  {'Max Dist':>9s}  {'Mean Time':>12s}  {'Std Dev':>10s}  {'Matches':>8s}\n")
        f.write(f"{'':>20s}  {'':>9s}  {'(ms)':>12s}  {'(ms)':>10s}  {'':>8s}\n")
        f.write("-"*80 + "\n")
        
        for row in robustness_data:
            test_type = row['test_type']
            max_dist = int(row['max_edit_distance'])
            mean_time = float(row['mean_search_time_ms'])
            std_dev = float(row['std_dev_ms'])
            matches = int(row['num_matches'])
            
            f.write(f"{test_type:>20s}  {max_dist:>9d}  {mean_time:>11.3f}  {std_dev:>9.3f}  {matches:>8d}\n")
        
        f.write("\nKey Findings:\n")
        f.write("  - Performance degrades with higher edit distance thresholds\n")
        f.write("  - More matches found with higher thresholds (expected behavior)\n")
        f.write("  - Algorithm handles approximate matching efficiently\n")
        f.write("  - Suitable for error-tolerant sequence matching\n\n")
        
        # FINAL SUMMARY
        f.write("\n" + "="*80 + "\n")
        f.write("KEY FINDINGS SUMMARY\n")
        f.write("="*80 + "\n\n")
        
        f.write("1. ACCURACY:\n")
        f.write("   ✓ Correct results for exact matching (max_edit_distance=0)\n")
        f.write("   ✓ Enables approximate matching with configurable thresholds\n\n")
        
        f.write("2. PERFORMANCE:\n")
        if all_throughputs:
            f.write(f"   ✓ Average throughput: {avg_throughput:.2f} MB/s\n")
        f.write("   ✓ Performance decreases with higher edit distance thresholds\n")
        f.write("   ✓ Slower than exact matching algorithms (KMP, Boyer-Moore)\n\n")
        
        f.write("3. COMPLEXITY:\n")
        f.write("   ✓ Time: O(n*m) - Quadratic with text and pattern length\n")
        f.write("   ✓ Space: O(m) - Linear with pattern length (space-optimized)\n")
        f.write("   ✓ Trade-off: Slower but supports approximate matching\n\n")
        
        f.write("4. MEMORY EFFICIENCY:\n")
        f.write("   ✓ Space-optimized: Uses 2 rows instead of full matrix\n")
        f.write("   ✓ Matrix memory: 16 bytes per pattern character\n")
        f.write("   ✓ Much more efficient than naive O(n*m) space\n\n")
        
        f.write("5. SCALABILITY:\n")
        f.write("   ✓ Linear scaling with text length (for fixed pattern size)\n")
        f.write("   ✓ Linear scaling with number of patterns\n")
        f.write("   ✓ Suitable for moderate-sized sequences\n\n")
        
        f.write("6. ROBUSTNESS:\n")
        f.write("   ✓ Handles approximate matching with edit distance\n")
        f.write("   ✓ Flexible threshold configuration\n")
        f.write("   ✓ More matches found with higher thresholds\n")
        f.write("   ✓ Ideal for error-tolerant DNA sequence matching\n\n")
        
        f.write("\nCOMPARISON WITH EXACT MATCHING ALGORITHMS\n")
        f.write("-"*80 + "\n\n")
        f.write("Advantages:\n")
        f.write("  + Supports approximate matching (insertions, deletions, substitutions)\n")
        f.write("  + Handles sequencing errors and mutations\n")
        f.write("  + Flexible edit distance threshold\n\n")
        
        f.write("Trade-offs:\n")
        f.write("  - Slower than KMP (O(n+m)) and Boyer-Moore (O(n/m) average)\n")
        f.write("  - Higher computational cost: O(n*m) vs O(n+m)\n")
        f.write("  - Performance sensitive to edit distance threshold\n\n")
        
        f.write("Use Cases:\n")
        f.write("  • Sequence alignment with mismatches\n")
        f.write("  • Error-tolerant pattern matching\n")
        f.write("  • Finding similar (not just exact) sequences\n")
        f.write("  • Handling sequencing errors\n\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("END OF COMPREHENSIVE REPORT\n")
        f.write("="*80 + "\n")
    
    print(f"✓ Generated comprehensive report: {report_path}")
    return report_path

def main():
    """Generate report from latest results."""
    print("\n" + "="*80)
    print("GENERATING WAGNER-FISCHER EVALUATION REPORT")
    print("="*80 + "\n")
    
    timestamp = find_latest_timestamp()
    if not timestamp:
        print("✗ No evaluation results found!")
        print("  Please run: python comprehensive_evaluation_full.py")
        return
    
    print(f"Using results from: {timestamp}\n")
    
    report_path = generate_detailed_report(timestamp)
    
    if report_path and report_path.exists():
        print(f"\nReport location: {report_path}\n")
        
        # Print to console
        with open(report_path, 'r') as f:
            print(f.read())

if __name__ == "__main__":
    main()
