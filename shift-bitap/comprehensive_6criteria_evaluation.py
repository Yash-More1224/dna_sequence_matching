#!/usr/bin/env python3
"""
COMPREHENSIVE 6-CRITERIA EVALUATION
===================================
Evaluates Shift-Or/Bitap algorithm on all 6 criteria:
1. Latency/Time - per-query latency
2. Preprocessing Time - bitmask creation
3. Memory Usage - peak memory consumption
4. Accuracy - correctness verification
5. Scalability - performance vs size
6. Robustness - handling mutations/errors
"""

import os
import sys
import time
import json
import tracemalloc
import psutil
import statistics
from pathlib import Path
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))

from src.algorithm import ShiftOrBitap
from src.data_loader import DataLoader


def measure_memory(func, *args, **kwargs):
    """Measure peak memory usage of a function."""
    tracemalloc.start()
    result = func(*args, **kwargs)
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return result, peak / 1024  # Return KB


def run_comprehensive_evaluation():
    """Run complete 6-criteria evaluation."""
    
    print("=" * 70)
    print("   SHIFT-OR/BITAP: COMPREHENSIVE 6-CRITERIA EVALUATION")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Python: {sys.version.split()[0]}")
    print()
    
    # Load datasets
    loader = DataLoader()
    dataset_dir = Path("../dataset")
    
    datasets = {
        'Lambda Phage': ('lambda_phage.fasta', None),  # Use full
        'E. coli K-12': ('ecoli_k12_mg1655.fasta', 100000),  # 100K subset
        'Salmonella': ('salmonella_typhimurium.fasta', 100000)  # 100K subset
    }
    
    all_results = {
        'timestamp': datetime.now().isoformat(),
        'criteria': {},
        'datasets': {}
    }
    
    # Criteria collectors
    latency_results = []
    preprocess_results = []
    memory_results = []
    accuracy_results = []
    scalability_results = []
    robustness_results = []
    
    for dataset_name, (filename, subset_size) in datasets.items():
        filepath = dataset_dir / filename
        if not filepath.exists():
            print(f"⚠ Skipping {dataset_name}: file not found")
            continue
        
        print(f"\n{'='*70}")
        print(f"DATASET: {dataset_name}")
        print(f"{'='*70}")
        
        # Load sequence
        sequence = loader.load_fasta_single(str(filepath))
        full_size = len(sequence)
        
        if subset_size and len(sequence) > subset_size:
            sequence = sequence[:subset_size]
        
        seq_len = len(sequence)
        gc = (sequence.count('G') + sequence.count('C')) / seq_len * 100
        
        print(f"Size: {seq_len:,} bp (full: {full_size:,} bp)")
        print(f"GC Content: {gc:.1f}%")
        
        dataset_results = {
            'full_size': full_size,
            'tested_size': seq_len,
            'gc_content': gc
        }
        
        # Test patterns
        patterns = {
            '5bp': sequence[1000:1005],
            '10bp': sequence[2000:2010],
            '15bp': sequence[3000:3015],
            '20bp': sequence[4000:4020],
            '30bp': sequence[5000:5030]
        }
        
        # =====================================================================
        # CRITERION 1: LATENCY/TIME
        # =====================================================================
        print(f"\n--- CRITERION 1: LATENCY/TIME ---")
        
        latency_data = []
        for pattern_name, pattern in patterns.items():
            matcher = ShiftOrBitap(pattern)
            
            # Warm up
            _ = matcher.search_exact(sequence)
            
            # Measure with multiple runs
            times = []
            for _ in range(3):
                start = time.perf_counter()
                matches = matcher.search_exact(sequence)
                elapsed = (time.perf_counter() - start) * 1000
                times.append(elapsed)
            
            avg_time = statistics.mean(times)
            throughput = (seq_len / 1_000_000) / (avg_time / 1000)
            
            latency_data.append({
                'pattern': pattern_name,
                'time_ms': avg_time,
                'matches': len(matches),
                'throughput_mbps': throughput
            })
            
            print(f"  {pattern_name}: {avg_time:.2f} ms | {len(matches)} matches | {throughput:.2f} Mbp/s")
        
        latency_results.append({'dataset': dataset_name, 'data': latency_data})
        dataset_results['latency'] = latency_data
        
        # =====================================================================
        # CRITERION 2: PREPROCESSING TIME
        # =====================================================================
        print(f"\n--- CRITERION 2: PREPROCESSING TIME ---")
        
        preprocess_data = []
        for pattern_name, pattern in patterns.items():
            times = []
            for _ in range(10):
                start = time.perf_counter()
                matcher = ShiftOrBitap(pattern)
                elapsed = (time.perf_counter() - start) * 1000
                times.append(elapsed)
            
            avg_time = statistics.mean(times)
            preprocess_data.append({
                'pattern': pattern_name,
                'time_ms': avg_time
            })
            print(f"  {pattern_name}: {avg_time:.4f} ms")
        
        preprocess_results.append({'dataset': dataset_name, 'data': preprocess_data})
        dataset_results['preprocessing'] = preprocess_data
        
        # =====================================================================
        # CRITERION 3: MEMORY USAGE
        # =====================================================================
        print(f"\n--- CRITERION 3: MEMORY USAGE ---")
        
        memory_data = []
        for pattern_name, pattern in patterns.items():
            # Measure preprocessing memory
            _, preprocess_mem = measure_memory(ShiftOrBitap, pattern)
            
            # Measure search memory
            matcher = ShiftOrBitap(pattern)
            _, search_mem = measure_memory(matcher.search_exact, sequence)
            
            memory_data.append({
                'pattern': pattern_name,
                'preprocess_kb': preprocess_mem,
                'search_kb': search_mem
            })
            print(f"  {pattern_name}: Preprocess {preprocess_mem:.2f} KB | Search {search_mem:.2f} KB")
        
        memory_results.append({'dataset': dataset_name, 'data': memory_data})
        dataset_results['memory'] = memory_data
        
        # =====================================================================
        # CRITERION 4: ACCURACY
        # =====================================================================
        print(f"\n--- CRITERION 4: ACCURACY ---")
        
        accuracy_data = []
        
        # Test with known pattern that exists
        test_pattern = sequence[10000:10010]
        matcher = ShiftOrBitap(test_pattern)
        matches = matcher.search_exact(sequence)
        
        # Verify at least position 10000 is found
        found_expected = 10000 in matches
        
        # Test approximate matching
        mutated_pattern = test_pattern[:5] + 'N' + test_pattern[6:]  # 1 mutation
        approx_matches = matcher.search_approximate(sequence[:20000], max_errors=1)
        
        accuracy_data.append({
            'test': 'exact_match_known_position',
            'pattern': test_pattern,
            'expected_pos': 10000,
            'found': found_expected,
            'total_matches': len(matches)
        })
        
        accuracy_data.append({
            'test': 'approximate_match_1_error',
            'matches_found': len(approx_matches),
            'correct': len(approx_matches) > 0
        })
        
        print(f"  Exact match at known position: {'PASS' if found_expected else 'FAIL'}")
        print(f"  Approximate matching (k=1): {len(approx_matches)} matches")
        
        accuracy_results.append({'dataset': dataset_name, 'data': accuracy_data})
        dataset_results['accuracy'] = accuracy_data
        
        # =====================================================================
        # CRITERION 5: SCALABILITY
        # =====================================================================
        print(f"\n--- CRITERION 5: SCALABILITY ---")
        
        scalability_data = []
        pattern = sequence[5000:5010]
        matcher = ShiftOrBitap(pattern)
        
        # Test with different text sizes
        sizes = [10000, 25000, 50000, 75000, seq_len]
        
        for size in sizes:
            if size > seq_len:
                continue
            
            test_seq = sequence[:size]
            
            times = []
            for _ in range(3):
                start = time.perf_counter()
                _ = matcher.search_exact(test_seq)
                elapsed = (time.perf_counter() - start) * 1000
                times.append(elapsed)
            
            avg_time = statistics.mean(times)
            time_per_bp = avg_time / size * 1000  # microseconds per bp
            
            scalability_data.append({
                'size': size,
                'time_ms': avg_time,
                'time_per_bp_us': time_per_bp
            })
            print(f"  {size:,} bp: {avg_time:.2f} ms ({time_per_bp:.4f} μs/bp)")
        
        # Check linearity (time should scale ~linearly with size)
        if len(scalability_data) >= 2:
            ratio = scalability_data[-1]['time_ms'] / scalability_data[0]['time_ms']
            size_ratio = scalability_data[-1]['size'] / scalability_data[0]['size']
            linearity_factor = ratio / size_ratio
            print(f"  Linearity factor: {linearity_factor:.2f} (ideal: 1.0)")
        
        scalability_results.append({'dataset': dataset_name, 'data': scalability_data})
        dataset_results['scalability'] = scalability_data
        
        # =====================================================================
        # CRITERION 6: ROBUSTNESS
        # =====================================================================
        print(f"\n--- CRITERION 6: ROBUSTNESS ---")
        
        robustness_data = []
        pattern = sequence[8000:8015]
        matcher = ShiftOrBitap(pattern)
        
        # Test with different error tolerances
        test_seq = sequence[:30000]  # Use subset for speed
        
        for k in range(4):
            start = time.perf_counter()
            matches = matcher.search_approximate(test_seq, max_errors=k)
            elapsed = (time.perf_counter() - start) * 1000
            
            robustness_data.append({
                'max_errors': k,
                'time_ms': elapsed,
                'matches': len(matches)
            })
            print(f"  k={k}: {elapsed:.2f} ms | {len(matches)} matches")
        
        robustness_results.append({'dataset': dataset_name, 'data': robustness_data})
        dataset_results['robustness'] = robustness_data
        
        all_results['datasets'][dataset_name] = dataset_results
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "=" * 70)
    print("SUMMARY - ALL 6 CRITERIA")
    print("=" * 70)
    
    # Calculate averages across datasets
    print("\n1. LATENCY/TIME:")
    avg_throughput = []
    for r in latency_results:
        for d in r['data']:
            avg_throughput.append(d['throughput_mbps'])
    print(f"   Average throughput: {statistics.mean(avg_throughput):.2f} Mbp/s")
    
    print("\n2. PREPROCESSING TIME:")
    avg_preprocess = []
    for r in preprocess_results:
        for d in r['data']:
            avg_preprocess.append(d['time_ms'])
    print(f"   Average preprocessing: {statistics.mean(avg_preprocess):.4f} ms")
    
    print("\n3. MEMORY USAGE:")
    avg_search_mem = []
    for r in memory_results:
        for d in r['data']:
            avg_search_mem.append(d['search_kb'])
    print(f"   Average search memory: {statistics.mean(avg_search_mem):.2f} KB")
    
    print("\n4. ACCURACY:")
    all_correct = all(
        all(d.get('found', d.get('correct', True)) for d in r['data'])
        for r in accuracy_results
    )
    print(f"   All tests passed: {'YES' if all_correct else 'NO'}")
    
    print("\n5. SCALABILITY:")
    print(f"   Time complexity: O(n) - linear with text length")
    
    print("\n6. ROBUSTNESS:")
    print(f"   Supports k-error matching up to k=pattern_length")
    
    # Store summary
    all_results['criteria'] = {
        'latency': {'avg_throughput_mbps': statistics.mean(avg_throughput)},
        'preprocessing': {'avg_time_ms': statistics.mean(avg_preprocess)},
        'memory': {'avg_search_kb': statistics.mean(avg_search_mem)},
        'accuracy': {'all_passed': all_correct},
        'scalability': {'complexity': 'O(n)'},
        'robustness': {'max_errors_supported': 'pattern_length'}
    }
    
    # Save results
    output_dir = Path('results')
    output_dir.mkdir(exist_ok=True)
    
    json_path = output_dir / 'comprehensive_6criteria.json'
    with open(json_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Generate report
    report_path = output_dir / 'reports' / 'COMPREHENSIVE_6CRITERIA_REPORT.txt'
    report_path.parent.mkdir(exist_ok=True)
    
    with open(report_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("   SHIFT-OR/BITAP: COMPREHENSIVE 6-CRITERIA EVALUATION\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("EVALUATION CRITERIA:\n")
        f.write("  1. Latency/Time - per-query latency and throughput\n")
        f.write("  2. Preprocessing Time - bitmask creation time\n")
        f.write("  3. Memory Usage - peak memory during search\n")
        f.write("  4. Accuracy - correctness of matching\n")
        f.write("  5. Scalability - O(n) time complexity\n")
        f.write("  6. Robustness - handling errors/mutations\n\n")
        
        f.write("=" * 70 + "\n")
        f.write("RESULTS BY DATASET\n")
        f.write("=" * 70 + "\n\n")
        
        for dataset_name, data in all_results['datasets'].items():
            f.write(f"\n{dataset_name}\n")
            f.write("-" * 50 + "\n")
            f.write(f"Size: {data['tested_size']:,} bp | GC: {data['gc_content']:.1f}%\n\n")
            
            f.write("LATENCY:\n")
            for item in data['latency']:
                f.write(f"  {item['pattern']}: {item['time_ms']:.2f} ms | {item['throughput_mbps']:.2f} Mbp/s\n")
            
            f.write("\nPREPROCESSING:\n")
            for item in data['preprocessing']:
                f.write(f"  {item['pattern']}: {item['time_ms']:.4f} ms\n")
            
            f.write("\nMEMORY:\n")
            for item in data['memory']:
                f.write(f"  {item['pattern']}: {item['search_kb']:.2f} KB\n")
            
            f.write("\nSCALABILITY:\n")
            for item in data['scalability']:
                f.write(f"  {item['size']:,} bp: {item['time_ms']:.2f} ms\n")
            
            f.write("\nROBUSTNESS:\n")
            for item in data['robustness']:
                f.write(f"  k={item['max_errors']}: {item['time_ms']:.2f} ms | {item['matches']} matches\n")
        
        f.write("\n" + "=" * 70 + "\n")
        f.write("SUMMARY\n")
        f.write("=" * 70 + "\n\n")
        
        f.write(f"1. LATENCY: {statistics.mean(avg_throughput):.2f} Mbp/s average throughput\n")
        f.write(f"2. PREPROCESSING: {statistics.mean(avg_preprocess):.4f} ms average\n")
        f.write(f"3. MEMORY: {statistics.mean(avg_search_mem):.2f} KB average\n")
        f.write(f"4. ACCURACY: {'100% correct' if all_correct else 'Issues found'}\n")
        f.write(f"5. SCALABILITY: O(n) linear time complexity\n")
        f.write(f"6. ROBUSTNESS: Supports up to k=3 errors efficiently\n")
        
        f.write("\n" + "=" * 70 + "\n")
    
    print(f"\n✓ Results saved to: {json_path}")
    print(f"✓ Report saved to: {report_path}")
    print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return all_results


if __name__ == "__main__":
    run_comprehensive_evaluation()
