"""
Experimental Workflows

Implements all planned experiments for Boyer-Moore analysis:
1. Latency vs Pattern Length
2. Scalability (varying text sizes)
3. Alphabet Size Effect
4. Heuristic Contribution
5. Preprocessing Overhead
6. Memory Footprint
7. Real Motif Search
8. Comparison with Python re
"""

import time
import json
from pathlib import Path
from typing import Dict, List, Tuple
import sys
sys.path.append('..')

from src.boyer_moore import BoyerMoore
from src.boyer_moore_variants import get_variant
from src.data_loader import DatasetManager
from src.data_generator import DNAGenerator
from experiments.benchmarks import Benchmarker, AggregatedResult


class ExperimentRunner:
    """Run comprehensive experiments on Boyer-Moore algorithm."""
    
    def __init__(self, output_dir: str = "results"):
        """
        Initialize experiment runner.
        
        Args:
            output_dir: Directory to save results
        """
        self.output_dir = Path(output_dir)
        self.tables_dir = self.output_dir / "tables"
        self.tables_dir.mkdir(parents=True, exist_ok=True)
        
        self.data_manager = DatasetManager()
        self.dna_generator = DNAGenerator(seed=42)
        self.benchmarker = Benchmarker(warmup_runs=3, min_runs=5)
        
        self.results = {}
    
    def experiment_1_pattern_length(self, 
                                   pattern_lengths: List[int] = None) -> Dict:
        """
        Experiment 1: Latency vs Pattern Length
        
        Test how search time varies with pattern length on E. coli genome.
        
        Args:
            pattern_lengths: List of pattern lengths to test
            
        Returns:
            Dictionary with results
        """
        print("\n" + "=" * 60)
        print("EXPERIMENT 1: Latency vs Pattern Length")
        print("=" * 60)
        
        if pattern_lengths is None:
            pattern_lengths = [4, 8, 16, 32, 64, 128, 256, 512]
        
        # Load E. coli genome
        print("Loading E. coli genome...")
        text = self.data_manager.load_ecoli_genome()[:1000000]  # First 1M bp
        print(f"Using {len(text):,} bp of E. coli genome")
        
        results = []
        
        for plen in pattern_lengths:
            print(f"\nTesting pattern length: {plen}")
            
            # Generate random pattern from text
            start_pos = len(text) // 2
            pattern = text[start_pos:start_pos + plen]
            
            # Benchmark full Boyer-Moore
            matcher = BoyerMoore(pattern)
            bench_results = self.benchmarker.run_multiple(
                lambda: self.benchmarker.benchmark_boyer_moore(matcher, text)
            )
            agg = self.benchmarker.aggregate_results(bench_results)
            
            result_dict = {
                'pattern_length': plen,
                'text_length': len(text),
                'mean_total_time': agg.mean_total_time,
                'std_total_time': agg.std_total_time,
                'mean_search_time': agg.mean_search_time,
                'mean_preprocessing_time': agg.mean_preprocessing_time,
                'mean_comparisons': agg.mean_comparisons,
                'mean_shifts': agg.mean_shifts,
                'matches_found': agg.mean_matches
            }
            results.append(result_dict)
            
            print(f"  Mean time: {agg.mean_total_time*1000:.3f} ms")
            print(f"  Comparisons: {agg.mean_comparisons:.0f}")
            print(f"  Matches: {agg.mean_matches:.0f}")
        
        # Save results
        output_file = self.tables_dir / "exp1_pattern_length.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n✓ Results saved to {output_file}")
        
        self.results['experiment_1'] = results
        return results
    
    def experiment_2_text_scaling(self, 
                                 text_sizes: List[int] = None) -> Dict:
        """
        Experiment 2: Scalability (varying text sizes)
        
        Test how performance scales with text size.
        
        Args:
            text_sizes: List of text sizes to test
            
        Returns:
            Dictionary with results
        """
        print("\n" + "=" * 60)
        print("EXPERIMENT 2: Text Size Scaling")
        print("=" * 60)
        
        if text_sizes is None:
            text_sizes = [10000, 50000, 100000, 500000, 1000000]
        
        pattern_length = 20
        pattern = self.dna_generator.generate_pattern(pattern_length)
        
        print(f"Pattern length: {pattern_length}")
        print(f"Pattern: {pattern}")
        
        results = []
        
        # Load full E. coli genome once
        print("\nLoading E. coli genome...")
        full_text = self.data_manager.load_ecoli_genome()
        
        for text_size in text_sizes:
            print(f"\nTesting text size: {text_size:,} bp")
            
            # Use subsequence of appropriate size
            text = full_text[:text_size]
            
            # Benchmark
            matcher = BoyerMoore(pattern)
            bench_results = self.benchmarker.run_multiple(
                lambda: self.benchmarker.benchmark_boyer_moore(matcher, text)
            )
            agg = self.benchmarker.aggregate_results(bench_results)
            
            # Calculate throughput
            throughput_mbps = (text_size / agg.mean_total_time) / 1_000_000
            
            result_dict = {
                'text_size': text_size,
                'pattern_length': pattern_length,
                'mean_total_time': agg.mean_total_time,
                'std_total_time': agg.std_total_time,
                'throughput_mbps': throughput_mbps,
                'mean_comparisons': agg.mean_comparisons,
                'mean_shifts': agg.mean_shifts,
                'matches_found': agg.mean_matches
            }
            results.append(result_dict)
            
            print(f"  Mean time: {agg.mean_total_time*1000:.3f} ms")
            print(f"  Throughput: {throughput_mbps:.2f} MB/s")
            print(f"  Matches: {agg.mean_matches:.0f}")
        
        # Save results
        output_file = self.tables_dir / "exp2_text_scaling.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n✓ Results saved to {output_file}")
        
        self.results['experiment_2'] = results
        return results
    
    def experiment_3_alphabet_effect(self) -> Dict:
        """
        Experiment 3: Alphabet Size Effect
        
        Compare performance on DNA (4 letters) vs random text (larger alphabet).
        
        Returns:
            Dictionary with results
        """
        print("\n" + "=" * 60)
        print("EXPERIMENT 3: Alphabet Size Effect")
        print("=" * 60)
        
        text_length = 500000
        pattern_length = 20
        
        results = []
        
        # Test 1: DNA sequence (4-letter alphabet)
        print("\n1. Testing DNA sequence (A,C,G,T)...")
        dna_text = self.dna_generator.generate_random_sequence(text_length)
        dna_pattern = self.dna_generator.generate_pattern(pattern_length)
        
        matcher = BoyerMoore(dna_pattern)
        bench_results = self.benchmarker.run_multiple(
            lambda: self.benchmarker.benchmark_boyer_moore(matcher, dna_text)
        )
        agg = self.benchmarker.aggregate_results(bench_results)
        
        results.append({
            'alphabet_type': 'DNA (4 letters)',
            'text_length': text_length,
            'pattern_length': pattern_length,
            'mean_total_time': agg.mean_total_time,
            'mean_comparisons': agg.mean_comparisons,
            'mean_shifts': agg.mean_shifts,
            'matches_found': agg.mean_matches
        })
        
        print(f"  Mean time: {agg.mean_total_time*1000:.3f} ms")
        print(f"  Mean comparisons: {agg.mean_comparisons:.0f}")
        
        # Test 2: Random ASCII text (larger alphabet)
        print("\n2. Testing random ASCII text (26 letters)...")
        import random
        import string
        random.seed(42)
        
        ascii_text = ''.join(random.choices(string.ascii_uppercase, k=text_length))
        ascii_pattern = ''.join(random.choices(string.ascii_uppercase, k=pattern_length))
        
        matcher = BoyerMoore(ascii_pattern)
        bench_results = self.benchmarker.run_multiple(
            lambda: self.benchmarker.benchmark_boyer_moore(matcher, ascii_text)
        )
        agg = self.benchmarker.aggregate_results(bench_results)
        
        results.append({
            'alphabet_type': 'ASCII (26 letters)',
            'text_length': text_length,
            'pattern_length': pattern_length,
            'mean_total_time': agg.mean_total_time,
            'mean_comparisons': agg.mean_comparisons,
            'mean_shifts': agg.mean_shifts,
            'matches_found': agg.mean_matches
        })
        
        print(f"  Mean time: {agg.mean_total_time*1000:.3f} ms")
        print(f"  Mean comparisons: {agg.mean_comparisons:.0f}")
        
        # Save results
        output_file = self.tables_dir / "exp3_alphabet_effect.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n✓ Results saved to {output_file}")
        
        self.results['experiment_3'] = results
        return results
    
    def experiment_4_heuristic_contribution(self) -> Dict:
        """
        Experiment 4: Heuristic Contribution
        
        Compare performance of BCR-only, GSR-only, and combined heuristics.
        
        Returns:
            Dictionary with results
        """
        print("\n" + "=" * 60)
        print("EXPERIMENT 4: Heuristic Contribution")
        print("=" * 60)
        
        # Load test data
        text = self.data_manager.load_ecoli_genome()[:500000]
        pattern_length = 30
        pattern = text[len(text)//2:len(text)//2 + pattern_length]
        
        print(f"Text length: {len(text):,} bp")
        print(f"Pattern length: {pattern_length}")
        
        variants = ['full', 'bcr_only', 'gsr_only', 'horspool']
        results = []
        
        for variant in variants:
            print(f"\nTesting variant: {variant}")
            
            matcher = get_variant(pattern, variant)
            bench_results = self.benchmarker.run_multiple(
                lambda: self.benchmarker.benchmark_boyer_moore(matcher, text)
            )
            agg = self.benchmarker.aggregate_results(bench_results)
            
            result_dict = {
                'variant': variant,
                'text_length': len(text),
                'pattern_length': pattern_length,
                'mean_total_time': agg.mean_total_time,
                'std_total_time': agg.std_total_time,
                'mean_comparisons': agg.mean_comparisons,
                'mean_shifts': agg.mean_shifts,
                'matches_found': agg.mean_matches
            }
            results.append(result_dict)
            
            print(f"  Mean time: {agg.mean_total_time*1000:.3f} ms")
            print(f"  Comparisons: {agg.mean_comparisons:.0f}")
            print(f"  Shifts: {agg.mean_shifts:.0f}")
        
        # Save results
        output_file = self.tables_dir / "exp4_heuristic_contribution.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n✓ Results saved to {output_file}")
        
        self.results['experiment_4'] = results
        return results
    
    def experiment_5_preprocessing_overhead(self, 
                                           pattern_lengths: List[int] = None) -> Dict:
        """
        Experiment 5: Preprocessing Overhead
        
        Measure preprocessing time vs search time for different pattern lengths.
        
        Args:
            pattern_lengths: List of pattern lengths to test
            
        Returns:
            Dictionary with results
        """
        print("\n" + "=" * 60)
        print("EXPERIMENT 5: Preprocessing Overhead")
        print("=" * 60)
        
        if pattern_lengths is None:
            pattern_lengths = [8, 16, 32, 64, 128, 256, 512]
        
        text = self.data_manager.load_ecoli_genome()[:500000]
        
        results = []
        
        for plen in pattern_lengths:
            print(f"\nTesting pattern length: {plen}")
            
            pattern = text[len(text)//2:len(text)//2 + plen]
            
            # Measure preprocessing time
            start = time.perf_counter()
            matcher = BoyerMoore(pattern)
            preprocess_time = time.perf_counter() - start
            
            # Measure search time
            search_times = []
            for _ in range(5):
                start = time.perf_counter()
                matches = matcher.search(text)
                search_time = time.perf_counter() - start
                search_times.append(search_time)
            
            import statistics
            mean_search_time = statistics.mean(search_times)
            
            result_dict = {
                'pattern_length': plen,
                'preprocessing_time': preprocess_time,
                'mean_search_time': mean_search_time,
                'total_time': preprocess_time + mean_search_time,
                'preprocess_ratio': preprocess_time / (preprocess_time + mean_search_time)
            }
            results.append(result_dict)
            
            print(f"  Preprocessing: {preprocess_time*1000:.3f} ms")
            print(f"  Search: {mean_search_time*1000:.3f} ms")
            print(f"  Ratio: {result_dict['preprocess_ratio']*100:.1f}% preprocessing")
        
        # Save results
        output_file = self.tables_dir / "exp5_preprocessing_overhead.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n✓ Results saved to {output_file}")
        
        self.results['experiment_5'] = results
        return results
    
    def experiment_6_memory_footprint(self, 
                                     pattern_lengths: List[int] = None) -> Dict:
        """
        Experiment 6: Memory Footprint
        
        Measure memory usage for different pattern lengths.
        
        Args:
            pattern_lengths: List of pattern lengths to test
            
        Returns:
            Dictionary with results
        """
        print("\n" + "=" * 60)
        print("EXPERIMENT 6: Memory Footprint")
        print("=" * 60)
        
        if pattern_lengths is None:
            pattern_lengths = [8, 16, 32, 64, 128, 256, 512]
        
        text = self.data_manager.load_ecoli_genome()[:500000]
        
        results = []
        
        for plen in pattern_lengths:
            print(f"\nTesting pattern length: {plen}")
            
            pattern = text[len(text)//2:len(text)//2 + plen]
            
            # Measure memory during preprocessing and search
            import tracemalloc
            
            tracemalloc.start()
            matcher = BoyerMoore(pattern)
            matches = matcher.search(text)
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            
            result_dict = {
                'pattern_length': plen,
                'peak_memory_bytes': peak,
                'peak_memory_kb': peak / 1024,
                'peak_memory_mb': peak / (1024 * 1024)
            }
            results.append(result_dict)
            
            print(f"  Peak memory: {peak / 1024:.2f} KB")
        
        # Save results
        output_file = self.tables_dir / "exp6_memory_footprint.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n✓ Results saved to {output_file}")
        
        self.results['experiment_6'] = results
        return results
    
    def experiment_7_real_motifs(self) -> Dict:
        """
        Experiment 7: Real Motif Search
        
        Search for real biological motifs in E. coli genome.
        
        Returns:
            Dictionary with results
        """
        print("\n" + "=" * 60)
        print("EXPERIMENT 7: Real Motif Search")
        print("=" * 60)
        
        # Common E. coli promoter/regulatory motifs
        motifs = {
            'Pribnow_box': 'TATAAT',  # -10 promoter element
            'UP_element': 'AAAWWT',  # -35 promoter element (W = A or T)
            'Shine_Dalgarno': 'AGGAGGT',  # Ribosome binding site
            'Rho_independent_terminator': 'GCGGCG',  # Terminator hairpin
            'CRP_binding': 'TGTGA',  # CAP/CRP binding site
            'LacI_operator': 'AATTGTGAGC',  # Lac repressor binding
        }
        
        # Load E. coli genome
        text = self.data_manager.load_ecoli_genome()
        print(f"Genome length: {len(text):,} bp\n")
        
        results = []
        
        for motif_name, pattern in motifs.items():
            # Skip patterns with ambiguities for now
            if 'W' in pattern:
                print(f"Skipping {motif_name} (contains ambiguous bases)")
                continue
            
            print(f"Searching for {motif_name}: {pattern}")
            
            matcher = BoyerMoore(pattern)
            
            start_time = time.perf_counter()
            matches = matcher.search(text)
            search_time = time.perf_counter() - start_time
            
            stats = matcher.get_statistics()
            
            result_dict = {
                'motif_name': motif_name,
                'pattern': pattern,
                'pattern_length': len(pattern),
                'matches_found': len(matches),
                'search_time': search_time,
                'comparisons': stats['comparisons'],
                'shifts': stats['shifts'],
                'matches_per_mb': len(matches) / (len(text) / 1_000_000)
            }
            results.append(result_dict)
            
            print(f"  Found: {len(matches)} occurrences")
            print(f"  Time: {search_time*1000:.3f} ms")
            print(f"  Density: {result_dict['matches_per_mb']:.1f} matches/Mb")
            
            # Show first few match positions
            if matches:
                print(f"  First matches at: {matches[:5]}")
            print()
        
        # Save results
        output_file = self.tables_dir / "exp7_real_motifs.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"✓ Results saved to {output_file}")
        
        self.results['experiment_7'] = results
        return results
    
    def experiment_8_compare_with_re(self, 
                                    pattern_lengths: List[int] = None) -> Dict:
        """
        Experiment 8: Comparison with Python re
        
        Compare Boyer-Moore with Python's built-in regex engine.
        
        Args:
            pattern_lengths: List of pattern lengths to test
            
        Returns:
            Dictionary with results
        """
        print("\n" + "=" * 60)
        print("EXPERIMENT 8: Comparison with Python re")
        print("=" * 60)
        
        if pattern_lengths is None:
            pattern_lengths = [8, 16, 32, 64, 128]
        
        text = self.data_manager.load_ecoli_genome()[:500000]
        print(f"Text length: {len(text):,} bp\n")
        
        results = []
        
        for plen in pattern_lengths:
            print(f"Testing pattern length: {plen}")
            
            pattern = text[len(text)//2:len(text)//2 + plen]
            
            # Test Boyer-Moore
            matcher = BoyerMoore(pattern)
            bm_results = self.benchmarker.run_multiple(
                lambda: self.benchmarker.benchmark_boyer_moore(matcher, text)
            )
            bm_agg = self.benchmarker.aggregate_results(bm_results)
            
            # Test Python re
            re_results = self.benchmarker.run_multiple(
                lambda: self.benchmarker.benchmark_python_re(pattern, text)
            )
            re_agg = self.benchmarker.aggregate_results(re_results)
            
            speedup = re_agg.mean_total_time / bm_agg.mean_total_time
            
            result_dict = {
                'pattern_length': plen,
                'bm_mean_time': bm_agg.mean_total_time,
                'bm_std_time': bm_agg.std_total_time,
                're_mean_time': re_agg.mean_total_time,
                're_std_time': re_agg.std_total_time,
                'speedup': speedup,
                'bm_matches': bm_agg.mean_matches,
                're_matches': re_agg.mean_matches
            }
            results.append(result_dict)
            
            print(f"  Boyer-Moore: {bm_agg.mean_total_time*1000:.3f} ms")
            print(f"  Python re:   {re_agg.mean_total_time*1000:.3f} ms")
            print(f"  Speedup:     {speedup:.2f}x")
            print()
        
        # Save results
        output_file = self.tables_dir / "exp8_compare_with_re.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"✓ Results saved to {output_file}")
        
        self.results['experiment_8'] = results
        return results
    
    def run_all_experiments(self):
        """Run all experiments in sequence."""
        print("\n" + "=" * 70)
        print(" " * 20 + "BOYER-MOORE EXPERIMENTS")
        print("=" * 70)
        
        experiments = [
            ('experiment_1', self.experiment_1_pattern_length),
            ('experiment_2', self.experiment_2_text_scaling),
            ('experiment_3', self.experiment_3_alphabet_effect),
            ('experiment_4', self.experiment_4_heuristic_contribution),
            ('experiment_5', self.experiment_5_preprocessing_overhead),
            ('experiment_6', self.experiment_6_memory_footprint),
            ('experiment_7', self.experiment_7_real_motifs),
            ('experiment_8', self.experiment_8_compare_with_re),
        ]
        
        for exp_name, exp_func in experiments:
            try:
                exp_func()
            except Exception as e:
                print(f"\n❌ Error in {exp_name}: {e}")
                import traceback
                traceback.print_exc()
        
        print("\n" + "=" * 70)
        print(" " * 20 + "ALL EXPERIMENTS COMPLETE")
        print("=" * 70)


if __name__ == "__main__":
    runner = ExperimentRunner()
    
    # Run individual experiment for testing
    # runner.experiment_1_pattern_length([8, 16, 32, 64])
    
    # Or run all experiments
    runner.run_all_experiments()
