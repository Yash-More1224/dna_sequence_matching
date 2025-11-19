"""
Experiment Orchestration for Suffix Array Analysis

This module provides functions to run comprehensive benchmark experiments,
comparing Suffix Array performance across different parameters and against Python's re module.
"""

import sys
import yaml
from pathlib import Path
from typing import List, Dict, Any, Optional
import time

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.suffix_array import SuffixArray
from src.data_loader import DatasetManager
from src.data_generator import DNAGenerator, generate_random_sequence, generate_pattern
from src.utils import (format_time, format_memory, format_throughput, save_json, 
                       save_csv, print_section, print_subsection, ensure_dir)
from experiments.benchmarks import Benchmarker, AggregatedResult


class ExperimentRunner:
    """
    Main class for running Suffix Array algorithm experiments.
    
    Attributes:
        results_dir: Directory to save results
        config: Configuration dictionary
        seed: Random seed for reproducibility
    """
    
    def __init__(self, results_dir: Path = None, config_file: str = "config.yaml"):
        """
        Initialize experiment runner.
        
        Args:
            results_dir: Directory to save results
            config_file: Path to configuration file
        """
        # Load configuration
        with open(config_file, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Set up directories
        if results_dir is None:
            results_dir = Path(self.config['output']['results_dir'])
        self.results_dir = results_dir
        self.plots_dir = Path(self.config['output']['plots_dir'])
        self.tables_dir = Path(self.config['output']['tables_dir'])
        self.reports_dir = Path(self.config['output']['reports_dir'])
        
        # Create directories
        ensure_dir(self.results_dir)
        ensure_dir(self.plots_dir)
        ensure_dir(self.tables_dir)
        ensure_dir(self.reports_dir)
        
        self.seed = 42
        self.benchmarker = Benchmarker(
            warmup_runs=self.config['benchmarking']['warmup_runs'],
            min_runs=self.config['benchmarking']['min_runs']
        )
        
        # Initialize data components
        self.data_manager = DatasetManager()
        self.dna_generator = DNAGenerator(seed=self.seed)
    
    def experiment_1_pattern_length(self) -> List[Dict]:
        """
        Experiment 1: Vary pattern length and measure performance.
        
        Tests how pattern length affects preprocessing and search time.
        """
        print_section("EXPERIMENT 1: Pattern Length Variation")
        
        pattern_lengths = self.config['experiments']['pattern_lengths']
        text_length = 100000
        repetitions = self.config['experiments']['repetitions']
        
        print(f"Text length: {text_length:,} bp")
        print(f"Testing pattern lengths: {pattern_lengths}")
        print(f"Repetitions per test: {repetitions}")
        print()
        
        # Generate text once
        text = generate_random_sequence(text_length, seed=self.seed)
        
        # Build suffix array once
        print("Building suffix array...")
        sa = SuffixArray(text, verbose=False)
        print(f"✓ Construction time: {format_time(sa.preprocessing_time)}")
        print(f"✓ Index memory: {format_memory(sa.memory_footprint)}")
        print()
        
        results = []
        for pattern_len in pattern_lengths:
            print(f"Testing pattern length: {pattern_len} bp", end=" ... ")
            
            # Generate pattern
            pattern = generate_pattern(pattern_len, seed=self.seed + pattern_len)
            
            # Benchmark
            result = self.benchmarker.benchmark_with_repetitions(
                sa, text, pattern, repetitions=repetitions
            )
            results.append(result.to_dict())
            
            print(f"✓ Search: {format_time(result.mean_search_time)}, "
                  f"Matches: {int(result.mean_matches)}")
        
        # Save results
        save_json(results, self.tables_dir / "exp1_pattern_length.json")
        save_csv(results, self.tables_dir / "exp1_pattern_length.csv")
        
        return results
    
    def experiment_2_text_scaling(self) -> List[Dict]:
        """
        Experiment 2: Vary text size and measure scalability.
        
        Tests how algorithm scales with text size.
        """
        print_section("EXPERIMENT 2: Text Size Scalability")
        
        text_sizes = self.config['experiments']['text_sizes']
        pattern_length = 32
        repetitions = self.config['experiments']['repetitions']
        
        print(f"Pattern length: {pattern_length} bp")
        print(f"Testing text sizes: {[f'{s:,}' for s in text_sizes]}")
        print(f"Repetitions per test: {repetitions}")
        print()
        
        pattern = generate_pattern(pattern_length, seed=self.seed)
        
        results = []
        for text_size in text_sizes:
            print(f"\nTesting text size: {text_size:,} bp")
            
            # Generate text
            text = generate_random_sequence(text_size, seed=self.seed + text_size)
            
            # Build suffix array and benchmark
            print("  Building index...", end=" ")
            start = time.perf_counter()
            sa = SuffixArray(text, verbose=False)
            build_time = time.perf_counter() - start
            print(f"✓ {format_time(build_time)}")
            
            # Benchmark search
            print("  Benchmarking search...", end=" ")
            result = self.benchmarker.benchmark_with_repetitions(
                sa, text, pattern, repetitions=repetitions
            )
            results.append(result.to_dict())
            
            throughput = text_size / result.mean_search_time / 1_000_000
            print(f"✓ {format_time(result.mean_search_time)}, "
                  f"{throughput:.2f} MB/s")
        
        # Save results
        save_json(results, self.tables_dir / "exp2_text_scaling.json")
        save_csv(results, self.tables_dir / "exp2_text_scaling.csv")
        
        return results
    
    def experiment_3_preprocessing_cost(self) -> List[Dict]:
        """
        Experiment 3: Analyze preprocessing overhead.
        
        Compares construction time vs search time for different text sizes.
        """
        print_section("EXPERIMENT 3: Preprocessing Overhead Analysis")
        
        text_sizes = [10000, 50000, 100000, 500000, 1000000]
        pattern_length = 32
        
        print(f"Pattern length: {pattern_length} bp")
        print(f"Testing text sizes: {[f'{s:,}' for s in text_sizes]}")
        print()
        
        pattern = generate_pattern(pattern_length, seed=self.seed)
        
        results = []
        for text_size in text_sizes:
            print(f"Testing text size: {text_size:,} bp")
            
            # Generate text
            text = generate_random_sequence(text_size, seed=self.seed + text_size)
            
            # Measure construction time
            start = time.perf_counter()
            sa = SuffixArray(text, verbose=False)
            construction_time = time.perf_counter() - start
            
            # Measure search time
            start = time.perf_counter()
            matches = sa.search(pattern)
            search_time = time.perf_counter() - start
            
            result = {
                'text_size': text_size,
                'pattern_length': pattern_length,
                'construction_time': construction_time,
                'search_time': search_time,
                'total_time': construction_time + search_time,
                'preprocessing_ratio': construction_time / (construction_time + search_time),
                'index_memory': sa.memory_footprint,
                'num_matches': len(matches)
            }
            results.append(result)
            
            print(f"  Construction: {format_time(construction_time)}")
            print(f"  Search: {format_time(search_time)}")
            print(f"  Ratio: {result['preprocessing_ratio']:.2%} preprocessing\n")
        
        # Save results
        save_json(results, self.tables_dir / "exp3_preprocessing.json")
        save_csv(results, self.tables_dir / "exp3_preprocessing.csv")
        
        return results
    
    def experiment_4_memory_footprint(self) -> List[Dict]:
        """
        Experiment 4: Analyze memory usage vs text size.
        
        Measures index memory footprint for different text sizes.
        """
        print_section("EXPERIMENT 4: Memory Footprint Analysis")
        
        text_sizes = [10000, 50000, 100000, 500000, 1000000, 5000000]
        
        print(f"Testing text sizes: {[f'{s:,}' for s in text_sizes]}")
        print()
        
        results = []
        for text_size in text_sizes:
            print(f"Testing text size: {text_size:,} bp...", end=" ")
            
            # Generate text
            text = generate_random_sequence(text_size, seed=self.seed + text_size)
            
            # Build suffix array
            sa = SuffixArray(text, verbose=False)
            
            result = {
                'text_size': text_size,
                'index_memory_bytes': sa.memory_footprint,
                'index_memory_mb': sa.memory_footprint / (1024**2),
                'memory_per_char': sa.memory_footprint / text_size,
                'construction_time': sa.preprocessing_time
            }
            results.append(result)
            
            print(f"✓ {format_memory(sa.memory_footprint)} "
                  f"({result['memory_per_char']:.1f} bytes/char)")
        
        # Save results
        save_json(results, self.tables_dir / "exp4_memory.json")
        save_csv(results, self.tables_dir / "exp4_memory.csv")
        
        return results
    
    def experiment_5_compare_with_re(self) -> Dict:
        """
        Experiment 5: Compare with Python's re module.
        
        Compares Suffix Array performance with regex-based search.
        """
        print_section("EXPERIMENT 5: Comparison with Python re Module")
        
        text_length = 1000000
        pattern_lengths = [8, 16, 32, 64, 128]
        repetitions = self.config['experiments']['repetitions']
        
        print(f"Text length: {text_length:,} bp")
        print(f"Pattern lengths: {pattern_lengths}")
        print(f"Repetitions: {repetitions}")
        print()
        
        # Generate text
        text = generate_random_sequence(text_length, seed=self.seed)
        
        # Build suffix array once
        print("Building suffix array...")
        sa = SuffixArray(text, verbose=False)
        print(f"✓ Construction: {format_time(sa.preprocessing_time)}\n")
        
        results = []
        for pattern_len in pattern_lengths:
            print(f"Pattern length {pattern_len}:")
            
            pattern = generate_pattern(pattern_len, seed=self.seed + pattern_len)
            
            # Benchmark Suffix Array
            sa_result = self.benchmarker.benchmark_with_repetitions(
                sa, text, pattern, repetitions=repetitions
            )
            
            # Benchmark Python re
            re_result = self.benchmarker.compare_with_re(
                text, pattern, repetitions=repetitions
            )
            
            speedup = re_result['mean_time'] / sa_result.mean_search_time
            
            result = {
                'pattern_length': pattern_len,
                'sa_search_time': sa_result.mean_search_time,
                'sa_total_time': sa_result.mean_total_time,
                'sa_throughput': sa_result.throughput_mbps,
                're_time': re_result['mean_time'],
                're_throughput': re_result['throughput_mbps'],
                'speedup': speedup,
                'num_matches': sa_result.mean_matches
            }
            results.append(result)
            
            print(f"  Suffix Array search: {format_time(sa_result.mean_search_time)}")
            print(f"  Python re:          {format_time(re_result['mean_time'])}")
            print(f"  Speedup:            {speedup:.2f}x {'(SA faster)' if speedup > 1 else '(re faster)'}\n")
        
        # Save results
        save_json(results, self.tables_dir / "exp5_compare_re.json")
        save_csv(results, self.tables_dir / "exp5_compare_re.csv")
        
        return results
    
    def experiment_6_repeat_discovery(self) -> List[Dict]:
        """
        Experiment 6: Test repeat/motif discovery using LCP array.
        
        Analyzes performance of finding repeated substrings.
        """
        print_section("EXPERIMENT 6: Repeat Discovery Performance")
        
        text_length = 500000
        min_lengths = self.config['experiments']['repeat_min_lengths']
        
        print(f"Text length: {text_length:,} bp")
        print(f"Testing minimum repeat lengths: {min_lengths}")
        print()
        
        # Generate text with some repeats
        text = generate_random_sequence(text_length, seed=self.seed)
        
        # Build suffix array
        print("Building suffix array...")
        sa = SuffixArray(text, verbose=False)
        print(f"✓ Construction: {format_time(sa.preprocessing_time)}\n")
        
        results = []
        for min_len in min_lengths:
            print(f"Min length {min_len}:", end=" ")
            
            start = time.perf_counter()
            repeats = sa.find_longest_repeats(min_length=min_len)
            discovery_time = time.perf_counter() - start
            
            result = {
                'min_length': min_len,
                'num_repeats_found': len(repeats),
                'discovery_time': discovery_time,
                'longest_repeat_length': repeats[0]['length'] if repeats else 0,
                'max_occurrences': max([r['count'] for r in repeats]) if repeats else 0
            }
            results.append(result)
            
            print(f"Found {len(repeats)} repeats in {format_time(discovery_time)}")
        
        # Save results
        save_json(results, self.tables_dir / "exp6_repeat_discovery.json")
        save_csv(results, self.tables_dir / "exp6_repeat_discovery.csv")
        
        return results
    
    def experiment_7_ecoli_genome(self) -> Dict:
        """
        Experiment 7: Test on real E. coli genome.
        
        Run comprehensive tests on actual biological data.
        """
        print_section("EXPERIMENT 7: E. coli Genome Analysis")
        
        # Load E. coli genome
        print("Loading E. coli genome...")
        genome = self.data_manager.load_ecoli_genome(download_if_missing=True)
        print(f"✓ Loaded {len(genome):,} bp\n")
        
        # Build suffix array
        print("Building suffix array for full genome...")
        start = time.perf_counter()
        sa = SuffixArray(genome, verbose=False)
        construction_time = time.perf_counter() - start
        
        print(f"✓ Construction: {format_time(construction_time)}")
        print(f"✓ Index memory: {format_memory(sa.memory_footprint)}\n")
        
        # Test various patterns
        test_patterns = [
            ("TATAAT", "Pribnow box (promoter)"),
            ("TTGACA", "-35 promoter element"),
            ("AGGAGG", "Ribosome binding site"),
            ("ATGATG", "Start codon region"),
            ("GCGATCGC", "Random 8-mer")
        ]
        
        results = []
        print("Searching for biological motifs:")
        print("-" * 70)
        
        for pattern, description in test_patterns:
            start = time.perf_counter()
            matches = sa.search(pattern)
            search_time = time.perf_counter() - start
            
            result = {
                'pattern': pattern,
                'description': description,
                'pattern_length': len(pattern),
                'num_matches': len(matches),
                'search_time': search_time,
                'throughput_mbps': len(genome) / search_time / 1_000_000
            }
            results.append(result)
            
            print(f"{pattern:12} ({description:25}): "
                  f"{len(matches):5} matches in {format_time(search_time)}")
        
        print()
        
        # Find repeats
        print("Finding long repeats (min_length=20)...")
        start = time.perf_counter()
        repeats = sa.find_longest_repeats(min_length=20)
        repeat_time = time.perf_counter() - start
        
        summary = {
            'genome_length': len(genome),
            'construction_time': construction_time,
            'index_memory': sa.memory_footprint,
            'pattern_searches': results,
            'repeat_discovery': {
                'min_length': 20,
                'num_repeats': len(repeats),
                'discovery_time': repeat_time,
                'top_5_repeats': repeats[:5] if len(repeats) >= 5 else repeats
            }
        }
        
        print(f"✓ Found {len(repeats)} repeats in {format_time(repeat_time)}")
        if repeats:
            print(f"  Longest repeat: {repeats[0]['length']} bp, "
                  f"{repeats[0]['count']} occurrences")
        print()
        
        # Save results
        save_json(summary, self.tables_dir / "exp7_ecoli_genome.json")
        
        return summary
    
    def experiment_8_pattern_complexity(self) -> List[Dict]:
        """
        Experiment 8: Analyze impact of pattern complexity.
        
        Tests patterns with different characteristics (repetitive, random, etc.).
        """
        print_section("EXPERIMENT 8: Pattern Complexity Analysis")
        
        text_length = 500000
        pattern_length = 32
        repetitions = 5
        
        # Generate text
        text = generate_random_sequence(text_length, seed=self.seed)
        sa = SuffixArray(text, verbose=False)
        
        # Test different pattern types
        pattern_types = [
            ("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA", "Highly repetitive (A only)"),
            ("ATATATATATATATATATATATATATATAT", "Alternating (AT)"),
            ("ACGTACGTACGTACGTACGTACGTACGTAC", "Repeating 4-mer"),
            (generate_pattern(pattern_length, seed=42), "Random"),
            ("ACGTNNNNNNNNNNNNNNNNNNNNNNNACGT", "With ambiguous bases")
        ]
        
        results = []
        print(f"Text length: {text_length:,} bp")
        print(f"Pattern length: {pattern_length} bp\n")
        
        for pattern, description in pattern_types:
            pattern = pattern[:pattern_length]  # Ensure correct length
            
            result_agg = self.benchmarker.benchmark_with_repetitions(
                sa, text, pattern, repetitions=repetitions
            )
            
            result = {
                'pattern_type': description,
                'pattern': pattern[:20] + "..." if len(pattern) > 20 else pattern,
                'search_time': result_agg.mean_search_time,
                'num_matches': int(result_agg.mean_matches),
                'comparisons': int(result_agg.mean_comparisons)
            }
            results.append(result)
            
            print(f"{description:30}: {format_time(result['search_time'])}, "
                  f"{result['num_matches']} matches")
        
        print()
        
        # Save results
        save_json(results, self.tables_dir / "exp8_pattern_complexity.json")
        save_csv(results, self.tables_dir / "exp8_pattern_complexity.csv")
        
        return results
    
    def run_all_experiments(self):
        """Run all experiments in sequence."""
        print_section("RUNNING ALL SUFFIX ARRAY EXPERIMENTS")
        print()
        
        experiments = [
            ("Experiment 1", self.experiment_1_pattern_length),
            ("Experiment 2", self.experiment_2_text_scaling),
            ("Experiment 3", self.experiment_3_preprocessing_cost),
            ("Experiment 4", self.experiment_4_memory_footprint),
            ("Experiment 5", self.experiment_5_compare_with_re),
            ("Experiment 6", self.experiment_6_repeat_discovery),
            ("Experiment 7", self.experiment_7_ecoli_genome),
            ("Experiment 8", self.experiment_8_pattern_complexity)
        ]
        
        for name, exp_func in experiments:
            try:
                print(f"\n{'='*70}")
                print(f"Running {name}")
                print('='*70)
                exp_func()
                print(f"\n✓ {name} completed successfully")
            except Exception as e:
                print(f"\n❌ Error in {name}: {e}")
                import traceback
                traceback.print_exc()
        
        print_section("ALL EXPERIMENTS COMPLETED")


if __name__ == "__main__":
    runner = ExperimentRunner()
    runner.run_all_experiments()
