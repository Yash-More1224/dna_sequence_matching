#!/usr/bin/env python3
"""
Wagner-Fischer Algorithm Implementation
Main entry point for DNA sequence matching experiments.

Author: AAD Project Team
Date: November 2025
"""

import argparse
import sys
from pathlib import Path
import yaml

from wf_core import WagnerFischer, levenshtein_distance
from wf_search import PatternSearcher, find_motifs
from data_loader import (
    FastaLoader, 
    SyntheticDataGenerator, 
    download_ecoli_genome,
    create_test_datasets
)
from benchmark import PerformanceBenchmark
from accuracy import AccuracyEvaluator
from visualization import ResultVisualizer


def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file."""
    if not Path(config_path).exists():
        print(f"Warning: Config file {config_path} not found, using defaults")
        return {}
    
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def cmd_distance(args):
    """Compute edit distance between two sequences."""
    wf = WagnerFischer(
        substitution_cost=args.sub_cost,
        insertion_cost=args.ins_cost,
        deletion_cost=args.del_cost
    )
    
    if args.show_alignment:
        distance, operations = wf.compute_with_traceback(args.source, args.target)
        print(f"Edit Distance: {distance}")
        print("\nAlignment Operations:")
        for op in operations:
            print(f"  {op}")
    else:
        distance, _ = wf.compute_distance(args.source, args.target)
        print(f"Edit Distance: {distance}")


def cmd_search(args):
    """Search for pattern in sequence."""
    searcher = PatternSearcher(
        max_distance=args.max_distance,
        substitution_cost=args.sub_cost,
        insertion_cost=args.ins_cost,
        deletion_cost=args.del_cost
    )
    
    # Load text
    if args.text_file:
        loader = FastaLoader()
        sequences = loader.load(args.text_file)
        text = sequences[0].sequence
        print(f"Loaded sequence: {sequences[0].id} ({len(text)} bp)")
    else:
        text = args.text
    
    # Search
    print(f"Searching for pattern: {args.pattern}")
    matches = searcher.search(args.pattern, text, return_alignment=args.show_alignment)
    
    print(f"\nFound {len(matches)} matches:")
    for i, match in enumerate(matches[:args.max_results], 1):
        print(f"\n  Match {i}:")
        print(f"    Position: {match.position}-{match.end_position}")
        print(f"    Edit Distance: {match.edit_distance}")
        print(f"    Matched Text: {match.matched_text}")
        
        if args.show_alignment and match.alignment:
            print(f"    Alignment: {' '.join(match.alignment[:10])}...")


def cmd_benchmark(args):
    """Run performance benchmarks."""
    config = load_config(args.config)
    benchmark_config = config.get('benchmark', {})
    
    output_dir = args.output_dir or benchmark_config.get('output_dir', 'results/benchmarks')
    benchmark = PerformanceBenchmark(output_dir=output_dir)
    
    if args.full:
        print("Running full benchmark suite...")
        benchmark.run_full_suite()
    else:
        print("Running custom benchmark...")
        
        if args.test_edit_distance:
            benchmark.benchmark_edit_distance(
                pattern_lengths=args.pattern_lengths or [10, 20, 50, 100],
                text_length=args.text_length or 10000,
                iterations=args.iterations or 10
            )
        
        if args.test_search:
            benchmark.benchmark_pattern_search(
                pattern_length=args.pattern_length or 30,
                text_lengths=args.text_lengths or [1000, 5000, 10000],
                max_distance=args.max_distance or 2,
                iterations=args.iterations or 5
            )
        
        if args.test_threshold:
            benchmark.benchmark_threshold_scaling(
                pattern_length=args.pattern_length or 50,
                text_length=args.text_length or 10000,
                thresholds=args.thresholds or [0, 1, 2, 3, 5],
                iterations=args.iterations or 5
            )
        
        if args.test_regex:
            benchmark.benchmark_regex_comparison(
                pattern_length=args.pattern_length or 20,
                text_length=args.text_length or 10000,
                iterations=args.iterations or 10
            )
        
        benchmark.save_results()


def cmd_accuracy(args):
    """Run accuracy evaluation."""
    config = load_config(args.config)
    accuracy_config = config.get('accuracy', {})
    
    output_dir = args.output_dir or accuracy_config.get('output_dir', 'results/accuracy')
    evaluator = AccuracyEvaluator(output_dir=output_dir)
    
    if args.full:
        print("Running full accuracy evaluation...")
        evaluator.run_full_evaluation()
    else:
        print("Running custom accuracy tests...")
        
        if args.test_exact:
            evaluator.evaluate_exact_matching(
                pattern_lengths=args.pattern_lengths or [10, 20, 50],
                text_length=args.text_length or 10000,
                num_tests=args.num_tests or 50
            )
        
        if args.test_mutations:
            evaluator.evaluate_synthetic_mutations(
                pattern_length=args.pattern_length or 50,
                text_length=args.text_length or 10000,
                num_patterns=args.num_tests or 100,
                mutation_rate=args.mutation_rate or 0.02,
                max_distance=args.max_distance or 2
            )
        
        evaluator.save_results()


def cmd_visualize(args):
    """Generate visualizations."""
    config = load_config(args.config)
    viz_config = config.get('visualization', {})
    
    output_dir = args.output_dir or viz_config.get('output_dir', 'results/plots')
    visualizer = ResultVisualizer(output_dir=output_dir)
    
    if args.benchmark_csv:
        print(f"Plotting benchmark results from {args.benchmark_csv}")
        visualizer.plot_benchmark_results(args.benchmark_csv)
        
        if args.comparison:
            visualizer.plot_comparison(args.benchmark_csv)
    
    if args.accuracy_csv:
        print(f"Plotting accuracy results from {args.accuracy_csv}")
        visualizer.plot_accuracy_results(args.accuracy_csv)
    
    print("Visualization complete!")


def cmd_data(args):
    """Manage datasets."""
    if args.download_ecoli:
        print("Downloading E. coli genome...")
        path = download_ecoli_genome(output_dir=args.data_dir)
        print(f"Genome saved to: {path}")
    
    if args.generate_synthetic:
        print("Generating synthetic test datasets...")
        create_test_datasets(output_dir=args.data_dir)
        print("Synthetic datasets created!")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Wagner-Fischer Algorithm for DNA Sequence Matching",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compute edit distance
  python main.py distance ATCG ATCG
  
  # Search for pattern
  python main.py search ATCG --text GGATCGGGATCG --max-distance 2
  
  # Run full benchmark suite
  python main.py benchmark --full
  
  # Run accuracy evaluation
  python main.py accuracy --full
  
  # Generate visualizations
  python main.py visualize --benchmark-csv results/benchmarks/benchmark_results.csv
  
  # Download E. coli genome
  python main.py data --download-ecoli
        """
    )
    
    parser.add_argument('--config', default='config.yaml', help='Config file path')
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Distance command
    distance_parser = subparsers.add_parser('distance', help='Compute edit distance')
    distance_parser.add_argument('source', help='Source sequence')
    distance_parser.add_argument('target', help='Target sequence')
    distance_parser.add_argument('--sub-cost', type=int, default=1, help='Substitution cost')
    distance_parser.add_argument('--ins-cost', type=int, default=1, help='Insertion cost')
    distance_parser.add_argument('--del-cost', type=int, default=1, help='Deletion cost')
    distance_parser.add_argument('--show-alignment', action='store_true', help='Show alignment')
    
    # Search command
    search_parser = subparsers.add_parser('search', help='Search for pattern')
    search_parser.add_argument('pattern', help='Pattern to search for')
    search_parser.add_argument('--text', help='Text to search in')
    search_parser.add_argument('--text-file', help='FASTA file with text')
    search_parser.add_argument('--max-distance', type=int, default=2, help='Max edit distance')
    search_parser.add_argument('--sub-cost', type=int, default=1, help='Substitution cost')
    search_parser.add_argument('--ins-cost', type=int, default=1, help='Insertion cost')
    search_parser.add_argument('--del-cost', type=int, default=1, help='Deletion cost')
    search_parser.add_argument('--show-alignment', action='store_true', help='Show alignments')
    search_parser.add_argument('--max-results', type=int, default=10, help='Max results to show')
    
    # Benchmark command
    benchmark_parser = subparsers.add_parser('benchmark', help='Run benchmarks')
    benchmark_parser.add_argument('--full', action='store_true', help='Run full suite')
    benchmark_parser.add_argument('--test-edit-distance', action='store_true')
    benchmark_parser.add_argument('--test-search', action='store_true')
    benchmark_parser.add_argument('--test-threshold', action='store_true')
    benchmark_parser.add_argument('--test-regex', action='store_true')
    benchmark_parser.add_argument('--pattern-lengths', type=int, nargs='+')
    benchmark_parser.add_argument('--pattern-length', type=int)
    benchmark_parser.add_argument('--text-lengths', type=int, nargs='+')
    benchmark_parser.add_argument('--text-length', type=int)
    benchmark_parser.add_argument('--thresholds', type=int, nargs='+')
    benchmark_parser.add_argument('--max-distance', type=int)
    benchmark_parser.add_argument('--iterations', type=int)
    benchmark_parser.add_argument('--output-dir', help='Output directory')
    
    # Accuracy command
    accuracy_parser = subparsers.add_parser('accuracy', help='Run accuracy tests')
    accuracy_parser.add_argument('--full', action='store_true', help='Run full evaluation')
    accuracy_parser.add_argument('--test-exact', action='store_true')
    accuracy_parser.add_argument('--test-mutations', action='store_true')
    accuracy_parser.add_argument('--pattern-lengths', type=int, nargs='+')
    accuracy_parser.add_argument('--pattern-length', type=int)
    accuracy_parser.add_argument('--text-length', type=int)
    accuracy_parser.add_argument('--num-tests', type=int)
    accuracy_parser.add_argument('--mutation-rate', type=float)
    accuracy_parser.add_argument('--max-distance', type=int)
    accuracy_parser.add_argument('--output-dir', help='Output directory')
    
    # Visualize command
    viz_parser = subparsers.add_parser('visualize', help='Generate visualizations')
    viz_parser.add_argument('--benchmark-csv', help='Benchmark results CSV')
    viz_parser.add_argument('--accuracy-csv', help='Accuracy results CSV')
    viz_parser.add_argument('--comparison', action='store_true', help='Plot comparisons')
    viz_parser.add_argument('--output-dir', help='Output directory')
    
    # Data command
    data_parser = subparsers.add_parser('data', help='Manage datasets')
    data_parser.add_argument('--download-ecoli', action='store_true', help='Download E. coli genome')
    data_parser.add_argument('--generate-synthetic', action='store_true', help='Generate synthetic data')
    data_parser.add_argument('--data-dir', default='data', help='Data directory')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    # Route to appropriate command
    commands = {
        'distance': cmd_distance,
        'search': cmd_search,
        'benchmark': cmd_benchmark,
        'accuracy': cmd_accuracy,
        'visualize': cmd_visualize,
        'data': cmd_data
    }
    
    try:
        commands[args.command](args)
        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
