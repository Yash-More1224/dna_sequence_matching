"""
Main Entry Point for Shift-Or/Bitap DNA Sequence Matching
=========================================================

This script provides a command-line interface for:
- Running the algorithm on DNA sequences
- Executing benchmark suites
- Running experiments
- Generating visualizations

Usage:
    python main.py search --pattern ACGT --text-file genome.fasta
    python main.py benchmark --pattern GATTACA
    python main.py experiments --full
    python main.py test

Author: DNA Sequence Matching Project
Date: November 2025
"""

import argparse
import sys
from pathlib import Path

from algorithm import ShiftOrBitap
from data_loader import DataLoader, SyntheticDataGenerator
from benchmark import Benchmarker
from evaluation import ApproximateMatchEvaluator
from experiments import ExperimentRunner, run_full_benchmark_suite


def cmd_search(args):
    """Execute search command."""
    print("Shift-Or/Bitap Pattern Search")
    print("=" * 60)
    
    # Load text
    if args.text_file:
        loader = DataLoader()
        try:
            text = loader.load_fasta_single(args.text_file)
            print(f"Loaded text from: {args.text_file}")
        except Exception as e:
            print(f"Error loading file: {e}")
            return 1
    elif args.text:
        text = args.text
    else:
        print("Error: Either --text or --text-file must be provided")
        return 1
    
    print(f"Text length: {len(text):,} characters")
    print(f"Pattern: {args.pattern}")
    print(f"Pattern length: {len(args.pattern)}")
    
    # Create matcher
    matcher = ShiftOrBitap(args.pattern, case_sensitive=args.case_sensitive)
    
    # Search
    if args.approximate:
        print(f"Approximate matching (max errors: {args.max_errors})")
        matches = matcher.search_approximate(text, args.max_errors)
        
        print(f"\nFound {len(matches)} matches:")
        for i, (pos, errors) in enumerate(matches[:args.max_results]):
            match_text = text[pos:pos+len(args.pattern)]
            print(f"  {i+1}. Position {pos}: '{match_text}' ({errors} error(s))")
        
        if len(matches) > args.max_results:
            print(f"  ... and {len(matches) - args.max_results} more")
    else:
        print("Exact matching")
        matches = matcher.search_exact(text)
        
        print(f"\nFound {len(matches)} matches:")
        for i, pos in enumerate(matches[:args.max_results]):
            match_text = text[pos:pos+len(args.pattern)]
            print(f"  {i+1}. Position {pos}: '{match_text}'")
        
        if len(matches) > args.max_results:
            print(f"  ... and {len(matches) - args.max_results} more")
    
    return 0


def cmd_benchmark(args):
    """Execute benchmark command."""
    print("Shift-Or/Bitap Benchmarking")
    print("=" * 60)
    
    # Generate or load text
    if args.text_file:
        loader = DataLoader()
        text = loader.load_fasta_single(args.text_file)
    else:
        print(f"Generating synthetic text ({args.text_length:,} characters)...")
        text = SyntheticDataGenerator.generate_random_sequence(args.text_length, seed=42)
    
    print(f"Text length: {len(text):,} characters")
    print(f"Pattern: {args.pattern}")
    print(f"Number of runs: {args.num_runs}")
    
    # Run benchmark
    benchmarker = Benchmarker()
    
    if args.compare_regex:
        results = benchmarker.compare_algorithms(args.pattern, text, args.num_runs)
        benchmarker.print_comparison(results)
    else:
        matcher = ShiftOrBitap(args.pattern)
        result = benchmarker.benchmark_shift_or(matcher, text, args.num_runs)
        print(f"\n{result}")
    
    return 0


def cmd_experiments(args):
    """Execute experiments command."""
    print("Running Experiments")
    print("=" * 60)
    
    if args.full:
        run_full_benchmark_suite()
    else:
        runner = ExperimentRunner(output_dir=args.output_dir)
        
        if args.pattern_scaling:
            runner.experiment_pattern_length_scaling(
                pattern_lengths=[5, 10, 15, 20, 25, 30, 40, 50],
                num_runs=args.num_runs
            )
        
        if args.text_scaling:
            runner.experiment_text_length_scaling(
                scale_factors=[1, 2, 5, 10, 20],
                num_runs=args.num_runs
            )
        
        if args.mutation_rates:
            runner.experiment_mutation_rates(
                mutation_rates=[0.0, 0.05, 0.1, 0.15, 0.2],
                num_runs=args.num_runs
            )
        
        if args.vs_regex:
            runner.experiment_vs_regex(
                patterns=["ACGT", "GATTACA", "TATAAA"],
                num_runs=args.num_runs
            )
        
        runner.save_results()
        
        for exp in runner.results['experiments']:
            runner.export_to_csv(exp['name'])
    
    return 0


def cmd_test(args):
    """Execute test command."""
    print("Running Tests")
    print("=" * 60)
    
    try:
        import pytest
        
        test_args = ["-v"]
        if args.verbose:
            test_args.append("-vv")
        if args.coverage:
            test_args.extend(["--cov=.", "--cov-report=html"])
        
        test_args.append("tests/")
        
        exit_code = pytest.main(test_args)
        return exit_code
    except ImportError:
        print("Error: pytest not installed. Install with: pip install pytest")
        return 1


def cmd_demo(args):
    """Execute demo command."""
    print("Shift-Or/Bitap Algorithm Demo")
    print("=" * 60)
    
    # Demo 1: Exact matching
    print("\n1. Exact Matching")
    print("-" * 60)
    
    pattern = "GATTACA"
    text = "CGATTACAGATGATTACATGATTXCA"
    
    print(f"Pattern: {pattern}")
    print(f"Text:    {text}")
    
    matcher = ShiftOrBitap(pattern)
    matches = matcher.search_exact(text)
    
    print(f"Exact matches at positions: {matches}")
    
    # Demo 2: Approximate matching
    print("\n2. Approximate Matching")
    print("-" * 60)
    
    print(f"Pattern: {pattern}")
    print(f"Text:    {text}")
    print(f"Max errors: 1")
    
    approx_matches = matcher.search_approximate(text, max_errors=1)
    
    print(f"Approximate matches:")
    for pos, errors in approx_matches:
        match_text = text[pos:pos+len(pattern)]
        print(f"  Position {pos}: '{match_text}' ({errors} error(s))")
    
    # Demo 3: Performance comparison
    print("\n3. Performance Comparison")
    print("-" * 60)
    
    synthetic_text = SyntheticDataGenerator.generate_random_sequence(10000, seed=42)
    print(f"Testing on {len(synthetic_text):,} character sequence...")
    
    benchmarker = Benchmarker()
    results = benchmarker.compare_algorithms(pattern, synthetic_text, num_runs=5)
    
    for name, result in results.items():
        print(f"\n{name}:")
        print(f"  Search time: {result.mean_search_time*1000:.3f} ms")
        print(f"  Throughput: {result.throughput_chars_per_sec/1e6:.2f} MB/s")
        print(f"  Matches found: {result.num_matches}")
    
    return 0


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Shift-Or/Bitap Algorithm for DNA Sequence Matching",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Search command
    search_parser = subparsers.add_parser('search', help='Search for pattern in text')
    search_parser.add_argument('--pattern', '-p', required=True, help='Pattern to search for')
    search_parser.add_argument('--text', '-t', help='Text to search in')
    search_parser.add_argument('--text-file', '-f', help='FASTA file containing text')
    search_parser.add_argument('--approximate', '-a', action='store_true', 
                              help='Use approximate matching')
    search_parser.add_argument('--max-errors', '-e', type=int, default=1,
                              help='Maximum edit distance for approximate matching')
    search_parser.add_argument('--case-sensitive', '-c', action='store_true',
                              help='Case-sensitive matching')
    search_parser.add_argument('--max-results', '-m', type=int, default=10,
                              help='Maximum number of results to display')
    
    # Benchmark command
    bench_parser = subparsers.add_parser('benchmark', help='Benchmark the algorithm')
    bench_parser.add_argument('--pattern', '-p', default='GATTACA', help='Pattern to test')
    bench_parser.add_argument('--text-file', '-f', help='FASTA file to use')
    bench_parser.add_argument('--text-length', '-l', type=int, default=10000,
                             help='Length of synthetic text to generate')
    bench_parser.add_argument('--num-runs', '-n', type=int, default=10,
                             help='Number of benchmark runs')
    bench_parser.add_argument('--compare-regex', '-r', action='store_true',
                             help='Compare with Python re module')
    
    # Experiments command
    exp_parser = subparsers.add_parser('experiments', help='Run experiments')
    exp_parser.add_argument('--full', action='store_true',
                           help='Run full benchmark suite')
    exp_parser.add_argument('--pattern-scaling', action='store_true',
                           help='Run pattern length scaling experiment')
    exp_parser.add_argument('--text-scaling', action='store_true',
                           help='Run text length scaling experiment')
    exp_parser.add_argument('--mutation-rates', action='store_true',
                           help='Run mutation rate experiment')
    exp_parser.add_argument('--vs-regex', action='store_true',
                           help='Compare with Python re')
    exp_parser.add_argument('--num-runs', '-n', type=int, default=5,
                           help='Number of runs per experiment')
    exp_parser.add_argument('--output-dir', '-o', default='./results',
                           help='Output directory for results')
    
    # Test command
    test_parser = subparsers.add_parser('test', help='Run unit tests')
    test_parser.add_argument('--verbose', '-v', action='store_true',
                            help='Verbose output')
    test_parser.add_argument('--coverage', '-c', action='store_true',
                            help='Generate coverage report')
    
    # Demo command
    demo_parser = subparsers.add_parser('demo', help='Run interactive demo')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    # Execute command
    command_map = {
        'search': cmd_search,
        'benchmark': cmd_benchmark,
        'experiments': cmd_experiments,
        'test': cmd_test,
        'demo': cmd_demo
    }
    
    try:
        return command_map[args.command](args)
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        return 130
    except Exception as e:
        print(f"\nError: {e}")
        if args.command == 'test':
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
