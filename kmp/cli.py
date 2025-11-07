"""
Command-Line Interface for KMP Algorithm.

This module provides a comprehensive CLI for running KMP searches,
benchmarks, experiments, and visualizations.
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

from .kmp_algorithm import KMP
from .data_loader import read_fasta, load_dataset, sequence_stats
from .synthetic_data import generate_synthetic_dataset
from .experiments import ExperimentRunner, run_quick_demo
from .evaluation import compare_with_re, print_comparison_summary
from .config import DATASETS_DIR, DATASETS, RESULTS_DIR
from .utils import format_time, format_memory


def cmd_search(args):
    """Handle search command."""
    print("\n" + "="*70)
    print("KMP SEARCH")
    print("="*70)
    
    # Load text
    if args.file:
        print(f"Loading text from {args.file}...")
        records = read_fasta(Path(args.file), max_sequences=1)
        if not records:
            print("Error: Could not read file")
            return 1
        text = records[0].sequence
    elif args.text:
        text = args.text
    else:
        print("Error: Must provide --file or --text")
        return 1
    
    # Get pattern
    pattern = args.pattern
    
    print(f"Text length: {len(text):,} bp")
    print(f"Pattern: {pattern} ({len(pattern)} bp)")
    print("-"*70)
    
    # Search
    kmp = KMP(pattern)
    stats = kmp.search_with_stats(text)
    
    # Display results
    print(f"\nResults:")
    print(f"  Preprocessing time: {format_time(stats['preprocessing_time'])}")
    print(f"  Search time:        {format_time(stats['search_time'])}")
    print(f"  Total time:         {format_time(stats['total_time'])}")
    print(f"  Matches found:      {stats['num_matches']}")
    
    if stats['matches'] and args.show_positions:
        print(f"\nMatch positions:")
        max_show = args.max_matches if args.max_matches else len(stats['matches'])
        for i, pos in enumerate(stats['matches'][:max_show]):
            print(f"    {i+1}. Position {pos}")
        if len(stats['matches']) > max_show:
            print(f"    ... and {len(stats['matches']) - max_show} more")
    
    print("="*70)
    return 0


def cmd_benchmark(args):
    """Handle benchmark command."""
    print("\n" + "="*70)
    print("KMP BENCHMARK")
    print("="*70)
    
    if args.dataset:
        # Benchmark on a real dataset
        print(f"Loading dataset: {args.dataset}")
        genome = load_dataset(args.dataset, DATASETS_DIR)
        
        if genome is None:
            print(f"Error: Dataset '{args.dataset}' not found.")
            print("Run 'python -m kmp.cli download' to download datasets.")
            return 1
        
        text = genome.sequence
        print(f"Loaded: {genome.id} ({len(text):,} bp)")
    else:
        # Generate synthetic data
        from .synthetic_data import generate_random_sequence
        text_size = args.text_size if args.text_size else 100000
        print(f"Generating synthetic text ({text_size:,} bp)...")
        text = generate_random_sequence(text_size, seed=42)
    
    # Generate or use provided pattern
    if args.pattern:
        pattern = args.pattern
    else:
        from .synthetic_data import generate_pattern
        pattern_len = args.pattern_length if args.pattern_length else 50
        print(f"Generating random pattern ({pattern_len} bp)...")
        pattern = generate_pattern(pattern_len, seed=42)
    
    print(f"Pattern: {pattern[:50]}{'...' if len(pattern) > 50 else ''}")
    print("-"*70)
    
    # Benchmark
    from .benchmarking import benchmark_kmp_search
    
    kmp = KMP(pattern)
    result = benchmark_kmp_search(kmp, text, num_runs=args.runs)
    
    # Display results
    print(f"\nBenchmark Results:")
    print(f"  Text length:        {result.text_length:,} bp")
    print(f"  Pattern length:     {result.pattern_length} bp")
    print(f"  Matches found:      {result.num_matches}")
    print(f"  Number of runs:     {result.num_runs}")
    print(f"\nTiming:")
    print(f"  Preprocessing:      {format_time(result.preprocessing_time)}")
    print(f"  Mean search time:   {format_time(result.mean_time)}")
    print(f"  Median search time: {format_time(result.median_time)}")
    print(f"  Std deviation:      {format_time(result.std_dev_time)}")
    print(f"  Min time:           {format_time(result.min_time)}")
    print(f"  Max time:           {format_time(result.max_time)}")
    print(f"\nMemory:")
    print(f"  Peak usage:         {format_memory(result.memory_used)}")
    
    throughput = result.text_length / result.mean_time / 1_000_000
    print(f"\nThroughput:           {throughput:.2f} MB/s")
    
    print("="*70)
    return 0


def cmd_compare(args):
    """Handle compare command."""
    print("\n" + "="*70)
    print("KMP vs Python re COMPARISON")
    print("="*70)
    
    # Load text
    if args.file:
        print(f"Loading text from {args.file}...")
        records = read_fasta(Path(args.file), max_sequences=1)
        if not records:
            print("Error: Could not read file")
            return 1
        text = records[0].sequence
    else:
        from .synthetic_data import generate_random_sequence
        text_size = args.text_size if args.text_size else 100000
        print(f"Generating synthetic text ({text_size:,} bp)...")
        text = generate_random_sequence(text_size, seed=42)
    
    # Get pattern
    if args.pattern:
        pattern = args.pattern
    else:
        from .synthetic_data import generate_pattern
        pattern_len = args.pattern_length if args.pattern_length else 50
        print(f"Generating random pattern ({pattern_len} bp)...")
        pattern = generate_pattern(pattern_len, seed=42)
    
    print(f"Text length: {len(text):,} bp")
    print(f"Pattern: {pattern[:50]}{'...' if len(pattern) > 50 else ''}")
    print("-"*70)
    
    # Compare
    result = compare_with_re(text, pattern)
    
    # Display results
    print_comparison_summary(result)
    
    return 0


def cmd_experiments(args):
    """Handle experiments command."""
    runner = ExperimentRunner(seed=args.seed if args.seed else 42)
    
    if args.all:
        # Run all experiments
        runner.run_all_experiments()
    elif args.pattern_length:
        runner.run_pattern_length_experiment()
    elif args.text_size:
        runner.run_text_size_experiment()
    elif args.comparison:
        runner.run_kmp_vs_re_experiment()
    elif args.real_genome:
        dataset = args.dataset if args.dataset else 'ecoli'
        runner.run_real_genome_experiment(dataset)
    elif args.correctness:
        runner.run_correctness_test()
    else:
        print("Please specify an experiment to run (use --all for all experiments)")
        return 1
    
    return 0


def cmd_generate(args):
    """Handle generate command."""
    print("\n" + "="*70)
    print("GENERATE SYNTHETIC DATA")
    print("="*70)
    
    dataset = generate_synthetic_dataset(
        text_length=args.length,
        num_patterns=args.num_patterns,
        pattern_length=args.pattern_length,
        num_injections=args.injections,
        mutation_rate=args.mutations,
        seed=args.seed if args.seed else 42
    )
    
    print(f"Generated synthetic dataset:")
    print(f"  Sequence length:  {len(dataset.sequence):,} bp")
    print(f"  Number of patterns: {len(dataset.patterns)}")
    print(f"  Pattern length:   {dataset.patterns[0] if dataset.patterns else 'N/A'}")
    print(f"  Injections:       {args.injections}")
    print(f"  Mutation rate:    {args.mutations:.2%}")
    
    if args.output:
        from .data_loader import write_fasta
        record = dataset.to_sequence_record("synthetic_generated")
        write_fasta([record], Path(args.output))
        print(f"\nSaved to: {args.output}")
    
    print("="*70)
    return 0


def cmd_download(args):
    """Handle download command."""
    from .datasets.download_datasets import download_dataset, download_all_datasets, list_downloaded_datasets
    
    if args.list:
        list_downloaded_datasets()
    elif args.dataset:
        if args.dataset == 'all':
            download_all_datasets()
        else:
            download_dataset(args.dataset)
    else:
        print("Please specify --dataset or --list")
        return 1
    
    return 0


def cmd_info(args):
    """Handle info command."""
    if args.dataset:
        # Show info about a dataset
        if args.dataset in DATASETS:
            info = DATASETS[args.dataset]
            print("\n" + "="*70)
            print(f"DATASET INFO: {args.dataset}")
            print("="*70)
            print(f"Name:        {info['name']}")
            print(f"Accession:   {info['accession']}")
            print(f"Size:        ~{info['size_mb']} MB")
            print(f"Description: {info['description']}")
            print(f"URL:         {info['url']}")
            print("="*70)
        else:
            print(f"Unknown dataset: {args.dataset}")
            print(f"Available: {list(DATASETS.keys())}")
            return 1
    elif args.file:
        # Show info about a FASTA file
        print(f"\nReading {args.file}...")
        records = read_fasta(Path(args.file))
        
        print("\n" + "="*70)
        print(f"FILE INFO: {args.file}")
        print("="*70)
        print(f"Number of sequences: {len(records)}")
        
        for i, record in enumerate(records[:5]):  # Show first 5
            print(f"\nSequence {i+1}:")
            print(f"  ID:          {record.id}")
            print(f"  Description: {record.description[:70]}...")
            print(f"  Length:      {record.length:,} bp")
            
            stats = sequence_stats(record.sequence)
            print(f"  GC content:  {stats['gc_content']:.2%}")
            print(f"  Base counts: A={stats['base_counts']['A']:,}, "
                  f"C={stats['base_counts']['C']:,}, "
                  f"G={stats['base_counts']['G']:,}, "
                  f"T={stats['base_counts']['T']:,}")
        
        if len(records) > 5:
            print(f"\n... and {len(records) - 5} more sequences")
        
        print("="*70)
    else:
        print("Please specify --dataset or --file")
        return 1
    
    return 0


def cmd_demo(args):
    """Handle demo command."""
    run_quick_demo()
    return 0


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="KMP Algorithm - DNA Sequence Matching",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Search for a pattern in a file
  python -m kmp.cli search --pattern ATCG --file genome.fasta
  
  # Benchmark on E. coli genome
  python -m kmp.cli benchmark --dataset ecoli --pattern-length 50
  
  # Compare KMP vs re
  python -m kmp.cli compare --text-size 100000 --pattern-length 50
  
  # Run all experiments
  python -m kmp.cli experiments --all
  
  # Download datasets
  python -m kmp.cli download --dataset ecoli
  
  # Generate synthetic data
  python -m kmp.cli generate --length 10000 --mutations 0.05 --output synthetic.fasta
  
  # Quick demo
  python -m kmp.cli demo
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Search command
    search_parser = subparsers.add_parser('search', help='Search for a pattern')
    search_parser.add_argument('--pattern', required=True, help='Pattern to search for')
    search_parser.add_argument('--file', help='FASTA file to search in')
    search_parser.add_argument('--text', help='Text to search in (alternative to --file)')
    search_parser.add_argument('--show-positions', action='store_true', help='Show match positions')
    search_parser.add_argument('--max-matches', type=int, help='Maximum matches to display')
    
    # Benchmark command
    bench_parser = subparsers.add_parser('benchmark', help='Benchmark KMP performance')
    bench_parser.add_argument('--dataset', choices=list(DATASETS.keys()), help='Dataset to benchmark on')
    bench_parser.add_argument('--text-size', type=int, help='Size of synthetic text (if not using dataset)')
    bench_parser.add_argument('--pattern', help='Pattern to use')
    bench_parser.add_argument('--pattern-length', type=int, help='Length of random pattern')
    bench_parser.add_argument('--runs', type=int, default=5, help='Number of benchmark runs')
    
    # Compare command
    compare_parser = subparsers.add_parser('compare', help='Compare KMP vs Python re')
    compare_parser.add_argument('--file', help='FASTA file to use')
    compare_parser.add_argument('--text-size', type=int, help='Size of synthetic text')
    compare_parser.add_argument('--pattern', help='Pattern to use')
    compare_parser.add_argument('--pattern-length', type=int, help='Length of random pattern')
    
    # Experiments command
    exp_parser = subparsers.add_parser('experiments', help='Run experiments')
    exp_parser.add_argument('--all', action='store_true', help='Run all experiments')
    exp_parser.add_argument('--pattern-length', action='store_true', help='Pattern length experiment')
    exp_parser.add_argument('--text-size', action='store_true', help='Text size experiment')
    exp_parser.add_argument('--comparison', action='store_true', help='KMP vs re experiment')
    exp_parser.add_argument('--real-genome', action='store_true', help='Real genome experiment')
    exp_parser.add_argument('--correctness', action='store_true', help='Correctness test')
    exp_parser.add_argument('--dataset', choices=list(DATASETS.keys()), help='Dataset for real genome experiment')
    exp_parser.add_argument('--seed', type=int, help='Random seed')
    
    # Generate command
    gen_parser = subparsers.add_parser('generate', help='Generate synthetic data')
    gen_parser.add_argument('--length', type=int, default=10000, help='Sequence length')
    gen_parser.add_argument('--num-patterns', type=int, default=10, help='Number of patterns')
    gen_parser.add_argument('--pattern-length', type=int, default=50, help='Pattern length')
    gen_parser.add_argument('--injections', type=int, default=20, help='Number of pattern injections')
    gen_parser.add_argument('--mutations', type=float, default=0.0, help='Mutation rate (0.0-1.0)')
    gen_parser.add_argument('--output', help='Output FASTA file')
    gen_parser.add_argument('--seed', type=int, help='Random seed')
    
    # Download command
    dl_parser = subparsers.add_parser('download', help='Download datasets')
    dl_parser.add_argument('--dataset', choices=list(DATASETS.keys()) + ['all'], help='Dataset to download')
    dl_parser.add_argument('--list', action='store_true', help='List downloaded datasets')
    
    # Info command
    info_parser = subparsers.add_parser('info', help='Show information')
    info_parser.add_argument('--dataset', choices=list(DATASETS.keys()), help='Show dataset info')
    info_parser.add_argument('--file', help='Show FASTA file info')
    
    # Demo command
    demo_parser = subparsers.add_parser('demo', help='Run quick demonstration')
    
    # Parse arguments
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 0
    
    # Execute command
    command_map = {
        'search': cmd_search,
        'benchmark': cmd_benchmark,
        'compare': cmd_compare,
        'experiments': cmd_experiments,
        'generate': cmd_generate,
        'download': cmd_download,
        'info': cmd_info,
        'demo': cmd_demo,
    }
    
    try:
        return command_map[args.command](args)
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        return 130
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
