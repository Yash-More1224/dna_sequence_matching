"""
Main entry point for running KMP algorithm experiments.

This script provides a simple way to run all experiments and generate
comprehensive results and visualizations.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from kmp.experiments import ExperimentRunner, run_quick_demo
from kmp.config import RESULTS_DIR


def main():
    """Main function to run all experiments."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Run KMP Algorithm Experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--demo',
        action='store_true',
        help='Run quick demonstration'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    parser.add_argument(
        '--results-dir',
        type=Path,
        default=RESULTS_DIR,
        help=f'Directory to save results (default: {RESULTS_DIR})'
    )
    
    args = parser.parse_args()
    
    if args.demo:
        # Run quick demo
        run_quick_demo()
    else:
        # Run full experiment suite
        print("\n" + "#"*70)
        print("# KMP ALGORITHM - COMPREHENSIVE EXPERIMENT SUITE")
        print("#"*70)
        print(f"\nRandom seed: {args.seed}")
        print(f"Results directory: {args.results_dir}")
        print("\nThis will run all experiments and may take several minutes...")
        print("You can interrupt at any time with Ctrl+C")
        
        try:
            input("\nPress Enter to continue or Ctrl+C to cancel...")
        except KeyboardInterrupt:
            print("\n\nCancelled by user")
            return 1
        
        # Create experiment runner and run all experiments
        runner = ExperimentRunner(results_dir=args.results_dir, seed=args.seed)
        results = runner.run_all_experiments()
        
        print("\n" + "#"*70)
        print("# EXPERIMENTS COMPLETED SUCCESSFULLY!")
        print("#"*70)
        print(f"\nResults have been saved to: {args.results_dir}")
        print("\nGenerated files:")
        print(f"  - Benchmark data: {args.results_dir}/benchmarks/")
        print(f"  - Visualizations: {args.results_dir}/plots/")
        print(f"  - Reports:        {args.results_dir}/reports/")
        print("\n" + "#"*70)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
