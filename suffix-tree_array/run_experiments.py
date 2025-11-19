#!/usr/bin/env python3
"""
Main Experiment Runner for Suffix Array Implementation

Run all experiments and generate comprehensive reports.
"""

import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from experiments.experiments import ExperimentRunner


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Run Suffix Array Algorithm Experiments"
    )
    
    parser.add_argument(
        '--experiments',
        nargs='+',
        type=int,
        choices=range(1, 9),
        help='Specific experiments to run (1-8). If not specified, runs all.'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results',
        help='Output directory for results (default: results)'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Configuration file (default: config.yaml)'
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print(" " * 15 + "SUFFIX ARRAY EXPERIMENTAL ANALYSIS")
    print("=" * 70)
    print()
    
    # Initialize runner
    runner = ExperimentRunner(config_file=args.config)
    
    # Run experiments
    if args.experiments:
        print(f"Running experiments: {args.experiments}\n")
        experiment_map = {
            1: runner.experiment_1_pattern_length,
            2: runner.experiment_2_text_scaling,
            3: runner.experiment_3_preprocessing_cost,
            4: runner.experiment_4_memory_footprint,
            5: runner.experiment_5_compare_with_re,
            6: runner.experiment_6_repeat_discovery,
            7: runner.experiment_7_ecoli_genome,
            8: runner.experiment_8_pattern_complexity
        }
        
        for exp_num in args.experiments:
            try:
                print(f"\n{'='*60}")
                print(f"Running Experiment {exp_num}")
                print('='*60)
                experiment_map[exp_num]()
            except Exception as e:
                print(f"❌ Error in Experiment {exp_num}: {e}")
                import traceback
                traceback.print_exc()
    else:
        print("Running all experiments...\n")
        try:
            runner.run_all_experiments()
        except Exception as e:
            print(f"❌ Error running experiments: {e}")
            import traceback
            traceback.print_exc()
            return 1
    
    print("\n" + "=" * 70)
    print("✓ EXPERIMENTS COMPLETED")
    print("=" * 70)
    print(f"\nResults saved to: {runner.tables_dir}")
    print("\nNext steps:")
    print("  - Check results/tables/ for data files")
    print("  - Run visualizations (if implemented)")
    print("  - See README.md for more information")
    print()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
