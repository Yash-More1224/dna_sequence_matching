#!/usr/bin/env python3
"""
Main Experiment Runner

Run all Boyer-Moore experiments and generate reports.
"""

import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from experiments.experiments import ExperimentRunner
from visualization.visualizations import Visualizer
from visualization.report_generator import ReportGenerator


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Run Boyer-Moore Algorithm Experiments"
    )
    
    parser.add_argument(
        '--experiments',
        nargs='+',
        type=int,
        choices=range(1, 9),
        help='Specific experiments to run (1-8). If not specified, runs all.'
    )
    
    parser.add_argument(
        '--skip-visualizations',
        action='store_true',
        help='Skip generating visualizations'
    )
    
    parser.add_argument(
        '--skip-report',
        action='store_true',
        help='Skip generating final report'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results',
        help='Output directory for results (default: results)'
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print(" " * 15 + "BOYER-MOORE EXPERIMENTAL ANALYSIS")
    print("=" * 70)
    print()
    
    # Initialize runner
    runner = ExperimentRunner(output_dir=args.output_dir)
    
    # Run experiments
    if args.experiments:
        print(f"Running experiments: {args.experiments}")
        experiment_map = {
            1: runner.experiment_1_pattern_length,
            2: runner.experiment_2_text_scaling,
            3: runner.experiment_3_alphabet_effect,
            4: runner.experiment_4_heuristic_contribution,
            5: runner.experiment_5_preprocessing_overhead,
            6: runner.experiment_6_memory_footprint,
            7: runner.experiment_7_real_motifs,
            8: runner.experiment_8_compare_with_re
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
        print("Running all experiments...")
        try:
            runner.run_all_experiments()
        except Exception as e:
            print(f"❌ Error running experiments: {e}")
            import traceback
            traceback.print_exc()
            return 1
    
    # Generate visualizations
    if not args.skip_visualizations:
        print("\n" + "=" * 70)
        print("Generating Visualizations...")
        print("=" * 70)
        try:
            viz = Visualizer(output_dir=f"{args.output_dir}/plots")
            viz.create_all_plots()
        except Exception as e:
            print(f"❌ Error generating visualizations: {e}")
            import traceback
            traceback.print_exc()
    
    # Generate report
    if not args.skip_report:
        print("\n" + "=" * 70)
        print("Generating Final Report...")
        print("=" * 70)
        try:
            report_gen = ReportGenerator(results_dir=args.output_dir)
            report_gen.generate_full_report()
            report_gen.generate_summary_report()
        except Exception as e:
            print(f"❌ Error generating report: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 70)
    print(" " * 20 + "✓ ALL TASKS COMPLETE")
    print("=" * 70)
    print(f"\nResults saved to: {args.output_dir}/")
    print(f"  - Tables: {args.output_dir}/tables/")
    print(f"  - Plots: {args.output_dir}/plots/")
    print(f"  - Reports: {args.output_dir}/reports/")
    print()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
