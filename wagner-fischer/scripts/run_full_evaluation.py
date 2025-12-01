#!/usr/bin/env python3


import subprocess
import sys
import os
from pathlib import Path


def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"\n{'='*80}")
    print(f"{description}")
    print(f"{'='*80}")
    print(f"Command: {' '.join(cmd)}")
    print()
    
    result = subprocess.run(cmd, capture_output=False, text=True)
    
    if result.returncode != 0:
        print(f"ERROR: {description} failed with code {result.returncode}")
        sys.exit(1)
    
    print(f"✓ {description} completed successfully")
    return result


def main():
    """Run complete evaluation pipeline."""
    
    # Configuration
    BASE_DIR = Path(__file__).parent.parent
    SCRIPTS_DIR = BASE_DIR / "scripts"
    DATASET_DIR = BASE_DIR.parent / "dataset"
    RESULTS_DIR = BASE_DIR / "results"
    
    # Create results directory
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Dataset files
    ECOLI_FASTA = DATASET_DIR / "ecoli_k12_mg1655.fasta"
    LAMBDA_FASTA = DATASET_DIR / "lambda_phage.fasta"
    SALMONELLA_FASTA = DATASET_DIR / "salmonella_typhimurium.fasta"
    
    # Select primary dataset (E. coli)
    PRIMARY_FASTA = ECOLI_FASTA
    GT_FILE = RESULTS_DIR / "ground_truth.json"
    
    print("=" * 80)
    print("WAGNER-FISCHER COMPREHENSIVE EVALUATION PIPELINE")
    print("=" * 80)
    print(f"\nBase Directory: {BASE_DIR}")
    print(f"Scripts Directory: {SCRIPTS_DIR}")
    print(f"Dataset Directory: {DATASET_DIR}")
    print(f"Results Directory: {RESULTS_DIR}")
    print(f"\nPrimary Dataset: {PRIMARY_FASTA}")
    
    # Check if dataset exists
    if not PRIMARY_FASTA.exists():
        print(f"\nERROR: Dataset not found: {PRIMARY_FASTA}")
        print("Please ensure FASTA files are in the dataset directory.")
        sys.exit(1)
    
    # Step 1: Generate Ground Truth
    print("\n" + "=" * 80)
    print("STEP 1: GENERATING GROUND TRUTH")
    print("=" * 80)
    
    gt_cmd = [
        sys.executable,
        str(SCRIPTS_DIR / "generate_ground_truth.py"),
        "--fasta", str(PRIMARY_FASTA),
        "--output", str(GT_FILE),
        "--num-patterns", "50",
        "--pattern-length", "30",
        "--mutation-rates", "0.0", "0.01", "0.02", "0.05", "0.1",
        "--seed", "42"
    ]
    
    run_command(gt_cmd, "Ground Truth Generation")
    
    # Step 2: Run Benchmarks
    print("\n" + "=" * 80)
    print("STEP 2: RUNNING BENCHMARKS")
    print("=" * 80)
    
    benchmark_cmd = [
        sys.executable,
        str(SCRIPTS_DIR / "fast_benchmark.py"),
        "--ground-truth", str(GT_FILE),
        "--output-dir", str(RESULTS_DIR),
        "--threshold", "3",
        "--pattern-counts", "10", "20", "50", "100", "200"
    ]
    
    run_command(benchmark_cmd, "Benchmark Execution")
    
    # Step 3: Generate Visualizations and Report
    print("\n" + "=" * 80)
    print("STEP 3: GENERATING VISUALIZATIONS AND REPORT")
    print("=" * 80)
    
    viz_cmd = [
        sys.executable,
        str(SCRIPTS_DIR / "generate_visualizations.py"),
        "--results-dir", str(RESULTS_DIR),
        "--metrics-file", "metrics.json"
    ]
    
    run_command(viz_cmd, "Visualization and Report Generation")
    
    # Summary
    print("\n" + "=" * 80)
    print("EVALUATION PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 80)
    print("\nGenerated Files:")
    print(f"  ├─ {GT_FILE}")
    print(f"  ├─ {RESULTS_DIR / 'metrics.json'}")
    print(f"  ├─ {RESULTS_DIR / 'wf_evaluation_report.txt'}")
    print(f"  ├─ {RESULTS_DIR / 'accuracy_vs_mutation_rate.png'}")
    print(f"  ├─ {RESULTS_DIR / 'scalability.png'}")
    print(f"  ├─ {RESULTS_DIR / 'robustness.png'}")
    print(f"  └─ {RESULTS_DIR / 'performance_summary.png'}")
    
    print("\n✓ All experiments completed and results saved!")
    print(f"\nView the full report at: {RESULTS_DIR / 'wf_evaluation_report.txt'}")


if __name__ == '__main__':
    main()
