#!/usr/bin/env python3
"""
Master Script to Run Complete Boyer-Moore Evaluation

This script orchestrates:
1. Comprehensive evaluation on all three datasets
2. Generation of visualizations
3. Creation of final report

Run this single script to reproduce all experiments.
"""

import sys
import subprocess
from pathlib import Path


def run_script(script_path: Path, description: str) -> bool:
    """
    Run a Python script and report results.
    
    Args:
        script_path: Path to script
        description: Description of what the script does
        
    Returns:
        True if successful, False otherwise
    """
    print("\n" + "="*80)
    print(f"RUNNING: {description}")
    print("="*80)
    print(f"Script: {script_path}")
    print()
    
    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            check=True,
            cwd=script_path.parent,
            capture_output=False
        )
        
        print(f"\n✓ {description} completed successfully")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error running {description}")
        print(f"Exit code: {e.returncode}")
        return False
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        return False


def main():
    """Main orchestration function."""
    print("\n" + "="*80)
    print(" " * 20 + "BOYER-MOORE COMPLETE EVALUATION")
    print("="*80)
    print("\nThis script will:")
    print("1. Run comprehensive evaluation on 3 datasets")
    print("2. Generate all visualizations")
    print("3. Create final comprehensive report")
    print("\nEstimated time: 5-15 minutes depending on hardware")
    print("="*80)
    
    input("\nPress Enter to continue or Ctrl+C to cancel...")
    
    script_dir = Path(__file__).parent
    
    # Define scripts to run
    scripts = [
        (script_dir / "comprehensive_evaluation.py", 
         "Comprehensive Evaluation on All Datasets"),
        (script_dir / "generate_visualizations.py", 
         "Visualization Generation"),
    ]
    
    results = []
    
    # Run each script
    for script_path, description in scripts:
        if not script_path.exists():
            print(f"\n❌ Script not found: {script_path}")
            results.append(False)
            continue
        
        success = run_script(script_path, description)
        results.append(success)
    
    # Print final summary
    print("\n" + "="*80)
    print(" " * 25 + "FINAL SUMMARY")
    print("="*80)
    
    for (script_path, description), success in zip(scripts, results):
        status = "✓ SUCCESS" if success else "❌ FAILED"
        print(f"{status:12} - {description}")
    
    # Overall status
    all_success = all(results)
    
    print("\n" + "="*80)
    if all_success:
        print("✓ ALL STEPS COMPLETED SUCCESSFULLY")
        print("\nResults are available in:")
        print(f"  - {script_dir / 'results' / 'comprehensive_evaluation_report.txt'}")
        print(f"  - {script_dir / 'results' / 'plots'} (visualizations)")
        print(f"  - {script_dir / 'results'} (JSON data)")
    else:
        print("❌ SOME STEPS FAILED")
        print("\nPlease check the error messages above.")
    
    print("="*80 + "\n")
    
    return all_success


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n❌ Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
