#!/usr/bin/env python3
"""
Quick start script for KMP algorithm demonstration.

This script provides an easy entry point for new users to see the KMP algorithm in action.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from kmp.experiments import run_quick_demo

if __name__ == '__main__':
    print("\n" + "="*70)
    print("  WELCOME TO KMP ALGORITHM - DNA SEQUENCE MATCHING")
    print("="*70)
    print("\nThis is a quick demonstration of the KMP algorithm.")
    print("For full functionality, see README.md or run: python -m kmp.cli --help")
    print("\n" + "="*70)
    
    run_quick_demo()
    
    print("\n" + "="*70)
    print("  NEXT STEPS:")
    print("="*70)
    print("\n1. Download datasets:")
    print("   python -m kmp.cli download --dataset ecoli")
    print("\n2. Run benchmarks:")
    print("   python -m kmp.cli benchmark --dataset ecoli")
    print("\n3. Run all experiments:")
    print("   python -m kmp.run_experiments")
    print("\n4. View help:")
    print("   python -m kmp.cli --help")
    print("\n" + "="*70 + "\n")
