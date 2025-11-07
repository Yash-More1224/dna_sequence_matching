"""
KMP (Knuth-Morris-Pratt) Algorithm Implementation for DNA Sequence Matching

This package provides a complete implementation of the KMP string matching algorithm
optimized for DNA sequences, along with benchmarking, visualization, and analysis tools.

Main Components:
- kmp_algorithm: Core KMP implementation with LPS array construction
- data_loader: FASTA/FASTQ file readers and dataset management
- synthetic_data: Synthetic DNA sequence generator with mutations
- benchmarking: Performance measurement utilities
- evaluation: Accuracy metrics and comparison with re module
- visualization: Plotting and match highlighting
- experiments: Experiment orchestration
- cli: Command-line interface

Author: DNA Sequence Matching Project
Date: November 2025
"""

__version__ = "1.0.0"
__author__ = "DNA Sequence Matching Project"

from .kmp_algorithm import KMP, kmp_search, build_lps_array

__all__ = [
    'KMP',
    'kmp_search',
    'build_lps_array',
]
