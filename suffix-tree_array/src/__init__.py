"""
Suffix Array Implementation for DNA Pattern Matching

This package provides a complete implementation of Suffix Array + LCP array
for efficient exact pattern matching in DNA sequences.
"""

from .suffix_array import SuffixArray
from .data_generator import DNAGenerator
from .utils import format_time, format_memory, save_json, save_csv

# Optional imports that require BioPython
try:
    from .data_loader import DatasetManager, load_sequence_from_file
    __all__ = [
        'SuffixArray',
        'DatasetManager',
        'load_sequence_from_file',
        'DNAGenerator',
        'format_time',
        'format_memory',
        'save_json',
        'save_csv'
    ]
except ImportError:
    __all__ = [
        'SuffixArray',
        'DNAGenerator',
        'format_time',
        'format_memory',
        'save_json',
        'save_csv'
    ]

__version__ = '1.0.0'
