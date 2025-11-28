"""
Shift-Or/Bitap Algorithm Implementation
=======================================

Core algorithm and data loading modules for the Shift-Or/Bitap
pattern matching algorithm applied to DNA sequence analysis.
"""

from .algorithm import ShiftOrBitap
from .data_loader import DataLoader, SyntheticDataGenerator
from . import utils

__all__ = ['ShiftOrBitap', 'DataLoader', 'SyntheticDataGenerator', 'utils']
