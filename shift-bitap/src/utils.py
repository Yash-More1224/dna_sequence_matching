"""
Utility Functions for Shift-Or/Bitap Implementation
===================================================

This module provides utility functions for timing, memory measurement,
and other helper functions used in the evaluation.
"""

import time
import tracemalloc
import psutil
import os
from typing import Callable, Dict, Any, List, Tuple
from functools import wraps


def measure_time(func: Callable) -> Callable:
    """
    Decorator to measure function execution time.
    
    Args:
        func: Function to measure
        
    Returns:
        Wrapped function that returns (result, elapsed_time)
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        elapsed = end_time - start_time
        return result, elapsed
    
    return wrapper


def measure_memory(func: Callable) -> Callable:
    """
    Decorator to measure peak memory usage.
    
    Args:
        func: Function to measure
        
    Returns:
        Wrapped function that returns (result, peak_memory_bytes)
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        tracemalloc.start()
        result = func(*args, **kwargs)
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        return result, peak
    
    return wrapper


def get_memory_usage() -> Dict[str, float]:
    """
    Get current memory usage statistics.
    
    Returns:
        Dictionary with memory usage in MB
    """
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    
    return {
        'rss_mb': mem_info.rss / (1024 * 1024),  # Resident Set Size
        'vms_mb': mem_info.vms / (1024 * 1024),  # Virtual Memory Size
    }


def format_time(seconds: float) -> str:
    """
    Format time in seconds to human-readable string.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted time string
    """
    if seconds < 0.001:
        return f"{seconds * 1_000_000:.2f} μs"
    elif seconds < 1:
        return f"{seconds * 1000:.2f} ms"
    else:
        return f"{seconds:.2f} s"


def format_size(bytes: int) -> str:
    """
    Format bytes to human-readable string.
    
    Args:
        bytes: Size in bytes
        
    Returns:
        Formatted size string
    """
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes < 1024.0:
            return f"{bytes:.2f} {unit}"
        bytes /= 1024.0
    return f"{bytes:.2f} TB"


def validate_dna_sequence(sequence: str) -> bool:
    """
    Validate that a sequence contains only valid DNA nucleotides.
    
    Args:
        sequence: DNA sequence string
        
    Returns:
        True if valid, False otherwise
    """
    valid_chars = set('ACGTN')  # N for unknown nucleotide
    return all(c.upper() in valid_chars for c in sequence)


def calculate_gc_content(sequence: str) -> float:
    """
    Calculate GC content of a DNA sequence.
    
    Args:
        sequence: DNA sequence string
        
    Returns:
        GC content as percentage (0-100)
    """
    sequence = sequence.upper()
    gc_count = sequence.count('G') + sequence.count('C')
    total = len(sequence)
    
    if total == 0:
        return 0.0
    
    return (gc_count / total) * 100


def batch_process(items: List[Any], batch_size: int = 100) -> List[List[Any]]:
    """
    Split items into batches for processing.
    
    Args:
        items: List of items to batch
        batch_size: Size of each batch
        
    Returns:
        List of batches
    """
    batches = []
    for i in range(0, len(items), batch_size):
        batches.append(items[i:i + batch_size])
    return batches


def timing_stats(times: List[float]) -> Dict[str, float]:
    """
    Calculate timing statistics from multiple runs.
    
    Args:
        times: List of timing measurements
        
    Returns:
        Dictionary with mean, median, min, max, std
    """
    import statistics
    
    if not times:
        return {'mean': 0, 'median': 0, 'min': 0, 'max': 0, 'std': 0}
    
    return {
        'mean': statistics.mean(times),
        'median': statistics.median(times),
        'min': min(times),
        'max': max(times),
        'std': statistics.stdev(times) if len(times) > 1 else 0
    }


__all__ = [
    'measure_time',
    'measure_memory',
    'get_memory_usage',
    'format_time',
    'format_size',
    'validate_dna_sequence',
    'calculate_gc_content',
    'batch_process',
    'timing_stats',
]
