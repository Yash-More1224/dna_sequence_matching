"""
Utility functions for the KMP DNA sequence matching project.

This module provides helper functions for string manipulation, file I/O,
timing, memory profiling, and other common tasks.
"""

import time
import tracemalloc
import functools
import random
import json
import csv
from typing import List, Dict, Any, Callable, Tuple, Generator
from pathlib import Path
import sys

from .config import DNA_BASES, DNA_COMPLEMENT


def reverse_complement(sequence: str) -> str:
    """
    Calculate the reverse complement of a DNA sequence.
    
    Args:
        sequence: DNA sequence string
        
    Returns:
        Reverse complement of the sequence
        
    Example:
        >>> reverse_complement("ATCG")
        'CGAT'
    """
    return ''.join(DNA_COMPLEMENT.get(base, 'N') for base in reversed(sequence.upper()))


def validate_dna_sequence(sequence: str, allow_n: bool = True) -> bool:
    """
    Validate that a sequence contains only valid DNA bases.
    
    Args:
        sequence: DNA sequence to validate
        allow_n: Whether to allow 'N' (unknown base)
        
    Returns:
        True if sequence is valid, False otherwise
    """
    if not sequence:
        return False
    
    valid_bases = set(DNA_BASES + (['N'] if allow_n else []))
    return all(base.upper() in valid_bases for base in sequence)


def generate_random_dna(length: int, seed: int = None) -> str:
    """
    Generate a random DNA sequence.
    
    Args:
        length: Length of sequence to generate
        seed: Random seed for reproducibility
        
    Returns:
        Random DNA sequence
    """
    if seed is not None:
        random.seed(seed)
    
    return ''.join(random.choice(DNA_BASES) for _ in range(length))


def format_time(seconds: float) -> str:
    """
    Format time in seconds to a human-readable string.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted time string
    """
    if seconds < 0.001:
        return f"{seconds * 1_000_000:.2f} μs"
    elif seconds < 1.0:
        return f"{seconds * 1000:.2f} ms"
    elif seconds < 60:
        return f"{seconds:.2f} s"
    else:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.2f}s"


def format_memory(bytes_size: int) -> str:
    """
    Format memory size in bytes to a human-readable string.
    
    Args:
        bytes_size: Memory size in bytes
        
    Returns:
        Formatted memory string
    """
    if bytes_size < 1024:
        return f"{bytes_size} B"
    elif bytes_size < 1024 ** 2:
        return f"{bytes_size / 1024:.2f} KB"
    elif bytes_size < 1024 ** 3:
        return f"{bytes_size / (1024 ** 2):.2f} MB"
    else:
        return f"{bytes_size / (1024 ** 3):.2f} GB"


def time_function(func: Callable) -> Callable:
    """
    Decorator to measure function execution time.
    
    Args:
        func: Function to time
        
    Returns:
        Wrapped function that prints execution time
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        print(f"{func.__name__} took {format_time(end - start)}")
        return result
    return wrapper


def measure_time(func: Callable, *args, **kwargs) -> Tuple[Any, float]:
    """
    Measure execution time of a function.
    
    Args:
        func: Function to measure
        *args: Positional arguments for func
        **kwargs: Keyword arguments for func
        
    Returns:
        Tuple of (function_result, elapsed_time_seconds)
    """
    start = time.perf_counter()
    result = func(*args, **kwargs)
    end = time.perf_counter()
    return result, end - start


def measure_memory(func: Callable, *args, **kwargs) -> Tuple[Any, int]:
    """
    Measure peak memory usage of a function.
    
    Args:
        func: Function to measure
        *args: Positional arguments for func
        **kwargs: Keyword arguments for func
        
    Returns:
        Tuple of (function_result, peak_memory_bytes)
    """
    tracemalloc.start()
    result = func(*args, **kwargs)
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return result, peak


def save_json(data: Any, filepath: Path) -> None:
    """
    Save data to a JSON file.
    
    Args:
        data: Data to save (must be JSON serializable)
        filepath: Path to save file
    """
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)


def load_json(filepath: Path) -> Any:
    """
    Load data from a JSON file.
    
    Args:
        filepath: Path to JSON file
        
    Returns:
        Loaded data
    """
    with open(filepath, 'r') as f:
        return json.load(f)


def save_csv(data: List[Dict[str, Any]], filepath: Path) -> None:
    """
    Save list of dictionaries to a CSV file.
    
    Args:
        data: List of dictionaries with consistent keys
        filepath: Path to save file
    """
    if not data:
        return
    
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=data[0].keys())
        writer.writeheader()
        writer.writerows(data)


def load_csv(filepath: Path) -> List[Dict[str, Any]]:
    """
    Load CSV file into list of dictionaries.
    
    Args:
        filepath: Path to CSV file
        
    Returns:
        List of dictionaries
    """
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        return list(reader)


def print_progress(current: int, total: int, prefix: str = '', suffix: str = '', 
                  bar_length: int = 50) -> None:
    """
    Print a progress bar to console.
    
    Args:
        current: Current progress value
        total: Total value
        prefix: Text before progress bar
        suffix: Text after progress bar
        bar_length: Length of progress bar in characters
    """
    percent = 100 * (current / float(total))
    filled = int(bar_length * current // total)
    bar = '█' * filled + '-' * (bar_length - filled)
    
    sys.stdout.write(f'\r{prefix} |{bar}| {percent:.1f}% {suffix}')
    sys.stdout.flush()
    
    if current == total:
        print()


def chunks(lst: List[Any], n: int) -> Generator[List[Any], None, None]:
    """
    Split a list into chunks of size n.
    
    Args:
        lst: List to split
        n: Chunk size
        
    Yields:
        Chunks of the list
    """
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def flatten(lst: List[List[Any]]) -> List[Any]:
    """
    Flatten a list of lists into a single list.
    
    Args:
        lst: List of lists
        
    Returns:
        Flattened list
    """
    return [item for sublist in lst for item in sublist]


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Safely divide two numbers, returning default if denominator is zero.
    
    Args:
        numerator: Numerator
        denominator: Denominator
        default: Value to return if denominator is zero
        
    Returns:
        Result of division or default
    """
    return numerator / denominator if denominator != 0 else default


def ensure_str(s: Any) -> str:
    """
    Ensure input is a string, converting if necessary.
    
    Args:
        s: Input value
        
    Returns:
        String representation
    """
    if isinstance(s, str):
        return s
    elif isinstance(s, bytes):
        return s.decode('utf-8')
    else:
        return str(s)
