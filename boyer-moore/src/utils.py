"""Utility functions for Boyer-Moore implementation."""


__all__ = []import time

import tracemalloc
from typing import Callable, Dict, Any, List
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


def format_time(seconds: float) -> str:
    """
    Format time in human-readable form.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted string (e.g., "1.23 ms")
    """
    if seconds < 1e-6:
        return f"{seconds * 1e9:.2f} ns"
    elif seconds < 1e-3:
        return f"{seconds * 1e6:.2f} µs"
    elif seconds < 1:
        return f"{seconds * 1e3:.2f} ms"
    else:
        return f"{seconds:.2f} s"


def format_size(bytes_: int) -> str:
    """
    Format byte size in human-readable form.
    
    Args:
        bytes_: Size in bytes
        
    Returns:
        Formatted string (e.g., "1.23 MB")
    """
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_ < 1024.0:
            return f"{bytes_:.2f} {unit}"
        bytes_ /= 1024.0
    return f"{bytes_:.2f} TB"


def calculate_gc_content(sequence: str) -> float:
    """
    Calculate GC content of DNA sequence.
    
    Args:
        sequence: DNA sequence
        
    Returns:
        GC content as fraction (0.0 to 1.0)
    """
    sequence = sequence.upper()
    gc_count = sequence.count('G') + sequence.count('C')
    total = len(sequence)
    
    return gc_count / total if total > 0 else 0.0


def reverse_complement(sequence: str) -> str:
    """
    Get reverse complement of DNA sequence.
    
    Args:
        sequence: DNA sequence
        
    Returns:
        Reverse complement
    """
    complement = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C', 'N': 'N'}
    return ''.join(complement.get(base, 'N') for base in reversed(sequence.upper()))


def count_nucleotides(sequence: str) -> Dict[str, int]:
    """
    Count occurrences of each nucleotide.
    
    Args:
        sequence: DNA sequence
        
    Returns:
        Dictionary with counts
    """
    sequence = sequence.upper()
    return {
        'A': sequence.count('A'),
        'C': sequence.count('C'),
        'G': sequence.count('G'),
        'T': sequence.count('T'),
        'N': sequence.count('N'),
        'Other': len(sequence) - sum([
            sequence.count(n) for n in ['A', 'C', 'G', 'T', 'N']
        ])
    }


def validate_dna_sequence(sequence: str, allow_n: bool = True) -> bool:
    """
    Check if sequence is valid DNA.
    
    Args:
        sequence: DNA sequence to validate
        allow_n: Whether to allow 'N' (ambiguous base)
        
    Returns:
        True if valid, False otherwise
    """
    valid_chars = set('ACGT')
    if allow_n:
        valid_chars.add('N')
    
    return all(c in valid_chars for c in sequence.upper())


def sliding_window(sequence: str, window_size: int, step: int = 1):
    """
    Generate sliding windows over sequence.
    
    Args:
        sequence: Input sequence
        window_size: Size of each window
        step: Step size between windows
        
    Yields:
        Tuples of (position, window_sequence)
    """
    for i in range(0, len(sequence) - window_size + 1, step):
        yield i, sequence[i:i + window_size]


def find_repeats(sequence: str, min_length: int = 10) -> Dict[str, List[int]]:
    """
    Find repeated subsequences.
    
    Args:
        sequence: DNA sequence
        min_length: Minimum length of repeats to find
        
    Returns:
        Dictionary mapping repeat sequences to their positions
    """
    repeats = {}
    
    for length in range(min_length, len(sequence) // 2 + 1):
        seen = {}
        
        for pos, window in sliding_window(sequence, length):
            if window in seen:
                if window not in repeats:
                    repeats[window] = [seen[window]]
                repeats[window].append(pos)
            else:
                seen[window] = pos
    
    return repeats


class Timer:
    """Context manager for timing code blocks."""
    
    def __init__(self, name: str = "Operation"):
        """
        Initialize timer.
        
        Args:
            name: Name of operation being timed
        """
        self.name = name
        self.start_time = None
        self.elapsed = None
    
    def __enter__(self):
        """Start timer."""
        self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, *args):
        """Stop timer and print elapsed time."""
        self.elapsed = time.perf_counter() - self.start_time
        print(f"{self.name} took {format_time(self.elapsed)}")


if __name__ == "__main__":
    # Test utilities
    print("Utility Functions Test")
    print("=" * 50)
    
    # Test sequence
    seq = "ACGTACGTACGT"
    
    print(f"\nSequence: {seq}")
    print(f"GC Content: {calculate_gc_content(seq) * 100:.1f}%")
    print(f"Reverse Complement: {reverse_complement(seq)}")
    print(f"Nucleotide Counts: {count_nucleotides(seq)}")
    print(f"Is Valid DNA: {validate_dna_sequence(seq)}")
    
    # Test timing
    print("\n=== Timing Test ===")
    with Timer("Test operation"):
        time.sleep(0.1)
    
    # Test formatting
    print("\n=== Formatting Test ===")
    print(f"1234567 bytes = {format_size(1234567)}")
    print(f"0.001234 seconds = {format_time(0.001234)}")
