"""
Utility Functions

Common utility functions for formatting, file I/O, and data manipulation.
"""

import json
import csv
from pathlib import Path
from typing import Any, Dict, List


def format_time(seconds: float) -> str:
    """
    Format time in human-readable format.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted string
    """
    if seconds < 1e-6:
        return f"{seconds * 1e9:.2f} ns"
    elif seconds < 1e-3:
        return f"{seconds * 1e6:.2f} μs"
    elif seconds < 1:
        return f"{seconds * 1e3:.2f} ms"
    elif seconds < 60:
        return f"{seconds:.2f} s"
    else:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.2f}s"


def format_memory(bytes_val: int) -> str:
    """
    Format memory size in human-readable format.
    
    Args:
        bytes_val: Memory in bytes
        
    Returns:
        Formatted string
    """
    if bytes_val < 1024:
        return f"{bytes_val} B"
    elif bytes_val < 1024**2:
        return f"{bytes_val / 1024:.2f} KB"
    elif bytes_val < 1024**3:
        return f"{bytes_val / (1024**2):.2f} MB"
    else:
        return f"{bytes_val / (1024**3):.2f} GB"


def format_throughput(bytes_per_second: float) -> str:
    """
    Format throughput in human-readable format.
    
    Args:
        bytes_per_second: Throughput in bytes/second
        
    Returns:
        Formatted string
    """
    if bytes_per_second < 1024:
        return f"{bytes_per_second:.2f} B/s"
    elif bytes_per_second < 1024**2:
        return f"{bytes_per_second / 1024:.2f} KB/s"
    elif bytes_per_second < 1024**3:
        return f"{bytes_per_second / (1024**2):.2f} MB/s"
    else:
        return f"{bytes_per_second / (1024**3):.2f} GB/s"


def save_json(data: Any, filepath: str, indent: int = 2):
    """
    Save data to JSON file.
    
    Args:
        data: Data to save
        filepath: Output file path
        indent: JSON indentation level
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=indent)
    
    print(f"✓ Saved JSON to {filepath}")


def load_json(filepath: str) -> Any:
    """
    Load data from JSON file.
    
    Args:
        filepath: Input file path
        
    Returns:
        Loaded data
    """
    with open(filepath, 'r') as f:
        return json.load(f)


def save_csv(data: List[Dict[str, Any]], filepath: str):
    """
    Save list of dictionaries to CSV file.
    
    Args:
        data: List of dictionaries with same keys
        filepath: Output file path
    """
    if not data:
        print("Warning: No data to save")
        return
    
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    # Get fieldnames from first dictionary
    fieldnames = list(data[0].keys())
    
    with open(filepath, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)
    
    print(f"✓ Saved CSV to {filepath}")


def load_csv(filepath: str) -> List[Dict[str, Any]]:
    """
    Load CSV file as list of dictionaries.
    
    Args:
        filepath: Input file path
        
    Returns:
        List of dictionaries
    """
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        return list(reader)


def print_table(headers: List[str], rows: List[List[Any]], 
                col_widths: List[int] = None):
    """
    Print formatted table.
    
    Args:
        headers: Column headers
        rows: Table rows
        col_widths: Column widths (auto-calculated if None)
    """
    if col_widths is None:
        # Calculate column widths
        col_widths = [len(h) for h in headers]
        for row in rows:
            for i, cell in enumerate(row):
                col_widths[i] = max(col_widths[i], len(str(cell)))
    
    # Print header
    header_line = " | ".join(
        str(h).ljust(w) for h, w in zip(headers, col_widths)
    )
    print(header_line)
    print("-" * len(header_line))
    
    # Print rows
    for row in rows:
        row_line = " | ".join(
            str(cell).ljust(w) for cell, w in zip(row, col_widths)
        )
        print(row_line)


def print_section(title: str, width: int = 70):
    """
    Print a formatted section header.
    
    Args:
        title: Section title
        width: Total width
    """
    print("\n" + "=" * width)
    print(title.center(width))
    print("=" * width)


def print_subsection(title: str, width: int = 70):
    """
    Print a formatted subsection header.
    
    Args:
        title: Subsection title
        width: Total width
    """
    print("\n" + "-" * width)
    print(title)
    print("-" * width)


def calculate_statistics(values: List[float]) -> Dict[str, float]:
    """
    Calculate basic statistics for a list of values.
    
    Args:
        values: List of numeric values
        
    Returns:
        Dictionary with mean, median, min, max, std
    """
    if not values:
        return {
            'mean': 0.0,
            'median': 0.0,
            'min': 0.0,
            'max': 0.0,
            'std': 0.0
        }
    
    import statistics
    
    return {
        'mean': statistics.mean(values),
        'median': statistics.median(values),
        'min': min(values),
        'max': max(values),
        'std': statistics.stdev(values) if len(values) > 1 else 0.0
    }


def ensure_dir(path: str) -> Path:
    """
    Ensure directory exists, create if not.
    
    Args:
        path: Directory path
        
    Returns:
        Path object
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_file_size(filepath: str) -> int:
    """
    Get file size in bytes.
    
    Args:
        filepath: Path to file
        
    Returns:
        File size in bytes
    """
    return Path(filepath).stat().st_size


def format_number(num: float, precision: int = 2) -> str:
    """
    Format large numbers with commas.
    
    Args:
        num: Number to format
        precision: Decimal precision
        
    Returns:
        Formatted string
    """
    if isinstance(num, int) or num.is_integer():
        return f"{int(num):,}"
    else:
        return f"{num:,.{precision}f}"
