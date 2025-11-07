"""
Visualization utilities for KMP algorithm analysis.

This module provides functions to create various plots and visualizations
for algorithm performance, match highlighting, and result analysis.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

from .config import VIZ_CONFIG, PLOTS_DIR
from .benchmarking import BenchmarkResult
from .evaluation import ComparisonResult


# Set default style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette(VIZ_CONFIG['color_palette'])


def plot_latency_vs_pattern_length(results: List[BenchmarkResult],
                                   save_path: Optional[Path] = None,
                                   show: bool = True) -> None:
    """
    Plot search latency vs pattern length.
    
    Args:
        results: List of BenchmarkResult objects
        save_path: Path to save the plot (if None, uses default)
        show: Whether to display the plot
    """
    if not results:
        print("No results to plot")
        return
    
    # Extract data
    pattern_lengths = [r.pattern_length for r in results]
    search_times = [r.search_time * 1000 for r in results]  # Convert to ms
    preprocessing_times = [r.preprocessing_time * 1000 for r in results]
    
    # Create figure
    fig, ax = plt.subplots(figsize=VIZ_CONFIG['figure_size'])
    
    # Plot
    ax.plot(pattern_lengths, search_times, 'o-', label='Search Time', linewidth=2, markersize=8)
    ax.plot(pattern_lengths, preprocessing_times, 's-', label='Preprocessing Time', linewidth=2, markersize=8)
    
    ax.set_xlabel('Pattern Length (bp)', fontsize=VIZ_CONFIG['font_size'])
    ax.set_ylabel('Time (ms)', fontsize=VIZ_CONFIG['font_size'])
    ax.set_title('KMP: Latency vs Pattern Length', fontsize=VIZ_CONFIG['font_size'] + 2, fontweight='bold')
    ax.legend(fontsize=VIZ_CONFIG['font_size'])
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path is None:
        save_path = PLOTS_DIR / 'latency_vs_pattern_length.png'
    
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=VIZ_CONFIG['dpi'], bbox_inches='tight')
    print(f"Saved plot to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_latency_vs_text_size(results: List[BenchmarkResult],
                              save_path: Optional[Path] = None,
                              show: bool = True) -> None:
    """
    Plot search latency vs text size.
    
    Args:
        results: List of BenchmarkResult objects
        save_path: Path to save the plot
        show: Whether to display the plot
    """
    if not results:
        print("No results to plot")
        return
    
    # Extract data
    text_sizes = [r.text_length / 1024 for r in results]  # Convert to KB
    search_times = [r.search_time * 1000 for r in results]  # Convert to ms
    
    # Create figure
    fig, ax = plt.subplots(figsize=VIZ_CONFIG['figure_size'])
    
    # Plot
    ax.plot(text_sizes, search_times, 'o-', linewidth=2, markersize=8, color='steelblue')
    
    ax.set_xlabel('Text Size (KB)', fontsize=VIZ_CONFIG['font_size'])
    ax.set_ylabel('Search Time (ms)', fontsize=VIZ_CONFIG['font_size'])
    ax.set_title('KMP: Search Latency vs Text Size', fontsize=VIZ_CONFIG['font_size'] + 2, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path is None:
        save_path = PLOTS_DIR / 'latency_vs_text_size.png'
    
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=VIZ_CONFIG['dpi'], bbox_inches='tight')
    print(f"Saved plot to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_memory_vs_text_size(results: List[BenchmarkResult],
                             save_path: Optional[Path] = None,
                             show: bool = True) -> None:
    """
    Plot memory usage vs text size.
    
    Args:
        results: List of BenchmarkResult objects
        save_path: Path to save the plot
        show: Whether to display the plot
    """
    if not results:
        print("No results to plot")
        return
    
    # Extract data
    text_sizes = [r.text_length / 1024 for r in results]  # Convert to KB
    memory_usage = [r.memory_used / 1024 for r in results]  # Convert to KB
    
    # Create figure
    fig, ax = plt.subplots(figsize=VIZ_CONFIG['figure_size'])
    
    # Plot
    ax.plot(text_sizes, memory_usage, 'o-', linewidth=2, markersize=8, color='coral')
    
    ax.set_xlabel('Text Size (KB)', fontsize=VIZ_CONFIG['font_size'])
    ax.set_ylabel('Memory Usage (KB)', fontsize=VIZ_CONFIG['font_size'])
    ax.set_title('KMP: Memory Usage vs Text Size', fontsize=VIZ_CONFIG['font_size'] + 2, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path is None:
        save_path = PLOTS_DIR / 'memory_vs_text_size.png'
    
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=VIZ_CONFIG['dpi'], bbox_inches='tight')
    print(f"Saved plot to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_kmp_vs_re_comparison(comparisons: List[ComparisonResult],
                              text_sizes: List[int],
                              save_path: Optional[Path] = None,
                              show: bool = True) -> None:
    """
    Plot KMP vs Python re comparison.
    
    Args:
        comparisons: List of ComparisonResult objects
        text_sizes: List of corresponding text sizes
        save_path: Path to save the plot
        show: Whether to display the plot
    """
    if not comparisons:
        print("No comparisons to plot")
        return
    
    # Extract data
    sizes_kb = [s / 1024 for s in text_sizes]
    kmp_times = [c.kmp_time * 1000 for c in comparisons]  # Convert to ms
    re_times = [c.re_time * 1000 for c in comparisons]
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Time comparison
    ax1.plot(sizes_kb, kmp_times, 'o-', label='KMP', linewidth=2, markersize=8)
    ax1.plot(sizes_kb, re_times, 's-', label='Python re', linewidth=2, markersize=8)
    ax1.set_xlabel('Text Size (KB)', fontsize=VIZ_CONFIG['font_size'])
    ax1.set_ylabel('Time (ms)', fontsize=VIZ_CONFIG['font_size'])
    ax1.set_title('Execution Time Comparison', fontsize=VIZ_CONFIG['font_size'] + 2, fontweight='bold')
    ax1.legend(fontsize=VIZ_CONFIG['font_size'])
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Speedup
    speedups = [c.speedup for c in comparisons]
    colors = ['green' if s > 1 else 'red' for s in speedups]
    ax2.bar(range(len(sizes_kb)), speedups, color=colors, alpha=0.7)
    ax2.axhline(y=1.0, color='black', linestyle='--', linewidth=2, label='No speedup')
    ax2.set_xlabel('Text Size Index', fontsize=VIZ_CONFIG['font_size'])
    ax2.set_ylabel('Speedup Factor (re/KMP)', fontsize=VIZ_CONFIG['font_size'])
    ax2.set_title('KMP Speedup vs Python re', fontsize=VIZ_CONFIG['font_size'] + 2, fontweight='bold')
    ax2.legend(fontsize=VIZ_CONFIG['font_size'])
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path is None:
        save_path = PLOTS_DIR / 'kmp_vs_re_comparison.png'
    
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=VIZ_CONFIG['dpi'], bbox_inches='tight')
    print(f"Saved plot to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_throughput(results: List[BenchmarkResult],
                   save_path: Optional[Path] = None,
                   show: bool = True) -> None:
    """
    Plot throughput (characters processed per second).
    
    Args:
        results: List of BenchmarkResult objects
        save_path: Path to save the plot
        show: Whether to display the plot
    """
    if not results:
        print("No results to plot")
        return
    
    # Calculate throughput
    text_sizes = [r.text_length / 1024 for r in results]  # KB
    throughputs = [r.text_length / r.search_time / 1_000_000 for r in results]  # MB/s
    
    # Create figure
    fig, ax = plt.subplots(figsize=VIZ_CONFIG['figure_size'])
    
    # Plot
    ax.plot(text_sizes, throughputs, 'o-', linewidth=2, markersize=8, color='purple')
    
    ax.set_xlabel('Text Size (KB)', fontsize=VIZ_CONFIG['font_size'])
    ax.set_ylabel('Throughput (MB/s)', fontsize=VIZ_CONFIG['font_size'])
    ax.set_title('KMP: Search Throughput', fontsize=VIZ_CONFIG['font_size'] + 2, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path is None:
        save_path = PLOTS_DIR / 'throughput.png'
    
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=VIZ_CONFIG['dpi'], bbox_inches='tight')
    print(f"Saved plot to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def highlight_matches_in_sequence(sequence: str,
                                  matches: List[int],
                                  pattern_length: int,
                                  context_length: int = 50,
                                  max_matches: int = 10,
                                  save_path: Optional[Path] = None,
                                  show: bool = True) -> None:
    """
    Visualize matches highlighted in the sequence.
    
    Args:
        sequence: The DNA sequence
        matches: List of match positions
        pattern_length: Length of the pattern
        context_length: How many bases to show around each match
        max_matches: Maximum number of matches to display
        save_path: Path to save the plot
        show: Whether to display the plot
    """
    if not matches:
        print("No matches to visualize")
        return
    
    # Limit matches to display
    display_matches = matches[:max_matches]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, len(display_matches) * 1.5))
    
    for idx, match_pos in enumerate(display_matches):
        # Extract context
        start = max(0, match_pos - context_length)
        end = min(len(sequence), match_pos + pattern_length + context_length)
        segment = sequence[start:end]
        
        # Highlight match
        match_start_in_segment = match_pos - start
        match_end_in_segment = match_start_in_segment + pattern_length
        
        # Create colored text
        y_pos = len(display_matches) - idx - 1
        
        # Plot context before match
        ax.text(0, y_pos, segment[:match_start_in_segment], 
               fontfamily='monospace', fontsize=8, va='center')
        
        # Plot match (highlighted)
        x_offset = match_start_in_segment
        ax.text(x_offset, y_pos, segment[match_start_in_segment:match_end_in_segment],
               fontfamily='monospace', fontsize=8, va='center',
               bbox=dict(boxstyle='round', facecolor=VIZ_CONFIG['match_highlight_color'], alpha=0.5))
        
        # Plot context after match
        x_offset = match_end_in_segment
        ax.text(x_offset, y_pos, segment[match_end_in_segment:],
               fontfamily='monospace', fontsize=8, va='center')
        
        # Add position label
        ax.text(-5, y_pos, f"{match_pos}:", fontsize=9, ha='right', va='center', fontweight='bold')
    
    ax.set_xlim(-10, context_length * 2 + pattern_length)
    ax.set_ylim(-0.5, len(display_matches))
    ax.axis('off')
    ax.set_title(f'Match Visualization (showing {len(display_matches)} of {len(matches)} matches)',
                fontsize=VIZ_CONFIG['font_size'] + 2, fontweight='bold', pad=20)
    
    plt.tight_layout()
    
    if save_path is None:
        save_path = PLOTS_DIR / 'match_highlights.png'
    
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=VIZ_CONFIG['dpi'], bbox_inches='tight')
    print(f"Saved plot to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_match_density_heatmap(sequence: str,
                               matches: List[int],
                               pattern_length: int,
                               window_size: int = 10000,
                               save_path: Optional[Path] = None,
                               show: bool = True) -> None:
    """
    Create a heatmap showing match density across the sequence.
    
    Args:
        sequence: The DNA sequence
        matches: List of match positions
        pattern_length: Length of the pattern
        window_size: Size of windows for density calculation
        save_path: Path to save the plot
        show: Whether to display the plot
    """
    if not matches:
        print("No matches to visualize")
        return
    
    # Calculate density in windows
    num_windows = (len(sequence) + window_size - 1) // window_size
    densities = np.zeros(num_windows)
    
    for match_pos in matches:
        window_idx = match_pos // window_size
        if window_idx < num_windows:
            densities[window_idx] += 1
    
    # Create figure
    fig, ax = plt.subplots(figsize=(16, 4))
    
    # Plot heatmap
    im = ax.imshow([densities], aspect='auto', cmap='YlOrRd', interpolation='nearest')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, orientation='vertical', pad=0.02)
    cbar.set_label('Match Count', fontsize=VIZ_CONFIG['font_size'])
    
    # Set labels
    ax.set_xlabel('Genomic Position (windows)', fontsize=VIZ_CONFIG['font_size'])
    ax.set_yticks([])
    ax.set_title(f'Match Density Heatmap (window size: {window_size:,} bp, {len(matches)} total matches)',
                fontsize=VIZ_CONFIG['font_size'] + 2, fontweight='bold')
    
    # Add x-axis labels for positions
    tick_positions = np.linspace(0, num_windows - 1, min(10, num_windows))
    tick_labels = [f"{int(pos * window_size / 1000)}k" for pos in tick_positions]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels)
    
    plt.tight_layout()
    
    if save_path is None:
        save_path = PLOTS_DIR / 'match_density_heatmap.png'
    
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=VIZ_CONFIG['dpi'], bbox_inches='tight')
    print(f"Saved plot to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_match_position_distribution(matches: List[int],
                                     sequence_length: int,
                                     bins: int = 50,
                                     save_path: Optional[Path] = None,
                                     show: bool = True) -> None:
    """
    Plot distribution of match positions across the sequence.
    
    Args:
        matches: List of match positions
        sequence_length: Total length of the sequence
        bins: Number of bins for histogram
        save_path: Path to save the plot
        show: Whether to display the plot
    """
    if not matches:
        print("No matches to visualize")
        return
    
    # Create figure
    fig, ax = plt.subplots(figsize=VIZ_CONFIG['figure_size'])
    
    # Plot histogram
    ax.hist(matches, bins=bins, color='skyblue', edgecolor='black', alpha=0.7)
    
    ax.set_xlabel('Position in Sequence', fontsize=VIZ_CONFIG['font_size'])
    ax.set_ylabel('Number of Matches', fontsize=VIZ_CONFIG['font_size'])
    ax.set_title(f'Match Position Distribution ({len(matches)} matches)',
                fontsize=VIZ_CONFIG['font_size'] + 2, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path is None:
        save_path = PLOTS_DIR / 'match_position_distribution.png'
    
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=VIZ_CONFIG['dpi'], bbox_inches='tight')
    print(f"Saved plot to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_performance_summary(results: List[BenchmarkResult],
                            save_path: Optional[Path] = None,
                            show: bool = True) -> None:
    """
    Create a comprehensive performance summary with multiple subplots.
    
    Args:
        results: List of BenchmarkResult objects
        save_path: Path to save the plot
        show: Whether to display the plot
    """
    if not results:
        print("No results to plot")
        return
    
    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Time breakdown
    pattern_lengths = [r.pattern_length for r in results]
    preprocessing = [r.preprocessing_time * 1000 for r in results]
    search = [r.search_time * 1000 for r in results]
    
    ax1.plot(pattern_lengths, preprocessing, 'o-', label='Preprocessing', linewidth=2, markersize=8)
    ax1.plot(pattern_lengths, search, 's-', label='Search', linewidth=2, markersize=8)
    ax1.set_xlabel('Pattern Length (bp)', fontsize=VIZ_CONFIG['font_size'])
    ax1.set_ylabel('Time (ms)', fontsize=VIZ_CONFIG['font_size'])
    ax1.set_title('Time Breakdown', fontsize=VIZ_CONFIG['font_size'] + 1, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Memory usage
    text_sizes = [r.text_length / 1024 for r in results]
    memory = [r.memory_used / 1024 for r in results]
    
    ax2.plot(text_sizes, memory, 'o-', linewidth=2, markersize=8, color='coral')
    ax2.set_xlabel('Text Size (KB)', fontsize=VIZ_CONFIG['font_size'])
    ax2.set_ylabel('Memory (KB)', fontsize=VIZ_CONFIG['font_size'])
    ax2.set_title('Memory Usage', fontsize=VIZ_CONFIG['font_size'] + 1, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Throughput
    throughputs = [r.text_length / r.search_time / 1_000_000 for r in results]
    
    ax3.bar(range(len(text_sizes)), throughputs, color='purple', alpha=0.7)
    ax3.set_xlabel('Test Index', fontsize=VIZ_CONFIG['font_size'])
    ax3.set_ylabel('Throughput (MB/s)', fontsize=VIZ_CONFIG['font_size'])
    ax3.set_title('Search Throughput', fontsize=VIZ_CONFIG['font_size'] + 1, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Match statistics
    num_matches = [r.num_matches for r in results]
    
    ax4.scatter(text_sizes, num_matches, s=100, alpha=0.6, color='green')
    ax4.set_xlabel('Text Size (KB)', fontsize=VIZ_CONFIG['font_size'])
    ax4.set_ylabel('Number of Matches', fontsize=VIZ_CONFIG['font_size'])
    ax4.set_title('Matches Found', fontsize=VIZ_CONFIG['font_size'] + 1, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle('KMP Algorithm Performance Summary', 
                fontsize=VIZ_CONFIG['font_size'] + 4, fontweight='bold', y=1.00)
    plt.tight_layout()
    
    if save_path is None:
        save_path = PLOTS_DIR / 'performance_summary.png'
    
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=VIZ_CONFIG['dpi'], bbox_inches='tight')
    print(f"Saved plot to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def create_all_visualizations(benchmark_results: List[BenchmarkResult],
                             comparison_results: Optional[List[ComparisonResult]] = None,
                             text_sizes: Optional[List[int]] = None,
                             sequence: Optional[str] = None,
                             matches: Optional[List[int]] = None,
                             pattern_length: Optional[int] = None) -> None:
    """
    Create all available visualizations.
    
    Args:
        benchmark_results: List of BenchmarkResult objects
        comparison_results: Optional list of ComparisonResult objects
        text_sizes: Optional list of text sizes (for comparison plots)
        sequence: Optional sequence for match highlighting
        matches: Optional match positions
        pattern_length: Optional pattern length for match visualization
    """
    print("Creating visualizations...")
    
    # Performance plots
    if benchmark_results:
        plot_latency_vs_pattern_length(benchmark_results, show=False)
        plot_latency_vs_text_size(benchmark_results, show=False)
        plot_memory_vs_text_size(benchmark_results, show=False)
        plot_throughput(benchmark_results, show=False)
        plot_performance_summary(benchmark_results, show=False)
    
    # Comparison plots
    if comparison_results and text_sizes:
        plot_kmp_vs_re_comparison(comparison_results, text_sizes, show=False)
    
    # Match visualizations
    if sequence and matches and pattern_length:
        plot_match_density_heatmap(sequence, matches, pattern_length, show=False)
        plot_match_position_distribution(matches, len(sequence), show=False)
        highlight_matches_in_sequence(sequence, matches, pattern_length, show=False)
    
    print("All visualizations created!")
