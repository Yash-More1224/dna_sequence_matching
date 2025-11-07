"""
Visualization Tools for Shift-Or/Bitap Analysis
==============================================

This module provides comprehensive visualization capabilities:
1. Match highlighting in DNA sequences
2. Motif density heatmaps
3. Performance plots (latency, memory, throughput)
4. Scalability curves
5. Comparison charts

Author: DNA Sequence Matching Project
Date: November 2025
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
from typing import List, Tuple, Dict, Optional
import seaborn as sns
from pathlib import Path


# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10


class SequenceVisualizer:
    """
    Visualize DNA sequences with match highlighting.
    """
    
    @staticmethod
    def highlight_matches(sequence: str, matches: List[int], pattern_length: int,
                         context_bp: int = 50, max_matches: int = 10,
                         output_file: Optional[str] = None):
        """
        Create a visual representation of matches in a sequence.
        
        Args:
            sequence: DNA sequence
            matches: List of match positions
            pattern_length: Length of matched pattern
            context_bp: Base pairs of context around each match
            max_matches: Maximum number of matches to display
            output_file: Optional output file path
        """
        fig, axes = plt.subplots(min(len(matches), max_matches), 1, 
                                figsize=(14, 2 * min(len(matches), max_matches)))
        
        if not isinstance(axes, np.ndarray):
            axes = [axes]
        
        for idx, match_pos in enumerate(matches[:max_matches]):
            ax = axes[idx]
            
            # Extract context
            start = max(0, match_pos - context_bp)
            end = min(len(sequence), match_pos + pattern_length + context_bp)
            context_seq = sequence[start:end]
            
            # Relative match position in context
            rel_match_start = match_pos - start
            rel_match_end = rel_match_start + pattern_length
            
            # Color mapping for DNA bases
            colors = {'A': 'red', 'C': 'blue', 'G': 'green', 'T': 'orange', 'N': 'gray'}
            
            # Plot each base
            for i, base in enumerate(context_seq):
                color = colors.get(base, 'black')
                weight = 'bold' if rel_match_start <= i < rel_match_end else 'normal'
                bgcolor = 'yellow' if rel_match_start <= i < rel_match_end else 'white'
                
                ax.text(i, 0, base, fontsize=10, ha='center', va='center',
                       color=color, weight=weight,
                       bbox=dict(boxstyle='square,pad=0.3', facecolor=bgcolor, 
                               edgecolor='black' if rel_match_start <= i < rel_match_end else 'none',
                               linewidth=1.5 if rel_match_start <= i < rel_match_end else 0))
            
            # Labels
            ax.set_xlim(-1, len(context_seq))
            ax.set_ylim(-0.5, 0.5)
            ax.set_title(f'Match at position {match_pos}', fontsize=12, weight='bold')
            ax.axis('off')
        
        plt.tight_layout()
        
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"Saved match visualization to {output_file}")
        else:
            plt.show()
        
        plt.close()
    
    @staticmethod
    def create_density_heatmap(sequence: str, matches: List[int], pattern_length: int,
                              window_size: int = 1000, output_file: Optional[str] = None):
        """
        Create a heatmap showing match density across the sequence.
        
        Args:
            sequence: DNA sequence
            matches: List of match positions
            pattern_length: Length of pattern
            window_size: Size of windows for density calculation
            output_file: Optional output file path
        """
        # Calculate density in windows
        num_windows = (len(sequence) + window_size - 1) // window_size
        densities = np.zeros(num_windows)
        
        for match_pos in matches:
            window_idx = match_pos // window_size
            if window_idx < num_windows:
                densities[window_idx] += 1
        
        # Create heatmap
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 6), 
                                       gridspec_kw={'height_ratios': [3, 1]})
        
        # Heatmap
        densities_2d = densities.reshape(1, -1)
        im = ax1.imshow(densities_2d, aspect='auto', cmap='YlOrRd', 
                       interpolation='nearest')
        ax1.set_ylabel('Matches', fontsize=12)
        ax1.set_yticks([])
        ax1.set_xticks(np.arange(0, num_windows, max(1, num_windows // 10)))
        ax1.set_xticklabels([f'{i*window_size//1000}kb' 
                            for i in range(0, num_windows, max(1, num_windows // 10))])
        ax1.set_title(f'Match Density Heatmap (window size: {window_size} bp)', 
                     fontsize=14, weight='bold')
        
        # Colorbar
        cbar = plt.colorbar(im, ax=ax1)
        cbar.set_label('Matches per window', rotation=270, labelpad=20)
        
        # Density plot
        window_positions = np.arange(num_windows) * window_size / 1000  # in kb
        ax2.fill_between(window_positions, densities, alpha=0.5, color='orange')
        ax2.plot(window_positions, densities, color='red', linewidth=2)
        ax2.set_xlabel('Position (kb)', fontsize=12)
        ax2.set_ylabel('Match count', fontsize=12)
        ax2.set_title('Match Density Profile', fontsize=12)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"Saved density heatmap to {output_file}")
        else:
            plt.show()
        
        plt.close()


class PerformanceVisualizer:
    """
    Visualize performance metrics and comparisons.
    """
    
    @staticmethod
    def plot_latency_comparison(results: Dict[str, 'BenchmarkResult'],
                               output_file: Optional[str] = None):
        """
        Plot latency comparison between algorithms.
        
        Args:
            results: Dictionary of algorithm names to BenchmarkResults
            output_file: Optional output file path
        """
        algorithms = list(results.keys())
        preprocessing = [results[alg].preprocessing_time * 1000 for alg in algorithms]
        search = [results[alg].mean_search_time * 1000 for alg in algorithms]
        
        x = np.arange(len(algorithms))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        bars1 = ax.bar(x - width/2, preprocessing, width, label='Preprocessing', alpha=0.8)
        bars2 = ax.bar(x + width/2, search, width, label='Search', alpha=0.8)
        
        ax.set_ylabel('Time (ms)', fontsize=12)
        ax.set_title('Algorithm Latency Comparison', fontsize=14, weight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(algorithms, rotation=15, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.2f}',
                       ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"Saved latency comparison to {output_file}")
        else:
            plt.show()
        
        plt.close()
    
    @staticmethod
    def plot_throughput_comparison(results: Dict[str, 'BenchmarkResult'],
                                   output_file: Optional[str] = None):
        """
        Plot throughput comparison.
        
        Args:
            results: Dictionary of algorithm names to BenchmarkResults
            output_file: Optional output file path
        """
        algorithms = list(results.keys())
        throughput = [results[alg].throughput_chars_per_sec / 1e6 for alg in algorithms]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        bars = ax.bar(algorithms, throughput, color='skyblue', alpha=0.8, edgecolor='navy')
        
        ax.set_ylabel('Throughput (MB/s)', fontsize=12)
        ax.set_title('Algorithm Throughput Comparison', fontsize=14, weight='bold')
        ax.set_xticklabels(algorithms, rotation=15, ha='right')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}',
                   ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"Saved throughput comparison to {output_file}")
        else:
            plt.show()
        
        plt.close()
    
    @staticmethod
    def plot_memory_comparison(results: Dict[str, 'BenchmarkResult'],
                              output_file: Optional[str] = None):
        """
        Plot memory usage comparison.
        
        Args:
            results: Dictionary of algorithm names to BenchmarkResults
            output_file: Optional output file path
        """
        algorithms = list(results.keys())
        peak_memory = [results[alg].peak_memory / 1024 for alg in algorithms]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        bars = ax.bar(algorithms, peak_memory, color='lightcoral', alpha=0.8, 
                     edgecolor='darkred')
        
        ax.set_ylabel('Peak Memory (KB)', fontsize=12)
        ax.set_title('Algorithm Memory Usage Comparison', fontsize=14, weight='bold')
        ax.set_xticklabels(algorithms, rotation=15, ha='right')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}',
                   ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"Saved memory comparison to {output_file}")
        else:
            plt.show()
        
        plt.close()
    
    @staticmethod
    def plot_scalability_text_length(scale_results: List['BenchmarkResult'],
                                     output_file: Optional[str] = None):
        """
        Plot how performance scales with text length.
        
        Args:
            scale_results: List of BenchmarkResults at different text lengths
            output_file: Optional output file path
        """
        text_lengths = [r.text_length for r in scale_results]
        times = [r.mean_search_time * 1000 for r in scale_results]
        throughputs = [r.throughput_chars_per_sec / 1e6 for r in scale_results]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Latency plot
        ax1.plot(text_lengths, times, 'o-', linewidth=2, markersize=8, color='blue')
        ax1.set_xlabel('Text Length (characters)', fontsize=12)
        ax1.set_ylabel('Search Time (ms)', fontsize=12)
        ax1.set_title('Scalability: Latency vs Text Length', fontsize=13, weight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Add trend line
        z = np.polyfit(text_lengths, times, 1)
        p = np.poly1d(z)
        ax1.plot(text_lengths, p(text_lengths), "--", alpha=0.5, color='red', 
                label=f'Linear fit: y={z[0]:.2e}x+{z[1]:.2f}')
        ax1.legend()
        
        # Throughput plot
        ax2.plot(text_lengths, throughputs, 'o-', linewidth=2, markersize=8, color='green')
        ax2.set_xlabel('Text Length (characters)', fontsize=12)
        ax2.set_ylabel('Throughput (MB/s)', fontsize=12)
        ax2.set_title('Scalability: Throughput vs Text Length', fontsize=13, weight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=np.mean(throughputs), color='red', linestyle='--', 
                   alpha=0.5, label=f'Mean: {np.mean(throughputs):.2f} MB/s')
        ax2.legend()
        
        plt.tight_layout()
        
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"Saved scalability plot to {output_file}")
        else:
            plt.show()
        
        plt.close()
    
    @staticmethod
    def plot_scalability_pattern_length(scale_results: List['BenchmarkResult'],
                                       output_file: Optional[str] = None):
        """
        Plot how performance scales with pattern length.
        
        Args:
            scale_results: List of BenchmarkResults at different pattern lengths
            output_file: Optional output file path
        """
        pattern_lengths = [r.pattern_length for r in scale_results]
        times = [r.mean_search_time * 1000 for r in scale_results]
        preprocessing = [r.preprocessing_time * 1000 for r in scale_results]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        ax.plot(pattern_lengths, times, 'o-', linewidth=2, markersize=8, 
               label='Search Time', color='blue')
        ax.plot(pattern_lengths, preprocessing, 's-', linewidth=2, markersize=8,
               label='Preprocessing Time', color='orange')
        
        ax.set_xlabel('Pattern Length (characters)', fontsize=12)
        ax.set_ylabel('Time (ms)', fontsize=12)
        ax.set_title('Scalability: Time vs Pattern Length', fontsize=14, weight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"Saved pattern length scalability to {output_file}")
        else:
            plt.show()
        
        plt.close()
    
    @staticmethod
    def plot_accuracy_vs_errors(accuracy_data: List[Tuple[int, 'AccuracyMetrics']],
                               output_file: Optional[str] = None):
        """
        Plot accuracy metrics vs maximum errors allowed.
        
        Args:
            accuracy_data: List of (max_errors, AccuracyMetrics) tuples
            output_file: Optional output file path
        """
        max_errors = [d[0] for d in accuracy_data]
        precision = [d[1].precision for d in accuracy_data]
        recall = [d[1].recall for d in accuracy_data]
        f1 = [d[1].f1_score for d in accuracy_data]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.plot(max_errors, precision, 'o-', linewidth=2, markersize=8, 
               label='Precision', color='blue')
        ax.plot(max_errors, recall, 's-', linewidth=2, markersize=8,
               label='Recall', color='green')
        ax.plot(max_errors, f1, '^-', linewidth=2, markersize=8,
               label='F1 Score', color='red')
        
        ax.set_xlabel('Maximum Edit Distance (k)', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title('Accuracy Metrics vs Edit Distance', fontsize=14, weight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1.05])
        
        plt.tight_layout()
        
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"Saved accuracy plot to {output_file}")
        else:
            plt.show()
        
        plt.close()


def create_summary_dashboard(benchmark_results: Dict, accuracy_metrics: Dict,
                             output_file: Optional[str] = None):
    """
    Create a comprehensive dashboard with multiple visualizations.
    
    Args:
        benchmark_results: Dictionary of benchmark results
        accuracy_metrics: Dictionary of accuracy metrics
        output_file: Optional output file path
    """
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Title
    fig.suptitle('Shift-Or/Bitap Algorithm Analysis Dashboard', 
                fontsize=16, weight='bold')
    
    # TODO: Add multiple subplots with different visualizations
    # This would include latency, throughput, memory, accuracy, etc.
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved dashboard to {output_file}")
    else:
        plt.show()
    
    plt.close()


if __name__ == "__main__":
    print("Visualization Demo")
    print("=" * 80)
    
    from algorithm import ShiftOrBitap
    from data_loader import SyntheticDataGenerator, create_motif_dataset
    
    # Demo 1: Match highlighting
    print("\n1. Match Highlighting")
    print("-" * 80)
    
    pattern = "GATTACA"
    text = create_motif_dataset([pattern], background_length=500, num_copies=5, seed=42)
    
    matcher = ShiftOrBitap(pattern)
    matches = matcher.search_exact(text)
    
    print(f"Pattern: {pattern}")
    print(f"Text length: {len(text)}")
    print(f"Matches found: {len(matches)} at positions {matches[:5]}")
    
    SequenceVisualizer.highlight_matches(text, matches, len(pattern), 
                                        context_bp=30, max_matches=3,
                                        output_file="match_highlight_demo.png")
    
    # Demo 2: Density heatmap
    print("\n2. Density Heatmap")
    print("-" * 80)
    
    long_text = create_motif_dataset([pattern], background_length=10000, 
                                    num_copies=50, seed=42)
    long_matches = matcher.search_exact(long_text)
    
    print(f"Long text length: {len(long_text)}")
    print(f"Matches found: {len(long_matches)}")
    
    SequenceVisualizer.create_density_heatmap(long_text, long_matches, len(pattern),
                                             window_size=500,
                                             output_file="density_heatmap_demo.png")
    
    print("\n✓ Visualization demos complete!")
    print("Check the generated PNG files in the current directory.")
