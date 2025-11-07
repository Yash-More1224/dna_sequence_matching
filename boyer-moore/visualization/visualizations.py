"""
Visualization Module

Create plots and visualizations for Boyer-Moore analysis:
- Performance plots (time, memory, scaling)
- Comparison plots (vs re, variant comparison)
- Match visualizations (highlighting, heatmaps)
"""

import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
import sys
sys.path.append('..')


class Visualizer:
    """Create visualizations for Boyer-Moore experiments."""
    
    def __init__(self, output_dir: str = "results/plots", style: str = "seaborn-v0_8"):
        """
        Initialize visualizer.
        
        Args:
            output_dir: Directory to save plots
            style: Matplotlib style
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style
        plt.style.use(style)
        sns.set_palette("husl")
        
        self.dpi = 300
        self.figsize = (12, 8)
    
    def load_experiment_data(self, experiment_name: str) -> List[Dict]:
        """
        Load experiment data from JSON file.
        
        Args:
            experiment_name: Name of experiment file (without extension)
            
        Returns:
            List of result dictionaries
        """
        filepath = Path("results/tables") / f"{experiment_name}.json"
        with open(filepath, 'r') as f:
            return json.load(f)
    
    def plot_pattern_length_vs_time(self, save: bool = True):
        """Plot latency vs pattern length (Experiment 1)."""
        data = self.load_experiment_data("exp1_pattern_length")
        
        pattern_lengths = [d['pattern_length'] for d in data]
        total_times = [d['mean_total_time'] * 1000 for d in data]  # Convert to ms
        search_times = [d['mean_search_time'] * 1000 for d in data]
        preprocess_times = [d['mean_preprocessing_time'] * 1000 for d in data]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Total time plot
        ax1.plot(pattern_lengths, total_times, 'o-', linewidth=2, markersize=8, label='Total Time')
        ax1.set_xlabel('Pattern Length (bp)', fontsize=12)
        ax1.set_ylabel('Time (ms)', fontsize=12)
        ax1.set_title('Boyer-Moore: Latency vs Pattern Length', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Stacked time breakdown
        ax2.plot(pattern_lengths, search_times, 's-', linewidth=2, markersize=8, label='Search Time')
        ax2.plot(pattern_lengths, preprocess_times, '^-', linewidth=2, markersize=8, label='Preprocessing Time')
        ax2.set_xlabel('Pattern Length (bp)', fontsize=12)
        ax2.set_ylabel('Time (ms)', fontsize=12)
        ax2.set_title('Time Breakdown', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        plt.tight_layout()
        
        if save:
            output_file = self.output_dir / "pattern_length_vs_time.png"
            plt.savefig(output_file, dpi=self.dpi, bbox_inches='tight')
            print(f"✓ Saved: {output_file}")
        
        plt.close()
    
    def plot_text_scaling(self, save: bool = True):
        """Plot scalability with text size (Experiment 2)."""
        data = self.load_experiment_data("exp2_text_scaling")
        
        text_sizes = [d['text_size'] / 1000 for d in data]  # Convert to KB
        times = [d['mean_total_time'] * 1000 for d in data]  # Convert to ms
        throughputs = [d['throughput_mbps'] for d in data]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Time vs text size
        ax1.plot(text_sizes, times, 'o-', linewidth=2, markersize=8, color='#2ecc71')
        ax1.set_xlabel('Text Size (KB)', fontsize=12)
        ax1.set_ylabel('Time (ms)', fontsize=12)
        ax1.set_title('Boyer-Moore: Time vs Text Size', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Throughput
        ax2.plot(text_sizes, throughputs, 's-', linewidth=2, markersize=8, color='#e74c3c')
        ax2.set_xlabel('Text Size (KB)', fontsize=12)
        ax2.set_ylabel('Throughput (MB/s)', fontsize=12)
        ax2.set_title('Search Throughput', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            output_file = self.output_dir / "text_scaling.png"
            plt.savefig(output_file, dpi=self.dpi, bbox_inches='tight')
            print(f"✓ Saved: {output_file}")
        
        plt.close()
    
    def plot_alphabet_effect(self, save: bool = True):
        """Plot alphabet size effect (Experiment 3)."""
        data = self.load_experiment_data("exp3_alphabet_effect")
        
        alphabets = [d['alphabet_type'] for d in data]
        times = [d['mean_total_time'] * 1000 for d in data]
        comparisons = [d['mean_comparisons'] for d in data]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Time comparison
        bars1 = ax1.bar(alphabets, times, color=['#3498db', '#9b59b6'], alpha=0.8, edgecolor='black')
        ax1.set_ylabel('Time (ms)', fontsize=12)
        ax1.set_title('Search Time by Alphabet Size', fontsize=14, fontweight='bold')
        ax1.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}', ha='center', va='bottom', fontsize=10)
        
        # Comparisons comparison
        bars2 = ax2.bar(alphabets, comparisons, color=['#3498db', '#9b59b6'], alpha=0.8, edgecolor='black')
        ax2.set_ylabel('Number of Comparisons', fontsize=12)
        ax2.set_title('Character Comparisons by Alphabet Size', fontsize=14, fontweight='bold')
        ax2.grid(axis='y', alpha=0.3)
        
        for bar in bars2:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height):,}', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        
        if save:
            output_file = self.output_dir / "alphabet_effect.png"
            plt.savefig(output_file, dpi=self.dpi, bbox_inches='tight')
            print(f"✓ Saved: {output_file}")
        
        plt.close()
    
    def plot_heuristic_contribution(self, save: bool = True):
        """Plot heuristic contribution comparison (Experiment 4)."""
        data = self.load_experiment_data("exp4_heuristic_contribution")
        
        variants = [d['variant'] for d in data]
        times = [d['mean_total_time'] * 1000 for d in data]
        comparisons = [d['mean_comparisons'] for d in data]
        shifts = [d['mean_shifts'] for d in data]
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Time comparison
        bars = axes[0, 0].bar(variants, times, color=['#e74c3c', '#3498db', '#2ecc71', '#f39c12'], alpha=0.8, edgecolor='black')
        axes[0, 0].set_ylabel('Time (ms)', fontsize=12)
        axes[0, 0].set_title('Search Time by Variant', fontsize=14, fontweight='bold')
        axes[0, 0].grid(axis='y', alpha=0.3)
        axes[0, 0].tick_params(axis='x', rotation=15)
        
        for bar in bars:
            height = bar.get_height()
            axes[0, 0].text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.2f}', ha='center', va='bottom', fontsize=9)
        
        # Comparisons
        bars = axes[0, 1].bar(variants, comparisons, color=['#e74c3c', '#3498db', '#2ecc71', '#f39c12'], alpha=0.8, edgecolor='black')
        axes[0, 1].set_ylabel('Number of Comparisons', fontsize=12)
        axes[0, 1].set_title('Character Comparisons by Variant', fontsize=14, fontweight='bold')
        axes[0, 1].grid(axis='y', alpha=0.3)
        axes[0, 1].tick_params(axis='x', rotation=15)
        
        for bar in bars:
            height = bar.get_height()
            axes[0, 1].text(bar.get_x() + bar.get_width()/2., height,
                           f'{int(height):,}', ha='center', va='bottom', fontsize=9)
        
        # Shifts
        bars = axes[1, 0].bar(variants, shifts, color=['#e74c3c', '#3498db', '#2ecc71', '#f39c12'], alpha=0.8, edgecolor='black')
        axes[1, 0].set_ylabel('Number of Shifts', fontsize=12)
        axes[1, 0].set_title('Pattern Shifts by Variant', fontsize=14, fontweight='bold')
        axes[1, 0].grid(axis='y', alpha=0.3)
        axes[1, 0].tick_params(axis='x', rotation=15)
        
        for bar in bars:
            height = bar.get_height()
            axes[1, 0].text(bar.get_x() + bar.get_width()/2., height,
                           f'{int(height):,}', ha='center', va='bottom', fontsize=9)
        
        # Efficiency: comparisons per shift
        efficiency = [c/s if s > 0 else 0 for c, s in zip(comparisons, shifts)]
        bars = axes[1, 1].bar(variants, efficiency, color=['#e74c3c', '#3498db', '#2ecc71', '#f39c12'], alpha=0.8, edgecolor='black')
        axes[1, 1].set_ylabel('Comparisons per Shift', fontsize=12)
        axes[1, 1].set_title('Search Efficiency', fontsize=14, fontweight='bold')
        axes[1, 1].grid(axis='y', alpha=0.3)
        axes[1, 1].tick_params(axis='x', rotation=15)
        
        for bar in bars:
            height = bar.get_height()
            axes[1, 1].text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.1f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        
        if save:
            output_file = self.output_dir / "heuristic_contribution.png"
            plt.savefig(output_file, dpi=self.dpi, bbox_inches='tight')
            print(f"✓ Saved: {output_file}")
        
        plt.close()
    
    def plot_preprocessing_overhead(self, save: bool = True):
        """Plot preprocessing overhead (Experiment 5)."""
        data = self.load_experiment_data("exp5_preprocessing_overhead")
        
        pattern_lengths = [d['pattern_length'] for d in data]
        preprocess_times = [d['preprocessing_time'] * 1000 for d in data]
        search_times = [d['mean_search_time'] * 1000 for d in data]
        ratios = [d['preprocess_ratio'] * 100 for d in data]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Stacked bar chart
        width = 0.6
        ax1.bar(pattern_lengths, preprocess_times, width, label='Preprocessing', alpha=0.8, color='#e74c3c')
        ax1.bar(pattern_lengths, search_times, width, bottom=preprocess_times, label='Search', alpha=0.8, color='#3498db')
        ax1.set_xlabel('Pattern Length (bp)', fontsize=12)
        ax1.set_ylabel('Time (ms)', fontsize=12)
        ax1.set_title('Preprocessing vs Search Time', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)
        
        # Ratio plot
        ax2.plot(pattern_lengths, ratios, 'o-', linewidth=2, markersize=8, color='#9b59b6')
        ax2.set_xlabel('Pattern Length (bp)', fontsize=12)
        ax2.set_ylabel('Preprocessing Overhead (%)', fontsize=12)
        ax2.set_title('Preprocessing as % of Total Time', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=50, color='r', linestyle='--', alpha=0.5, label='50% threshold')
        ax2.legend()
        
        plt.tight_layout()
        
        if save:
            output_file = self.output_dir / "preprocessing_overhead.png"
            plt.savefig(output_file, dpi=self.dpi, bbox_inches='tight')
            print(f"✓ Saved: {output_file}")
        
        plt.close()
    
    def plot_memory_footprint(self, save: bool = True):
        """Plot memory usage (Experiment 6)."""
        data = self.load_experiment_data("exp6_memory_footprint")
        
        pattern_lengths = [d['pattern_length'] for d in data]
        memory_kb = [d['peak_memory_kb'] for d in data]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.plot(pattern_lengths, memory_kb, 'o-', linewidth=2, markersize=8, color='#2ecc71')
        ax.set_xlabel('Pattern Length (bp)', fontsize=12)
        ax.set_ylabel('Peak Memory (KB)', fontsize=12)
        ax.set_title('Memory Footprint vs Pattern Length', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            output_file = self.output_dir / "memory_footprint.png"
            plt.savefig(output_file, dpi=self.dpi, bbox_inches='tight')
            print(f"✓ Saved: {output_file}")
        
        plt.close()
    
    def plot_real_motifs(self, save: bool = True):
        """Plot real motif search results (Experiment 7)."""
        data = self.load_experiment_data("exp7_real_motifs")
        
        motif_names = [d['motif_name'] for d in data]
        matches = [d['matches_found'] for d in data]
        densities = [d['matches_per_mb'] for d in data]
        times = [d['search_time'] * 1000 for d in data]
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Match counts
        bars = axes[0].barh(motif_names, matches, color='#3498db', alpha=0.8, edgecolor='black')
        axes[0].set_xlabel('Number of Matches', fontsize=12)
        axes[0].set_title('Motif Occurrences in E. coli', fontsize=14, fontweight='bold')
        axes[0].grid(axis='x', alpha=0.3)
        
        for i, bar in enumerate(bars):
            width = bar.get_width()
            axes[0].text(width, bar.get_y() + bar.get_height()/2.,
                        f' {int(width):,}', ha='left', va='center', fontsize=9)
        
        # Match density
        bars = axes[1].barh(motif_names, densities, color='#2ecc71', alpha=0.8, edgecolor='black')
        axes[1].set_xlabel('Matches per Mb', fontsize=12)
        axes[1].set_title('Motif Density', fontsize=14, fontweight='bold')
        axes[1].grid(axis='x', alpha=0.3)
        
        for i, bar in enumerate(bars):
            width = bar.get_width()
            axes[1].text(width, bar.get_y() + bar.get_height()/2.,
                        f' {width:.1f}', ha='left', va='center', fontsize=9)
        
        # Search times
        bars = axes[2].barh(motif_names, times, color='#e74c3c', alpha=0.8, edgecolor='black')
        axes[2].set_xlabel('Search Time (ms)', fontsize=12)
        axes[2].set_title('Search Performance', fontsize=14, fontweight='bold')
        axes[2].grid(axis='x', alpha=0.3)
        
        for i, bar in enumerate(bars):
            width = bar.get_width()
            axes[2].text(width, bar.get_y() + bar.get_height()/2.,
                        f' {width:.2f}', ha='left', va='center', fontsize=9)
        
        plt.tight_layout()
        
        if save:
            output_file = self.output_dir / "real_motifs.png"
            plt.savefig(output_file, dpi=self.dpi, bbox_inches='tight')
            print(f"✓ Saved: {output_file}")
        
        plt.close()
    
    def plot_comparison_with_re(self, save: bool = True):
        """Plot comparison with Python re (Experiment 8)."""
        data = self.load_experiment_data("exp8_compare_with_re")
        
        pattern_lengths = [d['pattern_length'] for d in data]
        bm_times = [d['bm_mean_time'] * 1000 for d in data]
        re_times = [d['re_mean_time'] * 1000 for d in data]
        speedups = [d['speedup'] for d in data]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Time comparison
        x = np.arange(len(pattern_lengths))
        width = 0.35
        
        bars1 = ax1.bar(x - width/2, bm_times, width, label='Boyer-Moore', alpha=0.8, color='#3498db', edgecolor='black')
        bars2 = ax1.bar(x + width/2, re_times, width, label='Python re', alpha=0.8, color='#e74c3c', edgecolor='black')
        
        ax1.set_xlabel('Pattern Length (bp)', fontsize=12)
        ax1.set_ylabel('Time (ms)', fontsize=12)
        ax1.set_title('Boyer-Moore vs Python re', fontsize=14, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(pattern_lengths)
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)
        
        # Speedup plot
        colors = ['#2ecc71' if s > 1 else '#e74c3c' for s in speedups]
        bars = ax2.bar(pattern_lengths, speedups, color=colors, alpha=0.8, edgecolor='black')
        ax2.axhline(y=1.0, color='black', linestyle='--', linewidth=1, label='Equal performance')
        ax2.set_xlabel('Pattern Length (bp)', fontsize=12)
        ax2.set_ylabel('Speedup (×)', fontsize=12)
        ax2.set_title('Boyer-Moore Speedup over Python re', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(axis='y', alpha=0.3)
        
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}×', ha='center', va='bottom' if height > 1 else 'top', fontsize=9)
        
        plt.tight_layout()
        
        if save:
            output_file = self.output_dir / "comparison_with_re.png"
            plt.savefig(output_file, dpi=self.dpi, bbox_inches='tight')
            print(f"✓ Saved: {output_file}")
        
        plt.close()
    
    def create_all_plots(self):
        """Generate all visualization plots."""
        print("\n" + "=" * 60)
        print("GENERATING VISUALIZATIONS")
        print("=" * 60)
        
        plots = [
            ("Pattern Length vs Time", self.plot_pattern_length_vs_time),
            ("Text Scaling", self.plot_text_scaling),
            ("Alphabet Effect", self.plot_alphabet_effect),
            ("Heuristic Contribution", self.plot_heuristic_contribution),
            ("Preprocessing Overhead", self.plot_preprocessing_overhead),
            ("Memory Footprint", self.plot_memory_footprint),
            ("Real Motifs", self.plot_real_motifs),
            ("Comparison with re", self.plot_comparison_with_re),
        ]
        
        for plot_name, plot_func in plots:
            try:
                print(f"\nGenerating: {plot_name}")
                plot_func(save=True)
            except Exception as e:
                print(f"❌ Error generating {plot_name}: {e}")
        
        print("\n" + "=" * 60)
        print("✓ ALL VISUALIZATIONS COMPLETE")
        print("=" * 60)


if __name__ == "__main__":
    viz = Visualizer()
    viz.create_all_plots()
