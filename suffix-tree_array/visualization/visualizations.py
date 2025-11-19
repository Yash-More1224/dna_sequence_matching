"""
Visualization Module for Suffix Array Experiments

Create plots and visualizations for performance analysis:
- Performance plots (time, memory, scaling)
- Comparison plots (vs re, pattern complexity)
- Match visualizations
"""

import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from typing import List, Dict, Any
import sys

sys.path.append('..')


class Visualizer:
    """Create visualizations for Suffix Array experiments."""
    
    def __init__(self, output_dir: str = "results/plots", style: str = "seaborn-v0_8-darkgrid"):
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
        if not filepath.exists():
            print(f"Warning: {filepath} not found")
            return []
        
        with open(filepath, 'r') as f:
            return json.load(f)
    
    def plot_pattern_length_vs_time(self, save: bool = True):
        """Plot latency vs pattern length (Experiment 1)."""
        data = self.load_experiment_data("exp1_pattern_length")
        if not data:
            return
        
        pattern_lengths = [d['pattern_length'] for d in data]
        total_times = [d['mean_total_time'] * 1000 for d in data]  # Convert to ms
        search_times = [d['mean_search_time'] * 1000 for d in data]
        preprocess_times = [d['mean_preprocessing_time'] * 1000 for d in data]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Search time plot
        ax1.plot(pattern_lengths, search_times, 'o-', linewidth=2, markersize=8, 
                label='Search Time', color='#2E86AB')
        ax1.set_xlabel('Pattern Length (bp)', fontsize=12)
        ax1.set_ylabel('Search Time (ms)', fontsize=12)
        ax1.set_title('Suffix Array: Search Time vs Pattern Length', 
                     fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Time breakdown
        ax2.plot(pattern_lengths, search_times, 's-', linewidth=2, markersize=8, 
                label='Search Time', color='#2E86AB')
        ax2.plot(pattern_lengths, preprocess_times, '^-', linewidth=2, markersize=8, 
                label='Preprocessing Time', color='#A23B72')
        ax2.set_xlabel('Pattern Length (bp)', fontsize=12)
        ax2.set_ylabel('Time (ms)', fontsize=12)
        ax2.set_title('Time Breakdown', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        ax2.set_yscale('log')
        
        plt.tight_layout()
        
        if save:
            output_file = self.output_dir / "exp1_pattern_length_vs_time.png"
            plt.savefig(output_file, dpi=self.dpi, bbox_inches='tight')
            print(f"✓ Saved: {output_file}")
        
        plt.close()
    
    def plot_text_scaling(self, save: bool = True):
        """Plot scalability with text size (Experiment 2)."""
        data = self.load_experiment_data("exp2_text_scaling")
        if not data:
            return
        
        text_sizes = [d['text_length'] / 1000 for d in data]  # Convert to KB
        search_times = [d['mean_search_time'] * 1000 for d in data]  # Convert to ms
        throughputs = [d['throughput_mbps'] for d in data]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Search time scaling
        ax1.plot(text_sizes, search_times, 'o-', linewidth=2, markersize=8, 
                color='#2E86AB')
        ax1.set_xlabel('Text Size (KB)', fontsize=12)
        ax1.set_ylabel('Search Time (ms)', fontsize=12)
        ax1.set_title('Suffix Array: Search Time vs Text Size', 
                     fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.set_xscale('log')
        
        # Throughput
        ax2.plot(text_sizes, throughputs, 's-', linewidth=2, markersize=8, 
                color='#F18F01')
        ax2.set_xlabel('Text Size (KB)', fontsize=12)
        ax2.set_ylabel('Throughput (MB/s)', fontsize=12)
        ax2.set_title('Search Throughput', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.set_xscale('log')
        
        plt.tight_layout()
        
        if save:
            output_file = self.output_dir / "exp2_text_scaling.png"
            plt.savefig(output_file, dpi=self.dpi, bbox_inches='tight')
            print(f"✓ Saved: {output_file}")
        
        plt.close()
    
    def plot_preprocessing_cost(self, save: bool = True):
        """Plot preprocessing overhead (Experiment 3)."""
        data = self.load_experiment_data("exp3_preprocessing")
        if not data:
            return
        
        text_sizes = [d['text_size'] / 1000 for d in data]  # KB
        construction_times = [d['construction_time'] * 1000 for d in data]  # ms
        search_times = [d['search_time'] * 1000 for d in data]  # ms
        ratios = [d['preprocessing_ratio'] * 100 for d in data]  # percentage
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Stacked bar chart
        ax1.bar(text_sizes, construction_times, label='Construction', 
               color='#A23B72', alpha=0.8)
        ax1.bar(text_sizes, search_times, bottom=construction_times, 
               label='Search', color='#2E86AB', alpha=0.8)
        ax1.set_xlabel('Text Size (KB)', fontsize=12)
        ax1.set_ylabel('Time (ms)', fontsize=12)
        ax1.set_title('Construction vs Search Time', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y')
        ax1.set_yscale('log')
        
        # Preprocessing ratio
        ax2.plot(text_sizes, ratios, 'o-', linewidth=2, markersize=8, 
                color='#C73E1D')
        ax2.set_xlabel('Text Size (KB)', fontsize=12)
        ax2.set_ylabel('Preprocessing Overhead (%)', fontsize=12)
        ax2.set_title('Preprocessing Cost Ratio', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='50%')
        ax2.legend()
        
        plt.tight_layout()
        
        if save:
            output_file = self.output_dir / "exp3_preprocessing_cost.png"
            plt.savefig(output_file, dpi=self.dpi, bbox_inches='tight')
            print(f"✓ Saved: {output_file}")
        
        plt.close()
    
    def plot_memory_footprint(self, save: bool = True):
        """Plot memory usage (Experiment 4)."""
        data = self.load_experiment_data("exp4_memory")
        if not data:
            return
        
        text_sizes = [d['text_size'] / 1000 for d in data]  # KB
        memory_mb = [d['index_memory_mb'] for d in data]
        memory_per_char = [d['memory_per_char'] for d in data]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Total memory
        ax1.plot(text_sizes, memory_mb, 'o-', linewidth=2, markersize=8, 
                color='#6A4C93')
        ax1.set_xlabel('Text Size (KB)', fontsize=12)
        ax1.set_ylabel('Index Memory (MB)', fontsize=12)
        ax1.set_title('Suffix Array: Memory Footprint', 
                     fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.set_xscale('log')
        ax1.set_yscale('log')
        
        # Memory per character
        ax2.plot(text_sizes, memory_per_char, 's-', linewidth=2, markersize=8, 
                color='#1982C4')
        ax2.set_xlabel('Text Size (KB)', fontsize=12)
        ax2.set_ylabel('Memory per Character (bytes)', fontsize=12)
        ax2.set_title('Memory Efficiency', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=16, color='gray', linestyle='--', alpha=0.5, 
                   label='Expected: 16 bytes')
        ax2.legend()
        
        plt.tight_layout()
        
        if save:
            output_file = self.output_dir / "exp4_memory_footprint.png"
            plt.savefig(output_file, dpi=self.dpi, bbox_inches='tight')
            print(f"✓ Saved: {output_file}")
        
        plt.close()
    
    def plot_comparison_with_re(self, save: bool = True):
        """Plot comparison with Python re (Experiment 5)."""
        data = self.load_experiment_data("exp5_compare_re")
        if not data:
            return
        
        pattern_lengths = [d['pattern_length'] for d in data]
        sa_times = [d['sa_search_time'] * 1000 for d in data]  # ms
        re_times = [d['re_time'] * 1000 for d in data]  # ms
        speedups = [d['speedup'] for d in data]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Time comparison
        x = np.arange(len(pattern_lengths))
        width = 0.35
        
        ax1.bar(x - width/2, sa_times, width, label='Suffix Array', 
               color='#2E86AB', alpha=0.8)
        ax1.bar(x + width/2, re_times, width, label='Python re', 
               color='#F18F01', alpha=0.8)
        ax1.set_xlabel('Pattern Length (bp)', fontsize=12)
        ax1.set_ylabel('Search Time (ms)', fontsize=12)
        ax1.set_title('Suffix Array vs Python re', fontsize=14, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(pattern_lengths)
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Speedup
        colors = ['green' if s > 1 else 'red' for s in speedups]
        ax2.bar(pattern_lengths, speedups, color=colors, alpha=0.7)
        ax2.axhline(y=1, color='black', linestyle='--', linewidth=2, 
                   label='Equal performance')
        ax2.set_xlabel('Pattern Length (bp)', fontsize=12)
        ax2.set_ylabel('Speedup Factor', fontsize=12)
        ax2.set_title('Speedup (>1 means SA faster)', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save:
            output_file = self.output_dir / "exp5_comparison_with_re.png"
            plt.savefig(output_file, dpi=self.dpi, bbox_inches='tight')
            print(f"✓ Saved: {output_file}")
        
        plt.close()
    
    def plot_repeat_discovery(self, save: bool = True):
        """Plot repeat discovery performance (Experiment 6)."""
        data = self.load_experiment_data("exp6_repeat_discovery")
        if not data:
            return
        
        min_lengths = [d['min_length'] for d in data]
        num_repeats = [d['num_repeats_found'] for d in data]
        discovery_times = [d['discovery_time'] * 1000 for d in data]  # ms
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Number of repeats found
        ax1.bar(min_lengths, num_repeats, color='#8338EC', alpha=0.8)
        ax1.set_xlabel('Minimum Repeat Length (bp)', fontsize=12)
        ax1.set_ylabel('Number of Repeats Found', fontsize=12)
        ax1.set_title('Repeat Discovery: Count vs Min Length', 
                     fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Discovery time
        ax2.plot(min_lengths, discovery_times, 'o-', linewidth=2, markersize=8, 
                color='#FB5607')
        ax2.set_xlabel('Minimum Repeat Length (bp)', fontsize=12)
        ax2.set_ylabel('Discovery Time (ms)', fontsize=12)
        ax2.set_title('Repeat Discovery Performance', 
                     fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            output_file = self.output_dir / "exp6_repeat_discovery.png"
            plt.savefig(output_file, dpi=self.dpi, bbox_inches='tight')
            print(f"✓ Saved: {output_file}")
        
        plt.close()
    
    def plot_pattern_complexity(self, save: bool = True):
        """Plot pattern complexity analysis (Experiment 8)."""
        data = self.load_experiment_data("exp8_pattern_complexity")
        if not data:
            return
        
        pattern_types = [d['pattern_type'] for d in data]
        search_times = [d['search_time'] * 1000 for d in data]  # ms
        num_matches = [d['num_matches'] for d in data]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Search times
        colors = plt.cm.Set3(np.linspace(0, 1, len(pattern_types)))
        ax1.barh(pattern_types, search_times, color=colors, alpha=0.8)
        ax1.set_xlabel('Search Time (ms)', fontsize=12)
        ax1.set_ylabel('Pattern Type', fontsize=12)
        ax1.set_title('Search Time by Pattern Complexity', 
                     fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='x')
        
        # Matches found
        ax2.barh(pattern_types, num_matches, color=colors, alpha=0.8)
        ax2.set_xlabel('Number of Matches', fontsize=12)
        ax2.set_ylabel('Pattern Type', fontsize=12)
        ax2.set_title('Matches Found by Pattern Type', 
                     fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        
        if save:
            output_file = self.output_dir / "exp8_pattern_complexity.png"
            plt.savefig(output_file, dpi=self.dpi, bbox_inches='tight')
            print(f"✓ Saved: {output_file}")
        
        plt.close()
    
    def create_all_visualizations(self):
        """Create all experiment visualizations."""
        print("\n" + "=" * 70)
        print("GENERATING VISUALIZATIONS")
        print("=" * 70)
        print()
        
        visualizations = [
            ("Pattern Length vs Time", self.plot_pattern_length_vs_time),
            ("Text Scaling", self.plot_text_scaling),
            ("Preprocessing Cost", self.plot_preprocessing_cost),
            ("Memory Footprint", self.plot_memory_footprint),
            ("Comparison with re", self.plot_comparison_with_re),
            ("Repeat Discovery", self.plot_repeat_discovery),
            ("Pattern Complexity", self.plot_pattern_complexity),
        ]
        
        for name, viz_func in visualizations:
            try:
                print(f"Creating {name}...", end=" ")
                viz_func()
                print("✓")
            except Exception as e:
                print(f"❌ Error: {e}")
        
        print()
        print("=" * 70)
        print(f"✓ Visualizations saved to: {self.output_dir}")
        print("=" * 70)


if __name__ == "__main__":
    visualizer = Visualizer()
    visualizer.create_all_visualizations()
