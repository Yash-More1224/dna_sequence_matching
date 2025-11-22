#!/usr/bin/env python3
"""
Generate Visualizations for Boyer-Moore Evaluation Results

Creates comprehensive plots and charts from evaluation results.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Any


class VisualizationGenerator:
    """Generate visualizations from evaluation results."""
    
    def __init__(self, results_dir: str = "results", output_dir: str = "results/plots"):
        """
        Initialize visualization generator.
        
        Args:
            results_dir: Directory containing result JSON files
            output_dir: Directory to save plots
        """
        self.results_dir = Path(results_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['figure.dpi'] = 300
    
    def load_results(self, dataset_name: str) -> Dict[str, Any]:
        """Load results for a dataset."""
        filepath = self.results_dir / f"evaluation_{dataset_name}.json"
        
        if not filepath.exists():
            raise FileNotFoundError(f"Results file not found: {filepath}")
        
        with open(filepath, 'r') as f:
            return json.load(f)
    
    def plot_pattern_length_scalability(self, results: Dict[str, Any], 
                                       dataset_name: str):
        """Plot pattern length vs performance metrics."""
        scalability = results['evaluation_results']['scalability']
        
        pattern_lengths = [r['pattern_length'] for r in scalability]
        times = [r['mean_time_ms'] for r in scalability]
        throughputs = [r['throughput_mbps'] for r in scalability]
        comparisons = [r['comparisons'] for r in scalability]
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Pattern Length Scalability - {dataset_name.upper()}', 
                    fontsize=16, fontweight='bold')
        
        # 1. Time vs Pattern Length
        ax1 = axes[0, 0]
        ax1.plot(pattern_lengths, times, 'o-', linewidth=2, markersize=8)
        ax1.set_xlabel('Pattern Length (bp)', fontsize=12)
        ax1.set_ylabel('Mean Time (ms)', fontsize=12)
        ax1.set_title('Execution Time vs Pattern Length')
        ax1.grid(True, alpha=0.3)
        ax1.set_xscale('log', base=2)
        
        # 2. Throughput vs Pattern Length
        ax2 = axes[0, 1]
        ax2.plot(pattern_lengths, throughputs, 's-', color='green', 
                linewidth=2, markersize=8)
        ax2.set_xlabel('Pattern Length (bp)', fontsize=12)
        ax2.set_ylabel('Throughput (MB/s)', fontsize=12)
        ax2.set_title('Throughput vs Pattern Length')
        ax2.grid(True, alpha=0.3)
        ax2.set_xscale('log', base=2)
        
        # 3. Comparisons vs Pattern Length
        ax3 = axes[1, 0]
        ax3.plot(pattern_lengths, comparisons, '^-', color='red', 
                linewidth=2, markersize=8)
        ax3.set_xlabel('Pattern Length (bp)', fontsize=12)
        ax3.set_ylabel('Number of Comparisons', fontsize=12)
        ax3.set_title('Character Comparisons vs Pattern Length')
        ax3.grid(True, alpha=0.3)
        ax3.set_xscale('log', base=2)
        ax3.set_yscale('log')
        
        # 4. Efficiency (comparisons per character)
        ax4 = axes[1, 1]
        text_length = scalability[0]['text_length']
        efficiency = [c / text_length for c in comparisons]
        ax4.plot(pattern_lengths, efficiency, 'd-', color='purple', 
                linewidth=2, markersize=8)
        ax4.set_xlabel('Pattern Length (bp)', fontsize=12)
        ax4.set_ylabel('Comparisons per Character', fontsize=12)
        ax4.set_title('Algorithm Efficiency')
        ax4.grid(True, alpha=0.3)
        ax4.set_xscale('log', base=2)
        
        plt.tight_layout()
        output_file = self.output_dir / f"{dataset_name}_pattern_scalability.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Saved: {output_file}")
    
    def plot_text_scaling(self, results: Dict[str, Any], dataset_name: str):
        """Plot text size scaling."""
        text_scaling = results['evaluation_results']['text_scaling']
        
        text_sizes = [r['text_length'] for r in text_scaling]
        times = [r['mean_time_ms'] for r in text_scaling]
        throughputs = [r['throughput_mbps'] for r in text_scaling]
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle(f'Text Size Scaling - {dataset_name.upper()}', 
                    fontsize=16, fontweight='bold')
        
        # 1. Time vs Text Size
        ax1 = axes[0]
        ax1.plot(text_sizes, times, 'o-', linewidth=2, markersize=8, color='blue')
        ax1.set_xlabel('Text Size (bp)', fontsize=12)
        ax1.set_ylabel('Mean Time (ms)', fontsize=12)
        ax1.set_title('Execution Time vs Text Size')
        ax1.grid(True, alpha=0.3)
        ax1.set_xscale('log')
        ax1.set_yscale('log')
        
        # Add linear reference line
        if len(text_sizes) > 1:
            # Fit line through first and last points
            x_ref = np.array([text_sizes[0], text_sizes[-1]])
            y_ref = np.array([times[0], times[-1]])
            ax1.plot(x_ref, y_ref, '--', alpha=0.5, color='red', 
                    label='Linear scaling')
            ax1.legend()
        
        # 2. Throughput vs Text Size
        ax2 = axes[1]
        ax2.plot(text_sizes, throughputs, 's-', linewidth=2, markersize=8, 
                color='green')
        ax2.set_xlabel('Text Size (bp)', fontsize=12)
        ax2.set_ylabel('Throughput (MB/s)', fontsize=12)
        ax2.set_title('Throughput vs Text Size')
        ax2.grid(True, alpha=0.3)
        ax2.set_xscale('log')
        
        plt.tight_layout()
        output_file = self.output_dir / f"{dataset_name}_text_scaling.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Saved: {output_file}")
    
    def plot_variant_comparison(self, results: Dict[str, Any], dataset_name: str):
        """Plot comparison of algorithm variants."""
        variants_data = results['evaluation_results']['variants']
        
        variants = [r['variant'] for r in variants_data]
        times = [r['mean_time_ms'] for r in variants_data]
        comparisons = [r['comparisons'] for r in variants_data]
        shifts = [r['shifts'] for r in variants_data]
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle(f'Algorithm Variants Comparison - {dataset_name.upper()}', 
                    fontsize=16, fontweight='bold')
        
        colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
        
        # 1. Execution Time
        ax1 = axes[0]
        bars1 = ax1.bar(variants, times, color=colors, alpha=0.8)
        ax1.set_ylabel('Mean Time (ms)', fontsize=12)
        ax1.set_title('Execution Time by Variant')
        ax1.tick_params(axis='x', rotation=15)
        ax1.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=10)
        
        # 2. Comparisons
        ax2 = axes[1]
        bars2 = ax2.bar(variants, comparisons, color=colors, alpha=0.8)
        ax2.set_ylabel('Number of Comparisons', fontsize=12)
        ax2.set_title('Character Comparisons by Variant')
        ax2.tick_params(axis='x', rotation=15)
        ax2.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bar in bars2:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height):,}', ha='center', va='bottom', fontsize=9)
        
        # 3. Shifts
        ax3 = axes[2]
        bars3 = ax3.bar(variants, shifts, color=colors, alpha=0.8)
        ax3.set_ylabel('Number of Shifts', fontsize=12)
        ax3.set_title('Pattern Shifts by Variant')
        ax3.tick_params(axis='x', rotation=15)
        ax3.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bar in bars3:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height):,}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        output_file = self.output_dir / f"{dataset_name}_variants.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Saved: {output_file}")
    
    def plot_motif_search(self, results: Dict[str, Any], dataset_name: str):
        """Plot biological motif search results."""
        motifs_data = results['evaluation_results']['motifs']
        
        patterns = [r['pattern'] for r in motifs_data]
        matches = [r['num_matches'] for r in motifs_data]
        times = [r['mean_time_ms'] for r in motifs_data]
        text_length = motifs_data[0]['text_length']
        densities = [m / (text_length / 1_000_000) for m in matches]
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle(f'Biological Motif Search - {dataset_name.upper()}', 
                    fontsize=16, fontweight='bold')
        
        # 1. Matches Found
        ax1 = axes[0]
        bars1 = ax1.bar(range(len(patterns)), matches, color='steelblue', alpha=0.8)
        ax1.set_xticks(range(len(patterns)))
        ax1.set_xticklabels(patterns, rotation=45, ha='right')
        ax1.set_ylabel('Number of Matches', fontsize=12)
        ax1.set_title('Motif Occurrences in Genome')
        ax1.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for i, bar in enumerate(bars1):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}', ha='center', va='bottom', fontsize=10)
        
        # 2. Match Density
        ax2 = axes[1]
        bars2 = ax2.bar(range(len(patterns)), densities, color='seagreen', alpha=0.8)
        ax2.set_xticks(range(len(patterns)))
        ax2.set_xticklabels(patterns, rotation=45, ha='right')
        ax2.set_ylabel('Matches per Megabase', fontsize=12)
        ax2.set_title('Motif Density')
        ax2.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for i, bar in enumerate(bars2):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        output_file = self.output_dir / f"{dataset_name}_motifs.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Saved: {output_file}")
    
    def plot_cross_dataset_comparison(self, all_results: Dict[str, Dict]):
        """Compare performance across all datasets."""
        dataset_names = list(all_results.keys())
        
        # Extract baseline performance (16bp pattern)
        baseline_times = []
        baseline_throughputs = []
        dataset_sizes = []
        
        for name in dataset_names:
            results = all_results[name]
            scalability = results['evaluation_results']['scalability']
            
            # Find 16bp result
            baseline = next((r for r in scalability if r['pattern_length'] == 16), 
                          scalability[2] if len(scalability) > 2 else scalability[0])
            
            baseline_times.append(baseline['mean_time_ms'])
            baseline_throughputs.append(baseline['throughput_mbps'])
            dataset_sizes.append(results['dataset_info']['length'])
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Cross-Dataset Performance Comparison', 
                    fontsize=16, fontweight='bold')
        
        # 1. Execution Time
        ax1 = axes[0, 0]
        bars1 = ax1.bar([n.upper() for n in dataset_names], baseline_times, 
                       color=['#3498db', '#e74c3c', '#2ecc71'], alpha=0.8)
        ax1.set_ylabel('Mean Time (ms)', fontsize=12)
        ax1.set_title('Execution Time (16bp pattern)')
        ax1.tick_params(axis='x', rotation=15)
        ax1.grid(axis='y', alpha=0.3)
        
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}', ha='center', va='bottom')
        
        # 2. Throughput
        ax2 = axes[0, 1]
        bars2 = ax2.bar([n.upper() for n in dataset_names], baseline_throughputs,
                       color=['#3498db', '#e74c3c', '#2ecc71'], alpha=0.8)
        ax2.set_ylabel('Throughput (MB/s)', fontsize=12)
        ax2.set_title('Data Throughput (16bp pattern)')
        ax2.tick_params(axis='x', rotation=15)
        ax2.grid(axis='y', alpha=0.3)
        
        for bar in bars2:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}', ha='center', va='bottom')
        
        # 3. Dataset Sizes
        ax3 = axes[1, 0]
        bars3 = ax3.bar([n.upper() for n in dataset_names], 
                       [s/1_000_000 for s in dataset_sizes],
                       color=['#3498db', '#e74c3c', '#2ecc71'], alpha=0.8)
        ax3.set_ylabel('Size (Mbp)', fontsize=12)
        ax3.set_title('Dataset Sizes')
        ax3.tick_params(axis='x', rotation=15)
        ax3.grid(axis='y', alpha=0.3)
        
        for i, bar in enumerate(bars3):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}', ha='center', va='bottom')
        
        # 4. GC Content
        ax4 = axes[1, 1]
        gc_contents = [all_results[n]['dataset_info']['gc_content'] 
                      for n in dataset_names]
        bars4 = ax4.bar([n.upper() for n in dataset_names], gc_contents,
                       color=['#3498db', '#e74c3c', '#2ecc71'], alpha=0.8)
        ax4.set_ylabel('GC Content (%)', fontsize=12)
        ax4.set_title('GC Content Comparison')
        ax4.tick_params(axis='x', rotation=15)
        ax4.grid(axis='y', alpha=0.3)
        ax4.set_ylim([0, 100])
        
        for bar in bars4:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%', ha='center', va='bottom')
        
        plt.tight_layout()
        output_file = self.output_dir / "cross_dataset_comparison.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Saved: {output_file}")
    
    def generate_all_visualizations(self):
        """Generate all visualizations."""
        print("\n" + "="*70)
        print("GENERATING VISUALIZATIONS")
        print("="*70)
        
        # Load all results
        all_results = {}
        
        for json_file in self.results_dir.glob("evaluation_*.json"):
            dataset_name = json_file.stem.replace('evaluation_', '')
            
            print(f"\nProcessing {dataset_name.upper()}...")
            
            try:
                results = self.load_results(dataset_name)
                all_results[dataset_name] = results
                
                # Generate individual plots
                self.plot_pattern_length_scalability(results, dataset_name)
                self.plot_text_scaling(results, dataset_name)
                self.plot_variant_comparison(results, dataset_name)
                self.plot_motif_search(results, dataset_name)
                
            except Exception as e:
                print(f"  ❌ Error: {e}")
        
        # Generate cross-dataset comparison
        if len(all_results) > 1:
            print(f"\nGenerating cross-dataset comparison...")
            self.plot_cross_dataset_comparison(all_results)
        
        print(f"\n{'='*70}")
        print(f"✓ All visualizations saved to {self.output_dir}")
        print(f"{'='*70}\n")


def main():
    """Main entry point."""
    script_dir = Path(__file__).parent
    results_dir = script_dir / "results"
    plots_dir = results_dir / "plots"
    
    print("\nBoyer-Moore Visualization Generator")
    print(f"Results directory: {results_dir}")
    print(f"Output directory: {plots_dir}")
    
    if not results_dir.exists():
        print(f"\n❌ Results directory not found: {results_dir}")
        print("Please run the evaluation script first.")
        return
    
    generator = VisualizationGenerator(
        results_dir=str(results_dir),
        output_dir=str(plots_dir)
    )
    
    generator.generate_all_visualizations()


if __name__ == "__main__":
    main()
