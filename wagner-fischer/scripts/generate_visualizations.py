"""
Visualization and Report Generation for Wagner-Fischer Evaluation
Generates plots and comprehensive evaluation report.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict, List
import sys


class WagnerFischerVisualizer:
    """
    Generate visualizations and reports for WF evaluation.
    """
    
    def __init__(self, results_dir: str = "results"):
        """
        Initialize visualizer.
        
        Args:
            results_dir: Directory containing results
        """
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Set matplotlib style
        plt.style.use('seaborn-v0_8-darkgrid' if 'seaborn-v0_8-darkgrid' in plt.style.available 
                     else 'default')
    
    def load_metrics(self, filename: str = 'metrics.json') -> Dict:
        """Load metrics from JSON file."""
        filepath = self.results_dir / filename
        with open(filepath, 'r') as f:
            return json.load(f)
    
    def plot_accuracy_vs_mutation_rate(self, metrics: Dict):
        """
        Plot precision, recall, and F1 vs mutation rate.
        
        Args:
            metrics: Metrics dictionary
        """
        accuracy_data = metrics['accuracy']
        
        mutation_rates = [a['mutation_rate'] for a in accuracy_data]
        precisions = [a['precision'] for a in accuracy_data]
        recalls = [a['recall'] for a in accuracy_data]
        f1_scores = [a['f1_score'] for a in accuracy_data]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        x = np.array(mutation_rates) * 100  # Convert to percentage
        
        ax.plot(x, precisions, 'o-', label='Precision', linewidth=2, markersize=8)
        ax.plot(x, recalls, 's-', label='Recall', linewidth=2, markersize=8)
        ax.plot(x, f1_scores, '^-', label='F1 Score', linewidth=2, markersize=8)
        
        ax.set_xlabel('Mutation Rate (%)', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title('Accuracy Metrics vs Mutation Rate', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1.05])
        
        plt.tight_layout()
        filepath = self.results_dir / 'accuracy_vs_mutation_rate.png'
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {filepath}")
    
    def plot_scalability(self, metrics: Dict):
        """
        Plot scalability metrics.
        
        Args:
            metrics: Metrics dictionary
        """
        scalability_data = metrics['scalability']
        
        pattern_counts = [s['pattern_count'] for s in scalability_data]
        runtimes = [s['total_runtime'] for s in scalability_data]
        throughputs = [s['throughput'] for s in scalability_data]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Runtime plot
        ax1.plot(pattern_counts, runtimes, 'o-', linewidth=2, markersize=8, color='#2E86AB')
        ax1.set_xlabel('Number of Patterns', fontsize=12)
        ax1.set_ylabel('Total Runtime (seconds)', fontsize=12)
        ax1.set_title('Scalability: Runtime vs Pattern Count', fontsize=13, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Throughput plot
        ax2.plot(pattern_counts, throughputs, 's-', linewidth=2, markersize=8, color='#A23B72')
        ax2.set_xlabel('Number of Patterns', fontsize=12)
        ax2.set_ylabel('Throughput (ops/sec)', fontsize=12)
        ax2.set_title('Scalability: Throughput vs Pattern Count', fontsize=13, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        filepath = self.results_dir / 'scalability.png'
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {filepath}")
    
    def plot_robustness(self, metrics: Dict):
        """
        Plot robustness metrics.
        
        Args:
            metrics: Metrics dictionary
        """
        robustness_data = metrics['robustness']
        
        mutation_rates = [r['mutation_rate'] for r in robustness_data]
        f1_scores = [r['f1_score'] for r in robustness_data]
        latencies = [r['mean_latency'] for r in robustness_data]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        x = np.array(mutation_rates) * 100
        
        # F1 score plot
        ax1.plot(x, f1_scores, 'o-', linewidth=2, markersize=8, color='#F18F01')
        ax1.set_xlabel('Mutation Rate (%)', fontsize=12)
        ax1.set_ylabel('F1 Score', fontsize=12)
        ax1.set_title('Robustness: F1 Score vs Mutation Rate', fontsize=13, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim([0, 1.05])
        
        # Latency plot
        ax2.plot(x, latencies, 's-', linewidth=2, markersize=8, color='#C73E1D')
        ax2.set_xlabel('Mutation Rate (%)', fontsize=12)
        ax2.set_ylabel('Mean Latency (seconds)', fontsize=12)
        ax2.set_title('Robustness: Latency vs Mutation Rate', fontsize=13, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        filepath = self.results_dir / 'robustness.png'
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {filepath}")
    
    def plot_performance_summary(self, metrics: Dict):
        """
        Create performance summary visualization.
        
        Args:
            metrics: Metrics dictionary
        """
        perf = metrics['performance']
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Latency distribution
        ax1 = axes[0, 0]
        latency_stats = [perf['min_latency'], perf['mean_latency'], 
                        perf['median_latency'], perf['max_latency']]
        labels = ['Min', 'Mean', 'Median', 'Max']
        colors = ['#06A77D', '#2E86AB', '#A23B72', '#F18F01']
        bars = ax1.bar(labels, latency_stats, color=colors, alpha=0.7, edgecolor='black')
        ax1.set_ylabel('Latency (seconds)', fontsize=11)
        ax1.set_title('Latency Statistics', fontsize=12, fontweight='bold')
        ax1.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.4f}s', ha='center', va='bottom', fontsize=9)
        
        # Memory usage
        ax2 = axes[0, 1]
        memory_data = [perf['peak_memory_mb'], perf['peak_rss_mb']]
        memory_labels = ['Peak Tracemalloc', 'Peak RSS']
        bars = ax2.bar(memory_labels, memory_data, color=['#2E86AB', '#A23B72'], 
                      alpha=0.7, edgecolor='black')
        ax2.set_ylabel('Memory (MB)', fontsize=11)
        ax2.set_title('Memory Usage', fontsize=12, fontweight='bold')
        ax2.grid(axis='y', alpha=0.3)
        
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f} MB', ha='center', va='bottom', fontsize=9)
        
        # Throughput
        ax3 = axes[1, 0]
        ax3.bar(['Throughput'], [perf['throughput']], color='#06A77D', 
               alpha=0.7, edgecolor='black', width=0.5)
        ax3.set_ylabel('Operations / Second', fontsize=11)
        ax3.set_title('Throughput', fontsize=12, fontweight='bold')
        ax3.grid(axis='y', alpha=0.3)
        ax3.text(0, perf['throughput'], f"{perf['throughput']:.2f} ops/s",
                ha='center', va='bottom', fontsize=10)
        
        # Summary text
        ax4 = axes[1, 1]
        ax4.axis('off')
        summary_text = f"""
Performance Summary
{'='*40}

Total Runtime: {perf['total_runtime']:.3f} seconds
Mean Latency: {perf['mean_latency']:.4f} seconds
Std Dev: {perf['std_latency']:.4f} seconds
Throughput: {perf['throughput']:.2f} ops/sec

Peak Memory: {perf['peak_memory_mb']:.2f} MB
Peak RSS: {perf['peak_rss_mb']:.2f} MB

Preprocessing: {perf['preprocessing_time']:.3f} seconds
(WF has no preprocessing overhead)
        """
        ax4.text(0.1, 0.5, summary_text, fontsize=10, family='monospace',
                verticalalignment='center')
        
        plt.tight_layout()
        filepath = self.results_dir / 'performance_summary.png'
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {filepath}")
    
    def generate_text_report(self, metrics: Dict, output_file: str = 'wf_evaluation_report.txt'):
        """
        Generate comprehensive text evaluation report.
        
        Args:
            metrics: Metrics dictionary
            output_file: Output filename
        """
        filepath = self.results_dir / output_file
        
        with open(filepath, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("WAGNER-FISCHER ALGORITHM EVALUATION REPORT\n")
            f.write("Comprehensive Performance and Accuracy Analysis\n")
            f.write("=" * 80 + "\n\n")
            
            # Dataset info
            f.write("DATASET INFORMATION\n")
            f.write("-" * 80 + "\n")
            f.write(f"Dataset: {metrics['dataset']}\n")
            f.write(f"Pattern Length: {metrics['pattern_length']} bp\n")
            f.write(f"Edit Distance Threshold: {metrics['threshold']}\n")
            f.write("\n")
            
            # Performance metrics
            f.write("PERFORMANCE METRICS\n")
            f.write("-" * 80 + "\n")
            perf = metrics['performance']
            f.write(f"Total Runtime: {perf['total_runtime']:.6f} seconds\n")
            f.write(f"Preprocessing Time: {perf['preprocessing_time']:.6f} seconds\n")
            f.write(f"  (Note: Wagner-Fischer has no preprocessing overhead)\n\n")
            
            f.write("Latency Statistics:\n")
            f.write(f"  Mean Latency: {perf['mean_latency']:.6f} seconds\n")
            f.write(f"  Median Latency: {perf['median_latency']:.6f} seconds\n")
            f.write(f"  Std Deviation: {perf['std_latency']:.6f} seconds\n")
            f.write(f"  Min Latency: {perf['min_latency']:.6f} seconds\n")
            f.write(f"  Max Latency: {perf['max_latency']:.6f} seconds\n\n")
            
            f.write(f"Throughput: {perf['throughput']:.2f} operations/second\n\n")
            
            f.write("Memory Usage:\n")
            f.write(f"  Peak Tracemalloc Memory: {perf['peak_memory_mb']:.2f} MB\n")
            f.write(f"  Peak RSS Memory: {perf['peak_rss_mb']:.2f} MB\n")
            f.write("\n")
            
            # Accuracy metrics
            f.write("ACCURACY EVALUATION\n")
            f.write("-" * 80 + "\n")
            f.write(f"{'Mutation Rate':<15} {'Precision':<12} {'Recall':<12} {'F1 Score':<12} {'TP':<8} {'FP':<8} {'FN':<8}\n")
            f.write("-" * 80 + "\n")
            
            for acc in metrics['accuracy']:
                f.write(f"{acc['mutation_rate']*100:>6.1f}%        ")
                f.write(f"{acc['precision']:>8.4f}    ")
                f.write(f"{acc['recall']:>8.4f}    ")
                f.write(f"{acc['f1_score']:>8.4f}    ")
                f.write(f"{acc['true_positives']:>4}    ")
                f.write(f"{acc['false_positives']:>4}    ")
                f.write(f"{acc['false_negatives']:>4}\n")
            f.write("\n")
            
            # Scalability
            f.write("SCALABILITY ANALYSIS\n")
            f.write("-" * 80 + "\n")
            f.write(f"{'Pattern Count':<15} {'Runtime (s)':<15} {'Throughput (ops/s)':<20} {'Memory (MB)':<15}\n")
            f.write("-" * 80 + "\n")
            
            for scal in metrics['scalability']:
                f.write(f"{scal['pattern_count']:<15} ")
                f.write(f"{scal['total_runtime']:<15.4f} ")
                f.write(f"{scal['throughput']:<20.2f} ")
                f.write(f"{scal['peak_memory_mb']:<15.2f}\n")
            f.write("\n")
            
            # Calculate weak/strong scaling
            if len(metrics['scalability']) > 1:
                first = metrics['scalability'][0]
                last = metrics['scalability'][-1]
                speedup = first['total_runtime'] / last['total_runtime'] * \
                         (last['pattern_count'] / first['pattern_count'])
                f.write(f"Weak Scaling Factor: {speedup:.3f}\n")
                f.write(f"  (Expected: 1.0 for perfect weak scaling)\n\n")
            
            # Robustness
            f.write("ROBUSTNESS EVALUATION\n")
            f.write("-" * 80 + "\n")
            f.write(f"{'Mutation Rate':<15} {'F1 Score':<12} {'Mean Latency (s)':<20} {'Throughput':<15}\n")
            f.write("-" * 80 + "\n")
            
            for rob in metrics['robustness']:
                f.write(f"{rob['mutation_rate']*100:>6.1f}%        ")
                f.write(f"{rob['f1_score']:>8.4f}    ")
                f.write(f"{rob['mean_latency']:>15.6f}     ")
                f.write(f"{rob['throughput']:>10.2f}\n")
            f.write("\n")
            
            # Algorithm details
            f.write("ALGORITHM IMPLEMENTATION DETAILS\n")
            f.write("-" * 80 + "\n")
            f.write("Wagner-Fischer Algorithm (Levenshtein Edit Distance)\n\n")
            f.write("Variants Implemented:\n")
            f.write("  1. Full Matrix DP: O(m*n) time, O(m*n) space\n")
            f.write("  2. Space-Optimized: O(m*n) time, O(min(m,n)) space\n")
            f.write("  3. Threshold-Based: Early termination with Ukkonen's optimization\n")
            f.write("  4. Banded DP: O(m*k) time, O(m*k) space (k=band width)\n\n")
            
            f.write("Features:\n")
            f.write("  - Traceback for alignment reconstruction\n")
            f.write("  - Configurable operation costs (substitution, insertion, deletion)\n")
            f.write("  - Sliding window approximate pattern matching\n")
            f.write("  - Support for DNA sequences (A, C, G, T alphabet)\n\n")
            
            # Generated plots
            f.write("GENERATED VISUALIZATIONS\n")
            f.write("-" * 80 + "\n")
            f.write("  - accuracy_vs_mutation_rate.png: Precision, Recall, F1 vs mutation rate\n")
            f.write("  - scalability.png: Runtime and throughput vs pattern count\n")
            f.write("  - robustness.png: F1 score and latency vs mutation rate\n")
            f.write("  - performance_summary.png: Comprehensive performance overview\n")
            f.write("\n")
            
            # Conclusions
            f.write("KEY FINDINGS\n")
            f.write("-" * 80 + "\n")
            
            # Best F1 score
            best_acc = max(metrics['accuracy'], key=lambda x: x['f1_score'])
            f.write(f"Best Accuracy: F1={best_acc['f1_score']:.4f} at {best_acc['mutation_rate']*100:.1f}% mutation rate\n")
            
            # Performance insights
            f.write(f"Average Query Latency: {perf['mean_latency']:.4f} seconds\n")
            f.write(f"Memory Efficiency: {perf['peak_memory_mb']:.2f} MB peak usage\n")
            
            # Scalability insights
            if len(metrics['scalability']) > 1:
                throughput_trend = metrics['scalability'][-1]['throughput'] / metrics['scalability'][0]['throughput']
                f.write(f"Scalability: {throughput_trend:.2f}x throughput change from "
                       f"{metrics['scalability'][0]['pattern_count']} to "
                       f"{metrics['scalability'][-1]['pattern_count']} patterns\n")
            
            f.write("\n")
            f.write("REPRODUCIBILITY\n")
            f.write("-" * 80 + "\n")
            f.write("All experiments can be reproduced using:\n")
            f.write("  1. scripts/generate_ground_truth.py - Generate ground truth datasets\n")
            f.write("  2. scripts/benchmark.py - Run comprehensive benchmarks\n")
            f.write("  3. scripts/generate_visualizations.py - Generate plots and report\n\n")
            f.write("Metrics saved in: results/metrics.json\n")
            f.write("\n")
            
            f.write("=" * 80 + "\n")
            f.write("End of Report\n")
            f.write("=" * 80 + "\n")
        
        print(f"Report saved to: {filepath}")
    
    def generate_all_visualizations(self, metrics_file: str = 'metrics.json'):
        """
        Generate all visualizations and report.
        
        Args:
            metrics_file: Metrics JSON filename
        """
        print("Loading metrics...")
        metrics = self.load_metrics(metrics_file)
        
        print("\nGenerating visualizations...")
        self.plot_accuracy_vs_mutation_rate(metrics)
        self.plot_scalability(metrics)
        self.plot_robustness(metrics)
        self.plot_performance_summary(metrics)
        
        print("\nGenerating text report...")
        self.generate_text_report(metrics)
        
        print("\n✓ All visualizations and report generated successfully!")


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate WF visualizations and report')
    parser.add_argument('--results-dir', default='results', help='Results directory')
    parser.add_argument('--metrics-file', default='metrics.json', help='Metrics JSON file')
    
    args = parser.parse_args()
    
    visualizer = WagnerFischerVisualizer(results_dir=args.results_dir)
    visualizer.generate_all_visualizations(metrics_file=args.metrics_file)


if __name__ == '__main__':
    main()
