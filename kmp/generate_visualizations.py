#!/usr/bin/env python3
"""
Generate visualizations from comprehensive evaluation results.
"""

import sys
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from glob import glob

# Setup
RESULTS_DIR = Path(__file__).parent / "results"
BENCHMARKS_DIR = RESULTS_DIR / "benchmarks"
PLOTS_DIR = RESULTS_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11

def load_latest_results():
    """Load the most recent evaluation results."""
    results = {}
    
    criteria = ['latency_time', 'preprocessing', 'memory', 'accuracy', 
                'scalability_text', 'scalability_patterns', 'robustness']
    
    for criterion in criteria:
        json_files = sorted(glob(str(BENCHMARKS_DIR / f"{criterion}_*.json")))
        if json_files:
            with open(json_files[-1], 'r') as f:
                results[criterion] = pd.DataFrame(json.load(f))
    
    return results

def plot_latency_vs_pattern_length(data):
    """Plot search latency vs pattern length."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Mean search time
    for dataset in data['dataset'].unique():
        df = data[data['dataset'] == dataset]
        grouped = df.groupby('pattern_length')['mean_time_ms'].mean()
        axes[0].plot(grouped.index, grouped.values, marker='o', label=dataset, linewidth=2)
    
    axes[0].set_xlabel('Pattern Length (bp)', fontsize=12)
    axes[0].set_ylabel('Mean Search Time (ms)', fontsize=12)
    axes[0].set_title('KMP Search Time vs Pattern Length', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Throughput
    for dataset in data['dataset'].unique():
        df = data[data['dataset'] == dataset]
        grouped = df.groupby('pattern_length')['throughput_mbps'].mean()
        axes[1].plot(grouped.index, grouped.values, marker='s', label=dataset, linewidth=2)
    
    axes[1].set_xlabel('Pattern Length (bp)', fontsize=12)
    axes[1].set_ylabel('Throughput (MB/s)', fontsize=12)
    axes[1].set_title('KMP Throughput vs Pattern Length', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'latency_vs_pattern_length.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: latency_vs_pattern_length.png")
    plt.close()

def plot_preprocessing_time(data):
    """Plot preprocessing time (LPS construction)."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(data['pattern_length'], data['mean_preprocessing_time_us'], 
            marker='o', linewidth=2, markersize=8, color='#2E86AB')
    ax.fill_between(data['pattern_length'], 
                     data['mean_preprocessing_time_us'] - data['std_dev_us'],
                     data['mean_preprocessing_time_us'] + data['std_dev_us'],
                     alpha=0.3, color='#2E86AB')
    
    ax.set_xlabel('Pattern Length (bp)', fontsize=12)
    ax.set_ylabel('LPS Construction Time (µs)', fontsize=12)
    ax.set_title('KMP Preprocessing Time (LPS Array Construction)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Add O(m) reference line
    x = data['pattern_length'].values
    y_ref = x * data['time_complexity_ratio'].mean()
    ax.plot(x, y_ref, '--', color='red', alpha=0.5, label='O(m) reference')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'preprocessing_time.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: preprocessing_time.png")
    plt.close()

def plot_memory_usage(data):
    """Plot memory usage."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: LPS memory footprint
    axes[0].plot(data['pattern_length'], data['lps_memory_bytes'] / 1024, 
                marker='o', linewidth=2, markersize=8, color='#A23B72')
    axes[0].set_xlabel('Pattern Length (bp)', fontsize=12)
    axes[0].set_ylabel('LPS Memory (KB)', fontsize=12)
    axes[0].set_title('LPS Array Memory Footprint', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Peak memory during operations
    axes[1].plot(data['pattern_length'], data['preprocessing_peak_kb'], 
                marker='s', linewidth=2, label='Preprocessing', markersize=8)
    axes[1].plot(data['pattern_length'], data['search_peak_kb'], 
                marker='^', linewidth=2, label='Search', markersize=8)
    axes[1].set_xlabel('Pattern Length (bp)', fontsize=12)
    axes[1].set_ylabel('Peak Memory (KB)', fontsize=12)
    axes[1].set_title('Peak Memory Usage', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'memory_usage.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: memory_usage.png")
    plt.close()

def plot_accuracy(data):
    """Plot accuracy metrics."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    datasets = data['dataset'].unique()
    x = range(len(datasets))
    width = 0.25
    
    precision = [data[data['dataset'] == ds]['precision'].mean() for ds in datasets]
    recall = [data[data['dataset'] == ds]['recall'].mean() for ds in datasets]
    f1 = [data[data['dataset'] == ds]['f1_score'].mean() for ds in datasets]
    
    ax.bar([i - width for i in x], precision, width, label='Precision', color='#06A77D')
    ax.bar(x, recall, width, label='Recall', color='#2E86AB')
    ax.bar([i + width for i in x], f1, width, label='F1 Score', color='#A23B72')
    
    ax.set_xlabel('Dataset', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('KMP Accuracy Metrics (vs Python re)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.set_ylim([0.95, 1.01])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'accuracy_metrics.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: accuracy_metrics.png")
    plt.close()

def plot_scalability_text(data):
    """Plot scalability with text length."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Time vs text length (log-log to show linearity)
    axes[0].loglog(data['text_length'], data['mean_search_time_ms'], 
                   marker='o', linewidth=2, markersize=8, color='#F18F01')
    axes[0].set_xlabel('Text Length (bp)', fontsize=12)
    axes[0].set_ylabel('Search Time (ms)', fontsize=12)
    axes[0].set_title('Scalability: Time vs Text Length (log-log)', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3, which='both')
    
    # Add O(n) reference line
    x = data['text_length'].values
    y_ref = x * data['time_per_char_ns'].mean() / 1e6
    axes[0].loglog(x, y_ref, '--', color='red', alpha=0.5, label='O(n) reference')
    axes[0].legend()
    
    # Plot 2: Time per character (should be constant)
    axes[1].semilogx(data['text_length'], data['time_per_char_ns'], 
                     marker='s', linewidth=2, markersize=8, color='#C73E1D')
    axes[1].set_xlabel('Text Length (bp)', fontsize=12)
    axes[1].set_ylabel('Time per Character (ns)', fontsize=12)
    axes[1].set_title('Time per Character (demonstrates O(n))', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].axhline(y=data['time_per_char_ns'].mean(), color='green', 
                    linestyle='--', alpha=0.5, label='Mean')
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'scalability_text_length.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: scalability_text_length.png")
    plt.close()

def plot_scalability_patterns(data):
    """Plot scalability with number of patterns."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Total time vs number of patterns
    axes[0].plot(data['num_patterns'], data['total_time_ms'], 
                marker='o', linewidth=2, markersize=8, color='#2E86AB')
    axes[0].set_xlabel('Number of Patterns', fontsize=12)
    axes[0].set_ylabel('Total Time (ms)', fontsize=12)
    axes[0].set_title('Total Search Time vs Number of Patterns', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Average time per pattern
    axes[1].plot(data['num_patterns'], data['avg_time_per_pattern_ms'], 
                marker='s', linewidth=2, markersize=8, color='#A23B72')
    axes[1].set_xlabel('Number of Patterns', fontsize=12)
    axes[1].set_ylabel('Avg Time per Pattern (ms)', fontsize=12)
    axes[1].set_title('Average Time per Pattern', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].axhline(y=data['avg_time_per_pattern_ms'].mean(), color='green', 
                    linestyle='--', alpha=0.5, label='Mean')
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'scalability_multiple_patterns.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: scalability_multiple_patterns.png")
    plt.close()

def plot_robustness(data):
    """Plot robustness across different pattern types."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Search time by pattern type
    pattern_types = data['pattern_type'].values
    search_times = data['mean_search_time_ms'].values
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#06A77D']
    
    bars = axes[0].barh(pattern_types, search_times, color=colors)
    axes[0].set_xlabel('Mean Search Time (ms)', fontsize=12)
    axes[0].set_ylabel('Pattern Type', fontsize=12)
    axes[0].set_title('Search Time by Pattern Type', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3, axis='x')
    
    # Plot 2: Preprocessing time by pattern type
    prep_times = data['preprocessing_time_us'].values
    axes[1].barh(pattern_types, prep_times, color=colors)
    axes[1].set_xlabel('Preprocessing Time (µs)', fontsize=12)
    axes[1].set_ylabel('Pattern Type', fontsize=12)
    axes[1].set_title('Preprocessing Time by Pattern Type', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'robustness_pattern_types.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: robustness_pattern_types.png")
    plt.close()

def create_summary_dashboard(results):
    """Create a summary dashboard with key metrics."""
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Title
    fig.suptitle('KMP Algorithm - Comprehensive Evaluation Dashboard', 
                 fontsize=16, fontweight='bold')
    
    # 1. Throughput
    ax1 = fig.add_subplot(gs[0, 0])
    if 'latency_time' in results:
        data = results['latency_time']
        throughputs = data.groupby('dataset')['throughput_mbps'].mean()
        ax1.bar(range(len(throughputs)), throughputs.values, color='#2E86AB')
        ax1.set_xticks(range(len(throughputs)))
        ax1.set_xticklabels(throughputs.index, rotation=45, ha='right')
        ax1.set_ylabel('Throughput (MB/s)')
        ax1.set_title('Average Throughput by Dataset')
        ax1.grid(True, alpha=0.3, axis='y')
    
    # 2. Accuracy
    ax2 = fig.add_subplot(gs[0, 1])
    if 'accuracy' in results:
        data = results['accuracy']
        f1_scores = data.groupby('dataset')['f1_score'].mean()
        ax2.bar(range(len(f1_scores)), f1_scores.values, color='#06A77D')
        ax2.set_xticks(range(len(f1_scores)))
        ax2.set_xticklabels(f1_scores.index, rotation=45, ha='right')
        ax2.set_ylabel('F1 Score')
        ax2.set_title('Average F1 Score by Dataset')
        ax2.set_ylim([0.95, 1.01])
        ax2.grid(True, alpha=0.3, axis='y')
    
    # 3. Memory Efficiency
    ax3 = fig.add_subplot(gs[0, 2])
    if 'memory' in results:
        data = results['memory']
        ax3.plot(data['pattern_length'], data['lps_memory_bytes'] / 1024, 
                marker='o', color='#A23B72')
        ax3.set_xlabel('Pattern Length (bp)')
        ax3.set_ylabel('LPS Memory (KB)')
        ax3.set_title('Memory Footprint')
        ax3.grid(True, alpha=0.3)
    
    # 4. Preprocessing Time
    ax4 = fig.add_subplot(gs[1, :])
    if 'preprocessing' in results:
        data = results['preprocessing']
        ax4.plot(data['pattern_length'], data['mean_preprocessing_time_us'], 
                marker='o', linewidth=2, color='#F18F01')
        ax4.set_xlabel('Pattern Length (bp)')
        ax4.set_ylabel('Preprocessing Time (µs)')
        ax4.set_title('LPS Array Construction Time (Linear O(m) complexity)')
        ax4.grid(True, alpha=0.3)
    
    # 5. Scalability
    ax5 = fig.add_subplot(gs[2, :2])
    if 'scalability_text' in results:
        data = results['scalability_text']
        ax5.loglog(data['text_length'], data['mean_search_time_ms'], 
                  marker='o', linewidth=2, color='#C73E1D')
        ax5.set_xlabel('Text Length (bp)')
        ax5.set_ylabel('Search Time (ms)')
        ax5.set_title('Scalability: Search Time vs Text Length (log-log)')
        ax5.grid(True, alpha=0.3, which='both')
    
    # 6. Pattern Type Performance
    ax6 = fig.add_subplot(gs[2, 2])
    if 'robustness' in results:
        data = results['robustness']
        ax6.barh(data['pattern_type'], data['mean_search_time_ms'], color='#06A77D')
        ax6.set_xlabel('Search Time (ms)')
        ax6.set_title('Performance by Pattern Type')
        ax6.grid(True, alpha=0.3, axis='x')
    
    plt.savefig(PLOTS_DIR / 'summary_dashboard.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: summary_dashboard.png")
    plt.close()

def main():
    """Generate all visualizations."""
    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80 + "\n")
    
    # Load results
    results = load_latest_results()
    
    if not results:
        print("✗ No results found!")
        return
    
    print(f"Loaded {len(results)} result sets\n")
    
    # Generate plots
    if 'latency_time' in results:
        plot_latency_vs_pattern_length(results['latency_time'])
    
    if 'preprocessing' in results:
        plot_preprocessing_time(results['preprocessing'])
    
    if 'memory' in results:
        plot_memory_usage(results['memory'])
    
    if 'accuracy' in results:
        plot_accuracy(results['accuracy'])
    
    if 'scalability_text' in results:
        plot_scalability_text(results['scalability_text'])
    
    if 'scalability_patterns' in results:
        plot_scalability_patterns(results['scalability_patterns'])
    
    if 'robustness' in results:
        plot_robustness(results['robustness'])
    
    # Summary dashboard
    create_summary_dashboard(results)
    
    print(f"\n{'='*80}")
    print(f"All visualizations saved to: {PLOTS_DIR}")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
