#!/usr/bin/env python3
"""
Generate comprehensive visualizations for Suffix Array evaluation results.
This script creates all the same plots as the KMP implementation for comparison.

All 8 plots generated:
1. latency_vs_pattern_length.png - Search latency and throughput vs pattern length
2. preprocessing_time.png - Index construction time (SA + LCP)
3. memory_usage.png - Memory footprint analysis
4. accuracy_metrics.png - Precision, recall, and F1 scores
5. scalability_text_length.png - Performance vs text size
6. scalability_multiple_patterns.png - Multi-pattern search performance
7. robustness_pattern_types.png - Performance across different pattern types
8. summary_dashboard.png - Comprehensive overview dashboard
"""

import sys
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from glob import glob
import numpy as np

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
                'scalability_patterns', 'robustness']
    
    for criterion in criteria:
        json_files = sorted(glob(str(BENCHMARKS_DIR / f"{criterion}_*.json")))
        if json_files:
            with open(json_files[-1], 'r') as f:
                data = json.load(f)
                results[criterion] = pd.DataFrame(data)
                print(f"✓ Loaded {criterion}: {len(data)} records from {Path(json_files[-1]).name}")
    
    return results

def plot_latency_vs_pattern_length(data):
    """Plot search latency vs pattern length."""
    print("\n[1/8] Generating latency_vs_pattern_length.png...")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Mean search time
    for dataset in data['dataset'].unique():
        df = data[data['dataset'] == dataset]
        grouped = df.groupby('pattern_length')['mean_search_time_ms'].mean()
        axes[0].plot(grouped.index, grouped.values, marker='o', label=dataset, linewidth=2.5)
    
    axes[0].set_xlabel('Pattern Length (bp)', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Mean Search Time (ms)', fontsize=12, fontweight='bold')
    axes[0].set_title('Suffix Array Search Time vs Pattern Length', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Throughput
    for dataset in data['dataset'].unique():
        df = data[data['dataset'] == dataset]
        grouped = df.groupby('pattern_length')['throughput_mbps'].mean()
        axes[1].plot(grouped.index, grouped.values, marker='s', label=dataset, linewidth=2.5)
    
    axes[1].set_xlabel('Pattern Length (bp)', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Throughput (MB/s)', fontsize=12, fontweight='bold')
    axes[1].set_title('Suffix Array Throughput vs Pattern Length', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'latency_vs_pattern_length.png', dpi=300, bbox_inches='tight')
    print("  ✓ Saved: latency_vs_pattern_length.png")
    plt.close()

def plot_preprocessing_time(data):
    """Plot preprocessing time (SA + LCP construction)."""
    print("\n[2/8] Generating preprocessing_time.png...")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot mean preprocessing time with error bars
    time_col = 'mean_construction_time_ms' if 'mean_construction_time_ms' in data.columns else 'construction_time_ms'
    ax.plot(data['text_length'], data[time_col], 
            marker='o', linewidth=2.5, markersize=8, color='#2E86AB', label='Actual')
    
    if 'std_dev_ms' in data.columns:
        ax.fill_between(data['text_length'], 
                         data[time_col] - data['std_dev_ms'],
                         data[time_col] + data['std_dev_ms'],
                         alpha=0.3, color='#2E86AB')
    
    ax.set_xlabel('Text Length (bp)', fontsize=12, fontweight='bold')
    ax.set_ylabel('SA + LCP Construction Time (ms)', fontsize=12, fontweight='bold')
    ax.set_title('Suffix Array Preprocessing Time (O(N log²N))', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Add O(N log²N) reference line
    x = data['text_length'].values
    y = data[time_col].values
    if len(x) > 1 and len(y) > 1:
        # Fit to last data point
        x_ref = x
        y_ref = (x_ref * np.log2(x_ref) * np.log2(x_ref)) * (y[-1] / (x[-1] * np.log2(x[-1]) * np.log2(x[-1])))
        ax.plot(x_ref, y_ref, '--', color='red', alpha=0.6, linewidth=2, label='O(N log²N) reference')
        ax.legend(fontsize=10)
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'preprocessing_time.png', dpi=300, bbox_inches='tight')
    print("  ✓ Saved: preprocessing_time.png")
    plt.close()

def plot_memory_usage(data):
    """Plot memory usage."""
    print("\n[3/8] Generating memory_usage.png...")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: SA + LCP memory footprint
    # Convert bytes to MB
    data['sa_memory_mb'] = data['sa_memory_bytes'] / (1024 * 1024)
    axes[0].plot(data['text_length'], data['sa_memory_mb'], 
                marker='o', linewidth=2.5, markersize=8, color='#A23B72')
    axes[0].set_xlabel('Text Length (bp)', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('SA + LCP Memory (MB)', fontsize=12, fontweight='bold')
    axes[0].set_title('Suffix Array Memory Footprint', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    # Add O(N) reference line
    x = data['text_length'].values
    y = data['sa_memory_mb'].values
    if len(x) > 0 and len(y) > 0:
        linear_coef = y[-1] / x[-1]
        axes[0].plot(x, x * linear_coef, '--', color='red', alpha=0.5, label='O(N) reference')
        axes[0].legend(fontsize=10)
    
    # Plot 2: Peak memory during operations (convert KB to MB)
    data['construction_peak_mb'] = data['construction_peak_kb'] / 1024
    data['search_peak_mb'] = data['search_peak_kb'] / 1024
    axes[1].plot(data['text_length'], data['construction_peak_mb'], 
                marker='s', linewidth=2.5, label='Construction', markersize=8, color='#2E86AB')
    axes[1].plot(data['text_length'], data['search_peak_mb'], 
                marker='^', linewidth=2.5, label='Search', markersize=8, color='#F18F01')
    axes[1].set_xlabel('Text Length (bp)', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Peak Memory (MB)', fontsize=12, fontweight='bold')
    axes[1].set_title('Peak Memory Usage', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'memory_usage.png', dpi=300, bbox_inches='tight')
    print("  ✓ Saved: memory_usage.png")
    plt.close()

def plot_accuracy(data):
    """Plot accuracy metrics."""
    print("\n[4/8] Generating accuracy_metrics.png...")
    
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
    
    ax.set_xlabel('Dataset', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Suffix Array Accuracy Metrics (vs Ground Truth)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, fontsize=10)
    ax.set_ylim([0.95, 1.01])
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'accuracy_metrics.png', dpi=300, bbox_inches='tight')
    print("  ✓ Saved: accuracy_metrics.png")
    plt.close()

def plot_scalability_text(data):
    """Plot scalability with text length."""
    print("\n[5/8] Generating scalability_text_length.png...")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Time vs text length (log-log to show logarithmic behavior)
    axes[0].loglog(data['text_length'], data['mean_search_time_ms'], 
                   marker='o', linewidth=2.5, markersize=8, color='#F18F01')
    axes[0].set_xlabel('Text Length (bp)', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Search Time (ms)', fontsize=12, fontweight='bold')
    axes[0].set_title('Scalability: Time vs Text Length (log-log)', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3, which='both')
    
    # Add O(P log N) reference line
    if 'pattern_length' in data.columns:
        x = data['text_length'].values
        p = data['pattern_length'].mean() if len(data['pattern_length']) > 0 else 20
        y_ref = p * np.log2(x) * data['mean_search_time_ms'].mean() / (p * np.log2(x.mean()))
        axes[0].loglog(x, y_ref, '--', color='red', alpha=0.5, label='O(P log N) reference')
        axes[0].legend(fontsize=10)
    
    # Plot 2: Time per log(N) (should be roughly constant for O(P log N))
    time_per_logn = data['mean_search_time_ms'] / np.log2(data['text_length'])
    axes[1].semilogx(data['text_length'], time_per_logn, 
                     marker='s', linewidth=2.5, markersize=8, color='#C73E1D')
    axes[1].set_xlabel('Text Length (bp)', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Time / log₂(N) (ms)', fontsize=12, fontweight='bold')
    axes[1].set_title('Time per log₂(N) (demonstrates O(P log N))', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].axhline(y=time_per_logn.mean(), color='green', 
                    linestyle='--', alpha=0.5, linewidth=2, label='Mean')
    axes[1].legend(fontsize=10)
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'scalability_text_length.png', dpi=300, bbox_inches='tight')
    print("  ✓ Saved: scalability_text_length.png")
    plt.close()

def plot_scalability_patterns(data):
    """Plot scalability with number of patterns."""
    print("\n[6/8] Generating scalability_multiple_patterns.png...")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Total time vs number of patterns
    axes[0].plot(data['num_patterns'], data['search_time_ms'], 
                marker='o', linewidth=2.5, markersize=8, color='#2E86AB')
    axes[0].set_xlabel('Number of Patterns', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Total Search Time (ms)', fontsize=12, fontweight='bold')
    axes[0].set_title('Total Search Time vs Number of Patterns', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    # Add linear reference
    x = data['num_patterns'].values
    y = data['search_time_ms'].values
    if len(x) > 0 and len(y) > 0:
        linear_coef = y[-1] / x[-1]
        axes[0].plot(x, x * linear_coef, '--', color='red', alpha=0.5, label='O(M) reference')
        axes[0].legend(fontsize=10)
    
    # Plot 2: Average time per pattern
    axes[1].plot(data['num_patterns'], data['avg_time_per_pattern_ms'], 
                marker='s', linewidth=2.5, markersize=8, color='#A23B72')
    axes[1].set_xlabel('Number of Patterns', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Avg Time per Pattern (ms)', fontsize=12, fontweight='bold')
    axes[1].set_title('Average Time per Pattern (Multi-pattern Search)', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].axhline(y=data['avg_time_per_pattern_ms'].mean(), color='green', 
                    linestyle='--', alpha=0.5, linewidth=2, label='Mean')
    axes[1].legend(fontsize=10)
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'scalability_multiple_patterns.png', dpi=300, bbox_inches='tight')
    print("  ✓ Saved: scalability_multiple_patterns.png")
    plt.close()

def plot_robustness(data):
    """Plot robustness across different pattern types."""
    print("\n[7/8] Generating robustness_pattern_types.png...")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Search time by pattern type
    pattern_types = data['pattern_type'].values
    search_times = data['mean_search_time_ms'].values
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#06A77D'][:len(pattern_types)]
    
    bars = axes[0].barh(pattern_types, search_times, color=colors)
    axes[0].set_xlabel('Mean Search Time (ms)', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Pattern Type', fontsize=12, fontweight='bold')
    axes[0].set_title('Search Time by Pattern Type', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for i, (pt, st) in enumerate(zip(pattern_types, search_times)):
        axes[0].text(st, i, f' {st:.3f}', va='center', fontsize=9)
    
    # Plot 2: Matches by pattern type
    match_counts = data['num_matches'].values
    axes[1].barh(pattern_types, match_counts, color=colors)
    axes[1].set_xlabel('Matches Found', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Pattern Type', fontsize=12, fontweight='bold')
    axes[1].set_title('Matches Found by Pattern Type', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for i, (pt, mc) in enumerate(zip(pattern_types, match_counts)):
        axes[1].text(mc, i, f' {int(mc)}', va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'robustness_pattern_types.png', dpi=300, bbox_inches='tight')
    print("  ✓ Saved: robustness_pattern_types.png")
    plt.close()

def create_summary_dashboard(results):
    """Create a summary dashboard with key metrics."""
    print("\n[8/8] Generating summary_dashboard.png...")
    
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Title
    fig.suptitle('Suffix Array - Comprehensive Performance Dashboard', 
                 fontsize=18, fontweight='bold', y=0.98)
    
    # 1. Latency overview
    if 'latency_time' in results:
        ax1 = fig.add_subplot(gs[0, 0])
        data = results['latency_time']
        for dataset in data['dataset'].unique():
            df = data[data['dataset'] == dataset]
            grouped = df.groupby('pattern_length')['mean_search_time_ms'].mean()
            ax1.plot(grouped.index, grouped.values, marker='o', label=dataset, linewidth=2)
        ax1.set_xlabel('Pattern Length', fontsize=10)
        ax1.set_ylabel('Search Time (ms)', fontsize=10)
        ax1.set_title('Search Latency', fontsize=11, fontweight='bold')
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)
    
    # 2. Preprocessing time
    if 'preprocessing' in results:
        ax2 = fig.add_subplot(gs[0, 1])
        data = results['preprocessing']
        time_col = 'mean_construction_time_ms' if 'mean_construction_time_ms' in data.columns else 'construction_time_ms'
        ax2.plot(data['text_length'], data[time_col], 
                marker='o', linewidth=2, color='#2E86AB')
        ax2.set_xlabel('Text Length', fontsize=10)
        ax2.set_ylabel('Time (ms)', fontsize=10)
        ax2.set_title('Construction (O(N log²N))', fontsize=11, fontweight='bold')
        ax2.grid(True, alpha=0.3)
    
    # 3. Memory usage
    if 'memory' in results:
        ax3 = fig.add_subplot(gs[0, 2])
        data = results['memory'].copy()
        data['sa_memory_mb'] = data['sa_memory_bytes'] / (1024 * 1024)
        ax3.plot(data['text_length'], data['sa_memory_mb'], 
                marker='s', linewidth=2, color='#A23B72')
        ax3.set_xlabel('Text Length', fontsize=10)
        ax3.set_ylabel('Memory (MB)', fontsize=10)
        ax3.set_title('Memory Footprint (O(N))', fontsize=11, fontweight='bold')
        ax3.grid(True, alpha=0.3)
    
    # 4. Accuracy
    if 'accuracy' in results:
        ax4 = fig.add_subplot(gs[1, 0])
        data = results['accuracy']
        datasets = data['dataset'].unique()
        f1_scores = [data[data['dataset'] == ds]['f1_score'].mean() for ds in datasets]
        colors = ['#06A77D', '#2E86AB', '#A23B72'][:len(datasets)]
        ax4.bar(range(len(datasets)), f1_scores, color=colors)
        ax4.set_xticks(range(len(datasets)))
        ax4.set_xticklabels(datasets, fontsize=8, rotation=15)
        ax4.set_ylabel('F1 Score', fontsize=10)
        ax4.set_title('Accuracy (F1 Score)', fontsize=11, fontweight='bold')
        ax4.set_ylim([0.95, 1.01])
        ax4.grid(True, alpha=0.3, axis='y')
    
    # 5. Scalability (patterns)
    if 'scalability_patterns' in results:
        ax5 = fig.add_subplot(gs[1, 1])
        data = results['scalability_patterns']
        ax5.plot(data['num_patterns'], data['search_time_ms'], 
                marker='o', linewidth=2, color='#F18F01')
        ax5.set_xlabel('Number of Patterns', fontsize=10)
        ax5.set_ylabel('Search Time (ms)', fontsize=10)
        ax5.set_title('Multi-Pattern Scalability', fontsize=11, fontweight='bold')
        ax5.grid(True, alpha=0.3)
    
    # 6. Robustness
    if 'robustness' in results:
        ax6 = fig.add_subplot(gs[1, 2])
        data = results['robustness']
        pattern_types = data['pattern_type'].values
        search_times = data['mean_search_time_ms'].values
        colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#06A77D'][:len(pattern_types)]
        ax6.barh(pattern_types, search_times, color=colors)
        ax6.set_xlabel('Time (ms)', fontsize=10)
        ax6.set_title('Robustness (Pattern Types)', fontsize=11, fontweight='bold')
        ax6.grid(True, alpha=0.3, axis='x')
    
    # 7-9. Key metrics summary
    ax7 = fig.add_subplot(gs[2, :])
    ax7.axis('off')
    
    summary_text = "KEY PERFORMANCE METRICS\n" + "="*80 + "\n\n"
    
    if 'latency_time' in results:
        avg_latency = results['latency_time']['mean_search_time_ms'].mean()
        avg_throughput = results['latency_time']['throughput_mbps'].mean()
        summary_text += f"• Average Search Latency: {avg_latency:.3f} ms\n"
        summary_text += f"• Average Throughput: {avg_throughput:.2f} MB/s\n"
    
    if 'preprocessing' in results:
        time_col = 'mean_construction_time_ms' if 'mean_construction_time_ms' in results['preprocessing'].columns else 'construction_time_ms'
        avg_prep = results['preprocessing'][time_col].mean()
        summary_text += f"• Average Construction Time: {avg_prep:.2f} ms (SA + LCP construction)\n"
    
    if 'memory' in results:
        avg_memory = (results['memory']['sa_memory_bytes'] / (1024 * 1024)).mean()
        summary_text += f"• Average Memory Footprint: {avg_memory:.2f} MB (SA + LCP arrays)\n"
    
    if 'accuracy' in results:
        avg_f1 = results['accuracy']['f1_score'].mean()
        summary_text += f"• Average F1 Score: {avg_f1:.4f} (Accuracy)\n"
    
    if 'scalability_patterns' in results:
        avg_per_pattern = results['scalability_patterns']['avg_time_per_pattern_ms'].mean()
        summary_text += f"• Average Time per Pattern: {avg_per_pattern:.3f} ms (Multi-pattern search)\n"
    
    summary_text += f"\nComplexity Guarantees:\n"
    summary_text += f"  - Construction: O(N log²N) time, O(N) space\n"
    summary_text += f"  - Single Search: O(|P| log |T|) time\n"
    summary_text += f"  - Multi-pattern: O(M × |P| log |T|) time\n"
    
    ax7.text(0.05, 0.5, summary_text, transform=ax7.transAxes,
             fontsize=11, verticalalignment='center', family='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.savefig(PLOTS_DIR / 'summary_dashboard.png', dpi=300, bbox_inches='tight')
    print("  ✓ Saved: summary_dashboard.png")
    plt.close()

def main():
    """Generate all visualizations."""
    print("="*70)
    print("SUFFIX ARRAY VISUALIZATION GENERATOR")
    print("="*70)
    print(f"Results directory: {BENCHMARKS_DIR}")
    print(f"Output directory: {PLOTS_DIR}")
    print("="*70)
    
    # Load data
    print("\nLoading evaluation results...")
    results = load_latest_results()
    
    if not results:
        print("❌ ERROR: No evaluation results found!")
        print(f"   Please run comprehensive_evaluation_balanced.py first")
        return 1
    
    print(f"\n✓ Loaded {len(results)} result sets")
    print("="*70)
    
    # Generate all plots
    print("\nGenerating visualizations...")
    
    if 'latency_time' in results:
        plot_latency_vs_pattern_length(results['latency_time'])
    
    if 'preprocessing' in results:
        plot_preprocessing_time(results['preprocessing'])
    
    if 'memory' in results:
        plot_memory_usage(results['memory'])
    
    if 'accuracy' in results:
        plot_accuracy(results['accuracy'])
    
    # For text length scalability, we can derive it from latency data
    if 'latency_time' in results:
        # Aggregate by text length
        text_scalability = results['latency_time'].groupby('text_length').agg({
            'mean_search_time_ms': 'mean',
            'pattern_length': 'mean'
        }).reset_index()
        plot_scalability_text(text_scalability)
    
    if 'scalability_patterns' in results:
        plot_scalability_patterns(results['scalability_patterns'])
    
    if 'robustness' in results:
        plot_robustness(results['robustness'])
    
    # Summary dashboard
    create_summary_dashboard(results)
    
    print("\n" + "="*70)
    print("✓ ALL VISUALIZATIONS GENERATED SUCCESSFULLY!")
    print(f"  Output location: {PLOTS_DIR}")
    print(f"  Total plots: 8")
    print("="*70)
    
    # List all generated files
    print("\nGenerated files:")
    for png_file in sorted(PLOTS_DIR.glob('*.png')):
        size_kb = png_file.stat().st_size / 1024
        print(f"  • {png_file.name} ({size_kb:.1f} KB)")
    
    print("\n" + "="*70)
    print("Ready for presentation and comparison with KMP!")
    print("="*70)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
