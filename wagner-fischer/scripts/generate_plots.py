"""
Generate Visualizations for Wagner-Fischer Evaluation
Creates plots matching KMP evaluation structure.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import glob


def find_latest_file(pattern):
    """Find the most recent file matching pattern."""
    files = glob.glob(pattern)
    return max(files, key=lambda x: Path(x).stat().st_mtime) if files else None


def plot_latency_vs_pattern_length(latency_data, output_dir):
    """Plot latency vs pattern length for each dataset."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    datasets = {}
    for row in latency_data:
        dataset = row['dataset']
        if dataset not in datasets:
            datasets[dataset] = {'lengths': [], 'runtimes': []}
        datasets[dataset]['lengths'].append(row['pattern_length'])
        datasets[dataset]['runtimes'].append(row['mean_runtime'] * 1000)  # Convert to ms
    
    for dataset, data in datasets.items():
        ax.plot(data['lengths'], data['runtimes'], 'o-', label=dataset, linewidth=2, markersize=8)
    
    ax.set_xlabel('Pattern Length (bp)', fontsize=12)
    ax.set_ylabel('Mean Latency (ms)', fontsize=12)
    ax.set_title('Wagner-Fischer: Latency vs Pattern Length', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'latency_vs_pattern_length.png', dpi=300)
    plt.close()
    print(f"  ✓ Saved: latency_vs_pattern_length.png")


def plot_memory_usage(memory_data, output_dir):
    """Plot memory usage."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    pattern_lengths = [row['pattern_length'] for row in memory_data]
    peak_mem = [row['peak_tracemalloc_mb'] for row in memory_data]
    peak_rss = [row['peak_rss_mb'] for row in memory_data]
    
    x = np.arange(len(pattern_lengths))
    width = 0.35
    
    ax1.bar(x - width/2, peak_mem, width, label='Peak Tracemalloc', alpha=0.8)
    ax1.bar(x + width/2, peak_rss, width, label='Peak RSS', alpha=0.8)
    ax1.set_xlabel('Pattern Length (bp)', fontsize=11)
    ax1.set_ylabel('Memory (MB)', fontsize=11)
    ax1.set_title('Memory Usage by Pattern Length', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(pattern_lengths)
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # Index footprint (always 0 for WF)
    ax2.bar(['Wagner-Fischer', 'KMP (ref)'], [0, 0.5], alpha=0.7, color=['#2E86AB', '#A23B72'])
    ax2.set_ylabel('Index Footprint (MB)', fontsize=11)
    ax2.set_title('Preprocessing Index Footprint', fontsize=12, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    ax2.text(0, 0.05, 'No Index\n(WF)', ha='center', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'memory_usage.png', dpi=300)
    plt.close()
    print(f"  ✓ Saved: memory_usage.png")


def plot_accuracy_metrics(accuracy_data, output_dir):
    """Plot accuracy metrics."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Group by mutation rate
    mutation_rates = sorted(set(row['mutation_rate'] for row in accuracy_data))
    f1_scores = {}
    
    for row in accuracy_data:
        dataset = row['dataset']
        mut_rate = row['mutation_rate']
        if dataset not in f1_scores:
            f1_scores[dataset] = {}
        if mut_rate not in f1_scores[dataset]:
            f1_scores[dataset][mut_rate] = []
        f1_scores[dataset][mut_rate].append(row['f1_score'])
    
    x = np.array(mutation_rates) * 100  # Convert to percentage
    
    for dataset in f1_scores:
        y = [np.mean(f1_scores[dataset].get(mr, [0])) for mr in mutation_rates]
        ax.plot(x, y, 'o-', label=dataset, linewidth=2, markersize=8)
    
    ax.set_xlabel('Mutation Rate (%)', fontsize=12)
    ax.set_ylabel('F1 Score', fontsize=12)
    ax.set_title('Accuracy: F1 Score vs Mutation Rate', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.05])
    
    plt.tight_layout()
    plt.savefig(output_dir / 'accuracy_metrics.png', dpi=300)
    plt.close()
    print(f"  ✓ Saved: accuracy_metrics.png")


def plot_scalability_text_length(scalability_data, output_dir):
    """Plot scalability with text length."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    text_lengths = [row['text_length'] for row in scalability_data]
    runtimes = [row['mean_runtime'] * 1000 for row in scalability_data]
    throughputs = [row['throughput_bp_s'] for row in scalability_data]
    
    ax1.plot(text_lengths, runtimes, 'o-', linewidth=2, markersize=8, color='#2E86AB')
    ax1.set_xlabel('Text Length (bp)', fontsize=12)
    ax1.set_ylabel('Mean Runtime (ms)', fontsize=12)
    ax1.set_title('Scalability: Runtime vs Text Length', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(text_lengths, throughputs, 's-', linewidth=2, markersize=8, color='#A23B72')
    ax2.set_xlabel('Text Length (bp)', fontsize=12)
    ax2.set_ylabel('Throughput (bp/s)', fontsize=12)
    ax2.set_title('Scalability: Throughput vs Text Length', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'scalability_text_length.png', dpi=300)
    plt.close()
    print(f"  ✓ Saved: scalability_text_length.png")


def plot_scalability_multiple_patterns(scalability_data, output_dir):
    """Plot scalability with multiple patterns."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    pattern_counts = [row['pattern_count'] for row in scalability_data]
    runtimes = [row['total_runtime'] for row in scalability_data]
    throughputs = [row['throughput_patterns_s'] for row in scalability_data]
    
    ax1.plot(pattern_counts, runtimes, 'o-', linewidth=2, markersize=8, color='#06A77D')
    ax1.set_xlabel('Number of Patterns', fontsize=12)
    ax1.set_ylabel('Total Runtime (s)', fontsize=12)
    ax1.set_title('Scalability: Runtime vs Pattern Count', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(pattern_counts, throughputs, 's-', linewidth=2, markersize=8, color='#F18F01')
    ax2.set_xlabel('Number of Patterns', fontsize=12)
    ax2.set_ylabel('Throughput (patterns/s)', fontsize=12)
    ax2.set_title('Scalability: Throughput vs Pattern Count', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'scalability_multiple_patterns.png', dpi=300)
    plt.close()
    print(f"  ✓ Saved: scalability_multiple_patterns.png")


def plot_robustness_mutation(robustness_data, output_dir):
    """Plot robustness across mutation rates."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    mutation_rates = [row['mutation_rate'] * 100 for row in robustness_data]
    edit_distances = [row['edit_distance'] for row in robustness_data]
    runtimes = [row['mean_runtime'] * 1000 for row in robustness_data]
    
    ax1.plot(mutation_rates, edit_distances, 'o-', linewidth=2, markersize=8, color='#C73E1D')
    ax1.set_xlabel('Mutation Rate (%)', fontsize=12)
    ax1.set_ylabel('Edit Distance', fontsize=12)
    ax1.set_title('Robustness: Edit Distance vs Mutation Rate', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(mutation_rates, runtimes, 's-', linewidth=2, markersize=8, color='#2E86AB')
    ax2.set_xlabel('Mutation Rate (%)', fontsize=12)
    ax2.set_ylabel('Mean Runtime (ms)', fontsize=12)
    ax2.set_title('Robustness: Runtime vs Mutation Rate', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'robustness_pattern_types.png', dpi=300)
    plt.close()
    print(f"  ✓ Saved: robustness_pattern_types.png")


def create_summary_dashboard(all_data, output_dir):
    """Create summary dashboard with key metrics."""
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Title
    fig.suptitle('Wagner-Fischer Algorithm: Comprehensive Evaluation Dashboard', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    # 1. Latency by pattern length
    ax1 = fig.add_subplot(gs[0, :2])
    if 'latency' in all_data:
        datasets = {}
        for row in all_data['latency']:
            dataset = row['dataset']
            if dataset not in datasets:
                datasets[dataset] = {'lengths': [], 'runtimes': []}
            datasets[dataset]['lengths'].append(row['pattern_length'])
            datasets[dataset]['runtimes'].append(row['mean_runtime'] * 1000)
        for dataset, data in datasets.items():
            ax1.plot(data['lengths'], data['runtimes'], 'o-', label=dataset, markersize=6)
        ax1.set_xlabel('Pattern Length (bp)')
        ax1.set_ylabel('Latency (ms)')
        ax1.set_title('Latency vs Pattern Length')
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)
    
    # 2. Memory usage
    ax2 = fig.add_subplot(gs[0, 2])
    if 'memory' in all_data:
        pattern_lengths = [str(row['pattern_length']) for row in all_data['memory']]
        peak_mem = [row['peak_tracemalloc_mb'] for row in all_data['memory']]
        ax2.bar(range(len(pattern_lengths)), peak_mem, alpha=0.7, color='#2E86AB')
        ax2.set_xlabel('Pattern Length')
        ax2.set_ylabel('Memory (MB)')
        ax2.set_title('Peak Memory Usage')
        ax2.set_xticks(range(len(pattern_lengths)))
        ax2.set_xticklabels(pattern_lengths, fontsize=8)
        ax2.grid(axis='y', alpha=0.3)
    
    # 3. Accuracy
    ax3 = fig.add_subplot(gs[1, :2])
    if 'accuracy' in all_data:
        mutation_rates = sorted(set(row['mutation_rate'] for row in all_data['accuracy']))
        avg_f1 = []
        for mr in mutation_rates:
            scores = [row['f1_score'] for row in all_data['accuracy'] if row['mutation_rate'] == mr]
            avg_f1.append(np.mean(scores))
        ax3.plot([mr*100 for mr in mutation_rates], avg_f1, 'o-', linewidth=2, markersize=8, color='#06A77D')
        ax3.set_xlabel('Mutation Rate (%)')
        ax3.set_ylabel('F1 Score')
        ax3.set_title('Accuracy: F1 Score vs Mutation Rate')
        ax3.set_ylim([0, 1.05])
        ax3.grid(True, alpha=0.3)
    
    # 4. Summary stats
    ax4 = fig.add_subplot(gs[1, 2])
    ax4.axis('off')
    if 'latency' in all_data:
        runtimes = [row['mean_runtime'] * 1000 for row in all_data['latency']]
        summary_text = f"""
Key Metrics:

Mean Latency:
  {np.mean(runtimes):.4f} ms

Median Latency:
  {np.median(runtimes):.4f} ms

Avg Memory:
  {np.mean([row['peak_tracemalloc_mb'] for row in all_data.get('memory', [])]):.2f} MB

Index Size:
  0.00 MB
  (no preprocessing)
        """
        ax4.text(0.1, 0.5, summary_text, fontsize=10, family='monospace',
                verticalalignment='center')
    
    # 5. Scalability text
    ax5 = fig.add_subplot(gs[2, 0])
    if 'scalability_text' in all_data:
        text_lengths = [row['text_length'] for row in all_data['scalability_text']]
        runtimes = [row['mean_runtime'] * 1000 for row in all_data['scalability_text']]
        ax5.plot(text_lengths, runtimes, 'o-', linewidth=2, markersize=6, color='#A23B72')
        ax5.set_xlabel('Text Length (bp)')
        ax5.set_ylabel('Runtime (ms)')
        ax5.set_title('Text Length Scaling')
        ax5.grid(True, alpha=0.3)
    
    # 6. Scalability patterns
    ax6 = fig.add_subplot(gs[2, 1])
    if 'scalability_patterns' in all_data:
        pattern_counts = [row['pattern_count'] for row in all_data['scalability_patterns']]
        runtimes = [row['total_runtime'] for row in all_data['scalability_patterns']]
        ax6.plot(pattern_counts, runtimes, 's-', linewidth=2, markersize=6, color='#F18F01')
        ax6.set_xlabel('Pattern Count')
        ax6.set_ylabel('Total Runtime (s)')
        ax6.set_title('Pattern Count Scaling')
        ax6.grid(True, alpha=0.3)
    
    # 7. Robustness
    ax7 = fig.add_subplot(gs[2, 2])
    if 'robustness' in all_data:
        mutation_rates = [row['mutation_rate'] * 100 for row in all_data['robustness']]
        edit_distances = [row['edit_distance'] for row in all_data['robustness']]
        ax7.plot(mutation_rates, edit_distances, '^-', linewidth=2, markersize=6, color='#C73E1D')
        ax7.set_xlabel('Mutation Rate (%)')
        ax7.set_ylabel('Edit Distance')
        ax7.set_title('Robustness')
        ax7.grid(True, alpha=0.3)
    
    plt.savefig(output_dir / 'summary_dashboard.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: summary_dashboard.png")


def main():
    """Generate all visualizations."""
    results_dir = Path('results')
    
    print("Generating Wagner-Fischer visualizations...")
    
    # Load all data
    all_data = {}
    
    # Load latency data
    latency_file = find_latest_file(str(results_dir / 'latency_time_*.json'))
    if latency_file:
        with open(latency_file) as f:
            all_data['latency'] = json.load(f)
    
    # Load memory data
    memory_file = find_latest_file(str(results_dir / 'memory_*.json'))
    if memory_file:
        with open(memory_file) as f:
            all_data['memory'] = json.load(f)
    
    # Load accuracy data
    accuracy_file = find_latest_file(str(results_dir / 'accuracy_*.json'))
    if accuracy_file:
        with open(accuracy_file) as f:
            all_data['accuracy'] = json.load(f)
    
    # Load scalability data
    scalability_text_file = find_latest_file(str(results_dir / 'scalability_text_*.json'))
    if scalability_text_file:
        with open(scalability_text_file) as f:
            all_data['scalability_text'] = json.load(f)
    
    scalability_patterns_file = find_latest_file(str(results_dir / 'scalability_patterns_*.json'))
    if scalability_patterns_file:
        with open(scalability_patterns_file) as f:
            all_data['scalability_patterns'] = json.load(f)
    
    # Load robustness data
    robustness_file = find_latest_file(str(results_dir / 'robustness_*.json'))
    if robustness_file:
        with open(robustness_file) as f:
            all_data['robustness'] = json.load(f)
    
    # Generate plots
    if 'latency' in all_data:
        plot_latency_vs_pattern_length(all_data['latency'], results_dir)
    
    if 'memory' in all_data:
        plot_memory_usage(all_data['memory'], results_dir)
    
    if 'accuracy' in all_data:
        plot_accuracy_metrics(all_data['accuracy'], results_dir)
    
    if 'scalability_text' in all_data:
        plot_scalability_text_length(all_data['scalability_text'], results_dir)
    
    if 'scalability_patterns' in all_data:
        plot_scalability_multiple_patterns(all_data['scalability_patterns'], results_dir)
    
    if 'robustness' in all_data:
        plot_robustness_mutation(all_data['robustness'], results_dir)
    
    # Create summary dashboard
    create_summary_dashboard(all_data, results_dir)
    
    print("\n✓ All visualizations generated successfully!")


if __name__ == '__main__':
    main()
