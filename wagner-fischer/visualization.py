"""
Visualization utilities for Wagner-Fischer results.
Creates plots, heatmaps, and alignment visualizations.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple

from wf_search import Match


class ResultVisualizer:
    """
    Visualize Wagner-Fischer algorithm results.
    """
    
    def __init__(self, output_dir: str = "results/plots"):
        """
        Initialize visualizer.
        
        Args:
            output_dir: Directory to save plots
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (10, 6)
        plt.rcParams['font.size'] = 10
    
    def plot_benchmark_results(self, 
                               csv_path: str,
                               save_name: str = "benchmark_summary.png"):
        """
        Plot benchmark results from CSV file.
        
        Args:
            csv_path: Path to benchmark results CSV
            save_name: Output filename
        """
        df = pd.read_csv(csv_path)
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Wagner-Fischer Performance Benchmarks', fontsize=16, fontweight='bold')
        
        # Plot 1: Time vs Pattern Length
        ax1 = axes[0, 0]
        edit_dist_data = df[df['test_name'] == 'edit_distance']
        if not edit_dist_data.empty:
            ax1.plot(edit_dist_data['pattern_length'], 
                    edit_dist_data['time_mean'] * 1000, 
                    marker='o', linewidth=2, markersize=8)
            ax1.fill_between(edit_dist_data['pattern_length'],
                            (edit_dist_data['time_mean'] - edit_dist_data['time_std']) * 1000,
                            (edit_dist_data['time_mean'] + edit_dist_data['time_std']) * 1000,
                            alpha=0.3)
            ax1.set_xlabel('Pattern Length (bp)', fontweight='bold')
            ax1.set_ylabel('Time (ms)', fontweight='bold')
            ax1.set_title('Edit Distance Computation Time')
            ax1.grid(True, alpha=0.3)
        
        # Plot 2: Time vs Text Length
        ax2 = axes[0, 1]
        search_data = df[df['test_name'] == 'pattern_search']
        if not search_data.empty:
            ax2.plot(search_data['text_length'], 
                    search_data['time_mean'] * 1000, 
                    marker='s', linewidth=2, markersize=8, color='orange')
            ax2.fill_between(search_data['text_length'],
                            (search_data['time_mean'] - search_data['time_std']) * 1000,
                            (search_data['time_mean'] + search_data['time_std']) * 1000,
                            alpha=0.3, color='orange')
            ax2.set_xlabel('Text Length (bp)', fontweight='bold')
            ax2.set_ylabel('Time (ms)', fontweight='bold')
            ax2.set_title('Pattern Search Time vs Text Length')
            ax2.grid(True, alpha=0.3)
        
        # Plot 3: Memory Usage
        ax3 = axes[1, 0]
        if not edit_dist_data.empty:
            ax3.bar(edit_dist_data['pattern_length'], 
                   edit_dist_data['memory_peak_mb'],
                   color='green', alpha=0.7, edgecolor='black')
            ax3.set_xlabel('Pattern Length (bp)', fontweight='bold')
            ax3.set_ylabel('Peak Memory (MB)', fontweight='bold')
            ax3.set_title('Memory Usage')
            ax3.grid(True, alpha=0.3, axis='y')
        
        # Plot 4: Threshold Scaling
        ax4 = axes[1, 1]
        threshold_data = df[df['test_name'] == 'threshold_scaling']
        if not threshold_data.empty:
            ax4_twin = ax4.twinx()
            
            line1 = ax4.plot(threshold_data['edit_distance_threshold'], 
                           threshold_data['time_mean'] * 1000, 
                           marker='D', linewidth=2, markersize=8, 
                           color='red', label='Time')
            line2 = ax4_twin.plot(threshold_data['edit_distance_threshold'], 
                                threshold_data['matches_found'], 
                                marker='o', linewidth=2, markersize=8, 
                                color='blue', label='Matches')
            
            ax4.set_xlabel('Edit Distance Threshold', fontweight='bold')
            ax4.set_ylabel('Time (ms)', fontweight='bold', color='red')
            ax4_twin.set_ylabel('Number of Matches', fontweight='bold', color='blue')
            ax4.set_title('Threshold vs Performance & Matches')
            ax4.tick_params(axis='y', labelcolor='red')
            ax4_twin.tick_params(axis='y', labelcolor='blue')
            ax4.grid(True, alpha=0.3)
            
            # Combined legend
            lines = line1 + line2
            labels = [l.get_label() for l in lines]
            ax4.legend(lines, labels, loc='upper left')
        
        plt.tight_layout()
        output_path = self.output_dir / save_name
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Benchmark plot saved to {output_path}")
    
    def plot_accuracy_results(self,
                             csv_path: str,
                             save_name: str = "accuracy_summary.png"):
        """
        Plot accuracy evaluation results.
        
        Args:
            csv_path: Path to accuracy results CSV
            save_name: Output filename
        """
        df = pd.read_csv(csv_path)
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle('Wagner-Fischer Accuracy Evaluation', fontsize=16, fontweight='bold')
        
        # Plot 1: Precision, Recall, F1 by threshold
        ax1 = axes[0]
        threshold_data = df[df['test_name'] == 'synthetic_mutations']
        if not threshold_data.empty:
            x = threshold_data['edit_distance_threshold']
            ax1.plot(x, threshold_data['precision'], marker='o', label='Precision', linewidth=2)
            ax1.plot(x, threshold_data['recall'], marker='s', label='Recall', linewidth=2)
            ax1.plot(x, threshold_data['f1_score'], marker='^', label='F1 Score', linewidth=2)
            ax1.set_xlabel('Edit Distance Threshold', fontweight='bold')
            ax1.set_ylabel('Score', fontweight='bold')
            ax1.set_title('Accuracy Metrics vs Threshold')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            ax1.set_ylim([0, 1.1])
        
        # Plot 2: Confusion Matrix (for one threshold)
        ax2 = axes[1]
        if not threshold_data.empty:
            sample = threshold_data.iloc[0]
            confusion = np.array([
                [sample['true_positives'], sample['false_positives']],
                [sample['false_negatives'], sample['true_negatives']]
            ])
            sns.heatmap(confusion, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=['Predicted +', 'Predicted -'],
                       yticklabels=['Actual +', 'Actual -'],
                       ax=ax2, cbar_kws={'label': 'Count'})
            ax2.set_title(f'Confusion Matrix (k={int(sample["edit_distance_threshold"])})')
        
        # Plot 3: True Positives vs False Positives
        ax3 = axes[2]
        if not threshold_data.empty:
            ax3.scatter(threshold_data['false_positives'], 
                       threshold_data['true_positives'],
                       s=100, alpha=0.6, c=threshold_data['edit_distance_threshold'],
                       cmap='viridis', edgecolors='black')
            ax3.set_xlabel('False Positives', fontweight='bold')
            ax3.set_ylabel('True Positives', fontweight='bold')
            ax3.set_title('TP vs FP Trade-off')
            ax3.grid(True, alpha=0.3)
            
            # Add colorbar
            cbar = plt.colorbar(ax3.collections[0], ax=ax3)
            cbar.set_label('Edit Distance Threshold', rotation=270, labelpad=20)
        
        plt.tight_layout()
        output_path = self.output_dir / save_name
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Accuracy plot saved to {output_path}")
    
    def plot_comparison(self,
                       csv_path: str,
                       save_name: str = "wf_vs_regex.png"):
        """
        Plot comparison between Wagner-Fischer and regex.
        
        Args:
            csv_path: Path to benchmark results CSV
            save_name: Output filename
        """
        df = pd.read_csv(csv_path)
        comparison_data = df[df['test_name'] == 'regex_comparison']
        
        if comparison_data.empty:
            print("No comparison data found")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle('Wagner-Fischer vs Python Regex', fontsize=16, fontweight='bold')
        
        # Prepare data
        wf_data = comparison_data[comparison_data['algorithm'] == 'wagner_fischer'].iloc[0]
        regex_data = comparison_data[comparison_data['algorithm'] == 'python_regex'].iloc[0]
        
        # Plot 1: Time comparison
        ax1 = axes[0]
        algorithms = ['Wagner-Fischer', 'Python Regex']
        times = [wf_data['time_mean'] * 1000, regex_data['time_mean'] * 1000]
        colors = ['#3498db', '#e74c3c']
        
        bars = ax1.bar(algorithms, times, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
        ax1.set_ylabel('Time (ms)', fontweight='bold')
        ax1.set_title('Execution Time Comparison')
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, time in zip(bars, times):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{time:.2f}ms',
                    ha='center', va='bottom', fontweight='bold')
        
        # Plot 2: Memory comparison
        ax2 = axes[1]
        memory = [wf_data['memory_peak_mb'], regex_data['memory_peak_mb']]
        
        bars = ax2.bar(algorithms, memory, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
        ax2.set_ylabel('Peak Memory (MB)', fontweight='bold')
        ax2.set_title('Memory Usage Comparison')
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, mem in zip(bars, memory):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{mem:.2f}MB',
                    ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        output_path = self.output_dir / save_name
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Comparison plot saved to {output_path}")
    
    def visualize_alignment(self,
                          source: str,
                          target: str,
                          operations: List[str],
                          save_name: str = "alignment.png"):
        """
        Visualize sequence alignment with operations.
        
        Args:
            source: Source sequence
            target: Target sequence
            operations: List of edit operations
            save_name: Output filename
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Build alignment strings
        source_align = []
        target_align = []
        markers = []
        
        for op in operations:
            if op.startswith('match'):
                base = op.split('_')[1]
                source_align.append(base)
                target_align.append(base)
                markers.append('|')
            elif op.startswith('substitute'):
                parts = op.split('_')[1].split('->')
                source_align.append(parts[0])
                target_align.append(parts[1])
                markers.append('X')
            elif op.startswith('delete'):
                base = op.split('_')[1]
                source_align.append(base)
                target_align.append('-')
                markers.append(' ')
            elif op.startswith('insert'):
                base = op.split('_')[1]
                source_align.append('-')
                target_align.append(base)
                markers.append(' ')
        
        # Plot alignment
        source_str = ''.join(source_align)
        target_str = ''.join(target_align)
        marker_str = ''.join(markers)
        
        # Display in monospace
        ax.text(0.05, 0.7, f"Source: {source_str}", family='monospace', fontsize=10)
        ax.text(0.05, 0.5, f"        {marker_str}", family='monospace', fontsize=10)
        ax.text(0.05, 0.3, f"Target: {target_str}", family='monospace', fontsize=10)
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        ax.set_title('Sequence Alignment Visualization', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        output_path = self.output_dir / save_name
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Alignment visualization saved to {output_path}")
    
    def plot_match_heatmap(self,
                          text: str,
                          matches: List[Match],
                          window_size: int = 1000,
                          save_name: str = "match_heatmap.png"):
        """
        Create heatmap showing match density across sequence.
        
        Args:
            text: DNA sequence
            matches: List of matches
            window_size: Size of bins for density calculation
            save_name: Output filename
        """
        # Calculate match density
        num_windows = (len(text) + window_size - 1) // window_size
        density = np.zeros(num_windows)
        
        for match in matches:
            window_idx = match.position // window_size
            if window_idx < num_windows:
                density[window_idx] += 1
        
        fig, ax = plt.subplots(figsize=(14, 4))
        
        # Plot as heatmap
        density_2d = density.reshape(1, -1)
        im = ax.imshow(density_2d, cmap='YlOrRd', aspect='auto', interpolation='nearest')
        
        ax.set_xlabel(f'Position (windows of {window_size} bp)', fontweight='bold')
        ax.set_title('Match Density Heatmap', fontsize=14, fontweight='bold')
        ax.set_yticks([])
        
        # Colorbar
        cbar = plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.1)
        cbar.set_label('Number of Matches', fontweight='bold')
        
        plt.tight_layout()
        output_path = self.output_dir / save_name
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Match heatmap saved to {output_path}")
    
    def highlight_matches_in_sequence(self,
                                     text: str,
                                     matches: List[Match],
                                     context_size: int = 50,
                                     save_name: str = "highlighted_matches.txt"):
        """
        Create text file with highlighted matches.
        
        Args:
            text: DNA sequence
            matches: List of matches
            context_size: Characters of context around match
            save_name: Output filename
        """
        output_path = self.output_dir / save_name
        
        with open(output_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("MATCH HIGHLIGHTING\n")
            f.write("=" * 80 + "\n\n")
            
            for i, match in enumerate(matches[:20], 1):  # Limit to first 20
                start = max(0, match.position - context_size)
                end = min(len(text), match.end_position + context_size)
                
                context_before = text[start:match.position]
                matched_text = text[match.position:match.end_position]
                context_after = text[match.end_position:end]
                
                f.write(f"Match {i}:\n")
                f.write(f"  Position: {match.position}-{match.end_position}\n")
                f.write(f"  Edit Distance: {match.edit_distance}\n")
                f.write(f"  Sequence:\n")
                f.write(f"    {context_before}[{matched_text}]{context_after}\n")
                f.write("\n")
        
        print(f"Highlighted matches saved to {output_path}")
