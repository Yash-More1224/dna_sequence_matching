"""
Benchmark Suite for DNA Pattern Matching

Runs experiments comparing Suffix Array performance against Python's native 're' module.
Measures:
1. Exact Preprocessing Time
2. Peak Resident Memory (RAM) using memory_profiler
3. Search Latency vs Pattern Length
"""

import time
import re
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import List, Dict

# Try to import memory_profiler for the "Section 5" requirement
try:
    from memory_profiler import memory_usage
    HAS_PROFILER = True
except ImportError:
    HAS_PROFILER = False
    print("Warning: memory_profiler not found. Peak RAM stats will be skipped.")

# Import your modules
from suffix_array import SuffixArray
from data_generator import DNAGenerator
from data_loader import DatasetManager
import utils

class BenchmarkSuite:
    def __init__(self, output_dir: str = "results"):
        self.output_dir = utils.ensure_dir(output_dir)
        self.generator = DNAGenerator(seed=42)
        self.manager = DatasetManager()
        
        # Setup plotting style
        sns.set_theme(style="whitegrid")
        
    def run_full_suite(self):
        """Run all benchmarks defined in the proposal."""
        print(f"\n{'='*60}\nSTARTING BENCHMARK SUITE\n{'='*60}")
        
        # 1. Load Data (E. coli or Synthetic)
        try:
            print("[1/4] Loading E. coli genome...")
            # Using E. coli genome
            text = self.manager.load_ecoli_genome(download_if_missing=True)
            print(f"      Loaded {len(text):,} base pairs.")
            
            # OPTIONAL: Slice it if your laptop runs out of RAM. 
            # text = text[:1_000_000] # Uncomment to test on just 1MB
        except Exception as e:
            print(f"      Could not load E. coli ({e}). Using synthetic data.")
            text = self.generator.generate_random_sequence(500_000)

        # 2. Build Index & Measure Memory
        print("\n[2/4] Building Suffix Array Index & Profiling Memory...")
        sa = SuffixArray(verbose=False)
        
        start_time = time.perf_counter()
        
        if HAS_PROFILER:
            # This runs the function and tracks peak memory usage in MiB
            # interval=0.1 takes a snapshot every 0.1s
            mem_usage = memory_usage((sa.build_index, (text,)), interval=0.1)
            peak_ram = max(mem_usage) - min(mem_usage) # Net increase
            
            # Calculate build time manually since we wrapped the call
            build_time = time.perf_counter() - start_time
            
            print(f"      Construction Time:  {utils.format_time(build_time)}")
            print(f"      Theoretical Size:   {utils.format_memory(sa.memory_footprint)} (sys.getsizeof)")
            print(f"      Actual RAM Usage:   {peak_ram:.2f} MiB (memory_profiler)")
            
            # Visualize Memory Profile? (Bonus points)
            # You could plot `mem_usage` here if you wanted.
        else:
            build_time, memory = sa.build_index(text)
            print(f"      Construction Time: {utils.format_time(build_time)}")
            print(f"      Memory Footprint:  {utils.format_memory(memory)}")

        # 3. Experiment: Search Latency vs Pattern Length
        print("\n[3/4] Running Search Latency Experiment...")
        self._benchmark_search_latency(sa, text)

        print(f"\n{'='*60}\nBENCHMARK COMPLETE. Results saved to {self.output_dir}\n{'='*60}")

    def _benchmark_search_latency(self, sa_obj: SuffixArray, text: str):
        """
        Compare Suffix Array search vs Python 're' module.
        Varying pattern lengths: 10, 20, 50, 100, 200, 500
        """
        lengths = [10, 20, 50, 100, 200, 500]
        results = []
        
        # Number of patterns to test per length
        num_trials = 20
        
        for p_len in lengths:
            print(f"      Testing pattern length: {p_len}...")
            
            # Generate patterns that definitely exist in the text
            patterns = []
            for _ in range(num_trials):
                # Pick a random spot in the text to grab a pattern
                start = self.generator.seed % (len(text) - p_len)
                patterns.append(text[start : start + p_len])
                self.generator.seed += 1

            # --- 1. Measure Suffix Array Time ---
            sa_start = time.perf_counter()
            for p in patterns:
                _ = sa_obj.search_exact(p)
            sa_total = time.perf_counter() - sa_start
            sa_avg = sa_total / num_trials

            # --- 2. Measure Python 're' Time ---
            re_start = time.perf_counter()
            for p in patterns:
                # escape pattern just in case it has special regex chars (unlikely in DNA)
                _ = list(re.finditer(p, text))
            re_total = time.perf_counter() - re_start
            re_avg = re_total / num_trials

            results.append({
                "Pattern_Length": p_len,
                "Algorithm": "Suffix Array",
                "Time_Seconds": sa_avg,
            })
            
            results.append({
                "Pattern_Length": p_len,
                "Algorithm": "Python re (Regex)",
                "Time_Seconds": re_avg,
            })

        # Save Raw Data
        utils.save_csv(results, self.output_dir / "latency_results.csv")
        
        # Generate Plot
        self._plot_latency(results)

    def _plot_latency(self, data: List[Dict]):
        """Generate comparison plot using Seaborn."""
        df = pd.DataFrame(data)
        
        plt.figure(figsize=(10, 6))
        
        # Use a nice color palette
        sns.lineplot(
            data=df, 
            x="Pattern_Length", 
            y="Time_Seconds", 
            hue="Algorithm",
            style="Algorithm",
            markers=True,
            dashes=False,
            linewidth=2.5,
            markersize=8
        )
        
        plt.title("Search Latency: Suffix Array vs Python Regex", fontsize=15)
        plt.xlabel("Pattern Length (bp)", fontsize=12)
        plt.ylabel("Time per Query (seconds)", fontsize=12)
        
        # Log scale is usually better for algorithmic comparisons
        plt.yscale("log") 
        plt.grid(True, which="both", ls="-", alpha=0.2)
        
        save_path = self.output_dir / "latency_benchmark.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"      ✓ Plot saved to {save_path}")
        plt.close()

if __name__ == "__main__":
    bench = BenchmarkSuite()
    bench.run_full_suite()