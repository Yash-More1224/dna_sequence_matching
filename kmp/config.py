"""
Configuration file for KMP DNA sequence matching project.

This module contains all configuration constants, file paths, URLs,
and parameter settings used throughout the project.
"""

import os
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent
WORKSPACE_ROOT = PROJECT_ROOT.parent

# Directory structure
DATASETS_DIR = PROJECT_ROOT / "datasets"
RESULTS_DIR = PROJECT_ROOT / "results"
PLOTS_DIR = RESULTS_DIR / "plots"
BENCHMARKS_DIR = RESULTS_DIR / "benchmarks"
REPORTS_DIR = RESULTS_DIR / "reports"
TESTS_DIR = PROJECT_ROOT / "tests"

# Create directories if they don't exist
for directory in [DATASETS_DIR, RESULTS_DIR, PLOTS_DIR, BENCHMARKS_DIR, REPORTS_DIR, TESTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# DNA Alphabet
DNA_BASES = ['A', 'C', 'G', 'T']
DNA_BASES_WITH_N = ['A', 'C', 'G', 'T', 'N']
DNA_COMPLEMENT = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C', 'N': 'N'}

# Dataset URLs (NCBI)
DATASETS = {
    'ecoli': {
        'name': 'Escherichia coli K-12 MG1655',
        'accession': 'NC_000913.3',
        'url': 'https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/005/845/GCF_000005845.2_ASM584v2/GCF_000005845.2_ASM584v2_genomic.fna.gz',
        'size_mb': 4.6,
        'description': 'E. coli K-12 substr. MG1655 complete genome'
    },
    'lambda_phage': {
        'name': 'Lambda Phage',
        'accession': 'NC_001416.1',
        'url': 'https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/840/245/GCF_000840245.1_ViralProj14204/GCF_000840245.1_ViralProj14204_genomic.fna.gz',
        'size_mb': 0.048,
        'description': 'Escherichia phage Lambda complete genome'
    },
    'salmonella': {
        'name': 'Salmonella enterica',
        'accession': 'NC_003197.2',
        'url': 'https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/006/945/GCF_000006945.2_ASM694v2/GCF_000006945.2_ASM694v2_genomic.fna.gz',
        'size_mb': 4.8,
        'description': 'Salmonella enterica subsp. enterica serovar Typhimurium str. LT2'
    },
    'bacillus': {
        'name': 'Bacillus subtilis',
        'accession': 'NC_000964.3',
        'url': 'https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/009/045/GCF_000009045.1_ASM904v1/GCF_000009045.1_ASM904v1_genomic.fna.gz',
        'size_mb': 4.2,
        'description': 'Bacillus subtilis subsp. subtilis str. 168 complete genome'
    }
}

# Benchmarking parameters
BENCHMARK_CONFIG = {
    'num_runs': 5,  # Number of runs for statistical significance
    'warmup_runs': 2,  # Number of warmup runs to discard
    'timeout_seconds': 300,  # Maximum time per experiment
    'memory_interval': 0.1,  # Memory profiling interval (seconds)
}

# Experiment parameters
EXPERIMENT_CONFIG = {
    # Pattern length variation (in base pairs)
    'pattern_lengths': [10, 20, 50, 100, 200, 500, 1000],
    
    # Text size scaling (in KB)
    'text_sizes': [1, 10, 50, 100, 500, 1000, 4600],  # Up to E.coli full genome
    
    # Number of patterns for multi-pattern experiments
    'num_patterns': [1, 10, 50, 100, 500, 1000],
    
    # Mutation rates for approximate matching experiments
    'mutation_rates': [0.0, 0.01, 0.05, 0.10, 0.20],
    
    # Random seed for reproducibility
    'random_seed': 42,
}

# Synthetic data generation parameters
SYNTHETIC_CONFIG = {
    'default_length': 10000,
    'default_num_patterns': 10,
    'default_pattern_length': 50,
    'default_mutation_rate': 0.05,
    'base_probabilities': {'A': 0.25, 'C': 0.25, 'G': 0.25, 'T': 0.25},
}

# Visualization parameters
VIZ_CONFIG = {
    'figure_size': (12, 8),
    'dpi': 300,
    'style': 'seaborn-v0_8-darkgrid',
    'color_palette': 'Set2',
    'match_highlight_color': 'red',
    'pattern_highlight_color': 'blue',
    'font_size': 12,
}

# File formats
OUTPUT_FORMATS = {
    'benchmark_results': 'csv',  # csv or json
    'plots': 'png',  # png, pdf, or svg
    'reports': 'txt',  # txt or markdown
}

# Logging configuration
LOG_LEVEL = 'INFO'  # DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

# Common DNA motifs for testing (if needed)
COMMON_MOTIFS = {
    'TATA_box': 'TATAAA',
    'kozak_sequence': 'GCCACCATGG',
    'poly_a_signal': 'AATAAA',
    'shine_dalgarno': 'AGGAGGT',
}

# Performance thresholds (for testing)
PERFORMANCE_THRESHOLDS = {
    'max_lps_time_ms': 100,  # Maximum time to build LPS array for 1000bp pattern
    'max_search_time_ms': 1000,  # Maximum time to search 1MB text
    'max_memory_mb': 100,  # Maximum memory usage for typical operations
}
