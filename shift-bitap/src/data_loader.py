"""
Data Loading and Synthetic Data Generation for DNA Sequence Matching
====================================================================

This module provides utilities for:
1. Loading DNA sequences from FASTA/FASTQ files
2. Downloading and caching genome datasets
3. Generating synthetic DNA sequences with controlled mutations
4. Creating test datasets for benchmarking

Supports:
- E. coli K-12 genome (primary target)
- Other bacterial genomes
- Viral genomes (lambda phage, etc.)
- Synthetic sequences with controlled mutation rates

Author: DNA Sequence Matching Project
Date: November 2025
"""

import os
import random
import gzip
from typing import List, Tuple, Optional, Dict
from pathlib import Path
import urllib.request
from io import StringIO


class DataLoader:
    """
    Handles loading and management of DNA sequence data.
    """
    
    def __init__(self, cache_dir: str = "./data/cache"):
        """
        Initialize the data loader.
        
        Args:
            cache_dir: Directory for caching downloaded datasets
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Common genome URLs
        self.genome_urls = {
            'ecoli': 'https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/005/845/GCF_000005845.2_ASM584v2/GCF_000005845.2_ASM584v2_genomic.fna.gz',
            'lambda': 'https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/840/245/GCF_000840245.1_ViralProj14204/GCF_000840245.1_ViralProj14204_genomic.fna.gz'
        }
    
    def load_fasta(self, filepath: str) -> Dict[str, str]:
        """
        Load sequences from a FASTA file.
        
        Args:
            filepath: Path to FASTA file (can be .gz compressed)
            
        Returns:
            Dictionary mapping sequence IDs to sequences
        """
        sequences = {}
        current_id = None
        current_seq = []
        
        # Determine if file is gzipped
        open_func = gzip.open if filepath.endswith('.gz') else open
        mode = 'rt' if filepath.endswith('.gz') else 'r'
        
        try:
            with open_func(filepath, mode) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    
                    if line.startswith('>'):
                        # Save previous sequence
                        if current_id is not None:
                            sequences[current_id] = ''.join(current_seq)
                        
                        # Start new sequence
                        current_id = line[1:].split()[0]  # Get ID without '>' and before first space
                        current_seq = []
                    else:
                        current_seq.append(line.upper())
                
                # Save last sequence
                if current_id is not None:
                    sequences[current_id] = ''.join(current_seq)
        
        except FileNotFoundError:
            raise FileNotFoundError(f"FASTA file not found: {filepath}")
        except Exception as e:
            raise RuntimeError(f"Error reading FASTA file: {e}")
        
        return sequences
    
    def load_fasta_single(self, filepath: str) -> str:
        """
        Load a single sequence from a FASTA file (takes first sequence).
        
        Args:
            filepath: Path to FASTA file
            
        Returns:
            First sequence as string
        """
        sequences = self.load_fasta(filepath)
        if not sequences:
            raise ValueError("No sequences found in FASTA file")
        return next(iter(sequences.values()))
    
    def download_genome(self, genome_name: str, force: bool = False) -> str:
        """
        Download a genome dataset if not already cached.
        
        Args:
            genome_name: Name of genome ('ecoli', 'lambda', etc.)
            force: If True, re-download even if cached
            
        Returns:
            Path to downloaded/cached genome file
        """
        if genome_name not in self.genome_urls:
            raise ValueError(f"Unknown genome: {genome_name}. Available: {list(self.genome_urls.keys())}")
        
        # Cache file path
        cache_file = self.cache_dir / f"{genome_name}_genome.fna.gz"
        
        # Download if needed
        if force or not cache_file.exists():
            url = self.genome_urls[genome_name]
            print(f"Downloading {genome_name} genome from NCBI...")
            print(f"URL: {url}")
            
            try:
                urllib.request.urlretrieve(url, cache_file)
                print(f"Downloaded to: {cache_file}")
            except Exception as e:
                raise RuntimeError(f"Failed to download genome: {e}")
        else:
            print(f"Using cached genome: {cache_file}")
        
        return str(cache_file)
    
    def load_genome(self, genome_name: str) -> str:
        """
        Load a genome (download if necessary).
        
        Args:
            genome_name: Name of genome to load
            
        Returns:
            Genome sequence as string
        """
        genome_file = self.download_genome(genome_name)
        return self.load_fasta_single(genome_file)
    
    def save_fasta(self, sequences: Dict[str, str], filepath: str, line_width: int = 80):
        """
        Save sequences to a FASTA file.
        
        Args:
            sequences: Dictionary mapping IDs to sequences
            filepath: Output file path
            line_width: Maximum characters per line for sequences
        """
        with open(filepath, 'w') as f:
            for seq_id, seq in sequences.items():
                f.write(f">{seq_id}\n")
                # Write sequence with line wrapping
                for i in range(0, len(seq), line_width):
                    f.write(seq[i:i+line_width] + '\n')


class SyntheticDataGenerator:
    """
    Generate synthetic DNA sequences with controlled mutations.
    """
    
    DNA_BASES = ['A', 'C', 'G', 'T']
    
    @staticmethod
    def generate_random_sequence(length: int, seed: Optional[int] = None) -> str:
        """
        Generate a random DNA sequence.
        
        Args:
            length: Length of sequence to generate
            seed: Random seed for reproducibility
            
        Returns:
            Random DNA sequence
        """
        if seed is not None:
            random.seed(seed)
        
        return ''.join(random.choice(SyntheticDataGenerator.DNA_BASES) for _ in range(length))
    
    @staticmethod
    def generate_gc_biased_sequence(length: int, gc_content: float = 0.5, seed: Optional[int] = None) -> str:
        """
        Generate DNA sequence with specific GC content.
        
        Args:
            length: Length of sequence
            gc_content: Desired GC content (0.0 to 1.0)
            seed: Random seed
            
        Returns:
            DNA sequence with approximately the specified GC content
        """
        if not 0 <= gc_content <= 1:
            raise ValueError("gc_content must be between 0 and 1")
        
        if seed is not None:
            random.seed(seed)
        
        sequence = []
        for _ in range(length):
            if random.random() < gc_content:
                sequence.append(random.choice(['G', 'C']))
            else:
                sequence.append(random.choice(['A', 'T']))
        
        return ''.join(sequence)
    
    @staticmethod
    def introduce_substitutions(sequence: str, rate: float, seed: Optional[int] = None) -> str:
        """
        Introduce random substitution mutations.
        
        Args:
            sequence: Original DNA sequence
            rate: Substitution rate (0.0 to 1.0)
            seed: Random seed
            
        Returns:
            Mutated sequence
        """
        if not 0 <= rate <= 1:
            raise ValueError("Substitution rate must be between 0 and 1")
        
        if seed is not None:
            random.seed(seed)
        
        mutated = list(sequence)
        bases = SyntheticDataGenerator.DNA_BASES
        
        for i in range(len(mutated)):
            if random.random() < rate:
                # Substitute with a different base
                original = mutated[i]
                new_base = random.choice([b for b in bases if b != original])
                mutated[i] = new_base
        
        return ''.join(mutated)
    
    @staticmethod
    def introduce_indels(sequence: str, insertion_rate: float = 0.01, 
                        deletion_rate: float = 0.01, seed: Optional[int] = None) -> str:
        """
        Introduce insertion and deletion mutations.
        
        Args:
            sequence: Original DNA sequence
            insertion_rate: Probability of insertion per position
            deletion_rate: Probability of deletion per position
            seed: Random seed
            
        Returns:
            Mutated sequence
        """
        if seed is not None:
            random.seed(seed)
        
        result = []
        bases = SyntheticDataGenerator.DNA_BASES
        
        for base in sequence:
            # Check for deletion
            if random.random() >= deletion_rate:
                result.append(base)
                
                # Check for insertion after this base
                if random.random() < insertion_rate:
                    result.append(random.choice(bases))
        
        return ''.join(result)
    
    @staticmethod
    def mutate_sequence(sequence: str, substitution_rate: float = 0.0,
                       insertion_rate: float = 0.0, deletion_rate: float = 0.0,
                       seed: Optional[int] = None) -> Tuple[str, Dict[str, int]]:
        """
        Apply multiple types of mutations to a sequence.
        
        Args:
            sequence: Original sequence
            substitution_rate: Substitution probability per base
            insertion_rate: Insertion probability per base
            deletion_rate: Deletion probability per base
            seed: Random seed
            
        Returns:
            Tuple of (mutated_sequence, mutation_stats)
        """
        if seed is not None:
            random.seed(seed)
        
        original_length = len(sequence)
        
        # Apply substitutions first
        mutated = SyntheticDataGenerator.introduce_substitutions(sequence, substitution_rate, seed)
        
        # Then apply indels
        mutated = SyntheticDataGenerator.introduce_indels(mutated, insertion_rate, deletion_rate, seed)
        
        stats = {
            'original_length': original_length,
            'mutated_length': len(mutated),
            'length_diff': len(mutated) - original_length,
            'substitution_rate': substitution_rate,
            'insertion_rate': insertion_rate,
            'deletion_rate': deletion_rate
        }
        
        return mutated, stats
    
    @staticmethod
    def create_test_dataset(num_sequences: int, seq_length: int, 
                          mutation_rates: List[float], seed: Optional[int] = None) -> List[Tuple[str, str, float]]:
        """
        Create a test dataset with sequences at various mutation levels.
        
        Args:
            num_sequences: Number of sequence pairs to generate
            seq_length: Length of each sequence
            mutation_rates: List of mutation rates to test
            seed: Random seed
            
        Returns:
            List of tuples (original_seq, mutated_seq, mutation_rate)
        """
        if seed is not None:
            random.seed(seed)
        
        dataset = []
        
        for rate in mutation_rates:
            for i in range(num_sequences):
                # Generate original sequence
                seq_seed = None if seed is None else seed + i
                original = SyntheticDataGenerator.generate_random_sequence(seq_length, seq_seed)
                
                # Mutate it
                mutated, _ = SyntheticDataGenerator.mutate_sequence(
                    original, substitution_rate=rate, seed=seq_seed
                )
                
                dataset.append((original, mutated, rate))
        
        return dataset


def create_motif_dataset(motifs: List[str], background_length: int = 10000, 
                        num_copies: int = 10, noise: float = 0.0, seed: Optional[int] = None) -> str:
    """
    Create a synthetic sequence with embedded motifs.
    
    Args:
        motifs: List of motif sequences to embed
        background_length: Length of background sequence
        num_copies: Number of times to embed each motif
        noise: Mutation rate for embedded motifs
        seed: Random seed
        
    Returns:
        Synthetic sequence with embedded motifs
    """
    if seed is not None:
        random.seed(seed)
    
    # Generate background
    background = SyntheticDataGenerator.generate_random_sequence(background_length, seed)
    sequence = list(background)
    
    # Embed motifs at random positions
    for motif in motifs:
        for _ in range(num_copies):
            # Random position (ensure we don't go out of bounds)
            if len(motif) < len(sequence):
                pos = random.randint(0, len(sequence) - len(motif))
                
                # Possibly mutate the motif
                if noise > 0:
                    motif_to_insert = SyntheticDataGenerator.introduce_substitutions(motif, noise, seed)
                else:
                    motif_to_insert = motif
                
                # Insert motif
                for i, base in enumerate(motif_to_insert):
                    sequence[pos + i] = base
    
    return ''.join(sequence)


if __name__ == "__main__":
    print("DNA Data Loader Demo")
    print("=" * 60)
    
    # Demo 1: Generate random sequences
    print("\n1. Random Sequence Generation:")
    random_seq = SyntheticDataGenerator.generate_random_sequence(50, seed=42)
    print(f"Random sequence (50bp): {random_seq}")
    
    # Demo 2: GC-biased sequence
    print("\n2. GC-Biased Sequence (70% GC):")
    gc_seq = SyntheticDataGenerator.generate_gc_biased_sequence(50, gc_content=0.7, seed=42)
    gc_actual = (gc_seq.count('G') + gc_seq.count('C')) / len(gc_seq)
    print(f"Sequence: {gc_seq}")
    print(f"Actual GC content: {gc_actual:.2%}")
    
    # Demo 3: Mutations
    print("\n3. Introducing Mutations:")
    original = "ACGTACGTACGTACGT"
    print(f"Original: {original}")
    
    mutated = SyntheticDataGenerator.introduce_substitutions(original, rate=0.2, seed=42)
    print(f"With substitutions (20%): {mutated}")
    
    mutated2 = SyntheticDataGenerator.introduce_indels(original, 0.1, 0.1, seed=42)
    print(f"With indels (10% each): {mutated2}")
    
    # Demo 4: Motif dataset
    print("\n4. Motif Dataset:")
    motifs = ["TATAAA", "GATTACA"]
    text = create_motif_dataset(motifs, background_length=200, num_copies=3, seed=42)
    print(f"Generated sequence with embedded motifs (length={len(text)}):")
    print(f"First 100bp: {text[:100]}")
    print(f"'TATAAA' appears {text.count('TATAAA')} times")
    print(f"'GATTACA' appears {text.count('GATTACA')} times")
    
    # Demo 5: Test dataset creation
    print("\n5. Test Dataset Creation:")
    test_data = SyntheticDataGenerator.create_test_dataset(
        num_sequences=2, 
        seq_length=30,
        mutation_rates=[0.0, 0.1],
        seed=42
    )
    print(f"Created {len(test_data)} test sequence pairs:")
    for i, (orig, mut, rate) in enumerate(test_data):
        print(f"  Pair {i+1} (rate={rate}):")
        print(f"    Original: {orig}")
        print(f"    Mutated:  {mut}")
