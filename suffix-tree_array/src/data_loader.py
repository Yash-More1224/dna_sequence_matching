"""
Data Loading and Management for Suffix Array Implementation

Handles FASTA/FASTQ parsing and automatic dataset downloading.
Downloads E. coli genome from NCBI automatically.
"""

import os
import gzip
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord


class DatasetManager:
    """Manage DNA sequence datasets."""
    
    def __init__(self, data_dir: str = "datasets"):
        """
        Initialize dataset manager.
        
        Args:
            data_dir: Directory to store datasets
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # E. coli genome configuration
        self.ecoli_url = (
            "https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/005/845/"
            "GCF_000005845.2_ASM584v2/GCF_000005845.2_ASM584v2_genomic.fna.gz"
        )
        self.ecoli_path = self.data_dir / "ecoli_genome.fasta"
        
    def download_ecoli_genome(self, force: bool = False) -> Path:
        """
        Download E. coli K-12 MG1655 genome from NCBI.
        
        Args:
            force: Force re-download even if file exists
            
        Returns:
            Path to downloaded FASTA file
        """
        if self.ecoli_path.exists() and not force:
            print(f"E. coli genome already exists at {self.ecoli_path}")
            return self.ecoli_path
        
        print("Downloading E. coli K-12 MG1655 genome from NCBI...")
        print(f"URL: {self.ecoli_url}")
        
        # Download compressed file
        gz_path = self.data_dir / "ecoli_genome.fna.gz"
        
        try:
            urllib.request.urlretrieve(self.ecoli_url, gz_path)
            print(f"Downloaded to {gz_path}")
            
            # Decompress
            print("Decompressing...")
            with gzip.open(gz_path, 'rb') as f_in:
                with open(self.ecoli_path, 'wb') as f_out:
                    f_out.write(f_in.read())
            
            # Remove compressed file
            gz_path.unlink()
            
            print(f"✓ E. coli genome saved to {self.ecoli_path}")
            
            # Show genome info
            self._show_genome_info(self.ecoli_path)
            
            return self.ecoli_path
            
        except Exception as e:
            print(f"❌ Error downloading E. coli genome: {e}")
            if gz_path.exists():
                gz_path.unlink()
            raise
    
    def _show_genome_info(self, fasta_path: Path):
        """Display basic genome information."""
        record = next(SeqIO.parse(fasta_path, "fasta"))
        seq_len = len(record.seq)
        
        # Count nucleotides
        counts = {
            'A': record.seq.count('A'),
            'C': record.seq.count('C'),
            'G': record.seq.count('G'),
            'T': record.seq.count('T'),
            'N': record.seq.count('N')
        }
        
        print("\n" + "=" * 60)
        print("GENOME INFORMATION")
        print("=" * 60)
        print(f"ID: {record.id}")
        print(f"Description: {record.description}")
        print(f"Length: {seq_len:,} bp")
        print(f"GC Content: {(counts['G'] + counts['C']) / seq_len * 100:.2f}%")
        print(f"Nucleotide counts:")
        print(f"  A = {counts['A']:,} ({counts['A']/seq_len*100:.2f}%)")
        print(f"  C = {counts['C']:,} ({counts['C']/seq_len*100:.2f}%)")
        print(f"  G = {counts['G']:,} ({counts['G']/seq_len*100:.2f}%)")
        print(f"  T = {counts['T']:,} ({counts['T']/seq_len*100:.2f}%)")
        if counts['N'] > 0:
            print(f"  N = {counts['N']:,} ({counts['N']/seq_len*100:.2f}%)")
        print("=" * 60 + "\n")
    
    def load_fasta(self, filepath: str) -> SeqRecord:
        """
        Load sequence from FASTA file.
        
        Args:
            filepath: Path to FASTA file
            
        Returns:
            First sequence record from file
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"FASTA file not found: {filepath}")
        
        # Read first record
        record = next(SeqIO.parse(filepath, "fasta"))
        
        return record
    
    def load_ecoli_genome(self, download_if_missing: bool = True) -> str:
        """
        Load E. coli genome sequence.
        
        Args:
            download_if_missing: Download if not found locally
            
        Returns:
            Genome sequence as string
        """
        if not self.ecoli_path.exists():
            if download_if_missing:
                self.download_ecoli_genome()
            else:
                raise FileNotFoundError(
                    f"E. coli genome not found at {self.ecoli_path}. "
                    "Set download_if_missing=True to download automatically."
                )
        
        record = self.load_fasta(self.ecoli_path)
        return str(record.seq).upper()
    
    def get_ecoli_subsequence(self, start: int, end: int, 
                              download_if_missing: bool = True) -> str:
        """
        Get subsequence from E. coli genome.
        
        Args:
            start: Start position (0-indexed)
            end: End position (exclusive)
            download_if_missing: Download if not found
            
        Returns:
            Subsequence as string
        """
        sequence = self.load_ecoli_genome(download_if_missing)
        return sequence[start:end]
    
    def save_fasta(self, sequence: str, filepath: str, 
                   seq_id: str = "sequence", 
                   description: str = ""):
        """
        Save sequence to FASTA file.
        
        Args:
            sequence: DNA sequence
            filepath: Output file path
            seq_id: Sequence ID
            description: Sequence description
        """
        record = SeqRecord(
            Seq(sequence),
            id=seq_id,
            description=description
        )
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            SeqIO.write(record, f, "fasta")
        
        print(f"✓ Saved sequence to {filepath}")
    
    def get_dataset_info(self) -> Dict[str, any]:
        """
        Get information about available datasets.
        
        Returns:
            Dictionary with dataset information
        """
        info = {
            'data_directory': str(self.data_dir),
            'ecoli_genome_path': str(self.ecoli_path),
            'ecoli_exists': self.ecoli_path.exists()
        }
        
        if self.ecoli_path.exists():
            record = self.load_fasta(self.ecoli_path)
            info['ecoli_length'] = len(record.seq)
            info['ecoli_id'] = record.id
            info['ecoli_description'] = record.description
        
        return info


def load_sequence_from_file(filepath: str) -> str:
    """
    Convenience function to load DNA sequence from FASTA file.
    
    Args:
        filepath: Path to FASTA file
        
    Returns:
        Sequence as string (uppercase)
    """
    record = next(SeqIO.parse(filepath, "fasta"))
    return str(record.seq).upper()


if __name__ == "__main__":
    # Test dataset manager
    manager = DatasetManager()
    
    print("\n" + "=" * 60)
    print("DATASET MANAGER TEST")
    print("=" * 60)
    
    # Download E. coli genome
    manager.download_ecoli_genome()
    
    # Load subsequence
    print("\nLoading subsequence (first 1000 bp)...")
    subseq = manager.get_ecoli_subsequence(0, 1000)
    print(f"Subsequence length: {len(subseq)}")
    print(f"First 100 bp: {subseq[:100]}")
    
    # Show dataset info
    print("\n" + "=" * 60)
    print("DATASET INFO")
    print("=" * 60)
    info = manager.get_dataset_info()
    for key, value in info.items():
        print(f"{key}: {value}")
    print("=" * 60 + "\n")
