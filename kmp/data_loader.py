"""
Data loading utilities for DNA sequences.

This module provides functions to read and parse FASTA and FASTQ files,
load genomic datasets, and manage sequence data efficiently.
"""

from typing import Dict, List, Tuple, Optional, Generator
from pathlib import Path
import gzip
from dataclasses import dataclass


@dataclass
class SequenceRecord:
    """
    A simple sequence record data class.
    
    Attributes:
        id: Sequence identifier
        description: Sequence description
        sequence: The actual DNA sequence
        length: Length of the sequence
    """
    id: str
    description: str
    sequence: str
    
    @property
    def length(self) -> int:
        """Get the length of the sequence."""
        return len(self.sequence)
    
    def __repr__(self) -> str:
        """String representation."""
        seq_preview = self.sequence[:50] + '...' if len(self.sequence) > 50 else self.sequence
        return f"SequenceRecord(id='{self.id}', length={self.length}, sequence='{seq_preview}')"


def read_fasta(filepath: Path, max_sequences: Optional[int] = None) -> List[SequenceRecord]:
    """
    Read sequences from a FASTA file.
    
    FASTA format:
        >sequence_id description
        ATCGATCG...
        
    Args:
        filepath: Path to FASTA file (.fasta, .fa, .fna, or .gz compressed)
        max_sequences: Maximum number of sequences to read (None = all)
        
    Returns:
        List of SequenceRecord objects
        
    Example:
        >>> records = read_fasta(Path("ecoli.fasta"))
        >>> print(records[0].id, records[0].length)
    """
    records = []
    
    # Handle gzip compressed files
    if str(filepath).endswith('.gz'):
        file_handle = gzip.open(filepath, 'rt')
    else:
        file_handle = open(filepath, 'r')
    
    try:
        current_id = None
        current_desc = None
        current_seq = []
        
        for line in file_handle:
            line = line.strip()
            
            if not line:
                continue
            
            if line.startswith('>'):
                # Save previous sequence if exists
                if current_id is not None:
                    records.append(SequenceRecord(
                        id=current_id,
                        description=current_desc,
                        sequence=''.join(current_seq)
                    ))
                    
                    if max_sequences and len(records) >= max_sequences:
                        break
                
                # Parse header line
                header = line[1:].strip()
                parts = header.split(maxsplit=1)
                current_id = parts[0]
                current_desc = parts[1] if len(parts) > 1 else ''
                current_seq = []
            else:
                # Sequence line
                current_seq.append(line.upper())
        
        # Don't forget the last sequence
        if current_id is not None and (not max_sequences or len(records) < max_sequences):
            records.append(SequenceRecord(
                id=current_id,
                description=current_desc,
                sequence=''.join(current_seq)
            ))
    
    finally:
        file_handle.close()
    
    return records


def read_fasta_streaming(filepath: Path) -> Generator[SequenceRecord, None, None]:
    """
    Read sequences from a FASTA file in a memory-efficient streaming manner.
    
    This is useful for very large FASTA files where loading all sequences
    into memory would be impractical.
    
    Args:
        filepath: Path to FASTA file
        
    Yields:
        SequenceRecord objects one at a time
        
    Example:
        >>> for record in read_fasta_streaming(Path("large_genome.fasta")):
        ...     process(record)
    """
    # Handle gzip compressed files
    if str(filepath).endswith('.gz'):
        file_handle = gzip.open(filepath, 'rt')
    else:
        file_handle = open(filepath, 'r')
    
    try:
        current_id = None
        current_desc = None
        current_seq = []
        
        for line in file_handle:
            line = line.strip()
            
            if not line:
                continue
            
            if line.startswith('>'):
                # Yield previous sequence if exists
                if current_id is not None:
                    yield SequenceRecord(
                        id=current_id,
                        description=current_desc,
                        sequence=''.join(current_seq)
                    )
                
                # Parse header line
                header = line[1:].strip()
                parts = header.split(maxsplit=1)
                current_id = parts[0]
                current_desc = parts[1] if len(parts) > 1 else ''
                current_seq = []
            else:
                # Sequence line
                current_seq.append(line.upper())
        
        # Don't forget the last sequence
        if current_id is not None:
            yield SequenceRecord(
                id=current_id,
                description=current_desc,
                sequence=''.join(current_seq)
            )
    
    finally:
        file_handle.close()


def read_fastq(filepath: Path, max_sequences: Optional[int] = None) -> List[SequenceRecord]:
    """
    Read sequences from a FASTQ file.
    
    FASTQ format:
        @sequence_id description
        ATCGATCG...
        +
        quality_scores...
        
    Args:
        filepath: Path to FASTQ file (.fastq, .fq, or .gz compressed)
        max_sequences: Maximum number of sequences to read (None = all)
        
    Returns:
        List of SequenceRecord objects (quality scores not included)
    """
    records = []
    
    # Handle gzip compressed files
    if str(filepath).endswith('.gz'):
        file_handle = gzip.open(filepath, 'rt')
    else:
        file_handle = open(filepath, 'r')
    
    try:
        while True:
            # Read 4 lines (FASTQ record)
            header = file_handle.readline().strip()
            if not header:
                break
            
            sequence = file_handle.readline().strip()
            plus = file_handle.readline().strip()
            quality = file_handle.readline().strip()
            
            if not header.startswith('@'):
                raise ValueError(f"Invalid FASTQ format: header doesn't start with '@': {header}")
            
            # Parse header
            header = header[1:].strip()
            parts = header.split(maxsplit=1)
            seq_id = parts[0]
            desc = parts[1] if len(parts) > 1 else ''
            
            records.append(SequenceRecord(
                id=seq_id,
                description=desc,
                sequence=sequence.upper()
            ))
            
            if max_sequences and len(records) >= max_sequences:
                break
    
    finally:
        file_handle.close()
    
    return records


def load_dataset(name: str, datasets_dir: Path) -> Optional[SequenceRecord]:
    """
    Load a dataset by name from the datasets directory.
    
    Args:
        name: Dataset name (e.g., 'ecoli', 'lambda_phage')
        datasets_dir: Path to datasets directory
        
    Returns:
        SequenceRecord object or None if not found
        
    Example:
        >>> from kmp.config import DATASETS_DIR
        >>> ecoli = load_dataset('ecoli', DATASETS_DIR)
    """
    # Look for files matching the dataset name
    patterns = [
        f"{name}.fasta",
        f"{name}.fa",
        f"{name}.fna",
        f"{name}.fasta.gz",
        f"{name}.fa.gz",
        f"{name}.fna.gz",
        f"*{name}*.fasta",
        f"*{name}*.fa",
        f"*{name}*.fna",
        f"*{name}*.fasta.gz",
        f"*{name}*.fa.gz",
        f"*{name}*.fna.gz",
    ]
    
    for pattern in patterns:
        matches = list(datasets_dir.glob(pattern))
        if matches:
            filepath = matches[0]
            print(f"Loading {filepath.name}...")
            records = read_fasta(filepath, max_sequences=1)
            if records:
                return records[0]
    
    return None


def get_subsequence(sequence: str, start: int, end: int) -> str:
    """
    Extract a subsequence from a sequence.
    
    Args:
        sequence: The full sequence
        start: Start position (0-indexed, inclusive)
        end: End position (0-indexed, exclusive)
        
    Returns:
        Subsequence
    """
    return sequence[start:end]


def split_sequence(sequence: str, chunk_size: int) -> List[str]:
    """
    Split a sequence into chunks of specified size.
    
    Args:
        sequence: The sequence to split
        chunk_size: Size of each chunk
        
    Returns:
        List of sequence chunks
    """
    return [sequence[i:i+chunk_size] for i in range(0, len(sequence), chunk_size)]


def concatenate_sequences(records: List[SequenceRecord], separator: str = '') -> str:
    """
    Concatenate multiple sequence records into one string.
    
    Args:
        records: List of SequenceRecord objects
        separator: String to insert between sequences
        
    Returns:
        Concatenated sequence string
    """
    return separator.join(record.sequence for record in records)


def sequence_stats(sequence: str) -> Dict[str, any]:
    """
    Calculate basic statistics for a DNA sequence.
    
    Args:
        sequence: DNA sequence string
        
    Returns:
        Dictionary with statistics:
            - length: Sequence length
            - gc_content: GC content (0-1)
            - base_counts: Count of each base
            - base_frequencies: Frequency of each base
    """
    length = len(sequence)
    if length == 0:
        return {
            'length': 0,
            'gc_content': 0.0,
            'base_counts': {},
            'base_frequencies': {}
        }
    
    # Count bases
    base_counts = {}
    for base in 'ACGTN':
        base_counts[base] = sequence.count(base)
    
    # Calculate frequencies
    base_frequencies = {base: count / length for base, count in base_counts.items()}
    
    # Calculate GC content
    gc_content = (base_counts['G'] + base_counts['C']) / length
    
    return {
        'length': length,
        'gc_content': gc_content,
        'base_counts': base_counts,
        'base_frequencies': base_frequencies
    }


def write_fasta(records: List[SequenceRecord], filepath: Path, line_width: int = 80) -> None:
    """
    Write sequences to a FASTA file.
    
    Args:
        records: List of SequenceRecord objects to write
        filepath: Output file path
        line_width: Number of characters per sequence line (0 = no wrapping)
    """
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'w') as f:
        for record in records:
            # Write header
            header = f">{record.id}"
            if record.description:
                header += f" {record.description}"
            f.write(header + '\n')
            
            # Write sequence (with line wrapping if specified)
            if line_width > 0:
                for i in range(0, len(record.sequence), line_width):
                    f.write(record.sequence[i:i+line_width] + '\n')
            else:
                f.write(record.sequence + '\n')
