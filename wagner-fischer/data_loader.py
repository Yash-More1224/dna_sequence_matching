
import os
import random
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from pathlib import Path
import gzip

try:
    from Bio import SeqIO
    from Bio.Seq import Seq
    BIOPYTHON_AVAILABLE = True
except ImportError:
    BIOPYTHON_AVAILABLE = False
    print("Warning: Biopython not available. Install with: pip install biopython")

@dataclass
class Sequence:
    id: str
    sequence: str
    description: str = ""
    
    def __len__(self):
        return len(self.sequence)


class FastaLoader:
    
    
    def __init__(self, cache_dir: Optional[str] = None):
        
        self.cache_dir = cache_dir
        if cache_dir:
            Path(cache_dir).mkdir(parents=True, exist_ok=True)
    
    def load(self, filepath: str) -> List[Sequence]:
        
        if not BIOPYTHON_AVAILABLE:
            return self._load_simple(filepath)
        
        sequences = []
        
        # Handle gzipped files
        if filepath.endswith('.gz'):
            with gzip.open(filepath, 'rt') as handle:
                for record in SeqIO.parse(handle, 'fasta'):
                    seq = Sequence(
                        id=record.id,
                        sequence=str(record.seq).upper(),
                        description=record.description
                    )
                    sequences.append(seq)
        else:
            for record in SeqIO.parse(filepath, 'fasta'):
                seq = Sequence(
                    id=record.id,
                    sequence=str(record.seq).upper(),
                    description=record.description
                )
                sequences.append(seq)
        
        return sequences
    
    def _load_simple(self, filepath: str) -> List[Sequence]:
       
        sequences = []
        current_id = None
        current_seq = []
        current_desc = ""
        
        open_func = gzip.open if filepath.endswith('.gz') else open
        mode = 'rt' if filepath.endswith('.gz') else 'r'
        
        with open_func(filepath, mode) as f:
            for line in f:
                line = line.strip()
                if line.startswith('>'):
                    # Save previous sequence
                    if current_id:
                        sequences.append(Sequence(
                            id=current_id,
                            sequence=''.join(current_seq).upper(),
                            description=current_desc
                        ))
                    
                    # Start new sequence
                    parts = line[1:].split(maxsplit=1)
                    current_id = parts[0]
                    current_desc = parts[1] if len(parts) > 1 else ""
                    current_seq = []
                else:
                    current_seq.append(line)
            
            # Save last sequence
            if current_id:
                sequences.append(Sequence(
                    id=current_id,
                    sequence=''.join(current_seq).upper(),
                    description=current_desc
                ))
        
        return sequences
    
    def load_single(self, filepath: str) -> Sequence:
       
        sequences = self.load(filepath)
        if not sequences:
            raise ValueError(f"No sequences found in {filepath}")
        return sequences[0]


class SyntheticDataGenerator:
    """
    Generate synthetic DNA sequences with controlled mutations.
    """
    
    DNA_BASES = ['A', 'C', 'G', 'T']
    
    def __init__(self, seed: Optional[int] = None):
        
        if seed is not None:
            random.seed(seed)
    
    def generate_random_sequence(self, length: int) -> str:
       
        return ''.join(random.choices(self.DNA_BASES, k=length))
    
    def mutate_sequence(self,
                       sequence: str,
                       substitution_rate: float = 0.01,
                       insertion_rate: float = 0.005,
                       deletion_rate: float = 0.005) -> Tuple[str, List[Dict]]:
        
        mutated = list(sequence)
        mutations = []
        i = 0
        
        while i < len(mutated):
            # Substitution
            if random.random() < substitution_rate:
                original = mutated[i]
                # Choose different base
                new_base = random.choice([b for b in self.DNA_BASES if b != original])
                mutated[i] = new_base
                mutations.append({
                    'type': 'substitution',
                    'position': i,
                    'original': original,
                    'new': new_base
                })
            
            # Insertion
            if random.random() < insertion_rate:
                new_base = random.choice(self.DNA_BASES)
                mutated.insert(i + 1, new_base)
                mutations.append({
                    'type': 'insertion',
                    'position': i + 1,
                    'base': new_base
                })
            
            # Deletion
            if random.random() < deletion_rate and len(mutated) > 1:
                deleted_base = mutated[i]
                del mutated[i]
                mutations.append({
                    'type': 'deletion',
                    'position': i,
                    'base': deleted_base
                })
                i -= 1  # Stay at same position
            
            i += 1
        
        return ''.join(mutated), mutations
    
    def generate_dataset(self,
                        num_sequences: int,
                        seq_length: int,
                        mutation_rate: float = 0.01) -> List[Tuple[str, str]]:
      
        dataset = []
        
        for _ in range(num_sequences):
            original = self.generate_random_sequence(seq_length)
            mutated, _ = self.mutate_sequence(
                original,
                substitution_rate=mutation_rate,
                insertion_rate=mutation_rate / 2,
                deletion_rate=mutation_rate / 2
            )
            dataset.append((original, mutated))
        
        return dataset
    
    def save_fasta(self, sequences: List[Tuple[str, str]], filepath: str):

        with open(filepath, 'w') as f:
            for seq_id, sequence in sequences:
                f.write(f">{seq_id}\n")
                # Write in 80-character lines
                for i in range(0, len(sequence), 80):
                    f.write(sequence[i:i+80] + '\n')


def download_ecoli_genome(output_dir: str = "data") -> str:
    
    import urllib.request
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    url = "https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/005/845/GCF_000005845.2_ASM584v2/GCF_000005845.2_ASM584v2_genomic.fna.gz"
    output_path = os.path.join(output_dir, "ecoli_k12.fna.gz")
    
    if os.path.exists(output_path):
        print(f"Genome already exists at {output_path}")
        return output_path
    
    print(f"Downloading E. coli genome to {output_path}...")
    urllib.request.urlretrieve(url, output_path)
    print("Download complete!")
    
    return output_path


def create_test_datasets(output_dir: str = "data"):
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    generator = SyntheticDataGenerator(seed=42)
    
    # Small test dataset
    print("Generating small test dataset...")
    small_seqs = [
        (f"seq_{i}", generator.generate_random_sequence(1000))
        for i in range(10)
    ]
    generator.save_fasta(small_seqs, os.path.join(output_dir, "synthetic_small.fasta"))
    
    # Medium dataset with mutations
    print("Generating medium mutated dataset...")
    medium_seqs = []
    for i in range(50):
        original = generator.generate_random_sequence(5000)
        mutated, _ = generator.mutate_sequence(original, substitution_rate=0.02)
        medium_seqs.append((f"original_{i}", original))
        medium_seqs.append((f"mutated_{i}", mutated))
    generator.save_fasta(medium_seqs, os.path.join(output_dir, "synthetic_medium.fasta"))
    
    print(f"Test datasets created in {output_dir}")
