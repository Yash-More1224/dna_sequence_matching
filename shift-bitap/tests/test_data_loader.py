"""
Unit tests for data loading and synthetic data generation.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from data_loader import DataLoader, SyntheticDataGenerator, create_motif_dataset


class TestSyntheticDataGenerator:
    """Test synthetic data generation."""
    
    def test_random_sequence_generation(self):
        """Test generating random DNA sequences."""
        seq = SyntheticDataGenerator.generate_random_sequence(100, seed=42)
        
        assert len(seq) == 100
        assert all(base in 'ACGT' for base in seq)
    
    def test_random_sequence_reproducibility(self):
        """Test that same seed produces same sequence."""
        seq1 = SyntheticDataGenerator.generate_random_sequence(50, seed=42)
        seq2 = SyntheticDataGenerator.generate_random_sequence(50, seed=42)
        
        assert seq1 == seq2
    
    def test_gc_biased_sequence(self):
        """Test GC-biased sequence generation."""
        seq = SyntheticDataGenerator.generate_gc_biased_sequence(1000, gc_content=0.7, seed=42)
        
        gc_count = seq.count('G') + seq.count('C')
        gc_actual = gc_count / len(seq)
        
        # Should be approximately 70% (within 10% tolerance)
        assert 0.6 <= gc_actual <= 0.8, f"GC content {gc_actual} not close to 0.7"
    
    def test_gc_content_boundaries(self):
        """Test GC content at boundaries."""
        # 0% GC
        seq0 = SyntheticDataGenerator.generate_gc_biased_sequence(100, gc_content=0.0, seed=42)
        assert all(base in 'AT' for base in seq0)
        
        # 100% GC
        seq100 = SyntheticDataGenerator.generate_gc_biased_sequence(100, gc_content=1.0, seed=42)
        assert all(base in 'GC' for base in seq100)
    
    def test_substitution_introduction(self):
        """Test introducing substitutions."""
        original = "AAAAAAAAAA"
        mutated = SyntheticDataGenerator.introduce_substitutions(original, rate=0.5, seed=42)
        
        differences = sum(1 for a, b in zip(original, mutated) if a != b)
        
        # With 50% rate, expect roughly half to be different
        assert 2 <= differences <= 8, f"Expected ~5 differences, got {differences}"
    
    def test_substitution_zero_rate(self):
        """Test that zero substitution rate produces identical sequence."""
        original = "ACGTACGT"
        mutated = SyntheticDataGenerator.introduce_substitutions(original, rate=0.0, seed=42)
        
        assert original == mutated
    
    def test_indel_introduction(self):
        """Test introducing indels."""
        original = "ACGT" * 10
        mutated = SyntheticDataGenerator.introduce_indels(original, 0.1, 0.1, seed=42)
        
        # Length should change due to indels
        assert len(mutated) != len(original)
    
    def test_mutate_sequence(self):
        """Test combined mutation function."""
        original = "ACGTACGTACGT"
        mutated, stats = SyntheticDataGenerator.mutate_sequence(
            original, substitution_rate=0.1, insertion_rate=0.05, 
            deletion_rate=0.05, seed=42
        )
        
        assert isinstance(mutated, str)
        assert 'original_length' in stats
        assert 'mutated_length' in stats
        assert stats['original_length'] == len(original)
        assert stats['mutated_length'] == len(mutated)
    
    def test_test_dataset_creation(self):
        """Test creating a test dataset."""
        dataset = SyntheticDataGenerator.create_test_dataset(
            num_sequences=2,
            seq_length=50,
            mutation_rates=[0.0, 0.1],
            seed=42
        )
        
        assert len(dataset) == 4  # 2 sequences * 2 rates
        
        for orig, mut, rate in dataset:
            assert len(orig) == 50
            assert isinstance(mut, str)
            assert rate in [0.0, 0.1]


class TestMotifDataset:
    """Test motif dataset creation."""
    
    def test_create_motif_dataset(self):
        """Test creating dataset with embedded motifs."""
        motifs = ["TATAAA", "GATTACA"]
        text = create_motif_dataset(motifs, background_length=1000, 
                                   num_copies=5, seed=42)
        
        assert len(text) >= 1000
        assert text.count("TATAAA") >= 1
        assert text.count("GATTACA") >= 1
    
    def test_motif_with_noise(self):
        """Test motif dataset with mutations."""
        motifs = ["ACGTACGT"]
        text = create_motif_dataset(motifs, background_length=500,
                                   num_copies=5, noise=0.2, seed=42)
        
        # With noise, exact motif count may be lower
        assert len(text) >= 500


class TestDataLoader:
    """Test FASTA file operations."""
    
    def test_dataloader_initialization(self):
        """Test DataLoader initialization."""
        loader = DataLoader()
        assert loader.cache_dir.exists()
    
    def test_save_and_load_fasta(self, tmp_path):
        """Test saving and loading FASTA files."""
        loader = DataLoader()
        
        # Create test sequences
        sequences = {
            'seq1': 'ACGTACGT',
            'seq2': 'GGGGCCCC'
        }
        
        # Save
        fasta_file = tmp_path / "test.fasta"
        loader.save_fasta(sequences, str(fasta_file))
        
        # Load
        loaded = loader.load_fasta(str(fasta_file))
        
        assert loaded == sequences
    
    def test_load_fasta_single(self, tmp_path):
        """Test loading single sequence from FASTA."""
        loader = DataLoader()
        
        sequences = {'seq1': 'ACGTACGT'}
        fasta_file = tmp_path / "test.fasta"
        loader.save_fasta(sequences, str(fasta_file))
        
        single = loader.load_fasta_single(str(fasta_file))
        assert single == 'ACGTACGT'
    
    def test_genome_urls(self):
        """Test that genome URLs are defined."""
        loader = DataLoader()
        assert 'ecoli' in loader.genome_urls
        assert 'lambda' in loader.genome_urls


class TestInputValidation:
    """Test input validation and error handling."""
    
    def test_invalid_gc_content(self):
        """Test invalid GC content raises error."""
        with pytest.raises(ValueError):
            SyntheticDataGenerator.generate_gc_biased_sequence(100, gc_content=1.5)
        
        with pytest.raises(ValueError):
            SyntheticDataGenerator.generate_gc_biased_sequence(100, gc_content=-0.1)
    
    def test_invalid_substitution_rate(self):
        """Test invalid substitution rate raises error."""
        with pytest.raises(ValueError):
            SyntheticDataGenerator.introduce_substitutions("ACGT", rate=1.5)
    
    def test_empty_fasta_file(self, tmp_path):
        """Test loading empty FASTA file."""
        loader = DataLoader()
        empty_file = tmp_path / "empty.fasta"
        empty_file.write_text("")
        
        sequences = loader.load_fasta(str(empty_file))
        assert sequences == {}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
