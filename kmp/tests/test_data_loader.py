"""
Unit tests for data loading functionality.
"""

import pytest
from pathlib import Path
import tempfile
import os

from kmp.data_loader import (
    SequenceRecord,
    read_fasta,
    write_fasta,
    get_subsequence,
    split_sequence,
    concatenate_sequences,
    sequence_stats
)


class TestSequenceRecord:
    """Tests for SequenceRecord dataclass."""
    
    def test_creation(self):
        """Test creating a SequenceRecord."""
        record = SequenceRecord(
            id="test_seq",
            description="Test sequence",
            sequence="ATCGATCG"
        )
        
        assert record.id == "test_seq"
        assert record.description == "Test sequence"
        assert record.sequence == "ATCGATCG"
        assert record.length == 8
    
    def test_length_property(self):
        """Test length property."""
        record = SequenceRecord(id="test", description="", sequence="ATCG")
        assert record.length == 4
        
        record2 = SequenceRecord(id="test2", description="", sequence="")
        assert record2.length == 0


class TestFastaIO:
    """Tests for FASTA file I/O."""
    
    def test_write_and_read_fasta(self):
        """Test writing and reading FASTA files."""
        # Create test data
        records = [
            SequenceRecord(id="seq1", description="First sequence", sequence="ATCGATCG"),
            SequenceRecord(id="seq2", description="Second sequence", sequence="GCTAGCTA")
        ]
        
        # Write to temporary file
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test.fasta"
            write_fasta(records, filepath)
            
            # Read back
            loaded_records = read_fasta(filepath)
            
            assert len(loaded_records) == 2
            assert loaded_records[0].id == "seq1"
            assert loaded_records[0].sequence == "ATCGATCG"
            assert loaded_records[1].id == "seq2"
            assert loaded_records[1].sequence == "GCTAGCTA"
    
    def test_read_fasta_with_max_sequences(self):
        """Test reading limited number of sequences."""
        records = [
            SequenceRecord(id=f"seq{i}", description=f"Seq {i}", sequence="ATCG" * 10)
            for i in range(10)
        ]
        
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test.fasta"
            write_fasta(records, filepath)
            
            # Read only first 3
            loaded_records = read_fasta(filepath, max_sequences=3)
            assert len(loaded_records) == 3
    
    def test_write_fasta_with_line_wrapping(self):
        """Test FASTA writing with line wrapping."""
        record = SequenceRecord(
            id="test",
            description="Test",
            sequence="A" * 100
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test.fasta"
            write_fasta([record], filepath, line_width=50)
            
            # Check file contents
            with open(filepath) as f:
                lines = f.readlines()
                # Should have header + 2 sequence lines (100/50)
                assert len(lines) == 3
                assert len(lines[1].strip()) == 50
                assert len(lines[2].strip()) == 50
    
    def test_write_fasta_no_wrapping(self):
        """Test FASTA writing without line wrapping."""
        record = SequenceRecord(
            id="test",
            description="Test",
            sequence="A" * 100
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test.fasta"
            write_fasta([record], filepath, line_width=0)
            
            # Check file contents
            with open(filepath) as f:
                lines = f.readlines()
                # Should have header + 1 sequence line
                assert len(lines) == 2
                assert len(lines[1].strip()) == 100


class TestSequenceOperations:
    """Tests for sequence manipulation functions."""
    
    def test_get_subsequence(self):
        """Test extracting subsequence."""
        seq = "ATCGATCGATCG"
        subseq = get_subsequence(seq, 2, 6)
        assert subseq == "CGAT"
    
    def test_split_sequence(self):
        """Test splitting sequence into chunks."""
        seq = "ATCGATCGATCG"
        chunks = split_sequence(seq, 4)
        assert chunks == ["ATCG", "ATCG", "ATCG"]
        
        # Test with non-divisible length
        chunks2 = split_sequence(seq, 5)
        assert chunks2 == ["ATCGA", "TCGAT", "CG"]
    
    def test_concatenate_sequences(self):
        """Test concatenating sequences."""
        records = [
            SequenceRecord(id="1", description="", sequence="ATCG"),
            SequenceRecord(id="2", description="", sequence="GCTA"),
        ]
        
        # No separator
        concat = concatenate_sequences(records)
        assert concat == "ATCGGCTA"
        
        # With separator
        concat_sep = concatenate_sequences(records, separator="NN")
        assert concat_sep == "ATCGNNGCTA"


class TestSequenceStats:
    """Tests for sequence statistics."""
    
    def test_sequence_stats_basic(self):
        """Test basic sequence statistics."""
        seq = "ATCGATCG"
        stats = sequence_stats(seq)
        
        assert stats['length'] == 8
        assert stats['base_counts']['A'] == 2
        assert stats['base_counts']['T'] == 2
        assert stats['base_counts']['C'] == 2
        assert stats['base_counts']['G'] == 2
        assert stats['gc_content'] == 0.5  # 50% GC
    
    def test_sequence_stats_gc_content(self):
        """Test GC content calculation."""
        # 100% GC
        seq = "GGGGCCCC"
        stats = sequence_stats(seq)
        assert stats['gc_content'] == 1.0
        
        # 0% GC
        seq2 = "AAAATTTT"
        stats2 = sequence_stats(seq2)
        assert stats2['gc_content'] == 0.0
    
    def test_sequence_stats_empty(self):
        """Test stats for empty sequence."""
        stats = sequence_stats("")
        assert stats['length'] == 0
        assert stats['gc_content'] == 0.0
    
    def test_sequence_stats_frequencies(self):
        """Test base frequency calculation."""
        seq = "AAATCG"
        stats = sequence_stats(seq)
        
        assert stats['base_frequencies']['A'] == pytest.approx(0.5)
        assert stats['base_frequencies']['T'] == pytest.approx(1/6)
        assert stats['base_frequencies']['C'] == pytest.approx(1/6)
        assert stats['base_frequencies']['G'] == pytest.approx(1/6)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
