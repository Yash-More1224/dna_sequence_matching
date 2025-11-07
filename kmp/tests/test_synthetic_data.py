"""
Unit tests for synthetic data generation.
"""

import pytest
from kmp.synthetic_data import (
    generate_random_sequence,
    generate_pattern,
    inject_pattern,
    introduce_substitution,
    introduce_insertion,
    introduce_deletion,
    mutate_sequence,
    generate_synthetic_dataset,
    generate_test_cases
)
from kmp.config import DNA_BASES


class TestRandomGeneration:
    """Tests for random sequence generation."""
    
    def test_generate_random_sequence(self):
        """Test generating random DNA sequence."""
        seq = generate_random_sequence(100, seed=42)
        
        assert len(seq) == 100
        assert all(base in DNA_BASES for base in seq)
    
    def test_generate_random_sequence_reproducible(self):
        """Test that same seed produces same sequence."""
        seq1 = generate_random_sequence(100, seed=42)
        seq2 = generate_random_sequence(100, seed=42)
        
        assert seq1 == seq2
    
    def test_generate_random_sequence_different_seeds(self):
        """Test that different seeds produce different sequences."""
        seq1 = generate_random_sequence(100, seed=42)
        seq2 = generate_random_sequence(100, seed=43)
        
        assert seq1 != seq2
    
    def test_generate_pattern(self):
        """Test pattern generation."""
        pattern = generate_pattern(50, seed=42)
        
        assert len(pattern) == 50
        assert all(base in DNA_BASES for base in pattern)


class TestPatternInjection:
    """Tests for pattern injection."""
    
    def test_inject_pattern_basic(self):
        """Test basic pattern injection."""
        sequence = "AAAAAAAAAA"
        pattern = "TCGT"
        position = 3
        
        result = inject_pattern(sequence, pattern, position)
        assert result == "AAATCGTAAA"
        assert result[position:position+len(pattern)] == pattern
    
    def test_inject_pattern_at_start(self):
        """Test injecting at start."""
        sequence = "AAAAAAAAAA"
        pattern = "TCGT"
        
        result = inject_pattern(sequence, pattern, 0)
        assert result.startswith(pattern)
    
    def test_inject_pattern_at_end(self):
        """Test injecting at end."""
        sequence = "AAAAAAAAAA"
        pattern = "TCGT"
        position = len(sequence) - len(pattern)
        
        result = inject_pattern(sequence, pattern, position)
        assert result.endswith(pattern)
    
    def test_inject_pattern_invalid_position(self):
        """Test that invalid position raises error."""
        sequence = "AAAA"
        pattern = "TCGT"
        
        with pytest.raises(ValueError):
            inject_pattern(sequence, pattern, 5)
        
        with pytest.raises(ValueError):
            inject_pattern(sequence, pattern, -1)


class TestMutations:
    """Tests for mutation functions."""
    
    def test_introduce_substitution(self):
        """Test substitution mutation."""
        sequence = "ATCGATCG"
        position = 2
        
        result = introduce_substitution(sequence, position, 'T')
        assert result == "ATTGATCG"
        assert len(result) == len(sequence)
    
    def test_introduce_substitution_random(self):
        """Test random substitution."""
        sequence = "ATCGATCG"
        position = 2
        original_base = sequence[position]
        
        result = introduce_substitution(sequence, position)
        assert len(result) == len(sequence)
        assert result[position] != original_base
        assert result[position] in DNA_BASES
    
    def test_introduce_insertion(self):
        """Test insertion mutation."""
        sequence = "ATCGATCG"
        position = 2
        
        result = introduce_insertion(sequence, position, 'G')
        assert result == "ATGCGATCG"
        assert len(result) == len(sequence) + 1
    
    def test_introduce_deletion(self):
        """Test deletion mutation."""
        sequence = "ATCGATCG"
        position = 2
        
        result = introduce_deletion(sequence, position)
        assert result == "ATGATCG"
        assert len(result) == len(sequence) - 1
    
    def test_mutate_sequence(self):
        """Test sequence mutation."""
        sequence = "A" * 1000
        mutation_rate = 0.1
        
        mutated, info = mutate_sequence(sequence, mutation_rate, seed=42)
        
        # Should have some mutations
        assert info['total_mutations'] > 0
        assert len(mutated) >= len(sequence) * 0.9  # Account for deletions
    
    def test_mutate_sequence_no_mutations(self):
        """Test with zero mutation rate."""
        sequence = "ATCGATCG"
        
        mutated, info = mutate_sequence(sequence, 0.0, seed=42)
        
        assert mutated == sequence
        assert info['total_mutations'] == 0
    
    def test_mutate_sequence_types(self):
        """Test different mutation types."""
        sequence = "A" * 1000
        mutation_types = {
            'substitution': 0.5,
            'insertion': 0.25,
            'deletion': 0.25
        }
        
        mutated, info = mutate_sequence(
            sequence, 0.1, mutation_types=mutation_types, seed=42
        )
        
        assert 'substitutions' in info
        assert 'insertions' in info
        assert 'deletions' in info


class TestSyntheticDataset:
    """Tests for synthetic dataset generation."""
    
    def test_generate_synthetic_dataset_basic(self):
        """Test basic synthetic dataset generation."""
        dataset = generate_synthetic_dataset(
            text_length=10000,
            num_patterns=5,
            pattern_length=20,
            num_injections=10,
            mutation_rate=0.0,
            seed=42
        )
        
        assert len(dataset.sequence) == 10000
        assert len(dataset.patterns) == 5
        assert all(len(p) == 20 for p in dataset.patterns)
        assert dataset.metadata['num_patterns'] == 5
    
    def test_generate_synthetic_dataset_with_mutations(self):
        """Test synthetic dataset with mutations."""
        dataset = generate_synthetic_dataset(
            text_length=10000,
            num_patterns=5,
            pattern_length=20,
            num_injections=10,
            mutation_rate=0.05,
            seed=42
        )
        
        assert dataset.mutations['total_mutations'] > 0
        assert dataset.metadata['mutation_rate'] == 0.05
    
    def test_generate_synthetic_dataset_reproducible(self):
        """Test that same seed produces same dataset."""
        dataset1 = generate_synthetic_dataset(
            text_length=1000,
            num_patterns=3,
            pattern_length=20,
            num_injections=5,
            seed=42
        )
        
        dataset2 = generate_synthetic_dataset(
            text_length=1000,
            num_patterns=3,
            pattern_length=20,
            num_injections=5,
            seed=42
        )
        
        assert dataset1.sequence == dataset2.sequence
        assert dataset1.patterns == dataset2.patterns
    
    def test_generate_synthetic_dataset_custom_patterns(self):
        """Test with custom patterns."""
        custom_patterns = ["ATCG", "GCTA", "TTAA"]
        
        dataset = generate_synthetic_dataset(
            text_length=1000,
            patterns=custom_patterns,
            num_injections=10,
            seed=42
        )
        
        assert dataset.patterns == custom_patterns
    
    def test_to_sequence_record(self):
        """Test converting to SequenceRecord."""
        dataset = generate_synthetic_dataset(
            text_length=100,
            num_patterns=2,
            seed=42
        )
        
        record = dataset.to_sequence_record("test_seq")
        assert record.id == "test_seq"
        assert record.sequence == dataset.sequence
        assert len(record.sequence) == len(dataset.sequence)


class TestTestCases:
    """Tests for test case generation."""
    
    def test_generate_test_cases(self):
        """Test generating standard test cases."""
        test_cases = generate_test_cases()
        
        assert len(test_cases) > 0
        
        # Check structure
        for text, pattern, expected_positions in test_cases:
            assert isinstance(text, str)
            assert isinstance(pattern, str)
            assert isinstance(expected_positions, list)
            assert all(isinstance(pos, int) for pos in expected_positions)
    
    def test_test_cases_validity(self):
        """Test that generated test cases are valid."""
        from kmp.kmp_algorithm import kmp_search
        
        test_cases = generate_test_cases()
        
        for text, pattern, expected in test_cases:
            # Verify expected results are correct
            actual = kmp_search(text, pattern)
            assert sorted(actual) == sorted(expected), \
                f"Test case failed: text='{text}', pattern='{pattern}'"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
