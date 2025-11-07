"""
Synthetic DNA sequence generator for testing and benchmarking.

This module provides functions to generate random DNA sequences,
inject patterns at known positions, and introduce controlled mutations
for testing approximate matching algorithms.
"""

import random
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass

from .config import DNA_BASES, SYNTHETIC_CONFIG
from .data_loader import SequenceRecord


@dataclass
class SyntheticSequence:
    """
    Container for synthetic sequence data with ground truth.
    
    Attributes:
        sequence: The generated DNA sequence
        patterns: List of patterns injected into the sequence
        pattern_positions: Positions where each pattern was injected
        mutations: Information about mutations applied
        metadata: Additional metadata
    """
    sequence: str
    patterns: List[str]
    pattern_positions: Dict[str, List[int]]
    mutations: Dict[str, any]
    metadata: Dict[str, any]
    
    def to_sequence_record(self, seq_id: str = "synthetic") -> SequenceRecord:
        """Convert to SequenceRecord object."""
        desc = f"Synthetic sequence, length={len(self.sequence)}, patterns={len(self.patterns)}"
        return SequenceRecord(id=seq_id, description=desc, sequence=self.sequence)


def generate_random_sequence(length: int, 
                            base_probabilities: Optional[Dict[str, float]] = None,
                            seed: Optional[int] = None) -> str:
    """
    Generate a random DNA sequence.
    
    Args:
        length: Length of sequence to generate
        base_probabilities: Dictionary of base probabilities (must sum to 1.0)
                          If None, uses uniform distribution
        seed: Random seed for reproducibility
        
    Returns:
        Random DNA sequence
        
    Example:
        >>> seq = generate_random_sequence(100, seed=42)
        >>> len(seq)
        100
    """
    if seed is not None:
        random.seed(seed)
    
    if base_probabilities is None:
        base_probabilities = SYNTHETIC_CONFIG['base_probabilities']
    
    # Validate probabilities
    total_prob = sum(base_probabilities.values())
    if abs(total_prob - 1.0) > 1e-6:
        raise ValueError(f"Base probabilities must sum to 1.0, got {total_prob}")
    
    # Generate sequence using weighted random choice
    bases = list(base_probabilities.keys())
    weights = list(base_probabilities.values())
    
    sequence = ''.join(random.choices(bases, weights=weights, k=length))
    return sequence


def generate_pattern(length: int, 
                    base_probabilities: Optional[Dict[str, float]] = None,
                    seed: Optional[int] = None) -> str:
    """
    Generate a random DNA pattern.
    
    Args:
        length: Length of pattern
        base_probabilities: Dictionary of base probabilities
        seed: Random seed
        
    Returns:
        Random DNA pattern
    """
    return generate_random_sequence(length, base_probabilities, seed)


def inject_pattern(sequence: str, pattern: str, position: int) -> str:
    """
    Inject a pattern into a sequence at a specific position.
    
    Args:
        sequence: The base sequence
        pattern: Pattern to inject
        position: Position to inject at (0-indexed)
        
    Returns:
        Modified sequence with pattern injected
        
    Raises:
        ValueError: If position is invalid or pattern doesn't fit
    """
    if position < 0 or position + len(pattern) > len(sequence):
        raise ValueError(f"Invalid position {position} for pattern of length {len(pattern)} "
                       f"in sequence of length {len(sequence)}")
    
    return sequence[:position] + pattern + sequence[position + len(pattern):]


def inject_patterns_random(sequence: str, 
                          patterns: List[str], 
                          num_injections: int,
                          seed: Optional[int] = None) -> Tuple[str, Dict[str, List[int]]]:
    """
    Inject patterns at random positions in a sequence.
    
    Args:
        sequence: The base sequence
        patterns: List of patterns to inject
        num_injections: Number of times to inject patterns
        seed: Random seed
        
    Returns:
        Tuple of (modified_sequence, pattern_positions_dict)
        where pattern_positions_dict maps pattern -> list of positions
    """
    if seed is not None:
        random.seed(seed)
    
    result = sequence
    pattern_positions = {p: [] for p in patterns}
    
    for _ in range(num_injections):
        pattern = random.choice(patterns)
        max_pos = len(result) - len(pattern)
        
        if max_pos <= 0:
            continue
        
        position = random.randint(0, max_pos)
        result = inject_pattern(result, pattern, position)
        pattern_positions[pattern].append(position)
    
    return result, pattern_positions


def introduce_substitution(sequence: str, position: int, new_base: Optional[str] = None) -> str:
    """
    Introduce a substitution mutation at a specific position.
    
    Args:
        sequence: DNA sequence
        position: Position to mutate
        new_base: Base to substitute (if None, random base different from original)
        
    Returns:
        Mutated sequence
    """
    if position < 0 or position >= len(sequence):
        raise ValueError(f"Invalid position {position} for sequence of length {len(sequence)}")
    
    if new_base is None:
        # Choose a random base different from the original
        original = sequence[position]
        choices = [b for b in DNA_BASES if b != original]
        new_base = random.choice(choices)
    
    return sequence[:position] + new_base + sequence[position + 1:]


def introduce_insertion(sequence: str, position: int, inserted_base: Optional[str] = None) -> str:
    """
    Introduce an insertion mutation at a specific position.
    
    Args:
        sequence: DNA sequence
        position: Position to insert at
        inserted_base: Base to insert (if None, random base)
        
    Returns:
        Mutated sequence
    """
    if position < 0 or position > len(sequence):
        raise ValueError(f"Invalid position {position} for sequence of length {len(sequence)}")
    
    if inserted_base is None:
        inserted_base = random.choice(DNA_BASES)
    
    return sequence[:position] + inserted_base + sequence[position:]


def introduce_deletion(sequence: str, position: int) -> str:
    """
    Introduce a deletion mutation at a specific position.
    
    Args:
        sequence: DNA sequence
        position: Position to delete
        
    Returns:
        Mutated sequence
    """
    if position < 0 or position >= len(sequence):
        raise ValueError(f"Invalid position {position} for sequence of length {len(sequence)}")
    
    return sequence[:position] + sequence[position + 1:]


def mutate_sequence(sequence: str, 
                   mutation_rate: float,
                   mutation_types: Optional[Dict[str, float]] = None,
                   seed: Optional[int] = None) -> Tuple[str, Dict[str, any]]:
    """
    Introduce random mutations into a sequence.
    
    Args:
        sequence: DNA sequence to mutate
        mutation_rate: Probability of mutation per base (0.0 to 1.0)
        mutation_types: Dictionary of mutation type probabilities
                       {'substitution': 0.7, 'insertion': 0.15, 'deletion': 0.15}
                       If None, uses only substitutions
        seed: Random seed
        
    Returns:
        Tuple of (mutated_sequence, mutation_info_dict)
    """
    if seed is not None:
        random.seed(seed)
    
    if mutation_types is None:
        mutation_types = {'substitution': 1.0}
    
    # Validate mutation types
    total = sum(mutation_types.values())
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"Mutation type probabilities must sum to 1.0, got {total}")
    
    result = sequence
    mutations = {
        'substitutions': [],
        'insertions': [],
        'deletions': [],
        'total_mutations': 0
    }
    
    i = 0
    while i < len(result):
        if random.random() < mutation_rate:
            # Choose mutation type
            mutation_type = random.choices(
                list(mutation_types.keys()),
                weights=list(mutation_types.values())
            )[0]
            
            if mutation_type == 'substitution':
                result = introduce_substitution(result, i)
                mutations['substitutions'].append(i)
                i += 1
            elif mutation_type == 'insertion':
                result = introduce_insertion(result, i)
                mutations['insertions'].append(i)
                i += 2  # Skip the inserted base
            elif mutation_type == 'deletion':
                result = introduce_deletion(result, i)
                mutations['deletions'].append(i)
                # Don't increment i, as we've deleted a base
            
            mutations['total_mutations'] += 1
        else:
            i += 1
    
    return result, mutations


def generate_synthetic_dataset(text_length: int,
                               patterns: Optional[List[str]] = None,
                               num_patterns: int = 10,
                               pattern_length: int = 50,
                               num_injections: int = 20,
                               mutation_rate: float = 0.0,
                               seed: Optional[int] = None) -> SyntheticSequence:
    """
    Generate a complete synthetic dataset with patterns and optional mutations.
    
    Args:
        text_length: Length of the base sequence
        patterns: List of patterns to inject (if None, generates random patterns)
        num_patterns: Number of random patterns to generate (if patterns is None)
        pattern_length: Length of random patterns
        num_injections: Number of pattern injections
        mutation_rate: Rate of mutations to introduce (0.0 = no mutations)
        seed: Random seed for reproducibility
        
    Returns:
        SyntheticSequence object with all information
        
    Example:
        >>> dataset = generate_synthetic_dataset(
        ...     text_length=10000,
        ...     num_patterns=5,
        ...     pattern_length=20,
        ...     num_injections=30,
        ...     mutation_rate=0.05,
        ...     seed=42
        ... )
        >>> print(len(dataset.sequence))
        >>> print(dataset.pattern_positions)
    """
    if seed is not None:
        random.seed(seed)
    
    # Generate base sequence
    sequence = generate_random_sequence(text_length, seed=seed)
    
    # Generate or use provided patterns
    if patterns is None:
        patterns = [generate_pattern(pattern_length, seed=seed+i if seed else None) 
                   for i in range(num_patterns)]
    
    # Inject patterns
    sequence, pattern_positions = inject_patterns_random(
        sequence, patterns, num_injections, seed=seed
    )
    
    # Apply mutations if requested
    mutations_info = {}
    if mutation_rate > 0:
        sequence, mutations_info = mutate_sequence(
            sequence, mutation_rate, seed=seed
        )
    
    metadata = {
        'original_length': text_length,
        'final_length': len(sequence),
        'num_patterns': len(patterns),
        'num_injections': num_injections,
        'mutation_rate': mutation_rate
    }
    
    return SyntheticSequence(
        sequence=sequence,
        patterns=patterns,
        pattern_positions=pattern_positions,
        mutations=mutations_info,
        metadata=metadata
    )


def generate_test_cases() -> List[Tuple[str, str, List[int]]]:
    """
    Generate standard test cases for algorithm validation.
    
    Returns:
        List of (text, pattern, expected_positions) tuples
    """
    test_cases = [
        # Basic cases
        ("ATCGATCG", "TCG", [1, 5]),
        ("AAAAAAA", "AAA", [0, 1, 2, 3, 4]),
        ("ATCG", "ATCG", [0]),
        ("ATCG", "GGGG", []),
        
        # Edge cases
        ("A", "A", [0]),
        ("ATCG", "ATCGATCG", []),  # Pattern longer than text
        
        # Overlapping patterns
        ("ABABABABAB", "ABAB", [0, 2, 4, 6]),
        
        # DNA-specific
        ("ATGCATGCATGC", "ATGC", [0, 4, 8]),
        ("CGCGCGCG", "GCGC", [1, 3, 5]),
    ]
    
    return test_cases
