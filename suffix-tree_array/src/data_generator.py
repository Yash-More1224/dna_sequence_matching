"""
Synthetic DNA Sequence Generator

Generate synthetic DNA sequences and patterns for testing:
- Random sequences
- Sequences with controlled mutations
- Test patterns with known positions
"""

import random
from typing import List, Tuple, Dict


class DNAGenerator:
    """Generate synthetic DNA sequences for testing."""
    
    def __init__(self, seed: int = 42):
        """
        Initialize DNA generator.
        
        Args:
            seed: Random seed for reproducibility
        """
        self.seed = seed
        random.seed(seed)
        self.nucleotides = ['A', 'C', 'G', 'T']
    
    def generate_random_sequence(self, length: int, 
                                 gc_content: float = 0.5) -> str:
        """
        Generate random DNA sequence with specified GC content.
        
        Args:
            length: Length of sequence
            gc_content: Desired GC content (0.0 to 1.0)
            
        Returns:
            Random DNA sequence
        """
        if not 0 <= gc_content <= 1:
            raise ValueError("GC content must be between 0 and 1")
        
        # Calculate probabilities
        gc_prob = gc_content / 2  # Split between G and C
        at_prob = (1 - gc_content) / 2  # Split between A and T
        
        nucleotide_probs = {
            'A': at_prob,
            'T': at_prob,
            'C': gc_prob,
            'G': gc_prob
        }
        
        sequence = random.choices(
            population=list(nucleotide_probs.keys()),
            weights=list(nucleotide_probs.values()),
            k=length
        )
        
        return ''.join(sequence)
    
    def generate_pattern(self, length: int) -> str:
        """
        Generate random pattern.
        
        Args:
            length: Pattern length
            
        Returns:
            Random DNA pattern
        """
        return ''.join(random.choices(self.nucleotides, k=length))
    
    def insert_pattern(self, text: str, pattern: str, 
                      positions: List[int]) -> Tuple[str, List[int]]:
        """
        Insert pattern at specified positions in text.
        
        Args:
            text: Base text sequence
            pattern: Pattern to insert
            positions: Positions where pattern should appear
            
        Returns:
            Tuple of (modified text, actual positions)
        """
        text_list = list(text)
        actual_positions = []
        
        for pos in sorted(positions):
            if pos + len(pattern) <= len(text):
                # Replace text at position with pattern
                for i, char in enumerate(pattern):
                    text_list[pos + i] = char
                actual_positions.append(pos)
        
        return ''.join(text_list), actual_positions
    
    def mutate_sequence(self, sequence: str, 
                       mutation_rate: float = 0.01) -> str:
        """
        Introduce random mutations (substitutions) into sequence.
        
        Args:
            sequence: Original sequence
            mutation_rate: Probability of mutation per nucleotide
            
        Returns:
            Mutated sequence
        """
        mutated = list(sequence)
        
        for i in range(len(mutated)):
            if random.random() < mutation_rate:
                # Substitute with different nucleotide
                current = mutated[i]
                others = [n for n in self.nucleotides if n != current]
                mutated[i] = random.choice(others)
        
        return ''.join(mutated)
    
    def introduce_indels(self, sequence: str, 
                        indel_rate: float = 0.01) -> str:
        """
        Introduce insertions and deletions.
        
        Args:
            sequence: Original sequence
            indel_rate: Probability of indel per position
            
        Returns:
            Sequence with indels
        """
        result = []
        
        for char in sequence:
            if random.random() < indel_rate:
                # Decide: skip (deletion) or insert
                if random.random() < 0.5:
                    # Deletion - skip this character
                    continue
                else:
                    # Insertion - add random nucleotide
                    result.append(random.choice(self.nucleotides))
            
            result.append(char)
        
        return ''.join(result)
    
    def generate_test_case(self, text_length: int, 
                          pattern_length: int,
                          num_occurrences: int = 5,
                          gc_content: float = 0.5) -> Tuple[str, str, List[int]]:
        """
        Generate test case with known pattern occurrences.
        
        Args:
            text_length: Length of text
            pattern_length: Length of pattern
            num_occurrences: Number of pattern occurrences
            gc_content: GC content of sequences
            
        Returns:
            Tuple of (text, pattern, occurrence_positions)
        """
        # Generate random text and pattern
        text = self.generate_random_sequence(text_length, gc_content)
        pattern = self.generate_pattern(pattern_length)
        
        # Choose random positions for pattern
        max_pos = text_length - pattern_length
        if max_pos <= 0:
            raise ValueError("Text too short for pattern")
        
        # Ensure positions don't overlap
        positions = []
        attempts = 0
        while len(positions) < num_occurrences and attempts < 1000:
            pos = random.randint(0, max_pos)
            
            # Check if position overlaps with existing ones
            overlap = False
            for existing_pos in positions:
                if abs(pos - existing_pos) < pattern_length:
                    overlap = True
                    break
            
            if not overlap:
                positions.append(pos)
            
            attempts += 1
        
        # Insert pattern at chosen positions
        text, actual_positions = self.insert_pattern(text, pattern, positions)
        
        return text, pattern, sorted(actual_positions)
    
    def generate_motif_variants(self, motif: str, 
                               num_variants: int = 10,
                               mutation_rate: float = 0.1) -> List[str]:
        """
        Generate variants of a motif with mutations.
        
        Args:
            motif: Original motif sequence
            num_variants: Number of variants to generate
            mutation_rate: Mutation rate for variants
            
        Returns:
            List of motif variants
        """
        variants = [motif]  # Include original
        
        for _ in range(num_variants - 1):
            variant = self.mutate_sequence(motif, mutation_rate)
            variants.append(variant)
        
        return variants
    
    def calculate_gc_content(self, sequence: str) -> float:
        """
        Calculate GC content of a sequence.
        
        Args:
            sequence: DNA sequence
            
        Returns:
            GC content as fraction (0.0 to 1.0)
        """
        if not sequence:
            return 0.0
        
        gc_count = sequence.count('G') + sequence.count('C')
        return gc_count / len(sequence)
    
    def get_nucleotide_counts(self, sequence: str) -> Dict[str, int]:
        """
        Count nucleotides in sequence.
        
        Args:
            sequence: DNA sequence
            
        Returns:
            Dictionary with counts for each nucleotide
        """
        counts = {
            'A': sequence.count('A'),
            'C': sequence.count('C'),
            'G': sequence.count('G'),
            'T': sequence.count('T'),
            'N': sequence.count('N'),
            'other': 0
        }
        
        total = sum(counts.values())
        counts['other'] = len(sequence) - total
        
        return counts


def generate_random_sequence(length: int, seed: int = None, 
                            gc_content: float = 0.5) -> str:
    """
    Convenience function to generate random DNA sequence.
    
    Args:
        length: Length of sequence
        seed: Random seed
        gc_content: GC content
        
    Returns:
        Random DNA sequence
    """
    if seed is not None:
        random.seed(seed)
    
    generator = DNAGenerator(seed=seed if seed else 42)
    return generator.generate_random_sequence(length, gc_content)


def generate_pattern(length: int, seed: int = None) -> str:
    """
    Convenience function to generate random pattern.
    
    Args:
        length: Pattern length
        seed: Random seed
        
    Returns:
        Random DNA pattern
    """
    if seed is not None:
        random.seed(seed)
    
    generator = DNAGenerator(seed=seed if seed else 42)
    return generator.generate_pattern(length)
