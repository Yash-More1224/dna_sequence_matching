"""
Ground Truth Generation for Wagner-Fischer Evaluation
Generates synthetic patterns with controlled mutation rates and ground truth locations.
"""

import random
import json
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass, asdict
from pathlib import Path
import sys


@dataclass
class MutatedPattern:
    """Represents a mutated pattern with ground truth."""
    original: str
    mutated: str
    mutation_rate: float
    operations: List[Dict[str, any]]
    position_in_target: int
    end_position_in_target: int
    pattern_id: int


@dataclass
class GroundTruth:
    """Complete ground truth dataset."""
    target_sequence: str
    target_id: str
    patterns: List[MutatedPattern]
    pattern_length: int
    num_patterns: int
    mutation_rates: List[float]


class GroundTruthGenerator:
    """
    Generates ground truth datasets for evaluating approximate pattern matching.
    """
    
    def __init__(self, seed: int = 42):
        """
        Initialize generator with random seed.
        
        Args:
            seed: Random seed for reproducibility
        """
        random.seed(seed)
        self.dna_alphabet = ['A', 'C', 'G', 'T']
    
    def extract_non_overlapping_patterns(self, 
                                        sequence: str, 
                                        num_patterns: int,
                                        pattern_length: int) -> List[Tuple[str, int]]:
        """
        Extract non-overlapping substrings from sequence.
        
        Args:
            sequence: Source sequence
            num_patterns: Number of patterns to extract
            pattern_length: Length of each pattern
            
        Returns:
            List of (pattern, position) tuples
        """
        if len(sequence) < num_patterns * pattern_length * 2:
            raise ValueError(f"Sequence too short for {num_patterns} non-overlapping patterns")
        
        patterns = []
        attempts = 0
        max_attempts = num_patterns * 100
        used_positions = set()
        
        while len(patterns) < num_patterns and attempts < max_attempts:
            pos = random.randint(0, len(sequence) - pattern_length)
            
            # Check if this position overlaps with already selected patterns
            overlap = False
            for used_pos in used_positions:
                if abs(pos - used_pos) < pattern_length * 2:
                    overlap = True
                    break
            
            if not overlap:
                pattern = sequence[pos:pos + pattern_length]
                # Ensure pattern contains only valid DNA bases
                if all(c in self.dna_alphabet for c in pattern):
                    patterns.append((pattern, pos))
                    used_positions.add(pos)
            
            attempts += 1
        
        if len(patterns) < num_patterns:
            print(f"Warning: Could only extract {len(patterns)} non-overlapping patterns")
        
        return patterns
    
    def mutate_pattern(self, 
                      pattern: str, 
                      mutation_rate: float) -> Tuple[str, List[Dict]]:
        """
        Apply random mutations to a pattern.
        
        Args:
            pattern: Original pattern
            mutation_rate: Fraction of bases to mutate (0.0 to 1.0)
            
        Returns:
            Tuple of (mutated_pattern, operations_list)
        """
        if mutation_rate == 0.0:
            return pattern, []
        
        num_mutations = max(1, int(len(pattern) * mutation_rate))
        mutated = list(pattern)
        operations = []
        
        for _ in range(num_mutations):
            op_type = random.choice(['substitute', 'insert', 'delete'])
            
            if op_type == 'substitute' and len(mutated) > 0:
                pos = random.randint(0, len(mutated) - 1)
                old_base = mutated[pos]
                new_base = random.choice([b for b in self.dna_alphabet if b != old_base])
                mutated[pos] = new_base
                operations.append({
                    'type': 'substitute',
                    'position': pos,
                    'old': old_base,
                    'new': new_base
                })
            
            elif op_type == 'insert':
                pos = random.randint(0, len(mutated))
                new_base = random.choice(self.dna_alphabet)
                mutated.insert(pos, new_base)
                operations.append({
                    'type': 'insert',
                    'position': pos,
                    'base': new_base
                })
            
            elif op_type == 'delete' and len(mutated) > 1:
                pos = random.randint(0, len(mutated) - 1)
                deleted_base = mutated[pos]
                del mutated[pos]
                operations.append({
                    'type': 'delete',
                    'position': pos,
                    'base': deleted_base
                })
        
        return ''.join(mutated), operations
    
    def generate_ground_truth(self,
                             target_sequence: str,
                             target_id: str,
                             num_patterns: int,
                             pattern_length: int,
                             mutation_rates: List[float]) -> GroundTruth:
        """
        Generate complete ground truth dataset.
        
        Args:
            target_sequence: Target sequence to search in
            target_id: Identifier for target sequence
            num_patterns: Number of patterns to generate
            pattern_length: Length of each pattern
            mutation_rates: List of mutation rates to test
            
        Returns:
            GroundTruth object
        """
        # Extract original patterns
        original_patterns = self.extract_non_overlapping_patterns(
            target_sequence, num_patterns, pattern_length
        )
        
        all_mutated_patterns = []
        pattern_id = 0
        
        for mutation_rate in mutation_rates:
            for original, position in original_patterns:
                # Apply mutations
                mutated, operations = self.mutate_pattern(original, mutation_rate)
                
                # Create mutated pattern object
                mutated_pattern = MutatedPattern(
                    original=original,
                    mutated=mutated,
                    mutation_rate=mutation_rate,
                    operations=operations,
                    position_in_target=position,
                    end_position_in_target=position + pattern_length,
                    pattern_id=pattern_id
                )
                
                all_mutated_patterns.append(mutated_pattern)
                pattern_id += 1
        
        return GroundTruth(
            target_sequence=target_sequence,
            target_id=target_id,
            patterns=all_mutated_patterns,
            pattern_length=pattern_length,
            num_patterns=num_patterns,
            mutation_rates=mutation_rates
        )
    
    def save_ground_truth(self, ground_truth: GroundTruth, filepath: str):
        """
        Save ground truth to JSON file.
        
        Args:
            ground_truth: GroundTruth object
            filepath: Output file path
        """
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to dictionary
        data = {
            'target_id': ground_truth.target_id,
            'target_length': len(ground_truth.target_sequence),
            'pattern_length': ground_truth.pattern_length,
            'num_patterns': ground_truth.num_patterns,
            'mutation_rates': ground_truth.mutation_rates,
            'patterns': [asdict(p) for p in ground_truth.patterns]
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Ground truth saved to {filepath}")
        print(f"Total patterns: {len(ground_truth.patterns)}")
        print(f"Mutation rates: {ground_truth.mutation_rates}")
    
    def load_ground_truth(self, filepath: str) -> Dict:
        """
        Load ground truth from JSON file.
        
        Args:
            filepath: Path to ground truth JSON file
            
        Returns:
            Dictionary containing ground truth data
        """
        with open(filepath, 'r') as f:
            data = json.load(f)
        return data


def load_fasta(filepath: str) -> Tuple[str, str]:
    """
    Simple FASTA loader.
    
    Args:
        filepath: Path to FASTA file
        
    Returns:
        Tuple of (sequence_id, sequence)
    """
    seq_id = ""
    sequence = []
    
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                seq_id = line[1:].split()[0]
            else:
                sequence.append(line.upper())
    
    return seq_id, ''.join(sequence)


def main():
    """Generate ground truth datasets."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate ground truth for WF evaluation')
    parser.add_argument('--fasta', required=True, help='Path to FASTA file')
    parser.add_argument('--output', required=True, help='Output JSON file')
    parser.add_argument('--num-patterns', type=int, default=100, 
                       help='Number of patterns to extract')
    parser.add_argument('--pattern-length', type=int, default=50,
                       help='Length of each pattern')
    parser.add_argument('--mutation-rates', type=float, nargs='+',
                       default=[0.0, 0.005, 0.01, 0.02, 0.05, 0.1],
                       help='Mutation rates to test')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    # Load target sequence
    print(f"Loading FASTA file: {args.fasta}")
    target_id, target_sequence = load_fasta(args.fasta)
    print(f"Loaded sequence: {target_id} ({len(target_sequence)} bp)")
    
    # Generate ground truth
    generator = GroundTruthGenerator(seed=args.seed)
    ground_truth = generator.generate_ground_truth(
        target_sequence=target_sequence,
        target_id=target_id,
        num_patterns=args.num_patterns,
        pattern_length=args.pattern_length,
        mutation_rates=args.mutation_rates
    )
    
    # Save ground truth
    generator.save_ground_truth(ground_truth, args.output)
    
    # Print summary
    print("\nGround Truth Summary:")
    print(f"Target: {target_id} ({len(target_sequence)} bp)")
    print(f"Pattern length: {args.pattern_length}")
    print(f"Patterns per mutation rate: {args.num_patterns}")
    print(f"Mutation rates: {args.mutation_rates}")
    print(f"Total patterns: {len(ground_truth.patterns)}")


if __name__ == '__main__':
    main()
