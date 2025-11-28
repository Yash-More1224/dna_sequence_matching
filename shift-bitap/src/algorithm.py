"""
Shift-Or / Bitap Algorithm Implementation
==========================================

This module implements the Shift-Or (also known as Bitap or Baeza-Yates-Gonnet) algorithm
for both exact and approximate string matching using bit-parallel operations.

The algorithm is particularly efficient for:
- Short to medium patterns (≤ 64 characters on 64-bit systems)
- Small alphabets (excellent for DNA: A, C, G, T)
- Approximate matching with small edit distances

Algorithm Overview:
-------------------
1. Preprocessing: Create bitmasks for each character in the alphabet
   - Each bit position represents a position in the pattern
   - Bit is 0 if character appears at that position, 1 otherwise

2. Exact Matching: Use bitwise operations to track potential matches
   - Maintain a state vector D where D[i] = 0 means pattern matches up to position i
   - Update: D = (D << 1) | pattern_mask[text[i]]
   - Match found when bit at pattern length position is 0

3. Approximate Matching: Maintain multiple state vectors for different edit distances
   - D[k] tracks matches with up to k errors
   - Update considers substitutions, insertions, and deletions
   - More computationally expensive but still efficient for small k

Performance Characteristics:
---------------------------
- Time Complexity: O(n) for text length n (preprocessing O(m) for pattern length m)
- Space Complexity: O(σ) for alphabet size σ (very small for DNA)
- Best for: patterns up to word size (typically 64), small alphabets
- Advantage over other algorithms: Simple, cache-efficient, handles approximate matching

Author: DNA Sequence Matching Project
Date: November 2025
"""

from typing import List, Tuple, Dict, Optional
import sys


class ShiftOrBitap:
    """
    Shift-Or / Bitap algorithm for exact and approximate pattern matching.
    
    This implementation uses bit-parallel operations for efficient searching,
    particularly optimized for DNA sequences (4-character alphabet).
    
    Attributes:
        pattern (str): The pattern to search for
        pattern_length (int): Length of the pattern
        max_pattern_length (int): Maximum supported pattern length (word size)
        pattern_masks (Dict[str, int]): Bitmasks for each character in alphabet
        case_sensitive (bool): Whether matching is case-sensitive
    """
    
    def __init__(self, pattern: str, case_sensitive: bool = False):
        """
        Initialize the Shift-Or/Bitap matcher with a pattern.
        
        Args:
            pattern: The pattern string to search for
            case_sensitive: If False, converts pattern to uppercase for DNA matching
            
        Raises:
            ValueError: If pattern is empty or exceeds maximum length
        """
        if not pattern:
            raise ValueError("Pattern cannot be empty")
        
        self.case_sensitive = case_sensitive
        self.pattern = pattern if case_sensitive else pattern.upper()
        self.pattern_length = len(self.pattern)
        
        # Maximum pattern length is word size (typically 64 bits)
        self.max_pattern_length = 64
        
        if self.pattern_length > self.max_pattern_length:
            raise ValueError(
                f"Pattern length ({self.pattern_length}) exceeds maximum "
                f"supported length ({self.max_pattern_length}). "
                f"Consider using a different algorithm for longer patterns."
            )
        
        # Preprocess pattern to create bitmasks
        self.pattern_masks = self._preprocess_pattern()
    
    def _preprocess_pattern(self) -> Dict[str, int]:
        """
        Create bitmasks for each character in the alphabet.
        
        For each character c, create a mask where bit i is 0 if pattern[i] == c,
        and 1 otherwise. This preprocessing enables efficient bit-parallel matching.
        
        Returns:
            Dictionary mapping each character to its bitmask
            
        Time Complexity: O(m) where m is pattern length
        Space Complexity: O(σ) where σ is alphabet size
        """
        # Initialize all masks to all 1s (no matches)
        # For DNA, we typically have: A, C, G, T, N (and sometimes ambiguous codes)
        masks = {}
        all_ones = (1 << self.pattern_length) - 1
        
        # Build character set from pattern
        for char in set(self.pattern):
            masks[char] = all_ones
        
        # Set bits to 0 where character appears in pattern
        for i, char in enumerate(self.pattern):
            # Bit i is 0 if pattern[i] == char
            masks[char] &= ~(1 << i)
        
        return masks
    
    def search_exact(self, text: str) -> List[int]:
        """
        Find all exact matches of the pattern in the text.
        
        Uses the Shift-Or algorithm with bit-parallel operations for efficient matching.
        
        Args:
            text: The text string to search in
            
        Returns:
            List of starting positions (0-indexed) where pattern matches exactly
            
        Time Complexity: O(n) where n is text length
        Space Complexity: O(1) - constant space for state vector
        
        Example:
            >>> matcher = ShiftOrBitap("ACGT")
            >>> positions = matcher.search_exact("AACGTCACGT")
            >>> print(positions)  # [1, 6]
        """
        if not text:
            return []
        
        text = text if self.case_sensitive else text.upper()
        matches = []
        
        # Initialize state vector: all 1s means no match yet
        state = (1 << self.pattern_length) - 1
        
        # Mask to check if we have a complete match
        match_mask = 1 << (self.pattern_length - 1)
        
        for i, char in enumerate(text):
            # Get the pattern mask for this character (default to all 1s if not in pattern)
            char_mask = self.pattern_masks.get(char, (1 << self.pattern_length) - 1)
            
            # Update state: shift left and OR with character mask
            state = (state << 1) | char_mask
            
            # Check if we have a complete match (bit at pattern_length-1 is 0)
            if (state & match_mask) == 0:
                # Match found at position (i - pattern_length + 1)
                match_pos = i - self.pattern_length + 1
                matches.append(match_pos)
        
        return matches
    
    def search_approximate(self, text: str, max_errors: int = 1) -> List[Tuple[int, int]]:
        """
        Find all approximate matches of the pattern in text with up to max_errors edits.
        
        Uses the Shift-Or algorithm extended for approximate matching. Tracks multiple
        state vectors for different error levels (0 errors, 1 error, ..., k errors).
        
        Edit operations considered:
        - Substitution: replace one character
        - Insertion: add one character
        - Deletion: remove one character
        
        Args:
            text: The text string to search in
            max_errors: Maximum number of errors (edit distance) allowed
            
        Returns:
            List of tuples (position, errors) where:
                - position: starting position of match
                - errors: number of errors in this match
                
        Time Complexity: O(k*n) where k is max_errors and n is text length
        Space Complexity: O(k) for k state vectors
        
        Example:
            >>> matcher = ShiftOrBitap("ACGT")
            >>> matches = matcher.search_approximate("AACGACCT", max_errors=1)
            >>> print(matches)  # [(1, 0), (4, 1)]  - exact match and 1-error match
        """
        if not text:
            return []
        
        if max_errors < 0:
            raise ValueError("max_errors must be non-negative")
        
        if max_errors > self.pattern_length:
            max_errors = self.pattern_length
        
        text = text if self.case_sensitive else text.upper()
        matches = []
        
        # State vectors for 0, 1, ..., max_errors errors
        # D[k] tracks matches with up to k errors
        D = [(1 << self.pattern_length) - 1 for _ in range(max_errors + 1)]
        
        # Mask to check for complete match
        match_mask = 1 << (self.pattern_length - 1)
        
        for i, char in enumerate(text):
            # Get the pattern mask for this character
            char_mask = self.pattern_masks.get(char, (1 << self.pattern_length) - 1)
            
            # Store previous states for computing next states
            old_D = D.copy()
            
            # Update D[0] - exact matching
            D[0] = (old_D[0] << 1) | char_mask
            
            # Update D[k] for k = 1 to max_errors
            for k in range(1, max_errors + 1):
                # D[k] is updated considering:
                # 1. Match/substitution: (old_D[k] << 1) | char_mask
                # 2. Insertion in text: old_D[k] << 1
                # 3. Deletion in text: D[k-1]
                # 4. Substitution: old_D[k-1] << 1
                
                D[k] = (
                    ((old_D[k] << 1) | char_mask) &      # Match or substitution
                    (old_D[k] << 1) &                     # Insertion in text
                    (D[k-1]) &                            # Deletion in text  
                    (old_D[k-1] << 1)                     # Substitution
                )
            
            # Check for matches at each error level
            for k in range(max_errors + 1):
                if (D[k] & match_mask) == 0:
                    match_pos = i - self.pattern_length + 1
                    # Only record the best match (fewest errors) at each position
                    if not matches or matches[-1][0] != match_pos:
                        matches.append((match_pos, k))
                    elif matches[-1][1] > k:
                        matches[-1] = (match_pos, k)
                    break  # Found match with k errors, no need to check higher k
        
        return matches
    
    def count_matches(self, text: str, approximate: bool = False, max_errors: int = 0) -> int:
        """
        Count the number of matches in the text.
        
        Args:
            text: The text string to search in
            approximate: If True, use approximate matching
            max_errors: Maximum errors for approximate matching
            
        Returns:
            Number of matches found
        """
        if approximate:
            return len(self.search_approximate(text, max_errors))
        else:
            return len(self.search_exact(text))
    
    def get_pattern_info(self) -> Dict:
        """
        Get information about the preprocessed pattern.
        
        Returns:
            Dictionary with pattern statistics and bitmask information
        """
        return {
            'pattern': self.pattern,
            'length': self.pattern_length,
            'max_supported_length': self.max_pattern_length,
            'alphabet_size': len(self.pattern_masks),
            'alphabet': sorted(self.pattern_masks.keys()),
            'case_sensitive': self.case_sensitive
        }


def create_matcher(pattern: str, case_sensitive: bool = False) -> ShiftOrBitap:
    """
    Factory function to create a Shift-Or/Bitap matcher.
    
    Args:
        pattern: Pattern to search for
        case_sensitive: Whether matching should be case-sensitive
        
    Returns:
        Configured ShiftOrBitap instance
    """
    return ShiftOrBitap(pattern, case_sensitive)


if __name__ == "__main__":
    # Simple demonstration
    print("Shift-Or / Bitap Algorithm Demo")
    print("=" * 50)
    
    # Example 1: Exact matching
    pattern = "ACGT"
    text = "AACGTCACGTGACGT"
    
    matcher = ShiftOrBitap(pattern)
    print(f"\nPattern: {pattern}")
    print(f"Text: {text}")
    
    exact_matches = matcher.search_exact(text)
    print(f"\nExact matches found at positions: {exact_matches}")
    
    # Example 2: Approximate matching
    print("\n" + "=" * 50)
    pattern2 = "GATTACA"
    text2 = "AGATTACAGATTXCAGATACA"
    
    matcher2 = ShiftOrBitap(pattern2)
    print(f"\nPattern: {pattern2}")
    print(f"Text: {text2}")
    
    approx_matches = matcher2.search_approximate(text2, max_errors=1)
    print(f"\nApproximate matches (max 1 error):")
    for pos, errors in approx_matches:
        match_text = text2[pos:pos+len(pattern2)]
        print(f"  Position {pos}: '{match_text}' ({errors} error(s))")
    
    # Example 3: Pattern info
    print("\n" + "=" * 50)
    info = matcher2.get_pattern_info()
    print("\nPattern Information:")
    for key, value in info.items():
        print(f"  {key}: {value}")
