"""
Boyer-Moore String Matching Algorithm

This module implements the Boyer-Moore algorithm with both:
- Bad Character Rule (BCR)
- Good Suffix Rule (GSR)

Optimized for DNA sequence matching (alphabet: A, C, G, T, N).
"""

from typing import List, Dict, Tuple, Optional


class BoyerMoore:
    """
    Boyer-Moore exact string matching algorithm.
    
    Implements both bad character and good suffix heuristics for
    efficient pattern matching in DNA sequences.
    """
    
    def __init__(self, pattern: str, use_bcr: bool = True, use_gsr: bool = True):
        """
        Initialize Boyer-Moore matcher with preprocessing.
        
        Args:
            pattern: Pattern string to search for
            use_bcr: Enable Bad Character Rule heuristic
            use_gsr: Enable Good Suffix Rule heuristic
        """
        if not pattern:
            raise ValueError("Pattern cannot be empty")
        
        self.pattern = pattern.upper()  # Normalize to uppercase
        self.m = len(pattern)
        self.use_bcr = use_bcr
        self.use_gsr = use_gsr
        
        # Statistics for analysis
        self.comparisons = 0
        self.shifts = 0
        
        # Preprocess pattern
        self.bad_char = self._build_bad_character_table() if use_bcr else {}
        self.good_suffix = self._build_good_suffix_table() if use_gsr else []
    
    def _build_bad_character_table(self) -> Dict[str, List[int]]:
        """
        Build bad character table (last occurrence of each character).
        
        For each character, stores the rightmost position where it occurs
        in the pattern (or -1 if not present).
        
        Returns:
            Dictionary mapping characters to their rightmost positions
        """
        bad_char = {}
        
        # For DNA sequences, we primarily care about A, C, G, T, N
        for char in 'ACGTN':
            bad_char[char] = -1
        
        # Record rightmost occurrence of each character
        for i, char in enumerate(self.pattern):
            bad_char[char] = i
        
        return bad_char
    
    def _build_good_suffix_table(self) -> List[int]:
        """
        Build good suffix shift table.
        
        For each position i, computes the shift distance based on:
        1. Finding the rightmost occurrence of the suffix pattern[i+1:]
        2. If not found, finding the longest prefix that matches a suffix
        
        Returns:
            List of shift values for each position
        """
        m = self.m
        good_suffix = [0] * m
        
        # Compute border array (similar to KMP failure function)
        border = self._compute_border_array()
        
        # Case 1: Shift based on borders
        j = border[0]
        for i in range(m):
            if good_suffix[i] == 0:
                good_suffix[i] = j
            if i == j:
                j = border[j]
        
        # Case 2: Shift for matching suffixes
        for i in range(m - 1):
            j = border[i + 1]
            good_suffix[m - 1 - j] = m - 1 - i
        
        return good_suffix
    
    def _compute_border_array(self) -> List[int]:
        """
        Compute border array for good suffix rule.
        
        Border array stores the length of the longest proper prefix
        that is also a suffix for each position.
        
        Returns:
            Border array
        """
        m = self.m
        border = [0] * (m + 1)
        border[0] = m
        
        i = m
        j = m + 1
        border[i] = j
        
        while i > 0:
            while j <= m and self.pattern[i - 1] != self.pattern[j - 1]:
                if border[j] == 0:
                    border[j] = j - i
                j = border[j]
            i -= 1
            j -= 1
            border[i] = j
        
        return border
    
    def _bad_character_shift(self, text_char: str, pattern_pos: int) -> int:
        """
        Calculate shift based on bad character rule.
        
        Args:
            text_char: Mismatched character in text
            pattern_pos: Current position in pattern
            
        Returns:
            Shift distance
        """
        if not self.use_bcr:
            return 1
        
        # Get last occurrence of text_char in pattern
        last_occurrence = self.bad_char.get(text_char, -1)
        
        # Shift so that the last occurrence aligns with text position
        shift = pattern_pos - last_occurrence
        
        return max(1, shift)
    
    def _good_suffix_shift(self, pattern_pos: int) -> int:
        """
        Calculate shift based on good suffix rule.
        
        Args:
            pattern_pos: Position of mismatch in pattern
            
        Returns:
            Shift distance
        """
        if not self.use_gsr or pattern_pos == self.m - 1:
            return 1
        
        return self.good_suffix[pattern_pos + 1]
    
    def search(self, text: str) -> List[int]:
        """
        Search for all occurrences of pattern in text.
        
        Args:
            text: Text string to search in
            
        Returns:
            List of starting positions where pattern is found
        """
        text = text.upper()  # Normalize to uppercase
        n = len(text)
        matches = []
        
        # Reset statistics
        self.comparisons = 0
        self.shifts = 0
        
        i = 0  # Position in text
        
        while i <= n - self.m:
            j = self.m - 1  # Start from end of pattern
            
            # Compare pattern from right to left
            while j >= 0 and self.pattern[j] == text[i + j]:
                self.comparisons += 1
                j -= 1
            
            if j < 0:
                # Match found
                matches.append(i)
                self.comparisons += self.m
                
                # Shift to next possible match
                if self.use_gsr and i + self.m < n:
                    shift = self.good_suffix[0]
                else:
                    shift = 1
                i += shift
                self.shifts += 1
            else:
                # Mismatch occurred
                self.comparisons += 1
                
                # Calculate shifts based on both heuristics
                bc_shift = self._bad_character_shift(text[i + j], j)
                gs_shift = self._good_suffix_shift(j)
                
                # Take maximum shift
                shift = max(bc_shift, gs_shift)
                i += shift
                self.shifts += 1
        
        return matches
    
    def search_first(self, text: str) -> Optional[int]:
        """
        Search for first occurrence of pattern in text.
        
        Args:
            text: Text string to search in
            
        Returns:
            Position of first match, or None if not found
        """
        text = text.upper()
        n = len(text)
        
        self.comparisons = 0
        self.shifts = 0
        
        i = 0
        
        while i <= n - self.m:
            j = self.m - 1
            
            while j >= 0 and self.pattern[j] == text[i + j]:
                self.comparisons += 1
                j -= 1
            
            if j < 0:
                self.comparisons += self.m
                return i
            
            self.comparisons += 1
            bc_shift = self._bad_character_shift(text[i + j], j)
            gs_shift = self._good_suffix_shift(j)
            shift = max(bc_shift, gs_shift)
            i += shift
            self.shifts += 1
        
        return None
    
    def get_statistics(self) -> Dict[str, int]:
        """
        Get search statistics for analysis.
        
        Returns:
            Dictionary with comparisons and shifts count
        """
        return {
            'comparisons': self.comparisons,
            'shifts': self.shifts,
            'pattern_length': self.m
        }


def boyer_moore_search(text: str, pattern: str, 
                       use_bcr: bool = True, 
                       use_gsr: bool = True) -> List[int]:
    """
    Convenience function for Boyer-Moore search.
    
    Args:
        text: Text to search in
        pattern: Pattern to search for
        use_bcr: Enable Bad Character Rule
        use_gsr: Enable Good Suffix Rule
        
    Returns:
        List of match positions
    """
    bm = BoyerMoore(pattern, use_bcr=use_bcr, use_gsr=use_gsr)
    return bm.search(text)


if __name__ == "__main__":
    # Example usage
    text = "GCATCGCAGAGAGTATACAGTACG"
    pattern = "GCAGAGAG"
    
    print(f"Text: {text}")
    print(f"Pattern: {pattern}")
    print()
    
    # Full Boyer-Moore
    bm = BoyerMoore(pattern)
    matches = bm.search(text)
    print(f"Matches found at positions: {matches}")
    print(f"Statistics: {bm.get_statistics()}")
