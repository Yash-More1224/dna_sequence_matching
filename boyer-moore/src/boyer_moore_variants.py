"""
Boyer-Moore Algorithm Variants

Implements different variants of Boyer-Moore for comparison:
- BCR-only (Bad Character Rule only)
- GSR-only (Good Suffix Rule only)
- Horspool (Simplified version with BCR only, different shift)
"""

from typing import List, Dict, Optional
try:
    from .boyer_moore import BoyerMoore
except ImportError:
    from boyer_moore import BoyerMoore


class BoyerMooreBCROnly(BoyerMoore):
    """Boyer-Moore with only Bad Character Rule."""
    
    def __init__(self, pattern: str):
        """Initialize with BCR only."""
        super().__init__(pattern, use_bcr=True, use_gsr=False)


class BoyerMooreGSROnly(BoyerMoore):
    """Boyer-Moore with only Good Suffix Rule."""
    
    def __init__(self, pattern: str):
        """Initialize with GSR only."""
        super().__init__(pattern, use_bcr=False, use_gsr=True)


class BoyerMooreHorspool:
    """
    Boyer-Moore-Horspool algorithm.
    
    Simplified version using only bad character rule with
    character at text position (not pattern position).
    """
    
    def __init__(self, pattern: str):
        """
        Initialize Horspool matcher.
        
        Args:
            pattern: Pattern string to search for
        """
        if not pattern:
            raise ValueError("Pattern cannot be empty")
        
        self.pattern = pattern.upper()
        self.m = len(pattern)
        
        # Statistics
        self.comparisons = 0
        self.shifts = 0
        
        # Build bad character table (Horspool version)
        self.bad_char = self._build_horspool_table()
    
    def _build_horspool_table(self) -> Dict[str, int]:
        """
        Build Horspool bad character table.
        
        For each character, store shift distance based on
        last occurrence in pattern (excluding last position).
        
        Returns:
            Dictionary mapping characters to shift distances
        """
        # Default shift is pattern length
        bad_char = {char: self.m for char in 'ACGTN'}
        
        # For each character in pattern (except last), store distance to end
        for i in range(self.m - 1):
            bad_char[self.pattern[i]] = self.m - 1 - i
        
        return bad_char
    
    def search(self, text: str) -> List[int]:
        """
        Search for all occurrences using Horspool algorithm.
        
        Args:
            text: Text string to search in
            
        Returns:
            List of starting positions where pattern is found
        """
        text = text.upper()
        n = len(text)
        matches = []
        
        self.comparisons = 0
        self.shifts = 0
        
        i = 0
        
        while i <= n - self.m:
            j = self.m - 1
            
            # Compare from right to left
            while j >= 0 and self.pattern[j] == text[i + j]:
                self.comparisons += 1
                j -= 1
            
            if j < 0:
                # Match found
                matches.append(i)
                self.comparisons += self.m
                i += 1
            else:
                # Mismatch - shift based on character at text[i + m - 1]
                self.comparisons += 1
                
                # Get shift for character aligned with last pattern position
                if i + self.m - 1 < n:
                    shift_char = text[i + self.m - 1]
                    shift = self.bad_char.get(shift_char, self.m)
                else:
                    shift = 1
                
                i += shift
            
            self.shifts += 1
        
        return matches
    
    def search_first(self, text: str) -> Optional[int]:
        """
        Search for first occurrence.
        
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
            
            if i + self.m - 1 < n:
                shift_char = text[i + self.m - 1]
                shift = self.bad_char.get(shift_char, self.m)
            else:
                shift = 1
            
            i += shift
            self.shifts += 1
        
        return None
    
    def get_statistics(self) -> Dict[str, int]:
        """Get search statistics."""
        return {
            'comparisons': self.comparisons,
            'shifts': self.shifts,
            'pattern_length': self.m
        }


def get_variant(pattern: str, variant: str = 'full'):
    """
    Factory function to get algorithm variant.
    
    Args:
        pattern: Pattern to search for
        variant: One of 'full', 'bcr_only', 'gsr_only', 'horspool'
        
    Returns:
        Matcher instance
        
    Raises:
        ValueError: If variant is unknown
    """
    variants = {
        'full': lambda p: BoyerMoore(p, use_bcr=True, use_gsr=True),
        'bcr_only': BoyerMooreBCROnly,
        'gsr_only': BoyerMooreGSROnly,
        'horspool': BoyerMooreHorspool
    }
    
    if variant not in variants:
        raise ValueError(f"Unknown variant: {variant}. "
                        f"Choose from {list(variants.keys())}")
    
    return variants[variant](pattern)


if __name__ == "__main__":
    # Test all variants
    text = "GCATCGCAGAGAGTATACAGTACG"
    pattern = "GCAGAGAG"
    
    print(f"Text: {text}")
    print(f"Pattern: {pattern}\n")
    
    for variant in ['full', 'bcr_only', 'gsr_only', 'horspool']:
        matcher = get_variant(pattern, variant)
        matches = matcher.search(text)
        stats = matcher.get_statistics()
        
        print(f"{variant.upper()}:")
        print(f"  Matches: {matches}")
        print(f"  Comparisons: {stats['comparisons']}")
        print(f"  Shifts: {stats['shifts']}\n")
