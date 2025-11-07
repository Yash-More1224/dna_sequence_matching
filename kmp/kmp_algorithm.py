"""
Knuth-Morris-Pratt (KMP) String Matching Algorithm

This module implements the KMP algorithm for efficient exact pattern matching
in DNA sequences. The algorithm runs in O(n + m) time where n is the text length
and m is the pattern length.

The KMP algorithm uses a preprocessed Longest Proper Prefix-Suffix (LPS) array
to avoid redundant comparisons during the search phase.

References:
    Knuth, D.E., Morris, J.H., and Pratt, V.R. (1977).
    "Fast pattern matching in strings". SIAM Journal on Computing, 6(2):323-350.
"""

from typing import List, Dict, Tuple, Optional
import time


def build_lps_array(pattern: str) -> List[int]:
    """
    Build the Longest Proper Prefix-Suffix (LPS) array for KMP algorithm.
    
    The LPS array stores the length of the longest proper prefix of pattern[0..i]
    which is also a suffix of pattern[0..i].
    
    Time Complexity: O(m) where m is the length of the pattern
    Space Complexity: O(m)
    
    Args:
        pattern: The pattern string to preprocess
        
    Returns:
        LPS array where lps[i] is the length of longest proper prefix-suffix
        for substring pattern[0..i]
        
    Example:
        >>> build_lps_array("ABABC")
        [0, 0, 1, 2, 0]
        >>> build_lps_array("AAAA")
        [0, 1, 2, 3]
    """
    m = len(pattern)
    lps = [0] * m
    
    # length of previous longest proper prefix-suffix
    length = 0
    i = 1
    
    # Build the LPS array
    while i < m:
        if pattern[i] == pattern[length]:
            length += 1
            lps[i] = length
            i += 1
        else:
            if length != 0:
                # Try the previous longest proper prefix-suffix
                length = lps[length - 1]
            else:
                lps[i] = 0
                i += 1
    
    return lps


def kmp_search(text: str, pattern: str, lps: Optional[List[int]] = None) -> List[int]:
    """
    Search for all occurrences of pattern in text using KMP algorithm.
    
    Time Complexity: O(n + m) where n is text length, m is pattern length
    Space Complexity: O(m) for LPS array
    
    Args:
        text: The text to search in
        pattern: The pattern to search for
        lps: Pre-computed LPS array (if None, will be computed)
        
    Returns:
        List of starting positions where pattern occurs in text (0-indexed)
        Returns empty list if pattern not found
        
    Example:
        >>> kmp_search("ABABDABACDABABCABAB", "ABABC")
        [10]
        >>> kmp_search("AAAAAAA", "AAA")
        [0, 1, 2, 3, 4]
    """
    n = len(text)
    m = len(pattern)
    
    # Handle edge cases
    if m == 0:
        return []
    if m > n:
        return []
    
    # Build LPS array if not provided
    if lps is None:
        lps = build_lps_array(pattern)
    
    matches = []
    i = 0  # index for text
    j = 0  # index for pattern
    
    while i < n:
        if pattern[j] == text[i]:
            i += 1
            j += 1
        
        if j == m:
            # Pattern found at index i - j
            matches.append(i - j)
            j = lps[j - 1]
        elif i < n and pattern[j] != text[i]:
            if j != 0:
                j = lps[j - 1]
            else:
                i += 1
    
    return matches


class KMP:
    """
    KMP algorithm class for efficient pattern matching.
    
    This class provides a convenient interface for performing KMP searches,
    with support for preprocessing and multiple searches with the same pattern.
    
    Attributes:
        pattern: The pattern to search for
        lps: The computed LPS array for the pattern
        preprocessing_time: Time taken to build LPS array (seconds)
        
    Example:
        >>> kmp = KMP("ABABC")
        >>> matches = kmp.search("ABABDABACDABABCABAB")
        >>> print(matches)
        [10]
    """
    
    def __init__(self, pattern: str):
        """
        Initialize KMP with a pattern.
        
        Args:
            pattern: The pattern to search for
            
        Raises:
            ValueError: If pattern is empty
        """
        if not pattern:
            raise ValueError("Pattern cannot be empty")
        
        self.pattern = pattern
        self.lps = None
        self.preprocessing_time = 0.0
        
        # Preprocess pattern
        self._preprocess()
    
    def _preprocess(self) -> None:
        """Build the LPS array and measure preprocessing time."""
        start = time.perf_counter()
        self.lps = build_lps_array(self.pattern)
        self.preprocessing_time = time.perf_counter() - start
    
    def search(self, text: str) -> List[int]:
        """
        Search for pattern in text.
        
        Args:
            text: The text to search in
            
        Returns:
            List of starting positions where pattern occurs in text
        """
        return kmp_search(text, self.pattern, self.lps)
    
    def search_with_stats(self, text: str) -> Dict[str, any]:
        """
        Search for pattern in text and return detailed statistics.
        
        Args:
            text: The text to search in
            
        Returns:
            Dictionary containing:
                - matches: List of match positions
                - num_matches: Number of matches found
                - search_time: Time taken for search (seconds)
                - preprocessing_time: Time taken for preprocessing
                - total_time: Total time (preprocessing + search)
                - text_length: Length of text
                - pattern_length: Length of pattern
        """
        start = time.perf_counter()
        matches = self.search(text)
        search_time = time.perf_counter() - start
        
        return {
            'matches': matches,
            'num_matches': len(matches),
            'search_time': search_time,
            'preprocessing_time': self.preprocessing_time,
            'total_time': self.preprocessing_time + search_time,
            'text_length': len(text),
            'pattern_length': len(self.pattern)
        }
    
    def get_lps_array(self) -> List[int]:
        """
        Get the LPS array for the pattern.
        
        Returns:
            LPS array
        """
        return self.lps.copy()
    
    def __repr__(self) -> str:
        """String representation of KMP object."""
        return f"KMP(pattern='{self.pattern[:20]}{'...' if len(self.pattern) > 20 else ''}')"


def kmp_search_multiple(text: str, patterns: List[str]) -> Dict[str, List[int]]:
    """
    Search for multiple patterns in text using KMP algorithm.
    
    This is a convenience function that searches for multiple patterns
    independently. Each pattern is searched separately.
    
    Args:
        text: The text to search in
        patterns: List of patterns to search for
        
    Returns:
        Dictionary mapping each pattern to its list of match positions
        
    Example:
        >>> results = kmp_search_multiple("ABABCABAB", ["AB", "CAB"])
        >>> print(results)
        {'AB': [0, 2, 5, 7], 'CAB': [4]}
    """
    results = {}
    for pattern in patterns:
        results[pattern] = kmp_search(text, pattern)
    return results


def kmp_count_matches(text: str, pattern: str) -> int:
    """
    Count the number of occurrences of pattern in text.
    
    Args:
        text: The text to search in
        pattern: The pattern to search for
        
    Returns:
        Number of pattern occurrences
        
    Example:
        >>> kmp_count_matches("AAAAAAA", "AAA")
        5
    """
    return len(kmp_search(text, pattern))


def kmp_find_first(text: str, pattern: str) -> Optional[int]:
    """
    Find the first occurrence of pattern in text.
    
    Args:
        text: The text to search in
        pattern: The pattern to search for
        
    Returns:
        Position of first match, or None if not found
        
    Example:
        >>> kmp_find_first("ABABCABAB", "CAB")
        4
    """
    matches = kmp_search(text, pattern)
    return matches[0] if matches else None


def kmp_contains(text: str, pattern: str) -> bool:
    """
    Check if pattern exists in text.
    
    Args:
        text: The text to search in
        pattern: The pattern to search for
        
    Returns:
        True if pattern found, False otherwise
        
    Example:
        >>> kmp_contains("ABABCABAB", "CAB")
        True
        >>> kmp_contains("ABABCABAB", "XYZ")
        False
    """
    return kmp_find_first(text, pattern) is not None
