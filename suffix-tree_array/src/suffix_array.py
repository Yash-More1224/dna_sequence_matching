"""
Suffix Array and LCP Array Implementation for DNA Pattern Matching

This module implements an efficient Suffix Array indexing structure with LCP (Longest Common Prefix) 
array for fast exact pattern matching and motif discovery in DNA sequences.

Design Choice: Suffix Array + LCP vs Suffix Tree
-----------------------------------------------
We chose Suffix Array + LCP over a full Suffix Tree (Ukkonen's algorithm) for the following reasons:
1. Simpler implementation with fewer edge cases
2. Better cache locality and lower memory overhead (2N integers vs ~10N for suffix trees)
3. Competitive search performance: O(|P| log |T|) for exact search
4. Easier integration with existing Python infrastructure
5. LCP array enables efficient repeat/motif discovery

Time/Space Complexity:
- Construction: O(N log N) for SA, O(N) for LCP
- Space: O(N) for SA and LCP arrays (2N integers total)
- Search: O(|P| log |T|) per pattern (binary search on SA)

Author: DNA Pattern Matching Team
Date: November 2025
"""

import time
import sys
from typing import List, Tuple, Optional, Dict


class SuffixArray:
    """
    Suffix Array-based indexing structure for DNA sequences.
    
    This class constructs a Suffix Array (SA) and Longest Common Prefix (LCP) array
    for efficient exact pattern matching and repeat discovery in DNA sequences.
    
    Attributes:
        text (str): The input DNA sequence (terminated internally with '$')
        sa (List[int]): Suffix Array - sorted indices of all suffixes
        lcp (List[int]): LCP Array - longest common prefix between consecutive suffixes
        n (int): Length of the text (including terminator)
        preprocessing_time (float): Time taken to build the index (seconds)
        memory_footprint (int): Memory used by SA and LCP arrays (bytes)
        comparisons (int): Number of character comparisons during search
        pattern (str): Last searched pattern (for statistics)
    """
    
    def __init__(self, text: str = "", verbose: bool = False):
        """
        Initialize the SuffixArray with a DNA sequence.
        
        Args:
            text (str): Input DNA sequence (should contain only A, C, G, T, N)
            verbose (bool): Whether to print construction information
        
        The constructor automatically builds the suffix array and LCP array if text provided.
        """
        self.text = ""
        self.sa: List[int] = []
        self.lcp: List[int] = []
        self.n = 0
        self.preprocessing_time = 0.0
        self.memory_footprint = 0
        self.comparisons = 0
        self.pattern = ""
        self.verbose = verbose
        
        if text:
            self.build_index(text)
    
    def build_index(self, text: str) -> Tuple[float, int]:
        """
        Build the Suffix Array and LCP Array for the given text.
        
        This method implements an O(N log N) suffix array construction using
        prefix doubling (also known as the Manber-Myers algorithm). The LCP
        array is then constructed in O(N) using the Kasai algorithm.
        
        Time Complexity: O(N log N) for SA construction + O(N) for LCP = O(N log N)
        Space Complexity: O(N)
        
        Args:
            text (str): The DNA sequence to index
            
        Returns:
            Tuple[float, int]: (preprocessing_time in seconds, memory_footprint in bytes)
        """
        start_time = time.perf_counter()
        
        # Add sentinel character to ensure proper suffix ordering
        self.text = text + '$'
        self.n = len(self.text)
        
        # Build suffix array using O(N log N) prefix doubling
        self.sa = self._build_suffix_array_prefix_doubling()
        
        # Build LCP array in O(N) using Kasai's algorithm
        self.lcp = self._build_lcp_array()
        
        end_time = time.perf_counter()
        self.preprocessing_time = end_time - start_time
        
        # Calculate memory footprint (SA + LCP arrays, each storing N integers)
        self.memory_footprint = (len(self.sa) + len(self.lcp)) * sys.getsizeof(int)
        
        # Print metrics for benchmarking
        if self.verbose:
            print(f"[SuffixArray] Index built successfully")
            print(f"  Text length: {self.n - 1} bases (+ 1 sentinel)")
            print(f"  Preprocessing time: {self.preprocessing_time:.4f} seconds")
            print(f"  Memory footprint: {self.memory_footprint / (1024**2):.2f} MB")
        
        return self.preprocessing_time, self.memory_footprint
    
    def _build_suffix_array_prefix_doubling(self) -> List[int]:
        """
        Build suffix array using prefix doubling (Manber-Myers algorithm).
        
        This is an O(N log N) algorithm that sorts suffixes by comparing increasingly
        longer prefixes (length 1, 2, 4, 8, ..., N). At each iteration, we use the
        previously computed ranks to sort in O(N log N).
        
        Returns:
            List[int]: The suffix array
        """
        n = self.n
        text = self.text
        
        # Initialize: rank by first character
        sa = list(range(n))
        rank = [ord(c) for c in text]
        tmp_rank = [0] * n
        
        k = 1  # Current comparison length
        while k < n:
            # Sort by (rank[i], rank[i+k]) pairs
            # We use Python's built-in sort with a key function
            sa.sort(key=lambda i: (rank[i], rank[i + k] if i + k < n else -1))
            
            # Recompute ranks based on sorted order
            tmp_rank[sa[0]] = 0
            for i in range(1, n):
                # Same rank if both pairs are equal
                prev_pair = (rank[sa[i-1]], rank[sa[i-1] + k] if sa[i-1] + k < n else -1)
                curr_pair = (rank[sa[i]], rank[sa[i] + k] if sa[i] + k < n else -1)
                
                tmp_rank[sa[i]] = tmp_rank[sa[i-1]]
                if prev_pair != curr_pair:
                    tmp_rank[sa[i]] += 1
            
            rank = tmp_rank[:]
            k *= 2
            
            # Early termination if all ranks are unique
            if rank[sa[-1]] == n - 1:
                break
        
        return sa
    
    def _build_lcp_array(self) -> List[int]:
        """
        Build LCP (Longest Common Prefix) array using Kasai's algorithm.
        
        The LCP array stores the length of the longest common prefix between
        consecutive suffixes in the sorted suffix array. This is computed in
        O(N) time using the inverse suffix array.
        
        Time Complexity: O(N)
        Space Complexity: O(N) for the inverse suffix array
        
        Returns:
            List[int]: The LCP array where lcp[i] = LCP(sa[i-1], sa[i])
        """
        n = self.n
        sa = self.sa
        text = self.text
        
        # Build inverse suffix array (rank array)
        rank = [0] * n
        for i in range(n):
            rank[sa[i]] = i
        
        lcp = [0] * n
        h = 0  # Length of current LCP
        
        for i in range(n):
            if rank[i] > 0:
                j = sa[rank[i] - 1]  # Previous suffix in sorted order
                
                # Extend the LCP from the previous comparison
                while i + h < n and j + h < n and text[i + h] == text[j + h]:
                    h += 1
                
                lcp[rank[i]] = h
                
                # Optimization: decrease h by at most 1 for next iteration
                if h > 0:
                    h -= 1
        
        return lcp
    
    def search(self, pattern: str) -> List[int]:
        """
        Search for exact occurrences of a pattern in the indexed text.
        Alias for search_exact() for consistency with other algorithms.
        
        Args:
            pattern (str): The DNA pattern to search for
            
        Returns:
            List[int]: Sorted list of starting positions where pattern occurs
        """
        return self.search_exact(pattern)
    
    def search_exact(self, pattern: str) -> List[int]:
        """
        Search for exact occurrences of a pattern in the indexed text.
        
        Uses binary search on the suffix array to find the range of suffixes
        that start with the given pattern, then returns all match positions.
        
        Time Complexity: O(|P| log |T|) where P is pattern length and T is text length
        Space Complexity: O(k) where k is the number of matches
        
        Args:
            pattern (str): The DNA pattern to search for
            
        Returns:
            List[int]: Sorted list of starting positions where pattern occurs
                      (0-indexed positions in the original text)
        """
        self.comparisons = 0
        self.pattern = pattern
        
        if not pattern or not self.sa:
            return []
        
        # Remove sentinel from text length for boundary checking
        text_len = self.n - 1
        
        # Pattern longer than text
        if len(pattern) > text_len:
            return []
        
        # Binary search for the leftmost suffix starting with pattern
        left = self._binary_search_left(pattern)
        
        # Binary search for the rightmost suffix starting with pattern
        right = self._binary_search_right(pattern)
        
        # Extract all match positions from the range
        if left <= right:
            matches = [self.sa[i] for i in range(left, right + 1) if self.sa[i] < text_len]
            return sorted(matches)
        
        return []
    
    def search_first(self, pattern: str) -> Optional[int]:
        """
        Search for the first occurrence of a pattern.
        
        Args:
            pattern (str): The DNA pattern to search for
            
        Returns:
            Optional[int]: Position of first match, or None if not found
        """
        matches = self.search_exact(pattern)
        return matches[0] if matches else None
    
    def _binary_search_left(self, pattern: str) -> int:
        """
        Find the leftmost position in SA where a suffix starts with pattern.
        
        Args:
            pattern (str): Pattern to search for
            
        Returns:
            int: Index in SA of leftmost matching suffix (or n if not found)
        """
        left, right = 0, len(self.sa)
        
        while left < right:
            mid = (left + right) // 2
            suffix_start = self.sa[mid]
            
            # Compare pattern with suffix starting at sa[mid]
            suffix = self.text[suffix_start:suffix_start + len(pattern)]
            self.comparisons += min(len(pattern), len(suffix))
            
            if suffix < pattern:
                left = mid + 1
            else:
                right = mid
        
        return left
    
    def _binary_search_right(self, pattern: str) -> int:
        """
        Find the rightmost position in SA where a suffix starts with pattern.
        
        Args:
            pattern (str): Pattern to search for
            
        Returns:
            int: Index in SA of rightmost matching suffix (or -1 if not found)
        """
        left, right = 0, len(self.sa)
        
        while left < right:
            mid = (left + right) // 2
            suffix_start = self.sa[mid]
            
            # Compare pattern with suffix starting at sa[mid]
            suffix = self.text[suffix_start:suffix_start + len(pattern)]
            self.comparisons += min(len(pattern), len(suffix))
            
            if suffix <= pattern:
                left = mid + 1
            else:
                right = mid
        
        return left - 1
    
    def find_longest_repeats(self, min_length: int = 15) -> List[Dict[str, any]]:
        """
        Find the longest exact repeats (substrings occurring at least twice) using LCP array.
        
        This method scans the LCP array to find maximal repeated substrings. A repeat
        is identified when LCP[i] >= min_length, indicating that suffixes sa[i-1] and
        sa[i] share a common prefix of at least min_length.
        
        Time Complexity: O(N) to scan LCP array + O(k log k) to sort results
        Space Complexity: O(k) where k is the number of repeats found
        
        Args:
            min_length (int): Minimum length of repeats to report (default: 15)
            
        Returns:
            List[Dict]: List of repeat dictionaries, each containing:
                - 'length': Length of the repeated substring
                - 'substring': The repeated substring itself
                - 'positions': List of positions where it occurs
                - 'count': Number of occurrences
                
        The results are sorted by length (descending), then by count (descending).
        """
        if not self.lcp or min_length <= 0:
            return []
        
        repeats_map = {}  # Maps substring to its positions
        n = self.n - 1  # Exclude sentinel
        
        # Scan LCP array to find repeats
        for i in range(1, len(self.lcp)):
            lcp_len = self.lcp[i]
            
            if lcp_len >= min_length:
                # Found a repeat between sa[i-1] and sa[i]
                pos1 = self.sa[i - 1]
                pos2 = self.sa[i]
                
                # Skip if positions include the sentinel
                if pos1 >= n or pos2 >= n:
                    continue
                
                # Extract the repeated substring
                # We use the full LCP length as the repeat
                repeat_str = self.text[pos1:pos1 + lcp_len]
                
                # Store positions for this repeat
                if repeat_str not in repeats_map:
                    repeats_map[repeat_str] = set()
                
                repeats_map[repeat_str].add(pos1)
                repeats_map[repeat_str].add(pos2)
        
        # Convert to list format and sort by length and count
        repeats_list = []
        for substring, positions in repeats_map.items():
            repeats_list.append({
                'length': len(substring),
                'substring': substring,
                'positions': sorted(list(positions)),
                'count': len(positions)
            })
        
        # Sort by length (desc), then by count (desc)
        repeats_list.sort(key=lambda x: (x['length'], x['count']), reverse=True)
        
        return repeats_list
    
    def get_statistics(self) -> Dict[str, any]:
        """
        Get comprehensive statistics about the index and last search.
        
        Returns:
            Dict: Dictionary containing:
                - text_length: Length of indexed text (excluding sentinel)
                - pattern_length: Length of last searched pattern
                - preprocessing_time: Time to build index (seconds)
                - memory_footprint_mb: Memory used by index (MB)
                - memory_footprint_bytes: Memory used by index (bytes)
                - comparisons: Character comparisons in last search
        """
        return {
            'text_length': self.n - 1,
            'pattern_length': len(self.pattern),
            'preprocessing_time': self.preprocessing_time,
            'memory_footprint_mb': self.memory_footprint / (1024**2),
            'memory_footprint_bytes': self.memory_footprint,
            'comparisons': self.comparisons
        }


def suffix_array_search(text: str, pattern: str, verbose: bool = False) -> List[int]:
    """
    Convenience function for one-time suffix array search.
    
    Args:
        text (str): Text to search in
        pattern (str): Pattern to search for
        verbose (bool): Whether to print information
        
    Returns:
        List[int]: List of match positions
    """
    sa = SuffixArray(text, verbose=verbose)
    return sa.search(pattern)
