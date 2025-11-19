"""
Wagner-Fischer Algorithm Implementation for DNA Sequence Matching
Comprehensive implementation with full matrix, space-optimized, and banded variants.
"""

import numpy as np
from typing import Tuple, List, Optional, Dict
from dataclasses import dataclass


@dataclass
class EditDistanceResult:
    """Result of edit distance computation."""
    distance: int
    operations: Optional[List[str]] = None
    aligned_source: Optional[str] = None
    aligned_target: Optional[str] = None


class WagnerFischer:
    """
    Wagner-Fischer algorithm for computing Levenshtein edit distance.
    Supports multiple variants: full matrix, space-optimized, banded, and threshold-based.
    """
    
    def __init__(self, 
                 substitution_cost: int = 1,
                 insertion_cost: int = 1,
                 deletion_cost: int = 1):
        """
        Initialize Wagner-Fischer with custom operation costs.
        
        Args:
            substitution_cost: Cost of substituting one character for another
            insertion_cost: Cost of inserting a character
            deletion_cost: Cost of deleting a character
        """
        self.sub_cost = substitution_cost
        self.ins_cost = insertion_cost
        self.del_cost = deletion_cost
    
    def compute_distance(self, 
                        source: str, 
                        target: str,
                        return_matrix: bool = False) -> Tuple[int, Optional[np.ndarray]]:
        """
        Compute the Levenshtein edit distance using full DP matrix.
        
        Args:
            source: Source string
            target: Target string
            return_matrix: If True, return the full DP matrix
            
        Returns:
            Tuple of (edit_distance, matrix) where matrix is None if return_matrix=False
        """
        m, n = len(source), len(target)
        
        # Initialize DP matrix
        dp = np.zeros((m + 1, n + 1), dtype=np.int32)
        
        # Base cases: converting from/to empty string
        for i in range(m + 1):
            dp[i][0] = i * self.del_cost
        for j in range(n + 1):
            dp[0][j] = j * self.ins_cost
        
        # Fill the DP matrix
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if source[i-1] == target[j-1]:
                    # Characters match, no operation needed
                    dp[i][j] = dp[i-1][j-1]
                else:
                    # Take minimum of three operations
                    substitute = dp[i-1][j-1] + self.sub_cost
                    delete = dp[i-1][j] + self.del_cost
                    insert = dp[i][j-1] + self.ins_cost
                    dp[i][j] = min(substitute, delete, insert)
        
        distance = int(dp[m][n])
        matrix = dp if return_matrix else None
        
        return distance, matrix
    
    def compute_distance_optimized(self, source: str, target: str) -> int:
        """
        Space-optimized version using only two rows.
        Uses O(min(m,n)) space instead of O(m*n).
        
        Args:
            source: Source string
            target: Target string
            
        Returns:
            Edit distance as integer
        """
        # Ensure source is the shorter string for space optimization
        if len(source) > len(target):
            source, target = target, source
        
        m, n = len(source), len(target)
        
        # Use only two rows
        prev_row = np.zeros(n + 1, dtype=np.int32)
        curr_row = np.zeros(n + 1, dtype=np.int32)
        
        # Initialize first row
        for j in range(n + 1):
            prev_row[j] = j * self.ins_cost
        
        # Fill row by row
        for i in range(1, m + 1):
            curr_row[0] = i * self.del_cost
            
            for j in range(1, n + 1):
                if source[i-1] == target[j-1]:
                    curr_row[j] = prev_row[j-1]
                else:
                    substitute = prev_row[j-1] + self.sub_cost
                    delete = prev_row[j] + self.del_cost
                    insert = curr_row[j-1] + self.ins_cost
                    curr_row[j] = min(substitute, delete, insert)
            
            # Swap rows
            prev_row, curr_row = curr_row, prev_row
        
        return int(prev_row[n])
    
    def compute_with_traceback(self, source: str, target: str) -> EditDistanceResult:
        """
        Compute edit distance and return the alignment operations.
        
        Args:
            source: Source string
            target: Target string
            
        Returns:
            EditDistanceResult with distance, operations, and aligned strings
        """
        m, n = len(source), len(target)
        distance, dp = self.compute_distance(source, target, return_matrix=True)
        
        # Traceback to find operations
        operations = []
        aligned_source = []
        aligned_target = []
        i, j = m, n
        
        while i > 0 or j > 0:
            if i == 0:
                operations.append(f'insert_{target[j-1]}')
                aligned_source.append('-')
                aligned_target.append(target[j-1])
                j -= 1
            elif j == 0:
                operations.append(f'delete_{source[i-1]}')
                aligned_source.append(source[i-1])
                aligned_target.append('-')
                i -= 1
            elif source[i-1] == target[j-1]:
                operations.append(f'match_{source[i-1]}')
                aligned_source.append(source[i-1])
                aligned_target.append(target[j-1])
                i -= 1
                j -= 1
            else:
                # Find which operation was used
                substitute = dp[i-1][j-1]
                delete = dp[i-1][j]
                insert = dp[i][j-1]
                
                min_op = min(substitute, delete, insert)
                
                if min_op == substitute:
                    operations.append(f'substitute_{source[i-1]}->{target[j-1]}')
                    aligned_source.append(source[i-1])
                    aligned_target.append(target[j-1])
                    i -= 1
                    j -= 1
                elif min_op == delete:
                    operations.append(f'delete_{source[i-1]}')
                    aligned_source.append(source[i-1])
                    aligned_target.append('-')
                    i -= 1
                else:
                    operations.append(f'insert_{target[j-1]}')
                    aligned_source.append('-')
                    aligned_target.append(target[j-1])
                    j -= 1
        
        operations.reverse()
        aligned_source.reverse()
        aligned_target.reverse()
        
        return EditDistanceResult(
            distance=distance,
            operations=operations,
            aligned_source=''.join(aligned_source),
            aligned_target=''.join(aligned_target)
        )
    
    def compute_with_threshold(self, 
                               source: str, 
                               target: str, 
                               threshold: int) -> Tuple[int, bool]:
        """
        Compute edit distance with early termination if threshold is exceeded.
        Uses simple optimization - just compute normally and check.
        
        Args:
            source: Source string
            target: Target string
            threshold: Maximum allowed edit distance
            
        Returns:
            Tuple of (distance, within_threshold) where within_threshold is
            False if distance exceeds threshold
        """
        m, n = len(source), len(target)
        
        # If length difference exceeds threshold, no need to compute
        if abs(m - n) > threshold:
            return abs(m - n), False
        
        # Use fast space-optimized version
        distance = self.compute_distance_optimized(source, target)
        within_threshold = distance <= threshold
        
        return distance, within_threshold
    
    def compute_banded(self, 
                       source: str, 
                       target: str, 
                       band_width: int) -> int:
        """
        Compute edit distance using banded DP (restricted diagonal band).
        More efficient for sequences with similar lengths.
        
        Args:
            source: Source string
            target: Target string
            band_width: Width of the diagonal band
            
        Returns:
            Edit distance as integer
        """
        m, n = len(source), len(target)
        
        if abs(m - n) > band_width:
            # Cannot compute within band
            return max(m, n)
        
        # Initialize band matrix
        k = band_width
        band = np.full((m + 1, 2 * k + 1), fill_value=float('inf'), dtype=np.float32)
        
        # Map j coordinate to band coordinate
        def to_band(j, i):
            return j - i + k
        
        band[0][k] = 0
        
        for i in range(m + 1):
            for j in range(max(0, i - k), min(n + 1, i + k + 1)):
                b = to_band(j, i)
                
                if i == 0 and j == 0:
                    continue
                
                candidates = []
                
                # Deletion
                if i > 0 and 0 <= to_band(j, i-1) < 2*k+1:
                    candidates.append(band[i-1][to_band(j, i-1)] + self.del_cost)
                
                # Insertion
                if j > 0 and 0 <= to_band(j-1, i) < 2*k+1:
                    candidates.append(band[i][to_band(j-1, i)] + self.ins_cost)
                
                # Match/Substitution
                if i > 0 and j > 0 and 0 <= to_band(j-1, i-1) < 2*k+1:
                    cost = 0 if source[i-1] == target[j-1] else self.sub_cost
                    candidates.append(band[i-1][to_band(j-1, i-1)] + cost)
                
                if candidates:
                    band[i][b] = min(candidates)
        
        result = band[m][to_band(n, m)]
        return int(result) if result != float('inf') else max(m, n)


def levenshtein_distance(s1: str, s2: str) -> int:
    """
    Convenience function to compute Levenshtein distance with default costs.
    
    Args:
        s1: First string
        s2: Second string
        
    Returns:
        Edit distance as integer
    """
    wf = WagnerFischer()
    return wf.compute_distance_optimized(s1, s2)


def similarity_ratio(s1: str, s2: str) -> float:
    """
    Compute similarity ratio between two strings (0.0 to 1.0).
    
    Args:
        s1: First string
        s2: Second string
        
    Returns:
        Similarity ratio where 1.0 means identical and 0.0 means completely different
    """
    distance = levenshtein_distance(s1, s2)
    max_len = max(len(s1), len(s2))
    
    if max_len == 0:
        return 1.0
    
    return 1.0 - (distance / max_len)
