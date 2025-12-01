from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from wf_core import WagnerFischer
import time


@dataclass
class Match:
    position: int           # Start position in text
    end_position: int       # End position in text
    matched_text: str       # The actual text that matched
    edit_distance: int      # Edit distance from pattern
    alignment: Optional[List[str]] = None  # Alignment operations


class PatternSearcher:
    
    def __init__(self, 
                 max_distance: int = 2,
                 substitution_cost: int = 1,
                 insertion_cost: int = 1,
                 deletion_cost: int = 1):
        
        self.max_distance = max_distance
        self.wf = WagnerFischer(substitution_cost, insertion_cost, deletion_cost)
    
    def search(self, 
               pattern: str, 
               text: str,
               return_alignment: bool = False) -> List[Match]:
        
        matches = []
        pattern_len = len(pattern)
        text_len = len(text)
        
        if pattern_len == 0 or text_len == 0:
            return matches
        
        # Sliding window with variable size to account for indels
        min_window = max(1, pattern_len - self.max_distance)
        max_window = pattern_len + self.max_distance
        
        i = 0
        while i <= text_len - min_window:
            # Try different window sizes
            for window_size in range(min_window, min(max_window + 1, text_len - i + 1)):
                window = text[i:i + window_size]
                
                # Compute edit distance
                distance, within_threshold = self.wf.compute_with_threshold(
                    pattern, window, self.max_distance
                )
                
                if within_threshold:
                    # Get alignment if requested
                    alignment = None
                    if return_alignment:
                        _, alignment = self.wf.compute_with_traceback(pattern, window)
                    
                    match = Match(
                        position=i,
                        end_position=i + window_size,
                        matched_text=window,
                        edit_distance=distance,
                        alignment=alignment
                    )
                    matches.append(match)
                    
                    # Skip overlapping matches
                    i += window_size
                    break
            else:
                i += 1
        
        return matches
    
    def search_exact(self, pattern: str, text: str) -> List[Match]:
        matches = []
        pattern_len = len(pattern)
        text_len = len(text)
        
        for i in range(text_len - pattern_len + 1):
            if text[i:i + pattern_len] == pattern:
                match = Match(
                    position=i,
                    end_position=i + pattern_len,
                    matched_text=pattern,
                    edit_distance=0
                )
                matches.append(match)
        
        return matches
    
    def search_multiple(self,
                       patterns: List[str],
                       text: str,
                       return_alignment: bool = False) -> Dict[str, List[Match]]:
        results = {}
        
        for pattern in patterns:
            matches = self.search(pattern, text, return_alignment)
            results[pattern] = matches
        
        return results
    
    def count_matches(self, pattern: str, text: str) -> int:
        return len(self.search(pattern, text, return_alignment=False))


class BenchmarkSearcher(PatternSearcher):
   
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stats = {
            'total_comparisons': 0,
            'total_time': 0.0,
            'matches_found': 0
        }
    
    def search_with_stats(self,
                         pattern: str,
                         text: str,
                         return_alignment: bool = False) -> Tuple[List[Match], Dict]:
        start_time = time.perf_counter()
        
        matches = self.search(pattern, text, return_alignment)
        
        end_time = time.perf_counter()
        elapsed = end_time - start_time
        
        stats = {
            'pattern_length': len(pattern),
            'text_length': len(text),
            'matches_found': len(matches),
            'time_seconds': elapsed,
            'time_per_match': elapsed / len(matches) if matches else 0,
            'throughput_chars_per_sec': len(text) / elapsed if elapsed > 0 else 0
        }
        
        return matches, stats
    
    def reset_stats(self):
        """Reset accumulated statistics."""
        self.stats = {
            'total_comparisons': 0,
            'total_time': 0.0,
            'matches_found': 0
        }


def find_motifs(pattern: str, 
                text: str, 
                max_distance: int = 2,
                min_similarity: float = 0.8) -> List[Match]:
    searcher = PatternSearcher(max_distance=max_distance)
    matches = searcher.search(pattern, text)
    
    # Filter by similarity
    filtered_matches = []
    for match in matches:
        similarity = 1.0 - (match.edit_distance / len(pattern))
        if similarity >= min_similarity:
            filtered_matches.append(match)
    
    return filtered_matches
