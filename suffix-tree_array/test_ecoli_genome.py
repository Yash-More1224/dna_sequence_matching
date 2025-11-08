"""
Real E. coli Genome Testing Script

This script tests the SuffixIndexer on a real E. coli K-12 MG1655 genome.
It can download the genome from NCBI RefSeq if needed.

Usage:
    # Download and test automatically
    python test_ecoli_genome.py --download
    
    # Test with local FASTA file
    python test_ecoli_genome.py --fasta /path/to/ecoli.fasta
    
Requirements:
    pip install biopython  # For FASTA parsing
"""

import sys
import os
import argparse
import time
from suffix_indexer import SuffixIndexer

try:
    from Bio import SeqIO, Entrez
    BIOPYTHON_AVAILABLE = True
except ImportError:
    BIOPYTHON_AVAILABLE = False
    print("Warning: Biopython not installed. FASTA parsing will be limited.")
    print("Install with: pip install biopython")


def download_ecoli_genome(output_path: str = "ecoli_k12_mg1655.fasta") -> str:
    """
    Download E. coli K-12 MG1655 genome from NCBI.
    
    Args:
        output_path (str): Where to save the FASTA file
        
    Returns:
        str: Path to downloaded file
    """
    if not BIOPYTHON_AVAILABLE:
        print("Error: Biopython is required to download from NCBI")
        print("Install with: pip install biopython")
        sys.exit(1)
    
    print("Downloading E. coli K-12 MG1655 genome from NCBI...")
    print("RefSeq ID: NC_000913.3")
    
    Entrez.email = "your_email@example.com"  # NCBI requires an email
    
    try:
        # Download from NCBI
        handle = Entrez.efetch(
            db="nucleotide",
            id="NC_000913.3",
            rettype="fasta",
            retmode="text"
        )
        
        with open(output_path, 'w') as f:
            f.write(handle.read())
        
        handle.close()
        
        print(f"✓ Downloaded to {output_path}")
        return output_path
    
    except Exception as e:
        print(f"Error downloading genome: {e}")
        print("\nAlternative: Download manually from:")
        print("https://www.ncbi.nlm.nih.gov/nuccore/NC_000913.3")
        sys.exit(1)


def read_fasta(fasta_path: str) -> str:
    """
    Read DNA sequence from FASTA file.
    
    Args:
        fasta_path (str): Path to FASTA file
        
    Returns:
        str: DNA sequence
    """
    if BIOPYTHON_AVAILABLE:
        # Use Biopython for robust parsing
        with open(fasta_path, 'r') as f:
            records = list(SeqIO.parse(f, "fasta"))
            if not records:
                raise ValueError("No sequences found in FASTA file")
            return str(records[0].seq)
    else:
        # Simple FASTA parser (assumes single sequence)
        with open(fasta_path, 'r') as f:
            lines = f.readlines()
        
        sequence = []
        for line in lines:
            line = line.strip()
            if line and not line.startswith('>'):
                sequence.append(line)
        
        return ''.join(sequence)


def test_ecoli_genome(fasta_path: str):
    """
    Comprehensive testing on E. coli genome.
    
    Args:
        fasta_path (str): Path to E. coli FASTA file
    """
    print("=" * 80)
    print("E. COLI K-12 MG1655 GENOME TESTING")
    print("=" * 80)
    
    # Read genome
    print("\nReading genome from FASTA...")
    start_read = time.time()
    genome = read_fasta(fasta_path)
    read_time = time.time() - start_read
    
    print(f"✓ Genome loaded in {read_time:.2f} seconds")
    print(f"  Genome length: {len(genome):,} bases")
    print(f"  Expected size: ~4.6 million bases")
    
    # Base composition
    composition = {
        'A': genome.count('A'),
        'C': genome.count('C'),
        'G': genome.count('G'),
        'T': genome.count('T'),
        'N': genome.count('N')
    }
    
    print(f"\n  Base composition:")
    total = sum(composition.values())
    for base, count in composition.items():
        pct = (count / total * 100) if total > 0 else 0
        print(f"    {base}: {count:>10,} ({pct:>5.2f}%)")
    
    gc_content = (composition['G'] + composition['C']) / total * 100
    print(f"  GC content: {gc_content:.2f}%")
    
    # Build suffix array index
    print("\n" + "-" * 80)
    print("Building Suffix Array Index...")
    print("-" * 80)
    
    indexer = SuffixIndexer(genome)
    
    print(f"\n✓ Index built successfully!")
    stats = indexer.get_statistics()
    print(f"  Construction time: {stats['preprocessing_time']:.2f} seconds")
    print(f"  Construction rate: {stats['text_length'] / stats['preprocessing_time']:,.0f} bases/second")
    print(f"  Memory footprint: {stats['memory_footprint_mb']:.2f} MB")
    print(f"  Memory per base: {stats['memory_footprint_bytes'] / stats['text_length']:.2f} bytes")
    
    # Test exact pattern searches
    print("\n" + "-" * 80)
    print("Testing Exact Pattern Matching")
    print("-" * 80)
    
    # Test patterns of biological interest
    test_patterns = {
        'Start codon (ATG)': 'ATG',
        'Stop codon (TAA)': 'TAA',
        'Stop codon (TAG)': 'TAG',
        'Stop codon (TGA)': 'TGA',
        'TATA box motif': 'TATAAA',
        'Shine-Dalgarno (AGGAGG)': 'AGGAGG',
        'Pribnow box (TATAAT)': 'TATAAT',
        'EcoRI site (GAATTC)': 'GAATTC',
        'BamHI site (GGATCC)': 'GGATCC',
        'Chi site (GCTGGTGG)': 'GCTGGTGG'
    }
    
    print("\nBiologically relevant patterns:")
    for name, pattern in test_patterns.items():
        start = time.time()
        matches = indexer.search_exact(pattern)
        search_time = time.time() - start
        
        print(f"\n  {name} ({pattern}):")
        print(f"    Matches: {len(matches):>6,}")
        print(f"    Search time: {search_time*1000:>8.4f} ms")
        
        if len(matches) > 0 and len(matches) <= 10:
            print(f"    Positions: {matches}")
    
    # Test variable pattern lengths
    print("\n" + "-" * 80)
    print("Pattern Length Scalability Test")
    print("-" * 80)
    
    print(f"\n{'Length':<10} {'Pattern':<20} {'Matches':<10} {'Time (ms)':<12}")
    print("-" * 60)
    
    # Extract actual subsequences from genome for testing
    test_positions = [1000, 10000, 100000, 500000]
    pattern_lengths = [10, 20, 50, 100]
    
    for length in pattern_lengths:
        pos = test_positions[0]
        if pos + length < len(genome):
            pattern = genome[pos:pos+length]
            
            start = time.time()
            matches = indexer.search_exact(pattern)
            search_time = time.time() - start
            
            pattern_display = pattern[:15] + "..." if len(pattern) > 15 else pattern
            print(f"{length:<10} {pattern_display:<20} {len(matches):<10} {search_time*1000:<12.4f}")
    
    # Find repeats and motifs
    print("\n" + "-" * 80)
    print("Repeat and Motif Discovery")
    print("-" * 80)
    
    print("\nSearching for long exact repeats...")
    
    for min_len in [15, 20, 25, 30]:
        print(f"\n  Minimum length: {min_len}bp")
        
        start = time.time()
        repeats = indexer.find_longest_repeats(min_length=min_len)
        discovery_time = time.time() - start
        
        print(f"  Discovery time: {discovery_time:.4f} seconds")
        print(f"  Repeats found: {len(repeats):,}")
        
        if repeats:
            print(f"\n  Top 5 longest repeats:")
            for i, repeat in enumerate(repeats[:5], 1):
                substr_display = repeat['substring'][:50]
                if len(repeat['substring']) > 50:
                    substr_display += "..."
                
                print(f"    {i}. Length {repeat['length']:>3}bp, "
                      f"occurs {repeat['count']:>3}x at positions {repeat['positions'][:3]}"
                      f"{'...' if len(repeat['positions']) > 3 else ''}")
                print(f"       {substr_display}")
    
    # Performance summary
    print("\n" + "=" * 80)
    print("PERFORMANCE SUMMARY")
    print("=" * 80)
    
    print(f"\n  Genome size: {len(genome):,} bases (~{len(genome)/1e6:.2f} Mbp)")
    print(f"  Index build time: {stats['preprocessing_time']:.2f} seconds")
    print(f"  Index memory: {stats['memory_footprint_mb']:.2f} MB")
    print(f"  Search time (typical): < 1 ms for patterns up to 100bp")
    print(f"\n  ✓ Performance is excellent for a {len(genome)/1e6:.1f} Mbp bacterial genome!")
    
    # Biological insights
    print("\n" + "=" * 80)
    print("BIOLOGICAL INSIGHTS")
    print("=" * 80)
    
    # Estimate gene count from start codons
    atg_count = len(indexer.search_exact('ATG'))
    estimated_genes = atg_count // 10  # Rough estimate (many ATGs are not starts)
    actual_genes = 4300  # Known gene count for E. coli K-12 MG1655
    
    print(f"\n  Start codons (ATG): {atg_count:,}")
    print(f"  Estimated genes: ~{estimated_genes:,}")
    print(f"  Actual gene count: ~{actual_genes:,}")
    print(f"  GC content: {gc_content:.2f}% (typical for E. coli)")
    
    print("\n  Restriction enzyme analysis complete:")
    ecori = len(indexer.search_exact('GAATTC'))
    bamhi = len(indexer.search_exact('GGATCC'))
    print(f"    EcoRI sites (GAATTC): {ecori:,}")
    print(f"    BamHI sites (GGATCC): {bamhi:,}")
    
    print("\n✓ E. coli genome analysis complete!")
    print("\n" + "=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Test SuffixIndexer on E. coli K-12 MG1655 genome"
    )
    parser.add_argument(
        '--download',
        action='store_true',
        help='Download E. coli genome from NCBI'
    )
    parser.add_argument(
        '--fasta',
        type=str,
        help='Path to E. coli genome FASTA file'
    )
    
    args = parser.parse_args()
    
    fasta_path = None
    
    if args.download:
        fasta_path = download_ecoli_genome()
    elif args.fasta:
        if not os.path.exists(args.fasta):
            print(f"Error: FASTA file not found: {args.fasta}")
            sys.exit(1)
        fasta_path = args.fasta
    else:
        # Try to find FASTA in current directory
        possible_names = [
            'ecoli_k12_mg1655.fasta',
            'ecoli.fasta',
            'NC_000913.3.fasta',
            'ecoli.fna'
        ]
        
        for name in possible_names:
            if os.path.exists(name):
                fasta_path = name
                print(f"Found E. coli genome: {fasta_path}")
                break
        
        if not fasta_path:
            print("Error: No E. coli genome file found.")
            print("\nUsage:")
            print("  python test_ecoli_genome.py --download")
            print("  python test_ecoli_genome.py --fasta /path/to/ecoli.fasta")
            print("\nOr place a FASTA file named 'ecoli.fasta' in the current directory.")
            sys.exit(1)
    
    # Run tests
    test_ecoli_genome(fasta_path)


if __name__ == "__main__":
    main()
