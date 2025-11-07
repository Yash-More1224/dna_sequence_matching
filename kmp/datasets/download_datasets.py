"""
Script to download genomic datasets for KMP algorithm testing.

This script downloads bacterial and viral genomes from NCBI
for use in benchmarking and analysis.
"""

import requests
import gzip
import shutil
from pathlib import Path
from typing import Optional
import sys

from ..config import DATASETS, DATASETS_DIR


def download_file(url: str, output_path: Path, chunk_size: int = 8192) -> bool:
    """
    Download a file from a URL with progress indication.
    
    Args:
        url: URL to download from
        output_path: Path to save the file
        chunk_size: Size of chunks to download
        
    Returns:
        True if successful, False otherwise
    """
    try:
        print(f"Downloading from {url}...")
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    
                    # Progress indicator
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        sys.stdout.write(f"\r  Progress: {percent:.1f}% ({downloaded / 1024 / 1024:.2f} MB)")
                        sys.stdout.flush()
        
        print(f"\n  Download complete: {output_path}")
        return True
        
    except Exception as e:
        print(f"\n  Error downloading: {e}")
        return False


def decompress_gzip(input_path: Path, output_path: Path) -> bool:
    """
    Decompress a gzip file.
    
    Args:
        input_path: Path to gzipped file
        output_path: Path to save decompressed file
        
    Returns:
        True if successful, False otherwise
    """
    try:
        print(f"Decompressing {input_path.name}...")
        with gzip.open(input_path, 'rb') as f_in:
            with open(output_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        print(f"  Decompressed: {output_path}")
        return True
    except Exception as e:
        print(f"  Error decompressing: {e}")
        return False


def download_dataset(dataset_name: str, keep_compressed: bool = False) -> bool:
    """
    Download a specific dataset.
    
    Args:
        dataset_name: Name of dataset from DATASETS config
        keep_compressed: Whether to keep the compressed .gz file
        
    Returns:
        True if successful, False otherwise
    """
    if dataset_name not in DATASETS:
        print(f"Unknown dataset: {dataset_name}")
        print(f"Available datasets: {list(DATASETS.keys())}")
        return False
    
    dataset_info = DATASETS[dataset_name]
    
    print("\n" + "="*70)
    print(f"DOWNLOADING: {dataset_info['name']}")
    print("="*70)
    print(f"Accession: {dataset_info['accession']}")
    print(f"Size: ~{dataset_info['size_mb']} MB")
    print(f"Description: {dataset_info['description']}")
    print("-"*70)
    
    # Create datasets directory
    DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Download compressed file
    compressed_path = DATASETS_DIR / f"{dataset_name}.fna.gz"
    if not download_file(dataset_info['url'], compressed_path):
        return False
    
    # Decompress
    decompressed_path = DATASETS_DIR / f"{dataset_name}.fna"
    if not decompress_gzip(compressed_path, decompressed_path):
        return False
    
    # Remove compressed file if requested
    if not keep_compressed:
        compressed_path.unlink()
        print(f"  Removed compressed file")
    
    print("="*70)
    print(f"SUCCESS: {dataset_name} downloaded to {decompressed_path}")
    print("="*70)
    
    return True


def download_all_datasets(keep_compressed: bool = False) -> None:
    """
    Download all available datasets.
    
    Args:
        keep_compressed: Whether to keep compressed files
    """
    print("\n" + "#"*70)
    print("# DOWNLOADING ALL DATASETS")
    print("#"*70)
    
    success_count = 0
    fail_count = 0
    
    for dataset_name in DATASETS.keys():
        if download_dataset(dataset_name, keep_compressed):
            success_count += 1
        else:
            fail_count += 1
    
    print("\n" + "#"*70)
    print(f"# DOWNLOAD SUMMARY: {success_count} succeeded, {fail_count} failed")
    print("#"*70)


def check_dataset_exists(dataset_name: str) -> bool:
    """
    Check if a dataset has already been downloaded.
    
    Args:
        dataset_name: Name of dataset
        
    Returns:
        True if dataset exists locally
    """
    # Check for various extensions
    for ext in ['.fna', '.fasta', '.fa', '.fna.gz', '.fasta.gz', '.fa.gz']:
        path = DATASETS_DIR / f"{dataset_name}{ext}"
        if path.exists():
            return True
    return False


def list_downloaded_datasets() -> None:
    """List all downloaded datasets."""
    print("\n" + "="*70)
    print("DOWNLOADED DATASETS")
    print("="*70)
    
    if not DATASETS_DIR.exists():
        print("No datasets directory found.")
        return
    
    fasta_files = list(DATASETS_DIR.glob("*.fna")) + list(DATASETS_DIR.glob("*.fasta")) + list(DATASETS_DIR.glob("*.fa"))
    
    if not fasta_files:
        print("No datasets found.")
        print(f"Run with --download to download datasets.")
    else:
        for fasta_file in fasta_files:
            size_mb = fasta_file.stat().st_size / 1024 / 1024
            print(f"  {fasta_file.name:<30} ({size_mb:.2f} MB)")
    
    print("="*70)


def main():
    """Main function for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Download genomic datasets for KMP algorithm testing"
    )
    parser.add_argument(
        '--dataset',
        choices=list(DATASETS.keys()) + ['all'],
        help='Dataset to download (or "all" for all datasets)'
    )
    parser.add_argument(
        '--list',
        action='store_true',
        help='List downloaded datasets'
    )
    parser.add_argument(
        '--keep-compressed',
        action='store_true',
        help='Keep compressed .gz files after extraction'
    )
    
    args = parser.parse_args()
    
    if args.list:
        list_downloaded_datasets()
    elif args.dataset:
        if args.dataset == 'all':
            download_all_datasets(args.keep_compressed)
        else:
            download_dataset(args.dataset, args.keep_compressed)
    else:
        parser.print_help()
        print("\n" + "="*70)
        print("AVAILABLE DATASETS:")
        print("="*70)
        for name, info in DATASETS.items():
            print(f"\n{name}:")
            print(f"  Name:        {info['name']}")
            print(f"  Accession:   {info['accession']}")
            print(f"  Size:        ~{info['size_mb']} MB")
            print(f"  Description: {info['description']}")


if __name__ == '__main__':
    main()
