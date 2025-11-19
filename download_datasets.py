#!/usr/bin/env python3
"""
Download DNA sequence datasets for KMP algorithm testing.

This script downloads E. coli genome, Lambda phage, and Salmonella genome
from NCBI and saves them to the dataset folder.
"""

import os
import gzip
import shutil
from pathlib import Path
import urllib.request
from urllib.error import URLError
import time


# Dataset URLs
DATASETS = {
    'ecoli': {
        'name': 'Escherichia coli K-12 MG1655',
        'url': 'https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/005/845/GCF_000005845.2_ASM584v2/GCF_000005845.2_ASM584v2_genomic.fna.gz',
        'output': 'ecoli_k12_mg1655.fasta',
        'description': 'E. coli K-12 substr. MG1655 complete genome (~4.6 MB)'
    },
    'lambda_phage': {
        'name': 'Enterobacteria phage lambda',
        'url': 'https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/840/245/GCF_000840245.1_ViralProj14204/GCF_000840245.1_ViralProj14204_genomic.fna.gz',
        'output': 'lambda_phage.fasta',
        'description': 'Lambda phage complete genome (~48 KB)'
    },
    'salmonella': {
        'name': 'Salmonella enterica',
        'url': 'https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/006/945/GCF_000006945.2_ASM694v2/GCF_000006945.2_ASM694v2_genomic.fna.gz',
        'output': 'salmonella_typhimurium.fasta',
        'description': 'Salmonella enterica serovar Typhimurium complete genome (~4.8 MB)'
    }
}


def download_file(url: str, output_path: Path, description: str = "") -> bool:
    """
    Download a file from URL to output path.
    
    Args:
        url: URL to download from
        output_path: Path to save the file
        description: Description for progress message
        
    Returns:
        True if successful, False otherwise
    """
    try:
        print(f"\nDownloading {description}...")
        print(f"URL: {url}")
        print(f"Saving to: {output_path}")
        
        # Download with progress
        def reporthook(count, block_size, total_size):
            if total_size > 0:
                percent = int(count * block_size * 100 / total_size)
                mb_downloaded = count * block_size / (1024 * 1024)
                mb_total = total_size / (1024 * 1024)
                print(f"\rProgress: {percent}% ({mb_downloaded:.2f}/{mb_total:.2f} MB)", end='')
        
        # Create temporary file path
        temp_path = output_path.parent / f"{output_path.name}.tmp.gz"
        
        # Download
        urllib.request.urlretrieve(url, temp_path, reporthook)
        print()  # New line after progress
        
        # Decompress if gzipped
        if temp_path.suffix == '.gz':
            print("Decompressing...")
            with gzip.open(temp_path, 'rb') as f_in:
                with open(output_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            
            # Remove temporary compressed file
            temp_path.unlink()
        else:
            # Just rename if not compressed
            temp_path.rename(output_path)
        
        # Get file size
        size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"✓ Successfully downloaded ({size_mb:.2f} MB)")
        
        return True
        
    except URLError as e:
        print(f"✗ Error downloading: {e}")
        return False
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return False


def main():
    """Download all datasets."""
    print("="*70)
    print("DNA Sequence Dataset Downloader")
    print("="*70)
    
    # Create dataset directory
    dataset_dir = Path(__file__).parent / "dataset"
    dataset_dir.mkdir(exist_ok=True)
    print(f"\nDataset directory: {dataset_dir}")
    
    # Track successes
    successful = []
    failed = []
    
    # Download each dataset
    for key, info in DATASETS.items():
        output_path = dataset_dir / info['output']
        
        # Check if file already exists
        if output_path.exists():
            size_mb = output_path.stat().st_size / (1024 * 1024)
            print(f"\n{info['name']}: Already exists ({size_mb:.2f} MB)")
            print(f"  Location: {output_path}")
            response = input("  Redownload? (y/N): ").strip().lower()
            if response != 'y':
                successful.append(key)
                continue
        
        # Download
        success = download_file(
            url=info['url'],
            output_path=output_path,
            description=info['name']
        )
        
        if success:
            successful.append(key)
        else:
            failed.append(key)
        
        # Small delay between downloads
        if key != list(DATASETS.keys())[-1]:  # Not the last one
            time.sleep(1)
    
    # Summary
    print("\n" + "="*70)
    print("Download Summary")
    print("="*70)
    print(f"✓ Successful: {len(successful)}/{len(DATASETS)}")
    for key in successful:
        print(f"  - {DATASETS[key]['name']}")
    
    if failed:
        print(f"\n✗ Failed: {len(failed)}/{len(DATASETS)}")
        for key in failed:
            print(f"  - {DATASETS[key]['name']}")
    
    print("\n" + "="*70)
    print("Dataset Information:")
    print("="*70)
    for key, info in DATASETS.items():
        output_path = dataset_dir / info['output']
        if output_path.exists():
            size_mb = output_path.stat().st_size / (1024 * 1024)
            print(f"\n{info['name']}:")
            print(f"  File: {info['output']}")
            print(f"  Size: {size_mb:.2f} MB")
            print(f"  Path: {output_path}")
            print(f"  Description: {info['description']}")
    
    print("\n" + "="*70)
    print("Downloads complete!")
    print("="*70)


if __name__ == "__main__":
    main()
