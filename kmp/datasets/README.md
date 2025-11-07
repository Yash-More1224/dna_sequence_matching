# KMP Datasets

This directory contains genomic datasets used for benchmarking and testing the KMP algorithm.

## Available Datasets

### E. coli K-12 MG1655
- **Accession**: NC_000913.3
- **Size**: ~4.6 MB
- **Description**: Complete genome of Escherichia coli K-12 substr. MG1655
- **Use**: Primary benchmark dataset

### Lambda Phage
- **Accession**: NC_001416.1
- **Size**: ~48 KB
- **Description**: Escherichia phage Lambda complete genome
- **Use**: Small genome for quick testing

### Salmonella enterica
- **Accession**: NC_003197.2
- **Size**: ~4.8 MB
- **Description**: Salmonella enterica subsp. enterica serovar Typhimurium str. LT2
- **Use**: Alternative bacterial genome for comparison

### Bacillus subtilis
- **Accession**: NC_000964.3
- **Size**: ~4.2 MB
- **Description**: Bacillus subtilis subsp. subtilis str. 168 complete genome
- **Use**: Alternative bacterial genome for comparison

## Downloading Datasets

### Using the download script:

```bash
# Download a specific dataset
python -m kmp.datasets.download_datasets --dataset ecoli

# Download all datasets
python -m kmp.datasets.download_datasets --dataset all

# List downloaded datasets
python -m kmp.datasets.download_datasets --list

# Keep compressed files after extraction
python -m kmp.datasets.download_datasets --dataset ecoli --keep-compressed
```

### Using the CLI:

```bash
# Download using the main CLI
python -m kmp.cli download --dataset ecoli
```

## Dataset Format

All datasets are stored in FASTA format (`.fna` or `.fasta` extension).

**FASTA Format Example:**
```
>NC_000913.3 Escherichia coli str. K-12 substr. MG1655, complete genome
AGCTTTTCATTCTGACTGCAACGGGCAATATGTCTCTGTGTGGATTAAAAAAAGAGTGTCTGATAGCAGC
TTCTGAACTGGTTACCTGCCGTGAGTAAATTAAAATTTTATTGACTTAGGTCACTAAATACTTTAACCAA
...
```

## File Naming Convention

- Compressed: `{dataset_name}.fna.gz`
- Decompressed: `{dataset_name}.fna`

## Storage Requirements

Total storage required for all datasets: ~15 MB (uncompressed)

## Data Sources

All datasets are downloaded from NCBI (National Center for Biotechnology Information):
- https://www.ncbi.nlm.nih.gov/genome/

## License

Genomic data from NCBI is in the public domain and freely available for research use.
