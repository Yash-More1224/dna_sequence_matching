# Wagner-Fischer Algorithm - Quick Setup Guide

## Installation Steps

### 1. Prerequisites Check
```bash
# Check Python version (need 3.8+)
python --version

# Check pip
pip --version
```

### 2. Create Virtual Environment
```bash
cd wagner-fischer
python -m venv venv
```

### 3. Activate Virtual Environment

**On Linux/Mac:**
```bash
source venv/bin/activate
```

**On Fish shell:**
```fish
source venv/bin/activate.fish
```

**On Windows:**
```cmd
venv\Scripts\activate
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

### 5. Verify Installation
```bash
# Run the demo
python demo.py

# Run tests
pytest tests/ -v

# Check help
python main.py --help
```

## Quick Start Commands

### 1. Run Demo
```bash
python demo.py
```

### 2. Compute Edit Distance
```bash
python main.py distance ATCG ATCG
python main.py distance ATCGATCG ATCGTTCG --show-alignment
```

### 3. Search for Pattern
```bash
python main.py search ATCG --text "GGATCGGGATCGAAA" --max-distance 2
```

### 4. Download E. coli Genome
```bash
python main.py data --download-ecoli
```

### 5. Generate Synthetic Test Data
```bash
python main.py data --generate-synthetic
```

### 6. Run Benchmarks
```bash
# Quick benchmark
python main.py benchmark --test-edit-distance --pattern-lengths 10 50 100

# Full benchmark suite (takes longer)
python main.py benchmark --full
```

### 7. Run Accuracy Tests
```bash
# Quick accuracy test
python main.py accuracy --test-exact --pattern-lengths 10 20

# Full accuracy evaluation
python main.py accuracy --full
```

### 8. Generate Visualizations
```bash
python main.py visualize \
    --benchmark-csv results/benchmarks/benchmark_results.csv \
    --accuracy-csv results/accuracy/accuracy_results.csv \
    --comparison
```

### 9. Run All Experiments (Fish shell)
```bash
./run_experiments.fish
```

## Project Structure

```
wagner-fischer/
├── README.md              # Main documentation
├── SETUP.md              # This file
├── requirements.txt      # Python dependencies
├── config.yaml           # Configuration
├── .gitignore           # Git ignore rules
│
├── main.py              # Main CLI
├── demo.py              # Demo script
├── wf_core.py           # Core algorithm
├── wf_search.py         # Pattern search
├── data_loader.py       # Data handling
├── benchmark.py         # Benchmarking
├── accuracy.py          # Accuracy tests
├── visualization.py     # Plotting
│
├── tests/               # Test suite
│   ├── test_wf_core.py
│   ├── test_search.py
│   └── test_integration.py
│
├── data/                # Datasets
├── results/             # Results
│   ├── benchmarks/
│   ├── accuracy/
│   └── plots/
│
└── venv/                # Virtual environment
```

## Common Issues

### 1. Import Errors
If you get import errors, make sure:
- Virtual environment is activated
- Dependencies are installed: `pip install -r requirements.txt`
- You're in the correct directory

### 2. Biopython Not Found
```bash
pip install biopython
```

### 3. Matplotlib/Seaborn Issues
```bash
pip install matplotlib seaborn
```

### 4. Memory Errors
If you run out of memory during benchmarks:
- Reduce text length: `--text-length 1000`
- Reduce iterations: `--iterations 5`
- Use smaller pattern lengths

### 5. Permission Denied
Make scripts executable:
```bash
chmod +x main.py demo.py run_experiments.fish
```

## Testing

### Run All Tests
```bash
pytest tests/ -v
```

### Run Specific Test File
```bash
pytest tests/test_wf_core.py -v
```

### Run with Coverage
```bash
pytest tests/ --cov=. --cov-report=html
```

## Development

### Adding New Features
1. Implement in appropriate module (e.g., `wf_core.py`)
2. Add tests in `tests/`
3. Update documentation
4. Run tests: `pytest`

### Code Style
Follow PEP 8 guidelines. Use docstrings for all functions.

## Getting Help

### Command Help
```bash
python main.py --help
python main.py distance --help
python main.py search --help
python main.py benchmark --help
```

### Documentation
See README.md for detailed documentation and examples.

## Next Steps

After setup:
1. ✅ Run `python demo.py` to see basic functionality
2. ✅ Run `python main.py data --generate-synthetic` to create test data
3. ✅ Run a small benchmark: `python main.py benchmark --test-edit-distance`
4. ✅ Explore the API in Python scripts
5. ✅ Run full experiments when ready

Good luck with your experiments!
