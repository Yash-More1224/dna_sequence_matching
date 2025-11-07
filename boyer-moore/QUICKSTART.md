# Boyer-Moore Algorithm: Quick Start Guide

## 🚀 Get Started in 5 Minutes

### Step 1: Install Dependencies

```bash
cd boyer-moore
pip install -r requirements.txt
```

### Step 2: Run a Simple Test

```bash
cd src
python boyer_moore.py
```

Expected output:
```
Text: GCATCGCAGAGAGTATACAGTACG
Pattern: GCAGAGAG

Matches found at positions: [6]
Statistics: {'comparisons': 34, 'shifts': 8, 'pattern_length': 8}
```

### Step 3: Run All Experiments

```bash
python run_experiments.py
```

This will:
- ✅ Download E. coli genome (~5 MB)
- ✅ Run 8 comprehensive experiments
- ✅ Generate 8 visualization plots
- ✅ Create detailed analysis report

**Time:** ~10-15 minutes depending on your system

### Step 4: View Results

```bash
# View plots
ls results/plots/

# Read analysis report
cat results/reports/ANALYSIS_REPORT.md
```

## 📋 Quick Commands

### Run Specific Experiments

```bash
# Run only experiments 1 and 4
python run_experiments.py --experiments 1 4
```

### Run Tests

```bash
cd tests
python test_boyer_moore.py
```

### Generate Only Visualizations

```bash
cd visualization
python visualizations.py
```

### Generate Only Report

```bash
cd visualization
python report_generator.py
```

## 💡 Common Use Cases

### Use Case 1: Find a Motif in E. coli

```python
from src.data_loader import DatasetManager
from src.boyer_moore import BoyerMoore

# Load genome
manager = DatasetManager()
genome = manager.load_ecoli_genome()

# Search for Pribnow box (TATAAT)
matcher = BoyerMoore("TATAAT")
matches = matcher.search(genome)

print(f"Found {len(matches)} Pribnow boxes")
```

### Use Case 2: Compare Algorithm Variants

```python
from src.boyer_moore_variants import get_variant
from experiments.benchmarks import Benchmarker

text = "ACGT" * 10000
pattern = "ACGTACGT"

benchmarker = Benchmarker()

for variant in ['full', 'bcr_only', 'gsr_only', 'horspool']:
    matcher = get_variant(pattern, variant)
    result = benchmarker.benchmark_boyer_moore(matcher, text)
    print(f"{variant}: {result.total_time*1000:.2f} ms")
```

### Use Case 3: Generate Test Data

```python
from src.data_generator import DNAGenerator

gen = DNAGenerator(seed=42)

# Create synthetic sequence with known matches
text, pattern, positions = gen.generate_test_case(
    text_length=50000,
    pattern_length=20,
    num_occurrences=10
)

print(f"Pattern: {pattern}")
print(f"Positions: {positions}")
```

## 🔧 Troubleshooting

### Issue: Import Errors

**Solution:**
```bash
# Make sure you're in the correct directory
cd boyer-moore

# Install dependencies
pip install -r requirements.txt
```

### Issue: E. coli Genome Download Fails

**Solution:**
```python
# Manually download from NCBI
# URL in config.yaml under datasets.ecoli.url
# Place in datasets/ecoli_genome.fasta
```

### Issue: Tests Failing

**Solution:**
```bash
# Ensure you're running from tests directory
cd tests
python test_boyer_moore.py
```

### Issue: Plots Not Generating

**Solution:**
```bash
# Check if matplotlib backend is configured
# If on headless server, set:
export MPLBACKEND=Agg

# Then run experiments
python run_experiments.py
```

## 📊 Expected Results

After running all experiments, you should see:

### Files Created
- ✅ 8 JSON result files in `results/tables/`
- ✅ 8 PNG plot files in `results/plots/`
- ✅ 2 Markdown reports in `results/reports/`
- ✅ E. coli genome in `datasets/`

### Typical Performance (on modern hardware)
- Pattern matching: ~10-50 ms for 1MB text
- Full experiments: ~10-15 minutes total
- Memory usage: < 100 MB for most tests

## 🎯 Next Steps

1. **Read the full README:** `README.md`
2. **Examine experiment results:** `results/reports/ANALYSIS_REPORT.md`
3. **Explore the code:** Start with `src/boyer_moore.py`
4. **Modify experiments:** Edit `config.yaml` to customize
5. **Run your own patterns:** Use the API examples above

## 📚 Additional Resources

- **Algorithm Paper:** Boyer & Moore (1977) - "A Fast String Searching Algorithm"
- **E. coli Genome:** NCBI RefSeq NC_000913.3
- **Python PEP 8:** Style guide followed throughout

## ✅ Verification Checklist

Before submitting or presenting:

- [ ] All dependencies installed
- [ ] Tests pass successfully
- [ ] All 8 experiments completed
- [ ] Visualizations generated
- [ ] Report created
- [ ] Results files present in `results/`
- [ ] Code follows PEP 8 style
- [ ] README reviewed

---

**Questions?** Review the main README.md for detailed documentation.

**Ready to go!** Run: `python run_experiments.py`
