# Wagner-Fischer Project File Analysis

## USED FILES (Keep These)

### Core Implementation Files
- ✅ `wf_core.py` - Core Wagner-Fischer algorithm implementation
- ✅ `wf_search.py` - Pattern search using WF algorithm
- ✅ `data_loader.py` - FASTA file loading utilities
- ✅ `requirements.txt` - Python dependencies

### Scripts Directory (NEW - ACTIVELY USED)
- ✅ `scripts/wf.py` - Complete WF implementation with all variants
- ✅ `scripts/fast_benchmark.py` - Fast synthetic benchmark (MAIN BENCHMARK)
- ✅ `scripts/run_comprehensive_eval.py` - Main evaluation runner (ACTIVELY USED)
- ✅ `scripts/generate_plots.py` - Plot generation script (ACTIVELY USED)
- ✅ `scripts/README.md` - Scripts documentation

### Documentation (Keep)
- ✅ `README.md` - Main project documentation
- ✅ `EVALUATION_SUMMARY.md` - Evaluation summary (NEWLY CREATED)
- ✅ `PROJECT_SUMMARY.md` - Project overview

### Results Directory (Keep All)
- ✅ `results/benchmarks/*.json` - Benchmark data (12 files)
- ✅ `results/benchmarks/*.csv` - CSV exports (12 files)
- ✅ `results/reports/wf_evaluation_report_*.txt` - Final report (REQUIRED OUTPUT)
- ✅ `results/plots/*.png` - All visualization plots (7 files)

### Tests (Keep for verification)
- ✅ `tests/__init__.py`
- ✅ `tests/test_wf_core.py`
- ✅ `tests/test_search.py`
- ✅ `tests/test_integration.py`

### Configuration
- ✅ `config.yaml` - Configuration file
- ✅ `.gitignore` - Git ignore rules

---

## POTENTIALLY UNUSED FILES (Can Delete)

### Duplicate/Old Scripts in Root (scripts/ has better versions)
- ❌ `benchmark.py` - Root version (scripts/fast_benchmark.py is better)
- ❌ `accuracy.py` - Old accuracy script (now in scripts/fast_benchmark.py)
- ❌ `visualization.py` - Old viz script (scripts/generate_plots.py is better)

### Old/Unused Utilities
- ⚠️ `demo.py` - Demo script (might be useful for demos, but not required)
- ⚠️ `main.py` - Old main script (scripts/run_comprehensive_eval.py is used instead)
- ⚠️ `verify.py` - Verification script (not used in current workflow)

### Unused Generated in Scripts
- ❌ `scripts/benchmark.py` - Incomplete/slow version (fast_benchmark.py is used)
- ❌ `scripts/generate_ground_truth.py` - Not used (using synthetic data instead)
- ❌ `scripts/generate_visualizations.py` - Not used (generate_plots.py is used)
- ❌ `scripts/run_full_evaluation.py` - Not used (run_comprehensive_eval.py is used)

### Data Directory
- ⚠️ `data/` - May contain test data (check if needed)

---

## RECOMMENDATION

### DELETE THESE (Safe to Remove)
```
Root directory duplicates:
- benchmark.py (duplicate of scripts/fast_benchmark.py)
- accuracy.py (integrated into scripts/fast_benchmark.py)
- visualization.py (replaced by scripts/generate_plots.py)

Unused scripts:
- scripts/benchmark.py (slow version, not used)
- scripts/generate_ground_truth.py (not used - using synthetic data)
- scripts/generate_visualizations.py (not used - using generate_plots.py)
- scripts/run_full_evaluation.py (not used - using run_comprehensive_eval.py)
```

### OPTIONALLY DELETE (Check First)
```
- demo.py (keep if you want demo capability)
- main.py (keep if you want old interface)
- verify.py (keep if you want verification tests)
- data/ (check contents first)
```

### KEEP EVERYTHING ELSE
All files in:
- results/ (all subdirectories)
- tests/
- Core implementation files
- Documentation files
- Active scripts in scripts/

---

## FILE COUNT SUMMARY

### Current State
- Total Python files: ~38
- Active scripts: 5 (in scripts/)
- Test files: 4
- Documentation: 3
- Result files: ~31 (JSON, CSV, PNG, TXT)

### After Cleanup (Recommended)
- Remove: 7 files (duplicates and unused)
- Keep: ~31 files (core + active + results)

---

## USED vs UNUSED BREAKDOWN

### ✅ CURRENTLY USED IN WORKFLOW
1. `scripts/run_comprehensive_eval.py` - Main runner
2. `scripts/fast_benchmark.py` - Benchmarking
3. `scripts/generate_plots.py` - Visualization
4. `scripts/wf.py` - Algorithm implementation
5. `wf_core.py` - Core algorithm
6. `wf_search.py` - Search functionality
7. `data_loader.py` - Data loading
8. All files in `results/`

### ❌ NOT USED IN CURRENT WORKFLOW
1. Root: `benchmark.py`, `accuracy.py`, `visualization.py`
2. Scripts: `benchmark.py`, `generate_ground_truth.py`, `generate_visualizations.py`, `run_full_evaluation.py`
3. Root: `demo.py`, `main.py`, `verify.py` (optional utilities)

---

## COMMANDS TO CLEAN UP

```bash
cd /home/shrish-kadam/Documents/SEM3/AAD/dna_sequence_matching/wagner-fischer

# Remove duplicate root files
rm -f benchmark.py accuracy.py visualization.py

# Remove unused scripts
rm -f scripts/benchmark.py
rm -f scripts/generate_ground_truth.py
rm -f scripts/generate_visualizations.py
rm -f scripts/run_full_evaluation.py

# Optional: Remove old utilities (uncomment if desired)
# rm -f demo.py main.py verify.py
```

This will reduce clutter while keeping all essential files for the project to run.
