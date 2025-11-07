"""
Integration tests for Wagner-Fischer implementation.
"""

import pytest
import os
from pathlib import Path
import tempfile

from wf_core import WagnerFischer
from wf_search import PatternSearcher
from data_loader import SyntheticDataGenerator, FastaLoader
from benchmark import PerformanceBenchmark
from accuracy import AccuracyEvaluator


class TestEndToEnd:
    """End-to-end integration tests."""
    
    def test_synthetic_workflow(self):
        """Test complete workflow with synthetic data."""
        # Generate data
        generator = SyntheticDataGenerator(seed=42)
        text = generator.generate_random_sequence(1000)
        pattern = generator.generate_random_sequence(20)
        
        # Search
        searcher = PatternSearcher(max_distance=2)
        matches = searcher.search(pattern, text)
        
        # Verify results
        assert isinstance(matches, list)
        assert all(isinstance(m.position, int) for m in matches)
    
    def test_mutation_workflow(self):
        """Test workflow with mutated sequences."""
        generator = SyntheticDataGenerator(seed=42)
        
        # Create sequence and mutate it
        original = generator.generate_random_sequence(100)
        mutated, mutations = generator.mutate_sequence(
            original,
            substitution_rate=0.02
        )
        
        # Compute distance
        wf = WagnerFischer()
        distance, _ = wf.compute_distance(original, mutated)
        
        # Distance should be related to mutations
        assert distance >= 0
    
    def test_fasta_loading_and_search(self):
        """Test loading FASTA and searching."""
        # Create temporary FASTA file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.fasta', delete=False) as f:
            f.write(">test_seq\n")
            f.write("ATCGATCGATCG\n")
            temp_path = f.name
        
        try:
            # Load
            loader = FastaLoader()
            sequences = loader.load(temp_path)
            
            assert len(sequences) == 1
            assert sequences[0].id == "test_seq"
            
            # Search in loaded sequence
            searcher = PatternSearcher(max_distance=1)
            matches = searcher.search("ATCG", sequences[0].sequence)
            
            assert len(matches) >= 1
        
        finally:
            os.unlink(temp_path)


class TestBenchmarkIntegration:
    """Integration tests for benchmarking."""
    
    def test_benchmark_runs(self):
        """Test that benchmarks run without errors."""
        with tempfile.TemporaryDirectory() as tmpdir:
            benchmark = PerformanceBenchmark(output_dir=tmpdir)
            
            # Run small benchmark
            benchmark.benchmark_edit_distance(
                pattern_lengths=[10, 20],
                text_length=100,
                iterations=2
            )
            
            assert len(benchmark.results) > 0
    
    def test_benchmark_saves_results(self):
        """Test that benchmark results are saved."""
        with tempfile.TemporaryDirectory() as tmpdir:
            benchmark = PerformanceBenchmark(output_dir=tmpdir)
            
            benchmark.benchmark_edit_distance(
                pattern_lengths=[10],
                text_length=100,
                iterations=2
            )
            
            benchmark.save_results()
            
            # Check files exist
            csv_file = Path(tmpdir) / "benchmark_results.csv"
            json_file = Path(tmpdir) / "benchmark_results.json"
            
            assert csv_file.exists()
            assert json_file.exists()


class TestAccuracyIntegration:
    """Integration tests for accuracy evaluation."""
    
    def test_accuracy_evaluation_runs(self):
        """Test that accuracy evaluation runs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            evaluator = AccuracyEvaluator(output_dir=tmpdir)
            
            # Run small evaluation
            metrics = evaluator.evaluate_exact_matching(
                pattern_lengths=[10, 20],
                text_length=100,
                num_tests=5
            )
            
            assert len(metrics) > 0
            assert all(m.precision >= 0 for m in metrics)
    
    def test_accuracy_saves_results(self):
        """Test that accuracy results are saved."""
        with tempfile.TemporaryDirectory() as tmpdir:
            evaluator = AccuracyEvaluator(output_dir=tmpdir)
            
            evaluator.evaluate_exact_matching(
                pattern_lengths=[10],
                text_length=100,
                num_tests=5
            )
            
            evaluator.save_results()
            
            # Check files exist
            csv_file = Path(tmpdir) / "accuracy_results.csv"
            assert csv_file.exists()


class TestDataGeneration:
    """Test data generation utilities."""
    
    def test_generate_dataset(self):
        """Test generating complete dataset."""
        generator = SyntheticDataGenerator(seed=42)
        
        dataset = generator.generate_dataset(
            num_sequences=10,
            seq_length=100,
            mutation_rate=0.01
        )
        
        assert len(dataset) == 10
        assert all(len(pair) == 2 for pair in dataset)
    
    def test_save_and_load_fasta(self):
        """Test saving and loading FASTA."""
        generator = SyntheticDataGenerator(seed=42)
        loader = FastaLoader()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Generate and save
            sequences = [
                ("seq1", generator.generate_random_sequence(100)),
                ("seq2", generator.generate_random_sequence(100))
            ]
            
            fasta_path = os.path.join(tmpdir, "test.fasta")
            generator.save_fasta(sequences, fasta_path)
            
            # Load
            loaded = loader.load(fasta_path)
            
            assert len(loaded) == 2
            assert loaded[0].id == "seq1"
            assert loaded[1].id == "seq2"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
