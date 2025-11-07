#!/usr/bin/env python3
"""
Quick verification script to test Wagner-Fischer implementation.
Run this to verify everything is working correctly.
"""

import sys

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    try:
        import wf_core
        import wf_search
        import data_loader
        import benchmark
        import accuracy
        import visualization
        print("✓ All modules imported successfully")
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False

def test_basic_functionality():
    """Test basic Wagner-Fischer functionality."""
    print("\nTesting basic functionality...")
    try:
        from wf_core import levenshtein_distance, WagnerFischer
        
        # Test 1: Identical strings
        d1 = levenshtein_distance("ATCG", "ATCG")
        assert d1 == 0, f"Expected 0, got {d1}"
        print("✓ Identical strings test passed")
        
        # Test 2: Single substitution
        d2 = levenshtein_distance("ATCG", "TTCG")
        assert d2 == 1, f"Expected 1, got {d2}"
        print("✓ Single substitution test passed")
        
        # Test 3: Traceback
        wf = WagnerFischer()
        distance, operations = wf.compute_with_traceback("AT", "TT")
        assert distance == 1, f"Expected distance 1, got {distance}"
        assert len(operations) > 0, "Expected operations"
        print("✓ Traceback test passed")
        
        return True
    except Exception as e:
        print(f"✗ Functionality test failed: {e}")
        return False

def test_pattern_search():
    """Test pattern search functionality."""
    print("\nTesting pattern search...")
    try:
        from wf_search import PatternSearcher
        
        searcher = PatternSearcher(max_distance=1)
        matches = searcher.search("AT", "GGATGG")
        
        assert isinstance(matches, list), "Expected list of matches"
        print(f"✓ Pattern search test passed (found {len(matches)} matches)")
        
        return True
    except Exception as e:
        print(f"✗ Pattern search test failed: {e}")
        return False

def test_data_generation():
    """Test synthetic data generation."""
    print("\nTesting data generation...")
    try:
        from data_loader import SyntheticDataGenerator
        
        gen = SyntheticDataGenerator(seed=42)
        seq = gen.generate_random_sequence(100)
        
        assert len(seq) == 100, f"Expected length 100, got {len(seq)}"
        assert all(c in 'ATCG' for c in seq), "Invalid characters in sequence"
        print("✓ Data generation test passed")
        
        return True
    except Exception as e:
        print(f"✗ Data generation test failed: {e}")
        return False

def main():
    """Run all verification tests."""
    print("=" * 60)
    print("Wagner-Fischer Implementation Verification")
    print("=" * 60)
    
    tests = [
        test_imports,
        test_basic_functionality,
        test_pattern_search,
        test_data_generation
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"✗ Test crashed: {e}")
            results.append(False)
    
    print("\n" + "=" * 60)
    passed = sum(results)
    total = len(results)
    
    if all(results):
        print(f"✅ ALL TESTS PASSED ({passed}/{total})")
        print("=" * 60)
        print("\nYour Wagner-Fischer implementation is working correctly!")
        print("\nNext steps:")
        print("  1. Run: python3 demo.py")
        print("  2. Run: pytest tests/ -v")
        print("  3. Run: python3 main.py benchmark --full")
        return 0
    else:
        print(f"❌ SOME TESTS FAILED ({passed}/{total} passed)")
        print("=" * 60)
        print("\nPlease check:")
        print("  1. All dependencies installed: pip install -r requirements.txt")
        print("  2. You're in the correct directory")
        print("  3. Python version is 3.8+")
        return 1

if __name__ == "__main__":
    sys.exit(main())
