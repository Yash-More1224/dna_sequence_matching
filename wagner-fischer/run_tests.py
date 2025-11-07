#!/usr/bin/env python3
"""
Comprehensive test runner for Wagner-Fischer implementation.
Runs all tests and generates detailed reports.
"""

import sys
import subprocess
import os
from pathlib import Path
import time


class Colors:
    """ANSI color codes."""
    RED = '\033[0;31m'
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    BLUE = '\033[0;34m'
    PURPLE = '\033[0;35m'
    CYAN = '\033[0;36m'
    NC = '\033[0m'  # No Color


def print_header(text):
    """Print section header."""
    print(f"\n{'='*60}")
    print(text)
    print('='*60)


def print_success(text):
    """Print success message."""
    print(f"{Colors.GREEN}✓ {text}{Colors.NC}")


def print_error(text):
    """Print error message."""
    print(f"{Colors.RED}✗ {text}{Colors.NC}")


def print_info(text):
    """Print info message."""
    print(f"{Colors.BLUE}→ {text}{Colors.NC}")


def print_warning(text):
    """Print warning message."""
    print(f"{Colors.YELLOW}⚠ {text}{Colors.NC}")


def run_command(cmd, timeout=None):
    """Run a command and return status."""
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        return result.returncode == 0, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return False, "", "Command timed out"
    except Exception as e:
        return False, "", str(e)


def test_environment():
    """Test Python environment."""
    print_header("Test 1: Environment Check")
    
    # Check Python version
    print_info("Checking Python version...")
    success, stdout, _ = run_command("python3 --version")
    if success:
        print(f"  {stdout.strip()}")
        print_success("Python 3 available")
    else:
        print_error("Python 3 not found")
        return False
    
    # Check if we're in the right directory
    if not Path("wf_core.py").exists():
        print_error("Not in wagner-fischer directory!")
        return False
    
    print_success("Environment check passed")
    return True


def test_dependencies():
    """Test if dependencies are installed."""
    print_header("Test 2: Dependencies")
    
    print_info("Checking dependencies...")
    
    required_packages = [
        'numpy', 'pandas', 'matplotlib', 'seaborn',
        'pytest', 'biopython', 'pyyaml', 'psutil'
    ]
    
    all_installed = True
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"  ✓ {package}")
        except ImportError:
            print(f"  ✗ {package} - NOT INSTALLED")
            all_installed = False
    
    if all_installed:
        print_success("All dependencies installed")
        return True
    else:
        print_warning("Some dependencies missing")
        print_info("Installing missing dependencies...")
        success, _, _ = run_command("pip install -q -r requirements.txt")
        if success:
            print_success("Dependencies installed")
            return True
        else:
            print_error("Failed to install dependencies")
            return False


def test_verification():
    """Run verification script."""
    print_header("Test 3: Quick Verification")
    
    print_info("Running verification script...")
    success, stdout, stderr = run_command("python3 verify.py", timeout=30)
    
    if success:
        print(stdout)
        print_success("Verification passed")
        return True
    else:
        print(stderr)
        print_error("Verification failed")
        return False


def test_unit_tests():
    """Run unit tests."""
    print_header("Test 4: Unit Tests")
    
    tests = [
        ("Core Algorithm", "tests/test_wf_core.py"),
        ("Pattern Search", "tests/test_search.py"),
        ("Integration", "tests/test_integration.py"),
    ]
    
    results = []
    
    for name, test_file in tests:
        print_info(f"Testing {name}...")
        success, stdout, stderr = run_command(
            f"pytest {test_file} -v --tb=short",
            timeout=60
        )
        
        if success:
            print_success(f"{name} tests passed")
            results.append(True)
        else:
            print_error(f"{name} tests failed")
            print(stderr[-500:] if len(stderr) > 500 else stderr)  # Last 500 chars
            results.append(False)
    
    return all(results)


def test_coverage():
    """Run tests with coverage."""
    print_header("Test 5: Test Coverage")
    
    print_info("Running tests with coverage...")
    success, stdout, stderr = run_command(
        "pytest tests/ -v --cov=. --cov-report=term --cov-report=html --tb=short",
        timeout=120
    )
    
    if success:
        print(stdout[-1000:] if len(stdout) > 1000 else stdout)  # Last 1000 chars
        print_success("Coverage report generated")
        print_info("View detailed report: htmlcov/index.html")
        return True
    else:
        print_error("Coverage test failed")
        print(stderr[-500:] if len(stderr) > 500 else stderr)
        return False


def test_cli():
    """Test CLI commands."""
    print_header("Test 6: CLI Functionality")
    
    commands = [
        ("distance", "python3 main.py distance ATCG ATCG"),
        ("search", "python3 main.py search ATCG --text 'GGATCGGGATCG' --max-distance 1"),
        ("help", "python3 main.py --help"),
    ]
    
    results = []
    
    for name, cmd in commands:
        print_info(f"Testing '{name}' command...")
        success, _, _ = run_command(cmd, timeout=10)
        
        if success:
            print_success(f"'{name}' command works")
            results.append(True)
        else:
            print_error(f"'{name}' command failed")
            results.append(False)
    
    return all(results)


def test_demo():
    """Test demo script."""
    print_header("Test 7: Demo Script")
    
    print_info("Running demo script...")
    success, stdout, _ = run_command("python3 demo.py", timeout=30)
    
    if success:
        print_success("Demo script completed")
        return True
    else:
        print_warning("Demo script failed or timed out")
        return False


def test_quick_benchmark():
    """Run quick benchmark."""
    print_header("Test 8: Quick Benchmark")
    
    print_info("Running quick benchmark (this may take a moment)...")
    success, _, _ = run_command(
        "python3 main.py benchmark --test-edit-distance "
        "--pattern-lengths 10 20 --text-length 100 --iterations 2",
        timeout=60
    )
    
    if success:
        print_success("Quick benchmark completed")
        return True
    else:
        print_warning("Benchmark failed or timed out")
        return False


def test_data_generation():
    """Test data generation."""
    print_header("Test 9: Data Generation")
    
    print_info("Testing synthetic data generation...")
    success, _, _ = run_command(
        "python3 main.py data --generate-synthetic --data-dir data",
        timeout=30
    )
    
    if success:
        if Path("data/synthetic_small.fasta").exists():
            print_success("Data generation works and files created")
            return True
        else:
            print_warning("Command succeeded but files not found")
            return False
    else:
        print_error("Data generation failed")
        return False


def generate_summary(results):
    """Generate test summary."""
    print_header("Test Summary")
    
    total = len(results)
    passed = sum(results.values())
    
    print(f"\nTotal Test Suites: {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {total - passed}")
    print(f"Success Rate: {passed/total*100:.1f}%\n")
    
    print("Detailed Results:")
    for test_name, result in results.items():
        status = f"{Colors.GREEN}PASS{Colors.NC}" if result else f"{Colors.RED}FAIL{Colors.NC}"
        print(f"  {test_name}: {status}")
    
    print()
    
    if passed == total:
        print_success("ALL TESTS PASSED! 🎉")
        print("\nYour Wagner-Fischer implementation is working perfectly!")
        print("\nNext steps:")
        print("  • View coverage: open htmlcov/index.html")
        print("  • Run benchmarks: python3 main.py benchmark --full")
        print("  • Run accuracy tests: python3 main.py accuracy --full")
        print("  • Try the demo: python3 demo.py")
        return True
    else:
        print_warning("Some tests failed")
        print("\nPlease review the error messages above.")
        print("You may need to:")
        print("  • Install dependencies: pip install -r requirements.txt")
        print("  • Check Python version (need 3.8+)")
        print("  • Review specific test failures")
        return False


def main():
    """Main test runner."""
    print_header("Wagner-Fischer Comprehensive Test Suite")
    print(f"Started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    start_time = time.time()
    results = {}
    
    # Run all tests
    results["Environment"] = test_environment()
    if not results["Environment"]:
        print_error("Environment check failed. Cannot continue.")
        return 1
    
    results["Dependencies"] = test_dependencies()
    results["Verification"] = test_verification()
    results["Unit Tests"] = test_unit_tests()
    results["Coverage"] = test_coverage()
    results["CLI"] = test_cli()
    results["Demo"] = test_demo()
    results["Benchmark"] = test_quick_benchmark()
    results["Data Generation"] = test_data_generation()
    
    # Generate summary
    elapsed = time.time() - start_time
    print(f"\nTotal time: {elapsed:.2f} seconds")
    
    success = generate_summary(results)
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
