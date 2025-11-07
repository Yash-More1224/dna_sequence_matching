#!/usr/bin/env fish
# Comprehensive test script for Wagner-Fischer implementation
# Tests installation, functionality, and runs full test suite

set -l RED '\033[0;31m'
set -l GREEN '\033[0;32m'
set -l YELLOW '\033[1;33m'
set -l BLUE '\033[0;34m'
set -l NC '\033[0m' # No Color

function print_header
    echo ""
    echo "=========================================="
    echo $argv[1]
    echo "=========================================="
end

function print_success
    echo -e "$GREEN✓ $argv[1]$NC"
end

function print_error
    echo -e "$RED✗ $argv[1]$NC"
end

function print_info
    echo -e "$BLUE→ $argv[1]$NC"
end

function print_warning
    echo -e "$YELLOW⚠ $argv[1]$NC"
end

# Start testing
print_header "Wagner-Fischer Test Suite"

# Check if we're in the right directory
if not test -f "wf_core.py"
    print_error "Not in wagner-fischer directory!"
    exit 1
end

# Test 1: Check Python version
print_header "Test 1: Environment Check"
print_info "Checking Python version..."
set python_version (python3 --version 2>&1)
echo "  $python_version"

if test $status -eq 0
    print_success "Python 3 is available"
else
    print_error "Python 3 not found"
    exit 1
end

# Test 2: Check if virtual environment exists
print_info "Checking virtual environment..."
if test -d "venv"
    print_success "Virtual environment found"
    print_info "Activating virtual environment..."
    source venv/bin/activate.fish
else
    print_warning "Virtual environment not found"
    print_info "Creating virtual environment..."
    python3 -m venv venv
    source venv/bin/activate.fish
    print_success "Virtual environment created and activated"
end

# Test 3: Install/check dependencies
print_header "Test 2: Dependencies"
print_info "Installing/checking dependencies..."
pip install -q -r requirements.txt
if test $status -eq 0
    print_success "Dependencies installed"
else
    print_error "Failed to install dependencies"
    exit 1
end

# Test 4: Quick verification
print_header "Test 3: Quick Verification"
print_info "Running verification script..."
python3 verify.py
set verify_status $status

if test $verify_status -eq 0
    print_success "Verification passed"
else
    print_error "Verification failed"
    exit 1
end

# Test 5: Unit Tests - Core Algorithm
print_header "Test 4: Unit Tests - Core Algorithm"
print_info "Testing wf_core.py..."
pytest tests/test_wf_core.py -v --tb=short
set core_status $status

if test $core_status -eq 0
    print_success "Core algorithm tests passed"
else
    print_error "Core algorithm tests failed"
end

# Test 6: Unit Tests - Pattern Search
print_header "Test 5: Unit Tests - Pattern Search"
print_info "Testing wf_search.py..."
pytest tests/test_search.py -v --tb=short
set search_status $status

if test $search_status -eq 0
    print_success "Pattern search tests passed"
else
    print_error "Pattern search tests failed"
end

# Test 7: Integration Tests
print_header "Test 6: Integration Tests"
print_info "Testing integration..."
pytest tests/test_integration.py -v --tb=short
set integration_status $status

if test $integration_status -eq 0
    print_success "Integration tests passed"
else
    print_error "Integration tests failed"
end

# Test 8: All tests with coverage
print_header "Test 7: Full Test Suite with Coverage"
print_info "Running all tests with coverage..."
pytest tests/ -v --cov=. --cov-report=term-missing --cov-report=html --tb=short
set full_status $status

if test $full_status -eq 0
    print_success "Full test suite passed"
    print_info "Coverage report saved to htmlcov/index.html"
else
    print_error "Some tests failed"
end

# Test 9: Test CLI commands
print_header "Test 8: CLI Functionality"

print_info "Testing 'distance' command..."
python3 main.py distance ATCG ATCG > /dev/null 2>&1
if test $status -eq 0
    print_success "Distance command works"
else
    print_error "Distance command failed"
end

print_info "Testing 'search' command..."
python3 main.py search ATCG --text "GGATCGGGATCG" --max-distance 1 > /dev/null 2>&1
if test $status -eq 0
    print_success "Search command works"
else
    print_error "Search command failed"
end

# Test 10: Demo script
print_header "Test 9: Demo Script"
print_info "Running demo script..."
timeout 30s python3 demo.py > /dev/null 2>&1
if test $status -eq 0
    print_success "Demo script completed successfully"
else
    print_warning "Demo script failed or timed out"
end

# Test 11: Quick benchmark
print_header "Test 10: Quick Benchmark"
print_info "Running quick benchmark test..."
timeout 60s python3 main.py benchmark \
    --test-edit-distance \
    --pattern-lengths 10 20 \
    --text-length 100 \
    --iterations 2 \
    > /dev/null 2>&1
if test $status -eq 0
    print_success "Quick benchmark completed"
else
    print_warning "Benchmark test failed or timed out"
end

# Test 12: Data generation
print_header "Test 11: Data Generation"
print_info "Testing synthetic data generation..."
python3 main.py data --generate-synthetic --data-dir data > /dev/null 2>&1
if test $status -eq 0
    print_success "Data generation works"
    if test -f "data/synthetic_small.fasta"
        print_success "Synthetic data files created"
    end
else
    print_error "Data generation failed"
end

# Summary
print_header "Test Summary"

set total_tests 0
set passed_tests 0

# Count results
if test $verify_status -eq 0
    set passed_tests (math $passed_tests + 1)
end
set total_tests (math $total_tests + 1)

if test $core_status -eq 0
    set passed_tests (math $passed_tests + 1)
end
set total_tests (math $total_tests + 1)

if test $search_status -eq 0
    set passed_tests (math $passed_tests + 1)
end
set total_tests (math $total_tests + 1)

if test $integration_status -eq 0
    set passed_tests (math $passed_tests + 1)
end
set total_tests (math $total_tests + 1)

if test $full_status -eq 0
    set passed_tests (math $passed_tests + 1)
end
set total_tests (math $total_tests + 1)

echo ""
echo "Test Results: $passed_tests/$total_tests test suites passed"
echo ""

if test $passed_tests -eq $total_tests
    print_success "ALL TESTS PASSED! 🎉"
    echo ""
    echo "Your Wagner-Fischer implementation is working perfectly!"
    echo ""
    echo "Next steps:"
    echo "  • View coverage report: open htmlcov/index.html"
    echo "  • Run full benchmarks: python3 main.py benchmark --full"
    echo "  • Run accuracy tests: python3 main.py accuracy --full"
    echo "  • Try the demo: python3 demo.py"
    echo ""
    exit 0
else
    print_warning "Some test suites failed"
    echo ""
    echo "Please check the output above for details."
    echo "You may need to:"
    echo "  • Install missing dependencies: pip install -r requirements.txt"
    echo "  • Check Python version (need 3.8+)"
    echo "  • Review error messages above"
    echo ""
    exit 1
end
