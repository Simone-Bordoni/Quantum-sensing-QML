# Continuous Integration (CI) Documentation

This project uses GitHub Actions for automated testing and continuous integration.

## Workflows

### 1. Test Installation and Run Tests (`.github/workflows/test-windows.yml`)

**Trigger:** Push to `main` or `develop` branches, pull requests, manual dispatch

**Purpose:** Primary Windows testing workflow

**Jobs:**

#### test-windows
- **Platform:** Windows (latest)
- **Python:** 3.13
- **Steps:**
  1. Checkout repository
  2. Set up Python 3.13
  3. Upgrade pip
  4. Install package with test dependencies: `pip install -e ".[test]"`
  5. Verify installation (qsopt, JAX, QuTiP versions)
  6. Run installation tests (excluding fresh environment tests)
  7. Run gate tests
  8. Run circuit tests
  9. Run all other tests
  10. Generate coverage report
  11. Upload coverage artifacts

#### test-installation-fresh-env
- **Platform:** Windows (latest)
- **Depends on:** test-windows
- **Purpose:** Test installation in a completely fresh virtual environment
- **Steps:**
  1. Create new virtual environment
  2. Install package
  3. Run fresh environment installation tests
  4. Automatically clean up test environment

### 2. Comprehensive CI Tests (`.github/workflows/ci.yml`)

**Trigger:** Push to `main` or `develop` branches, pull requests, manual dispatch

**Purpose:** Multi-platform testing and code quality checks

**Jobs:**

#### lint-and-format-check
- **Platform:** Ubuntu (latest)
- **Purpose:** Code quality checks
- **Checks:**
  - Black formatting (line length 100)
  - isort import sorting (black profile)
  - Can be extended with pylint checks

#### test-multi-platform
- **Platforms:** Windows, Ubuntu, macOS (latest versions)
- **Python:** 3.13
- **Matrix strategy:** Run tests on all platforms in parallel
- **Steps:**
  1. Install package on each platform
  2. Verify core dependencies
  3. Run installation tests
  4. Run gate and circuit tests
  5. Run remaining test suite

#### test-coverage
- **Platform:** Windows (latest)
- **Depends on:** test-multi-platform
- **Purpose:** Generate detailed coverage reports
- **Outputs:**
  - Terminal coverage report
  - HTML coverage report
  - XML coverage report (for tools like Codecov)
  - Artifacts retained for 14 days

#### test-fresh-installation
- **Platform:** Windows (latest)
- **Depends on:** test-multi-platform
- **Purpose:** Verify package can be installed from scratch

## Status Badges

The README includes status badges showing the current state of CI tests:

```markdown
[![Test Windows](https://github.com/Simone-Bordoni/Quantum-sensing-QML/workflows/Test%20Installation%20and%20Run%20Tests/badge.svg)](https://github.com/Simone-Bordoni/Quantum-sensing-QML/actions/workflows/test-windows.yml)
[![CI Tests](https://github.com/Simone-Bordoni/Quantum-sensing-QML/workflows/Comprehensive%20CI%20Tests/badge.svg)](https://github.com/Simone-Bordoni/Quantum-sensing-QML/actions/workflows/ci.yml)
```

## Local Testing

Before pushing, you can run the same tests locally:

### Quick Test
```bash
# Run installation tests
pytest src/qsopt/tests/test_installation.py -v -k "not TestFreshEnvironmentInstallation"

# Run core tests
pytest src/qsopt/tests/test_gates.py src/qsopt/tests/test_circuit.py -v
```

### Full Test Suite
```bash
# Run all tests
pytest src/qsopt/tests/ -v

# With coverage
pytest src/qsopt/tests/ --cov=src/qsopt --cov-report=html
```

### Code Formatting
```bash
# Check formatting
black --check src/ --line-length 100
isort --check-only --profile black src/

# Auto-format
black src/ --line-length 100
isort --profile black src/
```

## Viewing Results

### GitHub Actions UI
1. Go to the repository on GitHub
2. Click the "Actions" tab
3. Select a workflow run to see detailed logs
4. Download artifacts (coverage reports) from completed runs

### Coverage Reports
After a successful run with coverage:
1. Go to the workflow run
2. Scroll to "Artifacts" section
3. Download `coverage-report` or `coverage-report-windows-py3.13`
4. Extract and open `htmlcov/index.html` in a browser

## Test Strategy

### Installation Tests (`test_installation.py`)
- **30 core tests:** Package imports, dependencies, basic functionality
- **3 fresh environment tests:** Installation in isolated environment
- **Purpose:** Ensure package can be installed and imported correctly

### Gate Tests (`test_gates.py`)
- **53 tests:** All quantum gates verified against QuTiP
- **Coverage:** RX, RY, RZ, Hadamard, CNOT, CZ gates
- **Purpose:** Verify gate implementations match standard quantum gates

### Circuit Tests (`test_circuit.py`)
- **33 tests:** Circuit construction and parameter management
- **Coverage:** QuantumCircuit class, gate application, unitaries
- **Purpose:** Verify circuit builder works correctly

### Other Tests
- Experiment tests
- Parameter management tests
- Visualization tests
- Noise model tests
- Optimization tests

## Troubleshooting CI Failures

### Installation Failures
```bash
# Check if dependencies are compatible
pip install -e ".[test]"

# Verify package structure
python -c "import qsopt; print(qsopt.__version__)"
```

### Test Failures
```bash
# Run specific failing test locally
pytest src/qsopt/tests/test_gates.py::TestRotationGates::test_rx_gate -v

# Run with full traceback
pytest src/qsopt/tests/ --tb=long
```

### Platform-Specific Issues
- **Windows:** May need Visual C++ Build Tools
- **macOS:** May need Xcode command line tools
- **Linux:** May need additional system packages

## Adding New Tests

When adding new functionality:

1. **Write tests first** (TDD approach)
2. **Add to appropriate test file** in `src/qsopt/tests/`
3. **Run locally** before pushing
4. **Check CI results** after pushing

Example:
```python
# In src/qsopt/tests/test_myfeature.py
def test_new_feature():
    """Test description."""
    from qsopt.core.myfeature import new_function
    result = new_function()
    assert result == expected_value
```

## Extending CI

### Add More Python Versions
Edit workflow matrix:
```yaml
strategy:
  matrix:
    python-version: ['3.13', '3.14']  # Add more versions
```

### Add More Platforms
Edit workflow matrix:
```yaml
strategy:
  matrix:
    os: [windows-latest, ubuntu-latest, macos-latest, macos-13]
```

### Add Performance Tests
Create new workflow file:
```yaml
# .github/workflows/performance.yml
name: Performance Tests
on:
  schedule:
    - cron: '0 0 * * 0'  # Weekly
# ... test steps
```

## Best Practices

1. **Keep tests fast:** Aim for < 10 minutes total
2. **Use `continue-on-error: true`** for non-critical steps
3. **Cache dependencies** when possible (not needed for simple installs)
4. **Upload artifacts** for debugging
5. **Matrix testing** for multiple platforms/versions
6. **Fail-fast: false** to see all platform results

## Security

- **No secrets required** for current workflows
- **Read-only permissions** for basic CI
- **Artifacts expire** after 7-14 days

## Future Improvements

Potential additions:
- [ ] Codecov integration for coverage tracking
- [ ] Automatic package publishing to PyPI on release
- [ ] Documentation building and deployment
- [ ] Performance benchmarking
- [ ] Security scanning (Dependabot, CodeQL)
- [ ] Docker container testing

## Questions?

If CI tests fail:
1. Check the Actions tab for detailed logs
2. Download artifacts for coverage reports
3. Run tests locally to reproduce
4. Check this documentation for common issues

For help, open an issue or contact maintainers.
