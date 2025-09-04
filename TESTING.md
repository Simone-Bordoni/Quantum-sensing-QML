# GitHub Actions Badges

Add these badges to your README.md to show the status of your tests and coverage:

```markdown
[![Tests](https://github.com/Simone-Bordoni/Quantum-sensing-QML/actions/workflows/test.yml/badge.svg)](https://github.com/Simone-Bordoni/Quantum-sensing-QML/actions/workflows/test.yml)
[![codecov](https://codecov.io/gh/Simone-Bordoni/Quantum-sensing-QML/branch/main/graph/badge.svg)](https://codecov.io/gh/Simone-Bordoni/Quantum-sensing-QML)
```

## Setup Instructions for Coverage Reporting

### 1. Codecov Setup (Optional)
If you want coverage reporting on codecov.io:

1. Go to [codecov.io](https://codecov.io)
2. Sign in with your GitHub account
3. Add your repository `Simone-Bordoni/Quantum-sensing-QML`
4. The GitHub Action will automatically upload coverage reports

### 2. Local Coverage Testing
To run coverage tests locally:

```bash
# Basic CI validation test (works around JAX issues)
python ci_test.py

# Run tests with coverage (if JAX environment is working)
python -m pytest src/qsopt/tests/test_trainable_parameters.py --cov=qsopt.core.trainable_parameters --cov-report=html
```

### 3. GitHub Actions Features

The workflow includes:

- **Multi-Python Version Testing**: Tests on Python 3.9, 3.10, 3.11, and 3.12
- **Caching**: Pip dependencies are cached for faster builds
- **Coverage Reports**: Generates HTML, XML, and terminal coverage reports
- **Linting**: Runs flake8, black, isort, and mypy (optional)
- **Artifacts**: Coverage reports are saved as GitHub artifacts

### 4. Files Created

- `.github/workflows/test.yml` - Main CI/CD pipeline
- `.coveragerc` - Coverage configuration
- `requirements-test.txt` - Testing dependencies
- `ci_test.py` - Basic validation script for CI
- `run_coverage.py` - Local coverage testing script
