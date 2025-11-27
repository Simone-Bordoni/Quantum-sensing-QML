# Installation Guide

## Quick Installation

### Using pip (recommended)

```bash
# Install from source (editable mode)
pip install -e .

# Install with development dependencies
pip install -e ".[dev]"

# Install with test dependencies only
pip install -e ".[test]"

# Install with Jupyter support
pip install -e ".[jupyter]"
```

### Using Poetry (alternative)

Poetry provides simplified dependency management:

```bash
# Install Poetry if you don't have it
pip install poetry

# Install all dependencies
poetry install

# Install only main dependencies (no dev tools)
poetry install --only main

# Install with specific groups
poetry install --with test
poetry install --with dev
```

### Using requirements files

```bash
# Install main dependencies
pip install -r requirements.txt

# Install development dependencies
pip install -r requirements-dev.txt
```

## Installation Verification

Test that the installation was successful:

```bash
# Run installation tests
pytest src/qsopt/tests/test_installation.py -v

# Or run directly
python src/qsopt/tests/test_installation.py
```

## Fresh Environment Testing

The installation test suite includes automated testing in a fresh virtual environment:

```bash
# This will:
# 1. Create a new virtual environment
# 2. Install the package
# 3. Run all tests
# 4. Clean up the environment
pytest src/qsopt/tests/test_installation.py::TestFreshEnvironmentInstallation -v
```

## Requirements

- **Python**: 3.13 or higher
- **Operating System**: Windows, macOS, or Linux

### Core Dependencies

- numpy (2.3.4)
- jax (0.4.35)
- jaxlib (0.4.35)
- qutip (5.2.2)
- qutip-jax (0.1.1.dev6)
- optax (0.2.4)
- matplotlib (3.10.0)
- scipy (1.16.3)
- Additional scientific computing libraries

All versions are pinned to ensure compatibility.

## Development Setup

For contributing to the project:

```bash
# Clone the repository
git clone https://github.com/Simone-Bordoni/Quantum-sensing-QML.git
cd Quantum-sensing-QML

# Install in editable mode with dev dependencies
pip install -e ".[dev]"

# Or with Poetry
poetry install --with dev

# Run tests
pytest tests/ -v
pytest src/qsopt/tests/ -v

# Run code formatting
black src/ tests/
isort src/ tests/

# Run linting
pylint src/qsopt
```

## Troubleshooting

### JAX Installation Issues

If you encounter issues with JAX:

```bash
# For CPU-only version
pip install --upgrade "jax[cpu]==0.4.35"

# For GPU version (CUDA 12)
pip install --upgrade "jax[cuda12]==0.4.35"
```

### QuTiP Installation Issues

If QuTiP installation fails:

```bash
# Install dependencies first
pip install numpy scipy cython

# Then install QuTiP
pip install qutip==5.2.2
```

### Windows-Specific Issues

On Windows, you might need Microsoft Visual C++ Build Tools for some dependencies.

### Python Version

Ensure you're using Python 3.13+:

```bash
python --version
```

## Uninstallation

```bash
pip uninstall qsopt
```

## Next Steps

After installation, see the [main README](../README.md) for usage examples and documentation.
