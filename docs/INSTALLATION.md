# Installation Guide

## Quick Installation

### Using pip (recommended)

By default, the library installs with **CPU-based JAX**, which works on any system. Optionally, you can add GPU support or development tools.

#### Basic Installation (CPU - works everywhere)
```bash
pip install -e .
```

#### Add GPU Support (pick one)
```bash
# For NVIDIA GPUs with CUDA 12.x (recommended for most modern setups)
pip install -e ".[cuda12]"

# For NVIDIA GPUs with CUDA 11.x
pip install -e ".[cuda11]"
```


#### Other Extras (without GPU)
```bash
# Development tools (testing, formatting, linting)
pip install -e ".[dev]"

# Testing only
pip install -e ".[test]"

# Jupyter notebook support
pip install -e ".[jupyter]"
```


#### Combine Options (mix and match as needed)
Extras are composable—combine GPU, development tools, testing, Jupyter, etc. with commas:

```bash
# GPU + development tools
pip install -e ".[cuda12,dev]"

# GPU + testing
pip install -e ".[cuda12,test]"

# development + Jupyter (on CPU)
pip install -e ".[dev,jupyter]"

# GPU + all optional tools
pip install -e ".[cuda12,dev,test,jupyter]"

# Multiple non-GPU extras
pip install -e ".[dev,jupyter]"
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
