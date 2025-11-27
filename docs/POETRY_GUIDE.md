# Using Poetry for Dependency Management

Poetry provides a modern, simplified approach to Python dependency management and packaging.

## Why Use Poetry?

- **Simplified dependency management**: Single command to install everything
- **Automatic virtual environment**: No need to manually create venvs
- **Lock files**: Ensures reproducible installations across environments
- **Better dependency resolution**: Handles conflicts automatically
- **Modern tooling**: Integrates with modern Python development workflows

## Quick Start with Poetry

### 1. Install Poetry

```bash
# On Windows (PowerShell)
(Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | python -

# On macOS/Linux
curl -sSL https://install.python-poetry.org | python3 -

# Or with pip (not recommended but works)
pip install poetry
```

### 2. Install Project Dependencies

```bash
cd Quantum-sensing-QML

# Install all dependencies (including dev tools)
poetry install

# Install only main dependencies
poetry install --only main

# Install with specific groups
poetry install --with test
poetry install --with dev
```

### 3. Activate the Virtual Environment

```bash
# Spawn a shell within the virtual environment
poetry shell

# Or run commands directly
poetry run python your_script.py
poetry run pytest
```

## Common Poetry Commands

### Managing Dependencies

```bash
# Add a new dependency
poetry add numpy

# Add a development dependency
poetry add --group dev pytest

# Update dependencies
poetry update

# Update specific package
poetry update numpy

# Show installed packages
poetry show

# Show dependency tree
poetry show --tree
```

### Running Commands

```bash
# Run Python scripts
poetry run python script.py

# Run pytest
poetry run pytest tests/ -v

# Run Jupyter
poetry run jupyter notebook

# Format code
poetry run black src/
poetry run isort src/
```

### Building and Publishing

```bash
# Build the package
poetry build

# Check the package
poetry check

# Publish to PyPI (when ready)
poetry publish
```

## Poetry Configuration File

The project uses `poetry.lock.toml` which defines:

- **Project metadata**: name, version, authors
- **Dependencies**: exact versions for reproducibility
- **Development dependencies**: testing and formatting tools
- **Build settings**: how to package the library

## Advantages Over pip + requirements.txt

| Feature | Poetry | pip + requirements.txt |
|---------|--------|------------------------|
| Dependency resolution | Automatic | Manual |
| Virtual environment | Built-in | Manual (venv) |
| Lock files | Yes (.lock) | Manual (freeze) |
| Development deps | Separate groups | Separate files |
| Publishing | Built-in | Separate tools |
| Dependency tree | Built-in view | External tools |

## Migration from pip

If you're currently using pip and want to try Poetry:

```bash
# 1. Install Poetry
pip install poetry

# 2. Initialize (or use existing pyproject.toml)
poetry install

# 3. From now on, use poetry commands
poetry add package_name
poetry run pytest
```

Your existing `pip install -e .` installations will continue to work.

## Troubleshooting

### Poetry command not found

Add Poetry to your PATH:

```bash
# Windows
$Env:Path += ";$Env:APPDATA\Python\Scripts"

# macOS/Linux
export PATH="$HOME/.local/bin:$PATH"
```

### Virtual environment issues

```bash
# Remove existing venv and recreate
poetry env remove python
poetry install
```

### Lock file errors

```bash
# Regenerate lock file
poetry lock --no-update
```

## Integration with IDEs

### VS Code

1. Install Python extension
2. Select interpreter: `Ctrl+Shift+P` → "Python: Select Interpreter"
3. Choose the Poetry virtual environment

### PyCharm

1. Settings → Project → Python Interpreter
2. Add Interpreter → Poetry Environment
3. PyCharm will auto-detect the poetry configuration

## Comparison: Installation Methods

### Using pip (Traditional)

```bash
pip install -e .                     # Manual venv management
pip install -r requirements.txt      # No lock file
pip install -r requirements-dev.txt  # Separate dev deps
```

### Using Poetry (Modern)

```bash
poetry install              # Auto venv + lock file + all deps
poetry install --only main  # Just main deps
poetry install --with dev   # With dev tools
```

## Best Practices

1. **Commit `poetry.lock`**: Ensures everyone has same versions
2. **Use dependency groups**: Separate dev, test, and docs dependencies
3. **Update regularly**: `poetry update` to get security fixes
4. **Check before adding**: `poetry show <package>` to see if already included

## For Contributors

When contributing to the project:

```bash
# 1. Clone and install
git clone https://github.com/Simone-Bordoni/Quantum-sensing-QML.git
cd Quantum-sensing-QML
poetry install --with dev

# 2. Make changes and test
poetry run pytest
poetry run black src/ tests/
poetry run pylint src/qsopt

# 3. Submit PR
git commit -m "Your changes"
git push
```

## See Also

- [Poetry Documentation](https://python-poetry.org/docs/)
- [INSTALLATION.md](./INSTALLATION.md) - Complete installation guide
- [pyproject.toml](./pyproject.toml) - Project configuration
- [README.md](./README.md) - Main project documentation
