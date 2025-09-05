# Code Quality with Pylint

This document describes how to use Pylint for code quality improvement in the Quantum Sensing Optimization Library.

## Overview

We use Pylint along with other tools to maintain high code quality:

- **Pylint**: Static code analysis and style checking
- **Black**: Automatic code formatting 
- **isort**: Import statement sorting
- **autopep8**: PEP 8 compliance (when available)

## Quick Start

### Running Pylint Locally

```bash
# Run analysis only
python run_pylint.py

# Run analysis with automatic fixes
python run_pylint.py --fix

# Use the batch script (Windows)
run_pylint.bat

# Use the shell script (Unix/Linux)
./run_pylint.sh
```

### Command Line Options

```bash
python run_pylint.py --help
```

Available options:
- `--source-dir, -s`: Source directory to analyze (default: `src/qsopt`)
- `--config, -c`: Pylint configuration file (default: `.pylintrc`)
- `--fix, -f`: Apply automatic fixes (black formatting)
- `--report, -r`: Output file for detailed report (default: `pylint_report.txt`)
- `--format`: Output format - json, text, or parseable (default: `json`)

## Configuration

### Pylint Configuration (`.pylintrc`)

The `.pylintrc` file contains customized settings for scientific Python code:

**Disabled Checks:**
- `C0103`: Invalid name (allows mathematical variable names like `x`, `y`, `theta`, `phi`)
- `C0114/C0115/C0116`: Missing docstrings (relaxed for simple functions)
- `R0913/R0914`: Too many arguments/locals (common in scientific computing)
- `E1101`: No member (false positives with JAX/NumPy dynamic attributes)

**Good Variable Names:**
Mathematical and physics variables are allowed: `i`, `j`, `k`, `x`, `y`, `z`, `t`, `theta`, `phi`, `alpha`, `beta`, `gamma`, `chi`, `psi`, `omega`

**Line Length:** 100 characters (compatible with Black formatting)

### Integration with Development Workflow

#### 1. IDE Integration
Most IDEs can use the `.pylintrc` configuration:

- **VS Code**: Install the Python extension and Pylint will be used automatically
- **PyCharm**: Enable Pylint in Settings → Tools → External Tools
- **Vim/Neovim**: Use ALE or Syntastic plugins

#### 2. Pre-commit Hooks
Install the pre-commit hook for automatic checking:

```bash
# Copy the hook script
cp pre_commit_hook.py .git/hooks/pre-commit
chmod +x .git/hooks/pre-commit  # Unix/Linux only
```

This will automatically:
- Run Black formatting
- Sort imports with isort  
- Run Pylint analysis
- Add formatted files back to the commit

#### 3. GitHub Actions Integration
The CI pipeline automatically:
- Runs Pylint analysis on all pull requests
- Applies automatic formatting
- Uploads detailed reports as artifacts
- Continues even if style issues are found (won't block merges)

## Understanding Pylint Output

### Issue Categories

- **CONVENTION (C)**: Coding standard violations
- **REFACTOR (R)**: Code that could be refactored
- **WARNING (W)**: Potential bugs or suspicious code
- **ERROR (E)**: Definite bugs or syntax errors

### Common Issues and Fixes

#### C0301: Line too long
**Fix**: Break long lines or use Black formatting
```python
# Before
result = some_very_long_function_name(argument1, argument2, argument3, argument4)

# After  
result = some_very_long_function_name(
    argument1, argument2, argument3, argument4
)
```

#### C0411: Wrong import order
**Fix**: Use isort to automatically sort imports
```bash
python -m isort src/qsopt/
```

#### W0611: Unused import
**Fix**: Remove unused imports
```python
# Remove this if jax is not used
import jax  # W0611: Unused import
```

#### R0913: Too many arguments
**Fix**: Use dataclasses or parameter objects
```python
# Before
def complex_function(a, b, c, d, e, f, g, h):
    pass

# After
@dataclass
class Parameters:
    a: float
    b: float
    c: float
    # ... etc

def complex_function(params: Parameters):
    pass
```

## Automatic Fixes Applied

### Black Formatting
- Consistent indentation (4 spaces)
- Line length enforcement (100 chars)
- Quote normalization
- Trailing comma handling

### isort Import Sorting
- Groups imports by type (standard library, third-party, local)
- Alphabetical sorting within groups
- Consistent formatting

## Continuous Integration

### GitHub Actions Workflow

The `.github/workflows/test.yml` includes a `lint` job that:

1. **Initial Analysis**: Runs Pylint to identify issues
2. **Auto-formatting**: Applies Black and isort fixes  
3. **Final Analysis**: Re-runs Pylint to show improvements
4. **Report Upload**: Saves detailed reports as artifacts

### Viewing Reports

After CI runs, download the Pylint reports from GitHub Actions artifacts:
- `pylint_initial.txt`: Issues before formatting
- `pylint_final.txt`: Issues after automatic fixes

## Best Practices

### 1. Run Pylint Before Committing
```bash
python run_pylint.py --fix
```

### 2. Address High-Priority Issues First
Focus on ERROR and WARNING categories before CONVENTION and REFACTOR.

### 3. Use Type Hints
Pylint works better with type annotations:
```python
def calculate_something(x: float, y: float) -> float:
    return x + y
```

### 4. Document Complex Functions
Add docstrings for non-obvious functions:
```python
def complex_calculation(theta: float, phi: float) -> jnp.ndarray:
    """
    Calculate quantum state rotation angles.
    
    Args:
        theta: Rotation angle around X-axis (radians)
        phi: Rotation angle around Z-axis (radians)
        
    Returns:
        Rotation matrix as JAX array
    """
    # Implementation...
```

### 5. Suppress False Positives Carefully
Use `# pylint: disable=` comments sparingly:
```python
# Only when Pylint is definitely wrong
result = jax.numpy.array([1, 2, 3])  # pylint: disable=no-member
```

## Troubleshooting

### Common Issues

**"No module named 'lib2to3'" Error**
This affects autopep8 in some Python 3.12+ environments. The workflow uses Black instead, which works reliably.

**JAX/NumPy Member Warnings** 
These are configured to be ignored in `.pylintrc` but may still appear. They're usually false positives due to dynamic attribute generation.

**Import Errors for Optional Dependencies**
The configuration ignores import errors for optional packages. Ensure all required dependencies are in `requirements-test.txt`.

### Getting Help

- Check the detailed report file (`pylint_report.txt`)
- Review the GitHub Actions logs for CI failures
- Consult the [Pylint documentation](https://pylint.readthedocs.io/)

## Future Improvements

- [ ] Add more scientific computing specific rules
- [ ] Integrate with code coverage metrics
- [ ] Add performance linting rules
- [ ] Consider adding custom Pylint plugins for quantum computing patterns
