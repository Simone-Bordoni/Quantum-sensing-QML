#!/usr/bin/env python3
"""
Local coverage test script that works around JAX import issues.
"""

import os
import sys
import subprocess
import tempfile

def run_coverage_test():
    """Run coverage test on trainable_parameters module specifically."""
    
    # Create a temporary test file that imports just what we need
    test_content = '''
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

# Import pytest components
import pytest
import numpy as np
from unittest.mock import Mock

# Import our specific module directly
from qsopt.core.trainable_parameters import (
    ParameterType, ParameterConstraints, OptimizationConfig,
    ParameterGroup, TrainableParameters
)

# Re-import all our test cases but skip JAX imports
'''
    
    # Read our test file and append it
    with open("src/qsopt/tests/test_trainable_parameters.py", "r") as f:
        original_test = f.read()
    
    # Replace JAX imports with numpy equivalents for coverage testing
    modified_test = original_test.replace("import jax.numpy as jnp", "import numpy as jnp")
    modified_test = modified_test.replace("jnp.array", "np.array")
    modified_test = modified_test.replace("jnp.allclose", "np.allclose") 
    
    # Write temporary test file
    with tempfile.NamedTemporaryFile(mode='w', suffix='_test.py', delete=False) as f:
        f.write(test_content + "\n\n" + modified_test)
        temp_test_file = f.name
    
    try:
        # Run coverage on the temporary test
        cmd = [
            sys.executable, "-m", "pytest", temp_test_file,
            "--cov=src/qsopt/core/trainable_parameters",
            "--cov-report=term-missing",
            "--cov-report=html:coverage_html",
            "-v"
        ]
        
        print("Running coverage test...")
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=".")
        
        print("STDOUT:")
        print(result.stdout)
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
            
        return result.returncode == 0
        
    finally:
        # Clean up temporary file
        if os.path.exists(temp_test_file):
            os.unlink(temp_test_file)

if __name__ == "__main__":
    success = run_coverage_test()
    if success:
        print("\\n✅ Coverage test completed successfully!")
        print("Check 'coverage_html/index.html' for detailed coverage report.")
    else:
        print("\\n❌ Coverage test failed!")
        sys.exit(1)
