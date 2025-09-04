#!/usr/bin/env python3
"""
Simple test script to validate the package can be imported and basic functionality works.
This is useful for CI environments where full JAX functionality might not be available.
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_imports():
    """Test that all main modules can be imported."""
    try:
        from qsopt.core.trainable_parameters import (
            ParameterType, ParameterConstraints, OptimizationConfig,
            ParameterGroup, TrainableParameters
        )
        print("✅ Core trainable_parameters imports successful")
        
        from qsopt.core.experimental_parameters import (
            ExperimentalParameters, PhysicalConstants, SystemDimensions
        )
        print("✅ Core experimental_parameters imports successful")
        
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False

def test_basic_functionality():
    """Test basic functionality without JAX dependencies."""
    try:
        from qsopt.core.trainable_parameters import ParameterGroup, ParameterType
        import numpy as np
        
        # Test basic parameter group creation (without JAX)
        group = ParameterGroup("test", ParameterType.CUSTOM, [1.0, 2.0, 3.0])
        assert len(group) == 3
        assert group.name == "test"
        print("✅ Basic ParameterGroup functionality works")
        
        return True
    except Exception as e:
        print(f"❌ Basic functionality test failed: {e}")
        return False

if __name__ == "__main__":
    print("Running basic CI validation tests...")
    
    success = True
    success &= test_imports()
    success &= test_basic_functionality()
    
    if success:
        print("\n🎉 All basic tests passed!")
        sys.exit(0)
    else:
        print("\n💥 Some tests failed!")
        sys.exit(1)
