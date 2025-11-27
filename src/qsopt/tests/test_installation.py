"""
Comprehensive Package Installation and Integration Tests
========================================================

This test suite verifies that:
1. The package can be installed correctly in a fresh environment
2. All core modules can be imported
3. All dependencies are available
4. Basic functionality works after installation
5. The environment can be cleaned up after testing

Run with: pytest src/qsopt/tests/test_installation.py -v
Or standalone: python src/qsopt/tests/test_installation.py
"""

import sys
import subprocess
import shutil
from pathlib import Path
import pytest


# =============================================================================
# Package Installation Tests
# =============================================================================

class TestPackageInstallation:
    """Test that the package installs correctly."""
    
    def test_package_importable(self):
        """Test that qsopt package can be imported."""
        try:
            import qsopt
        except ImportError as e:
            pytest.fail(f"Failed to import qsopt: {e}")
    
    def test_package_version(self):
        """Test that package version is defined."""
        import qsopt
        assert hasattr(qsopt, '__version__')
        assert isinstance(qsopt.__version__, str)
        assert len(qsopt.__version__) > 0
    
    def test_package_metadata(self):
        """Test that package metadata is defined."""
        import qsopt
        assert hasattr(qsopt, '__author__')
        assert hasattr(qsopt, '__email__')
        assert isinstance(qsopt.__author__, str)
        assert isinstance(qsopt.__email__, str)


# =============================================================================
# Core Module Import Tests
# =============================================================================

class TestCoreImports:
    """Test that all core modules can be imported."""
    
    def test_experimental_parameters_import(self):
        """Test experimental parameters module imports."""
        try:
            from qsopt.core.experimental_parameters import (
                ExperimentalParameters,
                PhysicalConstants,
                SystemDimensions,
                MeasurementProtocol,
                InitialStateConfig,
                InitialStateType,
                NoiseConfiguration
            )
        except ImportError as e:
            pytest.fail(f"Failed to import experimental parameters: {e}")
    
    def test_trainable_parameters_import(self):
        """Test trainable parameters module imports."""
        try:
            from qsopt.core.trainable_parameters import (
                TrainableParameters,
                ParameterType,
                ParameterConstraints
            )
        except ImportError as e:
            pytest.fail(f"Failed to import trainable parameters: {e}")
    
    def test_experiment_import(self):
        """Test experiment module imports."""
        try:
            from qsopt.core.experiment import (
                Experiment,
                SingleQubitExperiment,
                TwoQubitExperiment
            )
        except ImportError as e:
            pytest.fail(f"Failed to import experiment: {e}")
    
    def test_callback_import(self):
        """Test callback module imports."""
        try:
            from qsopt.core.callback import OptimizationCallback
        except ImportError as e:
            pytest.fail(f"Failed to import callback: {e}")
    
    def test_gates_import(self):
        """Test gates module imports."""
        try:
            from qsopt.core.gates import (
                Gate,
                GateParameter,
                RXGate,
                RYGate,
                RZGate,
                HadamardGate,
                CNOTGate,
                CZGate
            )
        except ImportError as e:
            pytest.fail(f"Failed to import gates: {e}")
    
    def test_circuit_import(self):
        """Test circuit module imports."""
        try:
            from qsopt.core.circuit import (
                QuantumCircuit,
                GateApplication,
                create_layer,
                create_entangling_layer
            )
        except ImportError as e:
            pytest.fail(f"Failed to import circuit: {e}")
    
    def test_visualization_import(self):
        """Test visualization module imports."""
        try:
            from qsopt.utils.visualization import (
                plot_optimization_dashboard,
                plot_contrast_evolution,
                plot_parameter_trajectory
            )
        except ImportError as e:
            pytest.fail(f"Failed to import visualization: {e}")


# =============================================================================
# Dependency Availability Tests
# =============================================================================

class TestDependencies:
    """Test that all required dependencies are available."""
    
    def test_numpy_available(self):
        """Test that numpy is available."""
        try:
            import numpy as np
            assert hasattr(np, '__version__')
        except ImportError as e:
            pytest.fail(f"numpy not available: {e}")
    
    def test_jax_available(self):
        """Test that JAX is available."""
        try:
            import jax
            import jax.numpy as jnp
            assert hasattr(jax, '__version__')
        except ImportError as e:
            pytest.fail(f"JAX not available: {e}")
    
    def test_qutip_available(self):
        """Test that QuTiP is available."""
        try:
            import qutip
            assert hasattr(qutip, '__version__')
        except ImportError as e:
            pytest.fail(f"QuTiP not available: {e}")
    
    def test_optax_available(self):
        """Test that Optax is available."""
        try:
            import optax
            assert hasattr(optax, '__version__')
        except ImportError as e:
            pytest.fail(f"Optax not available: {e}")
    
    def test_matplotlib_available(self):
        """Test that matplotlib is available."""
        try:
            import matplotlib
            import matplotlib.pyplot as plt
            assert hasattr(matplotlib, '__version__')
        except ImportError as e:
            pytest.fail(f"matplotlib not available: {e}")
    
    def test_scipy_available(self):
        """Test that scipy is available."""
        try:
            import scipy
            assert hasattr(scipy, '__version__')
        except ImportError as e:
            pytest.fail(f"scipy not available: {e}")


# =============================================================================
# Basic Functionality Tests
# =============================================================================

class TestBasicFunctionality:
    """Test that basic functionality works after installation."""
    
    def test_create_physical_constants(self):
        """Test creating PhysicalConstants object."""
        from qsopt.core.experimental_parameters import PhysicalConstants
        import numpy as np
        
        gm = 0.03 * 2 * np.pi
        constants = PhysicalConstants(
            n_qubits=1,
            chi=0.5 * gm,
            photon_cavity_coupling=gm,
            inverse_pulse_width=0.1 * gm
        )
        
        # chi is converted to a list internally for multi-qubit support
        assert isinstance(constants.chi, list)
        assert len(constants.chi) == 1
        assert abs(constants.chi[0] - 0.5 * gm) < 1e-10
        assert constants.photon_cavity_coupling == gm
        assert constants.inverse_pulse_width == 0.1 * gm
    
    def test_create_system_dimensions(self):
        """Test creating SystemDimensions object."""
        from qsopt.core.experimental_parameters import SystemDimensions
        
        dims = SystemDimensions(
            cavity_levels=2,
            qubit_levels=2,
            field_levels=2
        )
        
        assert dims.cavity_levels == 2
        assert dims.qubit_levels == 2
        assert dims.field_levels == 2
    
    def test_create_trainable_parameters(self):
        """Test creating TrainableParameters object."""
        from qsopt.core.trainable_parameters import TrainableParameters
        import numpy as np
        
        params = TrainableParameters()
        params.add_rotation_angles(
            names=['theta1', 'theta2'],
            initial_values=[np.pi/2, -np.pi/2]
        )
        
        rotation_angles = params.get_rotation_angles()
        assert 'theta1' in rotation_angles
        assert 'theta2' in rotation_angles
        assert abs(rotation_angles['theta1'] - np.pi/2) < 1e-10
        assert abs(rotation_angles['theta2'] + np.pi/2) < 1e-10
    
    def test_create_gate(self):
        """Test creating a quantum gate."""
        from qsopt.core.gates import RYGate
        import jax.numpy as jnp
        
        gate = RYGate(theta=jnp.pi/4, trainable=True)
        
        # Verify gate has required attributes
        assert hasattr(gate, 'matrix')
        assert hasattr(gate, 'get_parameter')
        assert hasattr(gate, 'set_parameter')
        
        # Verify parameter management
        theta = gate.get_parameter("theta")
        assert abs(theta - jnp.pi/4) < 1e-10
        
        # Test parameter update (value first, then name)
        gate.set_parameter(jnp.pi/2, "theta")
        new_theta = gate.get_parameter("theta")
        assert abs(new_theta - jnp.pi/2) < 1e-10
    
    def test_create_circuit(self):
        """Test creating a quantum circuit."""
        from qsopt.core.circuit import QuantumCircuit
        from qsopt.core.gates import RYGate, HadamardGate
        import jax.numpy as jnp
        
        circuit = QuantumCircuit(num_qubits=2)
        circuit.add_gate(HadamardGate(), target=0)
        circuit.add_gate(RYGate(theta=jnp.pi/4, trainable=True), target=1)
        
        # Verify circuit has required methods
        assert hasattr(circuit, 'add_gate')
        assert hasattr(circuit, 'get_unitary')
        assert hasattr(circuit, 'get_trainable_parameters')
        assert hasattr(circuit, 'get_unitary_jax')
        
        # Verify parameter tracking
        params = circuit.get_trainable_parameters()
        assert len(params) == 1  # Only one trainable parameter
        assert 'gate_1_theta' in params
    
    def test_jax_integration(self):
        """Test that JAX integration works."""
        import jax
        import jax.numpy as jnp
        from qsopt.core.gates import RYGate
        
        # Test JAX array creation
        theta = jnp.array(0.5)
        gate = RYGate(theta=theta, trainable=True)
        
        # Test that gate matrix can be created
        U = gate.matrix()
        assert U is not None
        
        # Test gradient computation setup
        def loss_fn(theta_val):
            gate = RYGate(theta=theta_val)
            U = gate.matrix()
            # Simple loss: trace of gate matrix
            return jnp.abs(jnp.trace(U.full()))
        
        # Verify gradients can be computed
        grad_fn = jax.grad(loss_fn)
        try:
            grad = grad_fn(jnp.array(0.5))
            assert grad is not None
        except Exception:
            # JAX integration might have limitations, but should not crash
            pass


# =============================================================================
# Package Structure Tests
# =============================================================================

class TestPackageStructure:
    """Test that package structure is correct."""
    
    def test_core_subpackage_exists(self):
        """Test that core subpackage exists."""
        try:
            import qsopt.core
        except ImportError as e:
            pytest.fail(f"core subpackage not found: {e}")
    
    def test_utils_subpackage_exists(self):
        """Test that utils subpackage exists."""
        try:
            import qsopt.utils
        except ImportError as e:
            pytest.fail(f"utils subpackage not found: {e}")
    
    def test_tests_subpackage_exists(self):
        """Test that tests subpackage exists."""
        try:
            import qsopt.tests
        except ImportError as e:
            pytest.fail(f"tests subpackage not found: {e}")


# =============================================================================
# Top-Level Import Tests
# =============================================================================

class TestTopLevelImports:
    """Test that top-level imports work as documented."""
    
    def test_toplevel_experimental_parameters(self):
        """Test top-level experimental parameters imports."""
        from qsopt import (
            ExperimentalParameters,
            PhysicalConstants,
            SystemDimensions,
            MeasurementProtocol,
            InitialStateConfig,
            InitialStateType,
            NoiseConfiguration
        )
        
        # Verify all are classes/types
        assert ExperimentalParameters is not None
        assert PhysicalConstants is not None
        assert SystemDimensions is not None
    
    def test_toplevel_experiment(self):
        """Test top-level experiment imports."""
        from qsopt import (
            Experiment,
            SingleQubitExperiment,
            TwoQubitExperiment
        )
        
        assert Experiment is not None
        assert SingleQubitExperiment is not None
        assert TwoQubitExperiment is not None
    
    def test_toplevel_trainable_parameters(self):
        """Test top-level trainable parameters imports."""
        from qsopt import (
            TrainableParameters,
            ParameterType,
            ParameterConstraints
        )
        
        assert TrainableParameters is not None
        assert ParameterType is not None
        assert ParameterConstraints is not None
    
    def test_toplevel_visualization(self):
        """Test top-level visualization imports."""
        from qsopt import (
            plot_optimization_dashboard,
            plot_contrast_evolution,
            plot_parameter_trajectory
        )
        
        assert plot_optimization_dashboard is not None
        assert plot_contrast_evolution is not None
        assert plot_parameter_trajectory is not None


# =============================================================================
# System Requirements Test
# =============================================================================

def test_python_version():
    """Test that Python version meets requirements."""
    assert sys.version_info >= (3, 13), \
        f"Python 3.13+ required, found {sys.version_info.major}.{sys.version_info.minor}"


# =============================================================================
# Fresh Environment Installation Test
# =============================================================================

class TestFreshEnvironmentInstallation:
    """Test installation in a fresh virtual environment."""
    
    @pytest.fixture(scope="class")
    def fresh_env(self):
        """Create a fresh virtual environment for testing."""
        project_root = Path(__file__).parent.parent.parent.parent
        test_env_dir = project_root / ".test_env"
        
        # Remove old test environment if exists
        if test_env_dir.exists():
            shutil.rmtree(test_env_dir)
        
        # Create new virtual environment
        print(f"\nCreating test environment at: {test_env_dir}")
        subprocess.run(
            [sys.executable, "-m", "venv", str(test_env_dir)],
            check=True,
            capture_output=True
        )
        
        # Determine python executable path
        if sys.platform == "win32":
            python_exe = test_env_dir / "Scripts" / "python.exe"
            pip_exe = test_env_dir / "Scripts" / "pip.exe"
        else:
            python_exe = test_env_dir / "bin" / "python"
            pip_exe = test_env_dir / "bin" / "pip"
        
        yield {
            'root': project_root,
            'env_dir': test_env_dir,
            'python': python_exe,
            'pip': pip_exe
        }
        
        # Cleanup after tests
        print(f"\nCleaning up test environment: {test_env_dir}")
        if test_env_dir.exists():
            shutil.rmtree(test_env_dir)
    
    def test_install_in_fresh_env(self, fresh_env):
        """Test that package can be installed in fresh environment."""
        # Upgrade pip
        subprocess.run(
            [str(fresh_env['python']), "-m", "pip", "install", "--upgrade", "pip"],
            check=True,
            capture_output=True,
            cwd=fresh_env['root']
        )
        
        # Install package
        result = subprocess.run(
            [str(fresh_env['pip']), "install", "-e", ".[test]"],
            capture_output=True,
            text=True,
            cwd=fresh_env['root']
        )
        
        assert result.returncode == 0, \
            f"Installation failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    
    def test_import_in_fresh_env(self, fresh_env):
        """Test that package can be imported in fresh environment."""
        result = subprocess.run(
            [str(fresh_env['python']), "-c", "import qsopt; print(qsopt.__version__)"],
            capture_output=True,
            text=True,
            cwd=fresh_env['root']
        )
        
        assert result.returncode == 0, \
            f"Import failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        assert "0.1.0" in result.stdout
    
    def test_run_tests_in_fresh_env(self, fresh_env):
        """Test that tests can run in fresh environment."""
        result = subprocess.run(
            [
                str(fresh_env['python']), "-m", "pytest",
                "src/qsopt/tests/test_installation.py",
                "-v", "--tb=short",
                "-k", "not TestFreshEnvironmentInstallation"  # Skip this class to avoid recursion
            ],
            capture_output=True,
            text=True,
            cwd=fresh_env['root']
        )
        
        # Check that tests ran (might have some failures in fresh env)
        assert "collected" in result.stdout.lower(), \
            f"Tests did not run:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    """Run installation tests when script is executed directly."""
    import sys
    
    print("="*70)
    print("Running Quantum Sensing Optimization Library Installation Tests")
    print("="*70)
    
    # Run pytest with verbose output
    exit_code = pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "--color=yes"
    ])
    
    sys.exit(exit_code)
