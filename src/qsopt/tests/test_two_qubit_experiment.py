"""
Tests for Two Qubit Experiment Class
====================================

Test suite for the TwoQubitExperiment placeholder class.
"""

import pytest
import optax
from qsopt.core.experiment import TwoQubitExperiment
from qsopt.core.experimental_parameters import (
    ExperimentalParameters,
    PhysicalConstants,
    SystemDimensions,
    MeasurementProtocol,
    NoiseConfiguration,
    InitialStateConfig,
    InitialStateType
)
from qsopt.core.trainable_parameters import TrainableParameters


@pytest.fixture
def experimental_params():
    """Create experimental parameters for two-qubit testing."""
    return ExperimentalParameters(
        physical_constants=PhysicalConstants(
            chi=0.01,
            photon_cavity_coupling=0.1,
            inverse_pulse_width=1.0
        ),
        system_dims=SystemDimensions(
            cavity_levels=3,
            qubit_levels=2,  # For two qubits, this will be extended
            field_levels=3
        ),
        measurement=MeasurementProtocol(
            initial_time=0.0,
            final_time=10.0,
            time_interval=1.0
        ),
        noise_config=NoiseConfiguration(),
        initial_state=InitialStateConfig(state_type=InitialStateType.VACUUM)
    )


@pytest.fixture
def trainable_params():
    """Create trainable parameters for two-qubit testing."""
    params = TrainableParameters()
    params.add_rotation_angles(
        ["theta1", "theta2", "theta3", "theta4"],
        [0.0, 0.0, 0.0, 0.0],
        optimizer=optax.adam(learning_rate=0.01)
    )
    return params


class TestTwoQubitExperiment:
    """Tests for TwoQubitExperiment class."""
    
    def test_initialization(self, experimental_params, trainable_params):
        """Test that TwoQubitExperiment can be instantiated."""
        exp = TwoQubitExperiment(experimental_params, trainable_params)
        
        assert exp is not None
        assert exp.experimental_params is experimental_params
        assert exp.trainable_params is trainable_params
    
    def test_has_two_qubit_caches(self, experimental_params, trainable_params):
        """Test that two-qubit specific caches are initialized."""
        exp = TwoQubitExperiment(experimental_params, trainable_params)
        
        assert hasattr(exp, '_cached_initial_state')
        assert hasattr(exp, '_cached_two_qubit_gates')
        assert isinstance(exp._cached_two_qubit_gates, dict)
    
    def test_generate_operators_not_fully_implemented(self, experimental_params, trainable_params):
        """Test that _generate_operators calls two-qubit utility (not yet implemented)."""
        exp = TwoQubitExperiment(experimental_params, trainable_params)
        
        # Call _generate_operators explicitly - should raise NotImplementedError
        # because generate_two_qubit_operators is not yet implemented
        with pytest.raises(NotImplementedError, match="Two-qubit operator generation"):
            exp._generate_operators()
    
    def test_generate_hamiltonian_not_implemented(self, experimental_params, trainable_params):
        """Test that _generate_hamiltonian raises NotImplementedError."""
        exp = TwoQubitExperiment(experimental_params, trainable_params)
        
        with pytest.raises(NotImplementedError, match="TwoQubitExperiment._generate_hamiltonian"):
            exp._generate_hamiltonian()
    
    def test_initialize_caches_not_implemented(self, experimental_params, trainable_params):
        """Test that _initialize_caches raises NotImplementedError."""
        exp = TwoQubitExperiment(experimental_params, trainable_params)
        
        with pytest.raises(NotImplementedError, match="TwoQubitExperiment._initialize_caches"):
            exp._initialize_caches()
    
    def test_get_initial_state_not_implemented(self, experimental_params, trainable_params):
        """Test that get_initial_state raises NotImplementedError."""
        exp = TwoQubitExperiment(experimental_params, trainable_params)
        
        with pytest.raises(NotImplementedError, match="TwoQubitExperiment.get_initial_state"):
            exp.get_initial_state()
    
    def test_simulation_not_implemented(self, experimental_params, trainable_params):
        """Test that simulation raises NotImplementedError."""
        exp = TwoQubitExperiment(experimental_params, trainable_params)
        
        with pytest.raises(NotImplementedError, match="TwoQubitExperiment.simulation"):
            exp.simulation()
    
    def test_run_simulation_not_implemented(self, experimental_params, trainable_params):
        """Test that run_simulation raises NotImplementedError."""
        exp = TwoQubitExperiment(experimental_params, trainable_params)
        
        with pytest.raises(NotImplementedError, match="TwoQubitExperiment.run_simulation"):
            exp.run_simulation()
    
    def test_optimize_rotations_not_implemented(self, experimental_params, trainable_params):
        """Test that optimize_rotations raises NotImplementedError."""
        exp = TwoQubitExperiment(experimental_params, trainable_params)
        
        with pytest.raises(NotImplementedError, match="TwoQubitExperiment.optimize_rotations"):
            exp.optimize_rotations()
    
    def test_apply_two_qubit_gate_not_implemented(self, experimental_params, trainable_params):
        """Test that apply_two_qubit_gate raises NotImplementedError."""
        exp = TwoQubitExperiment(experimental_params, trainable_params)
        
        import qutip as qt
        rho = qt.basis(2, 0).proj()
        
        with pytest.raises(NotImplementedError, match="TwoQubitExperiment.apply_two_qubit_gate"):
            exp.apply_two_qubit_gate(rho, 'CNOT')
    
    def test_measure_both_qubits_not_implemented(self, experimental_params, trainable_params):
        """Test that measure_both_qubits raises NotImplementedError."""
        exp = TwoQubitExperiment(experimental_params, trainable_params)
        
        import qutip as qt
        rho = qt.basis(2, 0).proj()
        
        with pytest.raises(NotImplementedError, match="TwoQubitExperiment.measure_both_qubits"):
            exp.measure_both_qubits(rho)
    
    def test_inherits_from_experiment(self, experimental_params, trainable_params):
        """Test that TwoQubitExperiment inherits from Experiment."""
        from qsopt.core.experiment import Experiment
        
        exp = TwoQubitExperiment(experimental_params, trainable_params)
        assert isinstance(exp, Experiment)
    
    def test_has_base_attributes(self, experimental_params, trainable_params):
        """Test that TwoQubitExperiment has all base class attributes."""
        exp = TwoQubitExperiment(experimental_params, trainable_params)
        
        # Check base class attributes
        assert hasattr(exp, 'experimental_params')
        assert hasattr(exp, 'trainable_params')
        assert hasattr(exp, 'operators')
        assert hasattr(exp, 'hamiltonians')
        assert hasattr(exp, 'lindblad_operators')
        assert hasattr(exp, 'callback')
    
    def test_rotation_angles_property_works(self, experimental_params, trainable_params):
        """Test that rotation_angles property from base class works."""
        exp = TwoQubitExperiment(experimental_params, trainable_params)
        
        angles = exp.rotation_angles
        assert isinstance(angles, dict)
        assert len(angles) == 4  # We added 4 rotation angles
