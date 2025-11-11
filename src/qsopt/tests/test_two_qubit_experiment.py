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
            n_qubits=2,
            chi=[0.01, 0.015],  # Different chi for each qubit
            photon_cavity_coupling=0.1,
            inverse_pulse_width=1.0
        ),
        system_dims=SystemDimensions(
            cavity_levels=3,
            qubit_levels=[2, 2],  # Two qubits
            field_levels=3
        ),
        measurement=MeasurementProtocol(
            initial_time=0.0,
            final_time=10.0,
            time_interval=1.0
        ),
        noise_config=NoiseConfiguration(
            depolarizing=[0.0, 0.0],
            dephasing=[0.0, 0.0],
            relaxation=[0.0, 0.0]
        ),
        initial_state=InitialStateConfig(state_type=InitialStateType.SINGLE_PHOTON)
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
        assert hasattr(exp, '_cached_joint_projectors')
        assert hasattr(exp, '_cached_qubit1_projectors')
        assert hasattr(exp, '_cached_qubit2_projectors')
        assert isinstance(exp._cached_joint_projectors, dict)
        assert isinstance(exp._cached_qubit1_projectors, dict)
        assert isinstance(exp._cached_qubit2_projectors, dict)
    
    def test_generate_operators_implemented(self, experimental_params, trainable_params):
        """Test that _generate_operators creates two-qubit operators."""
        exp = TwoQubitExperiment(experimental_params, trainable_params)
        
        # Check that operators were generated
        assert exp.operators is not None
        assert isinstance(exp.operators, dict)
        
        # Check for two-qubit specific operators
        assert 'sigma_z1' in exp.operators
        assert 'sigma_z2' in exp.operators
        assert 'P00' in exp.operators
        assert 'P01' in exp.operators
        assert 'P10' in exp.operators
        assert 'P11' in exp.operators
        assert 'roty_q1' in exp.operators
        assert 'roty_q2' in exp.operators
    
    def test_generate_hamiltonian_implemented(self, experimental_params, trainable_params):
        """Test that _generate_hamiltonian creates two-qubit Hamiltonian."""
        exp = TwoQubitExperiment(experimental_params, trainable_params)
        
        # Check that Hamiltonians were created
        assert exp.hamiltonians is not None
        assert 'total' in exp.hamiltonians
        assert 'dispersive' in exp.hamiltonians
        assert 'dispersive1' in exp.hamiltonians
        assert 'dispersive2' in exp.hamiltonians
    
    def test_initialize_caches_implemented(self, experimental_params, trainable_params):
        """Test that _initialize_caches creates proper cache structures."""
        exp = TwoQubitExperiment(experimental_params, trainable_params)
        
        # Check that caches are initialized
        assert exp._cached_initial_state is not None
        assert exp._cached_joint_projectors is not None
        assert exp._cached_qubit1_projectors is not None
        assert exp._cached_qubit2_projectors is not None
        
        # Check joint projectors
        assert '00' in exp._cached_joint_projectors
        assert '01' in exp._cached_joint_projectors
        assert '10' in exp._cached_joint_projectors
        assert '11' in exp._cached_joint_projectors
        
        # Check individual qubit projectors
        assert '0' in exp._cached_qubit1_projectors
        assert '1' in exp._cached_qubit1_projectors
        assert '0' in exp._cached_qubit2_projectors
        assert '1' in exp._cached_qubit2_projectors
    
    def test_get_initial_state_implemented(self, experimental_params, trainable_params):
        """Test that get_initial_state returns cached state."""
        exp = TwoQubitExperiment(experimental_params, trainable_params)
        
        state = exp.get_initial_state()
        assert state is not None
        assert state.isoper  # Should be a density matrix
    
    def test_get_joint_projector(self, experimental_params, trainable_params):
        """Test that get_joint_projector returns correct projectors."""
        exp = TwoQubitExperiment(experimental_params, trainable_params)
        
        # Test each joint state
        P00 = exp.get_joint_projector('00')
        P01 = exp.get_joint_projector('01')
        P10 = exp.get_joint_projector('10')
        P11 = exp.get_joint_projector('11')
        
        assert P00 is not None
        assert P01 is not None
        assert P10 is not None
        assert P11 is not None
        
        # Test invalid state raises error
        with pytest.raises(ValueError, match="Invalid joint state"):
            exp.get_joint_projector('invalid')
    
    def test_get_qubit_projector(self, experimental_params, trainable_params):
        """Test that get_qubit_projector returns correct projectors."""
        exp = TwoQubitExperiment(experimental_params, trainable_params)
        
        # Test qubit 1
        P0_q1 = exp.get_qubit_projector(1, '0')
        P1_q1 = exp.get_qubit_projector(1, '1')
        
        assert P0_q1 is not None
        assert P1_q1 is not None
        
        # Test qubit 2
        P0_q2 = exp.get_qubit_projector(2, '0')
        P1_q2 = exp.get_qubit_projector(2, '1')
        
        assert P0_q2 is not None
        assert P1_q2 is not None
        
        # Test invalid qubit index
        with pytest.raises(ValueError, match="Invalid qubit index"):
            exp.get_qubit_projector(3, '0')
        
        # Test invalid state
        with pytest.raises(ValueError, match="Invalid qubit state"):
            exp.get_qubit_projector(1, 'invalid')
    
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
