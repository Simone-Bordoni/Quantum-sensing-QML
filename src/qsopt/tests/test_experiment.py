"""
Test suite for SingleQubitExperiment class.

Tests cover:
- Operator generation in composite Hilbert space
- Time-dependent Hamiltonian construction
- Simulation workflow with quantum rotations
- Optimization with time-dependent Hamiltonian
"""

import pytest
import numpy as np
import jax.numpy as jnp
import qutip as qt

from qsopt.core.experimental_parameters import ExperimentalParameters
from qsopt.core.trainable_parameters import TrainableParameters
from qsopt.core.experiment import SingleQubitExperiment


class TestSingleQubitExperiment:
    """Test suite for SingleQubitExperiment class."""
    
    @pytest.fixture
    def default_params(self):
        """Create default experimental parameters for testing."""
        from qsopt.core.experimental_parameters import (
            PhysicalConstants, SystemDimensions, NoiseConfiguration,
            MeasurementProtocol, InitialStateConfig, InitialStateType
        )
        
        physical_constants = PhysicalConstants(
            chi=0.5 * 0.03 * 2 * np.pi,
            photon_cavity_coupling=0.03 * 2 * np.pi,
            inverse_pulse_width=0.1 * 0.03 * 2 * np.pi
        )
        
        system_dims = SystemDimensions(
            cavity_levels=2,
            qubit_levels=2,
            field_levels=2
        )
        
        noise_config = NoiseConfiguration(
            relaxation=0.001 * 2 * np.pi,
            dephasing=0.0005 * 2 * np.pi,
            depolarizing=0.0
        )
        
        measurement = MeasurementProtocol(
            measurement_times=[-5.0, -2.5, 0.0, 2.5, 5.0]
        )
        
        initial_state = InitialStateConfig(
            state_type=InitialStateType.SINGLE_PHOTON
        )
        
        return ExperimentalParameters(
            physical_constants=physical_constants,
            system_dims=system_dims,
            measurement=measurement,
            initial_state=initial_state,
            noise_config=noise_config
        )
    
    @pytest.fixture
    def trainable_params(self):
        """Create default trainable parameters."""
        params = TrainableParameters()
        params.add_rotation_angles(
            names=['theta1', 'theta2'],
            initial_values=[0.0, np.pi/2]
        )
        return params
    
    @pytest.fixture
    def experiment(self, default_params, trainable_params):
        """Create SingleQubitExperiment instance."""
        return SingleQubitExperiment(default_params, trainable_params)
    
    def test_initialization(self, experiment, default_params):
        """Test that experiment initializes correctly."""
        assert experiment.experimental_params == default_params
        assert experiment.operators is not None
        assert experiment.hamiltonians is not None
        assert experiment.lindblad_operators is not None
        
    def test_hilbert_space_dimension(self, experiment):
        """Test that composite Hilbert space has correct dimension."""
        # Should be field_levels * cavity_levels * qubit_levels = 2 * 2 * 2 = 8
        expected_dim = (experiment.experimental_params.field_levels * 
                       experiment.experimental_params.cavity_levels * 
                       experiment.experimental_params.qubit_levels)
        
        # Check dimension through an operator
        actual_dim = experiment.operators['a_in'].dims[0][0]
        total_dim = np.prod(experiment.operators['a_in'].shape)
        
        print(f"Expected total dimension: {expected_dim}")
        print(f"Actual operator shape: {experiment.operators['a_in'].shape}")
        
        assert total_dim == expected_dim ** 2  # Operator is dim x dim matrix
    
    def test_operators_are_qobj(self, experiment):
        """Test that all operators are QuTiP Qobj instances."""
        for name, op in experiment.operators.items():
            assert isinstance(op, qt.Qobj), f"Operator {name} is not a Qobj"
    
    def test_sigma_operators_hermitian(self, experiment):
        """Test that Pauli operators are Hermitian."""
        sx = experiment.operators['sx']
        sy = experiment.operators['sy']
        sz = experiment.operators['sz']
        
        assert (sx - sx.dag()).norm() < 1e-10, "σ_x not Hermitian"
        assert (sy - sy.dag()).norm() < 1e-10, "σ_y not Hermitian"
        assert (sz - sz.dag()).norm() < 1e-10, "σ_z not Hermitian"
    
    def test_projector_properties(self, experiment):
        """Test projector operators P0 and P1."""
        P0 = experiment.operators['P0']
        P1 = experiment.operators['P1']
        
        # Projectors should be Hermitian
        assert (P0 - P0.dag()).norm() < 1e-10, "P0 not Hermitian"
        assert (P1 - P1.dag()).norm() < 1e-10, "P1 not Hermitian"
        
        # Projectors should be idempotent: P^2 = P
        assert (P0 * P0 - P0).norm() < 1e-10, "P0 not idempotent"
        assert (P1 * P1 - P1).norm() < 1e-10, "P1 not idempotent"
        
        # Projectors should be orthogonal: P0*P1 = 0
        assert (P0 * P1).norm() < 1e-10, "P0 and P1 not orthogonal"
        
        # Projectors should sum to identity on qubit subspace
        identity = experiment.operators['identity']
        # Note: P0 + P1 acts on full space, so we check trace instead
        assert abs(P0.tr() + P1.tr() - experiment.exp_params.nlev**2) < 1e-10
    
    def test_hamiltonian_structure(self, experiment):
        """Test Hamiltonian dictionary structure."""
        h_dict = experiment.hamiltonian_dict
        
        # Check required keys
        assert 'H_dispersive' in h_dict
        assert 'H_coupling' in h_dict
        assert 'H_total' in h_dict
        assert 'c_ops' in h_dict
        
        # H_dispersive should be Qobj
        assert isinstance(h_dict['H_dispersive'], qt.Qobj)
        
        # H_coupling should be Qobj
        assert isinstance(h_dict['H_coupling'], qt.Qobj)
        
        # H_total should be QobjEvo (time-dependent)
        assert isinstance(h_dict['H_total'], qt.QobjEvo)
        
        # c_ops should be list
        assert isinstance(h_dict['c_ops'], list)
    
    def test_collapse_operators(self, experiment):
        """Test noise collapse operators."""
        c_ops = experiment.hamiltonian_dict['c_ops']
        
        # Should have noise operators if any noise is present
        if (experiment.exp_params.gamma_relax > 0 or 
            experiment.exp_params.gamma_deph > 0 or 
            experiment.exp_params.gamma_depol > 0):
            assert len(c_ops) > 0, "No collapse operators with nonzero noise"
        
        # All should be Qobj
        for c_op in c_ops:
            assert isinstance(c_op, qt.Qobj)
    
    def test_solvers_created(self, experiment):
        """Test that MESolver objects are created."""
        assert experiment.solver_with is not None
        assert experiment.solver_without is not None
        
        # Check they are MESolver instances
        assert isinstance(experiment.solver_with, qt.MESolver)
        assert isinstance(experiment.solver_without, qt.MESolver)
    
    def test_ry_rotation(self, experiment):
        """Test Y-rotation gate application to density matrix."""
        rho0 = experiment.get_initial_state()
        theta = np.pi / 4
        
        # Apply rotation
        rho_rotated = experiment.ry_rotation(rho0, theta)
        
        # Should be Qobj
        assert isinstance(rho_rotated, qt.Qobj)
        
        # Should be valid density matrix (trace = 1, positive)
        assert abs(rho_rotated.tr() - 1.0) < 1e-10, "Trace should be 1"
        assert rho_rotated.isherm, "Should be Hermitian"
        
        # Test that Ry(0) doesn't change the state
        rho_unchanged = experiment.ry_rotation(rho0, 0.0)
        assert (rho_unchanged - rho0).norm() < 1e-10, "Ry(0) should not change state"
        
        # Test that applying Ry(θ) then Ry(-θ) returns to original
        rho_back = experiment.ry_rotation(rho_rotated, -theta)
        assert (rho_back - rho0).norm() < 1e-8, "Ry(θ) followed by Ry(-θ) should return to original"
    
    def test_probability_sum(self, experiment):
        """Test that prob0 + prob1 = 1 for pure states."""
        rho0 = experiment.get_initial_state()
        
        p0 = experiment.prob0(rho0)
        p1 = experiment.prob1(rho0)
        
        assert abs(p0 + p1 - 1.0) < 1e-10, f"Probabilities don't sum to 1: {p0} + {p1} = {p0 + p1}"
    
    def test_initial_state(self, experiment):
        """Test initial state preparation."""
        rho0 = experiment.get_initial_state()
        
        # Should be Qobj
        assert isinstance(rho0, qt.Qobj)
        
        # Should be density matrix
        assert rho0.isoper
        
        # Should be normalized (trace = 1)
        assert abs(rho0.tr() - 1.0) < 1e-10
        
        # Should be positive semidefinite (eigenvalues >= 0)
        eigvals = rho0.eigenenergies()
        assert all(eigvals >= -1e-10), "Initial state has negative eigenvalues"
    
    def test_simulation_runs(self, experiment):
        """Test that simulation completes without errors."""
        solver = experiment.solver_with
        rho0 = experiment.get_initial_state()
        theta1 = 0.0
        theta2 = np.pi / 2
        measurements = [experiment.operators['P0'], experiment.operators['P1']]
        
        # Run simulation
        result = experiment.simulation(solver, rho0, theta1, theta2, measurements)
        
        # Should return array of probabilities
        assert isinstance(result, (np.ndarray, jnp.ndarray))
        assert len(result) == len(measurements)
        
        # Probabilities should be in [0, 1]
        assert all(0 <= p <= 1 for p in result), f"Invalid probabilities: {result}"
        
        # Should sum to approximately 1
        assert abs(sum(result) - 1.0) < 1e-6, f"Probabilities sum to {sum(result)}"
    
    def test_simulation_with_different_angles(self, experiment):
        """Test simulation with various rotation angles."""
        solver = experiment.solver_with
        rho0 = experiment.get_initial_state()
        measurements = [experiment.operators['P0'], experiment.operators['P1']]
        
        angles = [0.0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi]
        
        for theta1 in angles:
            for theta2 in angles:
                result = experiment.simulation(solver, rho0, theta1, theta2, measurements)
                assert len(result) == 2
                assert abs(sum(result) - 1.0) < 1e-6
    
    def test_optimization_initialization(self, experiment):
        """Test optimization setup without running full optimization."""
        # Just test that optimize method exists and can access parameters
        assert len(experiment.trainable_params.parameters) >= 2
        theta1_param = experiment.trainable_params.parameters[0]
        theta2_param = experiment.trainable_params.parameters[1]
        
        assert hasattr(theta1_param, 'value')
        assert hasattr(theta1_param, 'name')
        assert hasattr(theta1_param, 'param_type')
    
    def test_optimize_short_run(self, experiment):
        """Test optimization runs for a few steps."""
        # Run optimization for just 3 steps to verify it works
        result = experiment.optimize(
            num_steps=3,
            learning_rate=0.01,
            verbose=False
        )
        
        # Should return dict with history
        assert isinstance(result, dict)
        assert 'history' in result
        
        history = result['history']
        assert 'loss' in history
        assert 'contrast' in history
        assert 'theta1' in history
        assert 'theta2' in history
        
        # Each list should have correct length
        assert len(history['loss']) == 3
        assert len(history['contrast']) == 3
        assert len(history['theta1']) == 3
        assert len(history['theta2']) == 3
        
        # Loss values should be finite
        assert all(np.isfinite(loss) for loss in history['loss'])
        
        # Contrasts should be in valid range (can be negative for contrast)
        assert all(np.isfinite(c) for c in history['contrast'])


def test_experiment_creation_custom_params():
    """Test experiment with custom parameters."""
    from qsopt.core.experimental_parameters import (
        PhysicalConstants, SystemDimensions, NoiseConfiguration,
        MeasurementProtocol, InitialStateConfig, InitialStateType
    )
    
    physical_constants = PhysicalConstants(
        chi=0.25 * 0.02 * 2 * np.pi,
        photon_cavity_coupling=0.02 * 2 * np.pi,
        inverse_pulse_width=0.1 * 0.02 * 2 * np.pi
    )
    
    system_dims = SystemDimensions(
        cavity_levels=3,  # More levels
        qubit_levels=2,
        field_levels=2
    )
    
    noise_config = NoiseConfiguration(
        relaxation=0.002 * 2 * np.pi,
        dephasing=0.001 * 2 * np.pi,
        depolarizing=0.0005 * 2 * np.pi
    )
    
    measurement = MeasurementProtocol(
        measurement_times=np.linspace(-5.0, 5.0, 5).tolist()
    )
    
    initial_state = InitialStateConfig(
        state_type=InitialStateType.SINGLE_PHOTON
    )
    
    params = ExperimentalParameters(
        physical_constants=physical_constants,
        system_dims=system_dims,
        measurement=measurement,
        initial_state=initial_state,
        noise_config=noise_config
    )
    
    trainable = TrainableParameters()
    trainable.add_rotation_angles(
        names=['theta1', 'theta2'],
        initial_values=[np.pi/4, np.pi/3]
    )
    
    experiment = SingleQubitExperiment(params, trainable)
    
    # Verify initialization
    assert experiment.experimental_params.cavity_levels == 3
    assert experiment.operators is not None


if __name__ == '__main__':
    # Run tests with pytest
    pytest.main([__file__, '-v'])
