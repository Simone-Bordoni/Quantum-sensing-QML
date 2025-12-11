"""
Test suite for SingleQubitExperiment class.

Tests cover:
- Operator generation in composite Hilbert space
- Time-dependent Hamiltonian construction
- Simulation workflow with quantum rotations
- Optimization with time-dependent Hamiltonian
"""

import jax.numpy as jnp
import numpy as np
import optax
import pytest
import qutip as qt

from qsopt.core.callback import OptimizationCallback
from qsopt.core.experiment import SingleQubitExperiment
from qsopt.core.experimental_parameters import ExperimentalParameters
from qsopt.core.trainable_parameters import ParameterType, TrainableParameters


class TestSingleQubitExperiment:
    """Test suite for SingleQubitExperiment class."""

    @pytest.fixture
    def default_params(self):
        """Create default experimental parameters for testing."""
        from qsopt.core.experimental_parameters import (
            InitialStateConfig,
            InitialStateType,
            MeasurementProtocol,
            NoiseConfiguration,
            PhysicalConstants,
            SystemDimensions,
        )

        physical_constants = PhysicalConstants(
            chi=0.5 * 0.03 * 2 * np.pi,
            photon_cavity_coupling=0.03 * 2 * np.pi,
            inverse_pulse_width=0.1 * 0.03 * 2 * np.pi,
        )

        system_dims = SystemDimensions(cavity_levels=2, qubit_levels=2, field_levels=2)

        noise_config = NoiseConfiguration(
            relaxation=0.001 * 2 * np.pi, dephasing=0.0005 * 2 * np.pi, depolarizing=0.0
        )

        measurement = MeasurementProtocol(measurement_times=[-5.0, -2.5, 0.0, 2.5, 5.0])

        initial_state = InitialStateConfig(state_type=InitialStateType.SINGLE_PHOTON)

        return ExperimentalParameters(
            physical_constants=physical_constants,
            system_dims=system_dims,
            measurement=measurement,
            initial_state=initial_state,
            noise_config=noise_config,
        )

    @pytest.fixture
    def trainable_params(self):
        """Create default trainable parameters."""
        params = TrainableParameters()
        params.add_rotation_angles(names=["theta1", "theta2"], initial_values=[0.0, np.pi / 2])
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
        # qubit_levels is now a list, so extract first element for single qubit
        qubit_levels = experiment.experimental_params.qubit_levels
        if isinstance(qubit_levels, list):
            qubit_levels = qubit_levels[0]

        expected_dim = (
            experiment.experimental_params.field_levels
            * experiment.experimental_params.cavity_levels
            * qubit_levels
        )

        # Check dimension through an operator
        actual_dim = experiment.operators["a_in"].dims[0][0]
        total_dim = np.prod(experiment.operators["a_in"].shape)

        print(f"Expected total dimension: {expected_dim}")
        print(f"Actual operator shape: {experiment.operators['a_in'].shape}")

        assert total_dim == expected_dim**2  # Operator is dim x dim matrix

    def test_operators_are_qobj(self, experiment):
        """Test that all operators are QuTiP Qobj instances."""
        for name, op in experiment.operators.items():
            assert isinstance(op, qt.Qobj), f"Operator {name} is not a Qobj"

    def test_sigma_operators_hermitian(self, experiment):
        """Test that Pauli operators are Hermitian."""
        sigma_x = experiment.operators["sigma_x"]
        sigma_y = experiment.operators["sigma_y"]
        sigma_z = experiment.operators["sigma_z"]

        assert (sigma_x - sigma_x.dag()).norm() < 1e-10, "σ_x not Hermitian"
        assert (sigma_y - sigma_y.dag()).norm() < 1e-10, "σ_y not Hermitian"
        assert (sigma_z - sigma_z.dag()).norm() < 1e-10, "σ_z not Hermitian"

    def test_projector_properties(self, experiment):
        """Test projector operators P0 and P1."""
        P0 = experiment.operators["P0"]
        P1 = experiment.operators["P1"]

        # Projectors should be Hermitian
        assert (P0 - P0.dag()).norm() < 1e-10, "P0 not Hermitian"
        assert (P1 - P1.dag()).norm() < 1e-10, "P1 not Hermitian"

        # Projectors should be idempotent: P^2 = P
        assert (P0 * P0 - P0).norm() < 1e-10, "P0 not idempotent"
        assert (P1 * P1 - P1).norm() < 1e-10, "P1 not idempotent"

        # Projectors should be orthogonal: P0*P1 = 0
        assert (P0 * P1).norm() < 1e-10, "P0 and P1 not orthogonal"

        # Projectors should sum to identity on qubit subspace (in full composite space)
        # P0 + P1 = I_field ⊗ I_cavity ⊗ I_qubit, check by trace
        qubit_levels = experiment.experimental_params.qubit_levels
        if isinstance(qubit_levels, list):
            qubit_levels = qubit_levels[0]

        total_dim = (
            experiment.experimental_params.field_levels
            * experiment.experimental_params.cavity_levels
            * qubit_levels
        )
        assert abs(P0.tr() + P1.tr() - total_dim) < 1e-10

    def test_hamiltonian_structure(self, experiment):
        """Test Hamiltonian dictionary structure."""
        h_dict = experiment.hamiltonians

        # Check required keys
        assert "dispersive" in h_dict
        assert "total" in h_dict

        # H_dispersive should be Qobj
        assert isinstance(h_dict["dispersive"], qt.Qobj)

        # H_total should be QobjEvo (time-dependent)
        assert isinstance(h_dict["total"], qt.QobjEvo)

    def test_collapse_operators(self, experiment):
        """Test noise collapse operators."""
        lindblad_ops = experiment.lindblad_operators

        # Should have interaction and no_interaction keys
        assert "interaction" in lindblad_ops
        assert "no_interaction" in lindblad_ops

        # Should have noise operators if any noise is present
        noise_config = experiment.experimental_params.noise_config

        # Handle both scalar and list noise rates
        relaxation = noise_config.relaxation
        dephasing = noise_config.dephasing
        depolarizing = noise_config.depolarizing

        if isinstance(relaxation, list):
            relaxation = relaxation[0]
        if isinstance(dephasing, list):
            dephasing = dephasing[0]
        if isinstance(depolarizing, list):
            depolarizing = depolarizing[0]

        if relaxation > 0 or dephasing > 0 or depolarizing > 0:
            assert (
                len(lindblad_ops["no_interaction"]) > 0
            ), "No collapse operators with nonzero noise"

        # All should be Qobj or QobjEvo
        for c_op in lindblad_ops["interaction"]:
            assert isinstance(c_op, (qt.Qobj, qt.QobjEvo))

    def test_solvers_created(self, experiment):
        """Test that MESolver objects are created."""
        solver_with = experiment.get_solver_with_interaction()
        solver_without = experiment.get_solver_no_interaction()

        # Check they are MESolver instances
        assert isinstance(solver_with, qt.MESolver)
        assert isinstance(solver_without, qt.MESolver)

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

        # Should be hermitian (valid density matrix)
        assert rho0.isherm

    def test_simulation_runs(self, experiment):
        """Test that simulation completes without errors."""
        solver = experiment.get_solver_with_interaction()
        rho0 = experiment.get_initial_state()
        theta1 = 0.0
        theta2 = np.pi / 2
        measurement_times = experiment.experimental_params.measurement_times

        # Run simulation
        result = experiment.simulation(solver, rho0, theta1, theta2, measurement_times)

        # Should return a single probability value (JAX array or float)
        assert isinstance(result, (float, np.ndarray, jnp.ndarray))

        # Probability should be in [0, 1]
        prob_val = float(result) if hasattr(result, "__float__") else result
        assert 0 <= prob_val <= 1, f"Invalid probability: {prob_val}"

    def test_simulation_with_different_angles(self, experiment):
        """Test simulation with various rotation angles."""
        solver = experiment.get_solver_with_interaction()
        rho0 = experiment.get_initial_state()
        measurement_times = experiment.experimental_params.measurement_times

        # Use a smaller set of angles to avoid solver timeouts
        # Set seed for reproducibility
        np.random.seed(42)
        test_angles = [
            (0.0, 0.0),  # Identity
            (np.pi / 4, np.pi / 4),  # Equal rotations
            (np.pi / 2, 0.0),  # First gate only
            (0.0, np.pi / 2),  # Second gate only
            (np.pi / 4, -np.pi / 4),  # Opposite rotations
        ]

        for theta1, theta2 in test_angles:
            result = experiment.simulation(solver, rho0, theta1, theta2, measurement_times)
            prob_val = float(result) if hasattr(result, "__float__") else result
            # Allow small numerical errors (e.g., -1e-15 is effectively 0)
            assert (
                -1e-10 <= prob_val <= 1 + 1e-10
            ), f"Invalid probability for θ1={theta1:.3f}, θ2={theta2:.3f}: {prob_val}"

    def test_optimization_initialization(self, experiment):
        """Test optimization setup without running full optimization."""
        # Just test that optimize method exists and can access parameters
        assert len(experiment.trainable_params.parameters) >= 2
        theta1_param = experiment.trainable_params.parameters[0]
        theta2_param = experiment.trainable_params.parameters[1]

        assert hasattr(theta1_param, "value")
        assert hasattr(theta1_param, "name")
        assert hasattr(theta1_param, "param_type")

    def test_run_simulation(self, experiment):
        """Test run_simulation returns callback with single epoch."""
        # Run simulation with current parameters
        result = experiment.run_simulation()

        # Should return OptimizationCallback
        assert isinstance(result, OptimizationCallback)

        # Should have exactly 1 epoch
        assert result.epoch == 1
        assert len(result.history["epochs"]) == 1
        assert len(result.history["contrast"]) == 1

        # Should have valid metrics
        assert "prob_with" in result.history
        assert "prob_without" in result.history
        assert len(result.history["prob_with"]) == 1
        assert len(result.history["prob_without"]) == 1

        # Probabilities should be in [0, 1]
        assert 0 <= result.history["prob_with"][0] <= 1
        assert 0 <= result.history["prob_without"][0] <= 1

        # Optimization-related attributes should be False/None (not from optimization)
        assert result.converged is False
        assert result.final_grad_norm is None

        # Should have best_trainable_params set
        assert result.best_trainable_params is not None
        assert result.best_metrics is not None

        # Test __repr__ works without errors
        repr_str = repr(result)
        assert "MODE: Single Simulation" in repr_str
        assert "Current Parameters" in repr_str

    def test_optimize_short_run(self, experiment):
        """Test optimization runs for a few steps."""
        # Run optimization for just 3 steps to verify it works
        result = experiment.optimize_rotations(num_steps=3, verbose=False)

        # Should return OptimizationCallback directly
        assert isinstance(result, OptimizationCallback)

        # Verify callback is the same instance as experiment.callback
        assert result is experiment.callback

        # Check that callback has tracked epochs (may converge early)
        assert result.epoch >= 1  # At least one iteration
        assert result.epoch <= 3  # No more than requested
        assert len(result.history["epochs"]) == result.epoch
        assert len(result.history["contrast"]) == result.epoch

        # Contrasts should be finite and in valid range (can be negative for contrast)
        assert all(np.isfinite(c) for c in result.history["contrast"])

        # Verify convergence info is set
        assert hasattr(result, "converged")
        assert hasattr(result, "final_grad_norm")
        assert isinstance(result.converged, bool)
        assert result.final_grad_norm is not None

        # Test __repr__ works for optimization callback
        repr_str = repr(result)
        assert "MODE: Optimization" in repr_str
        assert "Best Parameters" in repr_str


def test_experiment_creation_custom_params():
    """Test experiment with custom parameters."""
    from qsopt.core.experimental_parameters import (
        InitialStateConfig,
        InitialStateType,
        MeasurementProtocol,
        NoiseConfiguration,
        PhysicalConstants,
        SystemDimensions,
    )

    physical_constants = PhysicalConstants(
        chi=0.25 * 0.02 * 2 * np.pi,
        photon_cavity_coupling=0.02 * 2 * np.pi,
        inverse_pulse_width=0.1 * 0.02 * 2 * np.pi,
    )

    system_dims = SystemDimensions(cavity_levels=3, qubit_levels=2, field_levels=2)  # More levels

    noise_config = NoiseConfiguration(
        relaxation=0.002 * 2 * np.pi, dephasing=0.001 * 2 * np.pi, depolarizing=0.0005 * 2 * np.pi
    )

    measurement = MeasurementProtocol(measurement_times=np.linspace(-5.0, 5.0, 5).tolist())

    initial_state = InitialStateConfig(state_type=InitialStateType.SINGLE_PHOTON)

    params = ExperimentalParameters(
        physical_constants=physical_constants,
        system_dims=system_dims,
        measurement=measurement,
        initial_state=initial_state,
        noise_config=noise_config,
    )

    trainable = TrainableParameters()
    trainable.add_rotation_angles(names=["theta1", "theta2"], initial_values=[np.pi / 4, np.pi / 3])

    experiment = SingleQubitExperiment(params, trainable)

    # Verify initialization
    assert experiment.experimental_params.cavity_levels == 3
    assert experiment.operators is not None


def test_initial_state_vacuum():
    """Test VACUUM initial state generation."""
    from qsopt.core.experimental_parameters import InitialStateConfig, InitialStateType

    initial_state = InitialStateConfig(state_type=InitialStateType.VACUUM)
    params = ExperimentalParameters(initial_state=initial_state)
    trainable = TrainableParameters()
    trainable.add_rotation_angles(["ry1", "ry2"], [0.5, 1.0])

    experiment = SingleQubitExperiment(params, trainable)
    rho0 = experiment.get_initial_state()

    # Verify it's a valid density matrix
    assert np.isclose(rho0.tr(), 1.0, atol=1e-10)
    assert rho0.isherm
    assert rho0.dims == [[2, 2, 2], [2, 2, 2]]


def test_initial_state_single_photon():
    """Test SINGLE_PHOTON initial state generation."""
    from qsopt.core.experimental_parameters import InitialStateConfig, InitialStateType

    initial_state = InitialStateConfig(state_type=InitialStateType.SINGLE_PHOTON)
    params = ExperimentalParameters(initial_state=initial_state)
    trainable = TrainableParameters()
    trainable.add_rotation_angles(["ry1", "ry2"], [0.5, 1.0])

    experiment = SingleQubitExperiment(params, trainable)
    rho0 = experiment.get_initial_state()

    # Verify it's a valid density matrix
    assert np.isclose(rho0.tr(), 1.0, atol=1e-10)
    assert rho0.isherm
    assert rho0.dims == [[2, 2, 2], [2, 2, 2]]


def test_initial_state_coherent():
    """Test COHERENT initial state generation."""
    from qsopt.core.experimental_parameters import InitialStateConfig, InitialStateType

    alpha = 0.5 + 0.3j
    initial_state = InitialStateConfig(state_type=InitialStateType.COHERENT, coherent_alpha=alpha)
    params = ExperimentalParameters(initial_state=initial_state)
    trainable = TrainableParameters()
    trainable.add_rotation_angles(["ry1", "ry2"], [0.5, 1.0])

    experiment = SingleQubitExperiment(params, trainable)
    rho0 = experiment.get_initial_state()

    # Verify it's a valid density matrix
    assert np.isclose(rho0.tr(), 1.0, atol=1e-10)
    assert rho0.isherm
    assert rho0.dims == [[2, 2, 2], [2, 2, 2]]


def test_initial_state_thermal():
    """Test THERMAL initial state generation."""
    from qsopt.core.experimental_parameters import InitialStateConfig, InitialStateType

    initial_state = InitialStateConfig(state_type=InitialStateType.THERMAL, thermal_n_bar=1.5)
    params = ExperimentalParameters(initial_state=initial_state)
    trainable = TrainableParameters()
    trainable.add_rotation_angles(["ry1", "ry2"], [0.5, 1.0])

    experiment = SingleQubitExperiment(params, trainable)
    rho0 = experiment.get_initial_state()

    # Verify it's a valid density matrix
    assert np.isclose(rho0.tr(), 1.0, atol=1e-10)
    assert rho0.isherm
    assert rho0.dims == [[2, 2, 2], [2, 2, 2]]


def test_initial_state_custom():
    """Test CUSTOM initial state generation."""
    from qsopt.core.experimental_parameters import InitialStateConfig, InitialStateType

    # Create superposition: (|0,0,0⟩ + |1,0,0⟩)/√2
    custom_amplitudes = {
        (0, 0, 0): 1.0 / np.sqrt(2),
        (1, 0, 0): 1.0 / np.sqrt(2),
    }

    initial_state = InitialStateConfig(
        state_type=InitialStateType.CUSTOM, custom_amplitudes=custom_amplitudes
    )
    params = ExperimentalParameters(initial_state=initial_state)
    trainable = TrainableParameters()
    trainable.add_rotation_angles(["ry1", "ry2"], [0.5, 1.0])

    experiment = SingleQubitExperiment(params, trainable)
    rho0 = experiment.get_initial_state()

    # Verify it's a valid density matrix
    assert np.isclose(rho0.tr(), 1.0, atol=1e-10)
    assert rho0.isherm
    assert rho0.dims == [[2, 2, 2], [2, 2, 2]]


def test_initial_state_custom_complex():
    """Test CUSTOM initial state with complex amplitudes."""
    from qsopt.core.experimental_parameters import InitialStateConfig, InitialStateType

    # Create complex superposition with phases
    custom_amplitudes = {
        (0, 0, 0): 0.5,
        (1, 0, 0): 0.5,
        (0, 1, 0): 0.5 * np.exp(1j * np.pi / 4),
        (0, 0, 1): 0.5 * np.exp(1j * np.pi / 2),
    }

    initial_state = InitialStateConfig(
        state_type=InitialStateType.CUSTOM, custom_amplitudes=custom_amplitudes
    )
    params = ExperimentalParameters(initial_state=initial_state)
    trainable = TrainableParameters()
    trainable.add_rotation_angles(["ry1", "ry2"], [0.5, 1.0])

    experiment = SingleQubitExperiment(params, trainable)
    rho0 = experiment.get_initial_state()

    # Verify it's a valid density matrix
    assert np.isclose(rho0.tr(), 1.0, atol=1e-10)
    assert rho0.isherm
    assert rho0.dims == [[2, 2, 2], [2, 2, 2]]


def test_optimize_with_theta_init():
    """Test optimization with custom initial angles."""
    params = ExperimentalParameters()
    trainable = TrainableParameters()
    trainable.add_rotation_angles(
        ["ry1", "ry2"], [np.pi / 2, -np.pi / 2], optimizer=optax.adam(0.05)
    )

    experiment = SingleQubitExperiment(params, trainable)

    # Test with theta_init parameter
    result = experiment.optimize_rotations(
        num_steps=2, theta_init=[np.pi / 4, -np.pi / 4], verbose=False
    )

    assert isinstance(result, OptimizationCallback)
    assert result.epoch == 2


def test_optimize_with_property_theta_init():
    """Test optimization using theta_init parameter."""
    params = ExperimentalParameters()
    trainable = TrainableParameters()
    trainable.add_rotation_angles(
        ["ry1", "ry2"], [np.pi / 2, -np.pi / 2], optimizer=optax.sgd(0.01)
    )

    experiment = SingleQubitExperiment(params, trainable)

    # Pass theta_init directly to optimize
    result = experiment.optimize_rotations(
        num_steps=2, theta_init=[np.pi / 3, -np.pi / 3], verbose=False
    )

    assert isinstance(result, OptimizationCallback)


def test_rotation_angles_property():
    """Test rotation_angles property getter and setter."""
    params = ExperimentalParameters()
    trainable = TrainableParameters()
    trainable.add_rotation_angles(["ry1", "ry2"], [1.0, 2.0])

    experiment = SingleQubitExperiment(params, trainable)

    # Get angles
    angles = experiment.rotation_angles
    assert "ry1" in angles
    assert "ry2" in angles
    assert np.isclose(angles["ry1"], 1.0)
    assert np.isclose(angles["ry2"], 2.0)

    # Set angles
    experiment.rotation_angles = {"ry1": 0.5, "ry2": 1.5}
    angles = experiment.rotation_angles
    assert np.isclose(angles["ry1"], 0.5)
    assert np.isclose(angles["ry2"], 1.5)


def test_parameter_constraints_applied():
    """Test that parameter constraints (0 to 2π) are applied after optimization."""
    params = ExperimentalParameters()
    trainable = TrainableParameters()
    # Start with angle > 2π
    trainable.add_rotation_angles(["ry1", "ry2"], [3 * np.pi, -np.pi], optimizer=optax.sgd(0.001))

    experiment = SingleQubitExperiment(params, trainable)

    # Run optimization (even for 1 step)
    experiment.optimize_rotations(num_steps=1, verbose=False)

    # Check that constraints were applied (angles should be in [0, 2π])
    angles = experiment.rotation_angles
    for angle in angles.values():
        assert 0 <= angle < 2 * np.pi, f"Angle {angle} not in [0, 2π)"


def test_optimizer_from_trainable_params():
    """Test that optimizer is taken from trainable_params."""
    params = ExperimentalParameters()
    trainable = TrainableParameters()

    # Add with specific optimizer
    custom_optimizer = optax.rmsprop(0.02)
    trainable.add_rotation_angles(
        ["ry1", "ry2"], [np.pi / 2, -np.pi / 2], optimizer=custom_optimizer
    )

    experiment = SingleQubitExperiment(params, trainable)

    # Should use the custom optimizer
    result = experiment.optimize_rotations(num_steps=2, verbose=False)
    assert isinstance(result, OptimizationCallback)


def test_optimize_measurement_times_updates_interval():
    """Test measurement time optimization applies the best interval."""
    params = ExperimentalParameters()
    params.measurement.initial_time = -2.0
    params.measurement.final_time = 2.0
    params.measurement.time_interval = 0.6
    params.measurement.measurement_times = None
    params._update_measurement_times()

    trainable = TrainableParameters()
    trainable.add_rotation_angles(["ry1", "ry2"], [np.pi / 2, -np.pi / 2])
    trainable.add_measurement_interval("time_interval", params.measurement.time_interval)

    experiment = SingleQubitExperiment(params, trainable)

    results = experiment.optimize_measurement_times(
        resolution=4,
        mode="discrete",
        batch_size=2,
        verbose=False,
    )

    assert "best_interval" in results
    best_interval = float(results["best_interval"])
    assert best_interval > 0
    assert np.isclose(experiment.experimental_params.measurement.time_interval, best_interval)

    measurement_params = [
        p for p in trainable.parameters if p.param_type == ParameterType.MEASUREMENT_TIME
    ]
    assert measurement_params
    assert np.isclose(measurement_params[0].value, best_interval)

    times_list = experiment.experimental_params._measurement_times_list
    assert times_list is not None and len(times_list) >= 2


    def test_get_measurement_interval(self, experiment):
        """Test get_measurement_interval method."""
        interval = experiment.get_measurement_interval()
        assert isinstance(interval, (float, np.floating))
        assert interval > 0

    def test_get_initial_state(self, experiment):
        """Test get_initial_state returns correct state."""
        state = experiment.get_initial_state()
        assert isinstance(state, qt.Qobj)
        assert state.isket

    def test_get_solver_with_interaction(self, experiment):
        """Test solver with interaction can be created."""
        solver = experiment.get_solver_with_interaction()
        assert isinstance(solver, qt.MESolver)

    def test_get_solver_no_interaction(self, experiment):
        """Test solver without interaction can be created."""
        solver = experiment.get_solver_no_interaction()
        assert isinstance(solver, qt.MESolver)

    def test_run_simulation_basic(self, experiment):
        """Test basic run_simulation."""
        result = experiment.run_simulation(batch_size=1)
        assert isinstance(result, OptimizationCallback)
        assert result.epoch == 1

    def test_run_simulation_with_batch(self, experiment):
        """Test run_simulation with batch processing."""
        result = experiment.run_simulation(batch_size=2)
        assert isinstance(result, OptimizationCallback)
        assert result.epoch == 1


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
