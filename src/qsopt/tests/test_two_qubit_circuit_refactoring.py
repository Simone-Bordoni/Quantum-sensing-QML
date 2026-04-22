"""
Tests for TwoQubitExperiment refactoring with QuantumCircuit integration.

Verifies that using QuantumCircuit produces the same results as the original
hardcoded RY gate implementation.

.. deprecated::
    TwoQubitExperiment has been deprecated.
"""

import jax.numpy as jnp
import numpy as np
import pytest

pytestmark = pytest.mark.skip(reason="TwoQubitExperiment has been deprecated")

from qsopt.core.callback import OptimizationCallback
from qsopt.core.circuit import QuantumCircuit, create_ry_circuit_layer
from qsopt.core.gates import RYGate
from qsopt.core.experimental_parameters import (
    ExperimentalParameters,
    InitialStateConfig,
    InitialStateType,
    MeasurementProtocol,
    NoiseConfiguration,
    PhysicalSetup,
    SystemDimensions,
)
# from qsopt.core.experiment import TwoQubitExperiment


@pytest.fixture
def two_qubit_experimental_params():
    """Create standard two-qubit experimental parameters."""
    physical_setup = PhysicalSetup(
        n_qubits=2, chi=[0.5, 0.5], photon_cavity_coupling=10.0, inverse_pulse_width=1.0
    )

    system_dims = SystemDimensions(field_levels=2, cavity_levels=2, qubit_levels=[2, 2])

    measurement = MeasurementProtocol(measurement_times=[-2.0, -1.0, 0.0, 1.0, 2.0])

    initial_state = InitialStateConfig(state_type=InitialStateType.SINGLE_PHOTON)
    noise_config = NoiseConfiguration(
        depolarizing=[0.0, 0.0], dephasing=[0.0, 0.0], relaxation=[0.0, 0.0]
    )

    return ExperimentalParameters(
        physical_setup=physical_setup,
        system_dims=system_dims,
        measurement=measurement,
        initial_state=initial_state,
        noise_config=noise_config,
    )


def test_two_qubit_default_circuits(two_qubit_experimental_params):
    """Test that TwoQubitExperiment creates default 2-qubit RY circuits when none provided."""
    experiment = TwoQubitExperiment(experimental_params=two_qubit_experimental_params)

    # Should have circuits with 2 trainable parameters each
    assert experiment.initial_circuit is not None
    assert experiment.final_circuit is not None

    initial_params = experiment.initial_circuit.get_trainable_parameters()
    final_params = experiment.final_circuit.get_trainable_parameters()

    assert len(initial_params) == 2, "Initial circuit should have 2 parameters (one per qubit)"
    assert len(final_params) == 2, "Final circuit should have 2 parameters (one per qubit)"

    # Default values should be ±π/2
    initial_values = list(initial_params.values())
    final_values = list(final_params.values())
    np.testing.assert_allclose(initial_values, [np.pi/2, np.pi/2], rtol=1e-5)
    np.testing.assert_allclose(final_values, [-np.pi/2, -np.pi/2], rtol=1e-5)

    # TrainableParameters should have 4 total parameters
    all_params = experiment.trainable_params.get_parameter_vector()
    assert len(all_params) == 4, "Should have 4 total trainable parameters"


def test_two_qubit_custom_circuits(two_qubit_experimental_params):
    """Test TwoQubitExperiment with custom circuit initialization."""
    # Create custom circuits with specific angles
    initial_circuit = create_ry_circuit_layer(num_qubits=2, theta_values=[0.1, 0.2])
    final_circuit = create_ry_circuit_layer(num_qubits=2, theta_values=[0.3, 0.4])

    experiment = TwoQubitExperiment(
        experimental_params=two_qubit_experimental_params,
        initial_circuit=initial_circuit,
        final_circuit=final_circuit,
    )

    # Verify circuits were set correctly
    initial_params = experiment.initial_circuit.get_trainable_parameters()
    final_params = experiment.final_circuit.get_trainable_parameters()

    initial_values = list(initial_params.values())
    final_values = list(final_params.values())
    np.testing.assert_allclose(initial_values, [0.1, 0.2], rtol=1e-5)
    np.testing.assert_allclose(final_values, [0.3, 0.4], rtol=1e-5)


def test_two_qubit_parameter_updates(two_qubit_experimental_params):
    """Test that _update_circuits_from_trainable_params works correctly."""
    # Create experiment with custom initial values
    initial_circuit = create_ry_circuit_layer(num_qubits=2, theta_values=[0.1, 0.2])
    final_circuit = create_ry_circuit_layer(num_qubits=2, theta_values=[0.3, 0.4])

    experiment = TwoQubitExperiment(
        experimental_params=two_qubit_experimental_params,
        initial_circuit=initial_circuit,
        final_circuit=final_circuit,
    )

    # Get initial trainable parameters (should match circuit values)
    params = experiment.trainable_params.get_parameter_vector()
    np.testing.assert_allclose(params, [0.1, 0.2, 0.3, 0.4], rtol=1e-5)

    # Modify circuit parameters directly
    experiment.initial_circuit.set_trainable_parameters({'gate_0_theta': 0.5, 'gate_1_theta': 0.6})
    experiment.final_circuit.set_trainable_parameters({'gate_0_theta': 0.7, 'gate_1_theta': 0.8})

    # Verify circuits were updated
    initial_params = experiment.initial_circuit.get_trainable_parameters()
    final_params = experiment.final_circuit.get_trainable_parameters()

    initial_values = list(initial_params.values())
    final_values = list(final_params.values())
    np.testing.assert_allclose(initial_values, [0.5, 0.6], rtol=1e-5)
    np.testing.assert_allclose(final_values, [0.7, 0.8], rtol=1e-5)


def test_two_qubit_prepare_rotation_gates(two_qubit_experimental_params):
    """Test _prepare_rotation_gates with circuit-based implementation."""
    experiment = TwoQubitExperiment(experimental_params=two_qubit_experimental_params)

    # Call _prepare_rotation_gates with specific angles
    theta1_q1, theta2_q1 = 0.1, 0.2
    theta1_q2, theta2_q2 = 0.3, 0.4

    # Now returns (R1_full, R2_full) - embedded circuit unitaries
    R1_full, R2_full = experiment._prepare_rotation_gates(
        theta1_q1, theta2_q1, theta1_q2, theta2_q2
    )

    # Verify we got QuTiP operators
    import qutip as qt
    assert isinstance(R1_full, qt.Qobj)
    assert isinstance(R2_full, qt.Qobj)

    # Verify the operators act on the full composite Hilbert space
    field_levels = two_qubit_experimental_params.field_levels
    cavity_levels = two_qubit_experimental_params.cavity_levels
    expected_dims = [[field_levels, cavity_levels, 2, 2], [field_levels, cavity_levels, 2, 2]]
    assert R1_full.dims == expected_dims
    assert R2_full.dims == expected_dims

    # Verify circuits were updated with these parameters
    initial_params = experiment.initial_circuit.get_trainable_parameters()
    final_params = experiment.final_circuit.get_trainable_parameters()

    initial_values = list(initial_params.values())
    final_values = list(final_params.values())
    np.testing.assert_allclose(initial_values, [theta1_q1, theta1_q2], rtol=1e-5)
    np.testing.assert_allclose(final_values, [theta2_q1, theta2_q2], rtol=1e-5)


def test_two_qubit_single_shot_with_circuits(two_qubit_experimental_params):
    """Test that single shot measurement works with circuit-based implementation."""
    experiment = TwoQubitExperiment(experimental_params=two_qubit_experimental_params)

    # Run simulation with default parameters
    callback = experiment.run_simulation(batch_size=1)

    # Verify callback has results
    assert len(callback.history["detection_with"]) > 0
    assert len(callback.history["detection_without"]) > 0
    assert len(callback.history["metric"]) > 0

    # Get the latest results
    detection_with = callback.history["detection_with"][-1]
    detection_without = callback.history["detection_without"][-1]

    # Check that detection measures are valid
    assert 0 <= detection_with <= 1
    assert 0 <= detection_without <= 1


def test_two_qubit_optimization_step(two_qubit_experimental_params):
    """Test that optimization can take steps with circuit-based implementation."""
    experiment = TwoQubitExperiment(experimental_params=two_qubit_experimental_params)

    # Store initial parameters
    initial_params = experiment.trainable_params.get_parameter_vector().copy()

    # Take one optimization step
    callback = experiment.optimize_rotations(num_steps=1, verbose=False)

    # Verify parameters changed
    final_params = experiment.trainable_params.get_parameter_vector()
    assert not np.allclose(initial_params, final_params), "Parameters should change after optimization"

    # Verify history was recorded
    assert len(callback.history["metric"]) == 1


def test_two_qubit_multi_gate_circuits(two_qubit_experimental_params):
    """Test TwoQubitExperiment with multi-gate circuits (not just RY)."""
    # Create circuits with multiple gates per qubit
    initial_circuit = QuantumCircuit(num_qubits=2)
    initial_circuit.add_gate(RYGate(theta=0.1, trainable=True), target=0)
    initial_circuit.add_gate(RYGate(theta=0.2, trainable=True), target=1)

    final_circuit = QuantumCircuit(num_qubits=2)
    final_circuit.add_gate(RYGate(theta=0.3, trainable=True), target=0)
    final_circuit.add_gate(RYGate(theta=0.4, trainable=True), target=1)

    experiment = TwoQubitExperiment(
        experimental_params=two_qubit_experimental_params,
        initial_circuit=initial_circuit,
        final_circuit=final_circuit,
    )

    # Run simulation to verify everything works
    callback = experiment.run_simulation(batch_size=1)

    assert len(callback.history["detection_with"]) > 0
    assert len(callback.history["detection_without"]) > 0
    detection_with = callback.history["detection_with"][-1]
    detection_without = callback.history["detection_without"][-1]
    assert 0 <= detection_with <= 1
    assert 0 <= detection_without <= 1


def test_two_qubit_callback_integration(two_qubit_experimental_params):
    """Test that OptimizationCallback works with circuit-based TwoQubitExperiment."""
    callback = OptimizationCallback()
    experiment = TwoQubitExperiment(experimental_params=two_qubit_experimental_params)

    # Run optimization for a few steps with the callback
    result_callback = experiment.optimize_rotations(num_steps=3, callback=callback, verbose=False, tolerance=0.0)

    # Verify callback recorded history (may terminate early if converged)
    num_steps = len(result_callback.history["metric"])
    assert num_steps >= 1, "Should have at least one optimization step"
    assert num_steps <= 3, "Should not exceed requested steps"

    # Verify we have matching history lengths
    assert len(result_callback.history["detection_with"]) == num_steps
    assert len(result_callback.history["detection_without"]) == num_steps

    # Verify all values are valid
    for detection_with, detection_without in zip(
        result_callback.history["detection_with"], result_callback.history["detection_without"]
    ):
        assert 0 <= detection_with <= 1
        assert 0 <= detection_without <= 1


def test_two_qubit_wrong_parameter_count():
    """Test that error is raised if circuits don't have 2 parameters each."""
    physical_setup = PhysicalSetup(n_qubits=2, chi=[0.5, 0.5])
    system_dims = SystemDimensions(field_levels=2, cavity_levels=2, qubit_levels=[2, 2])
    params = ExperimentalParameters(
        physical_setup=physical_setup,
        system_dims=system_dims,
    )

    # Create circuit with wrong number of parameters (3 instead of 2)
    bad_circuit = QuantumCircuit(num_qubits=3)
    bad_circuit.add_gate(RYGate(theta=0.1, trainable=True), target=0)
    bad_circuit.add_gate(RYGate(theta=0.2, trainable=True), target=1)
    bad_circuit.add_gate(RYGate(theta=0.3, trainable=True), target=2)

    good_circuit = create_ry_circuit_layer(num_qubits=2)

    # Should raise ValueError due to wrong parameter count
    with pytest.raises(ValueError, match="Expected 2 trainable parameters"):
        TwoQubitExperiment(
            experimental_params=params,
            initial_circuit=bad_circuit,
            final_circuit=good_circuit,
        )


if __name__ == "__main__":
    # Run a simple smoke test
    print("Running two-qubit circuit refactoring tests...")

    physical_setup = PhysicalSetup(
        n_qubits=2, chi=[0.5, 0.5], photon_cavity_coupling=10.0, inverse_pulse_width=1.0
    )

    system_dims = SystemDimensions(field_levels=2, cavity_levels=2, qubit_levels=[2, 2])

    measurement = MeasurementProtocol(measurement_times=[-2.0, -1.0, 0.0, 1.0, 2.0])

    initial_state = InitialStateConfig(state_type=InitialStateType.SINGLE_PHOTON)
    noise_config = NoiseConfiguration(
        depolarizing=[0.0, 0.0], dephasing=[0.0, 0.0], relaxation=[0.0, 0.0]
    )

    params = ExperimentalParameters(
        physical_setup=physical_setup,
        system_dims=system_dims,
        measurement=measurement,
        initial_state=initial_state,
        noise_config=noise_config,
    )

    # Test default circuits
    print("\n1. Testing default circuit creation...")
    exp = TwoQubitExperiment(experimental_params=params)
    callback = exp.run_simulation(batch_size=1)
    detection_with = callback.history["detection_with"][-1]
    detection_without = callback.history["detection_without"][-1]
    metric_value = callback.history["metric"][-1]
    print(f"   Detection measure with photon: {detection_with:.6f}")
    print(f"   Detection measure without photon: {detection_without:.6f}")
    print(f"   Sensing metric: {metric_value:.6f}")

    # Test optimization
    print("\n2. Testing optimization...")
    callback = OptimizationCallback()
    result_callback = exp.optimize_rotations(num_steps=5, callback=callback, verbose=False)

    # Use result_callback which is returned by the optimization
    if "loss" in result_callback.history:
        print(f"   Initial loss: {result_callback.history['loss'][0]:.6f}")
        print(f"   Final loss: {result_callback.history['loss'][-1]:.6f}")
        print(f"   Loss improved: {result_callback.history['loss'][0] > result_callback.history['loss'][-1]}")
    else:
        # Fallback to metric for this experiment.
        print(f"   Initial metric: {result_callback.history['metric'][0]:.6f}")
        print(f"   Final metric: {result_callback.history['metric'][-1]:.6f}")
        print(f"   Metric improved: {result_callback.history['metric'][0] < result_callback.history['metric'][-1]}")

    print("\n✅ All manual tests passed!")
