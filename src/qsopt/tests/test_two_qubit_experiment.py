"""Tests for TwoQubitExperiment class."""  # Two qubit tests

import numpy as np
import pytest
import qutip as qt

from qsopt.core.experiment.two_qubit_experiment import TwoQubitExperiment
from qsopt.core.experimental_parameters import (
    ExperimentalParameters,
    InitialStateConfig,
    InitialStateType,
    MeasurementProtocol,
    NoiseConfiguration,
    PhysicalConstants,
    SystemDimensions,
)
from qsopt.core.trainable_parameters import TrainableParameters


@pytest.fixture
def two_qubit_params():
    """Create experimental parameters for two-qubit system."""
    physical_constants = PhysicalConstants(
        n_qubits=2, chi=[10.0, 10.0], photon_cavity_coupling=10.0, inverse_pulse_width=1.0
    )

    system_dims = SystemDimensions(field_levels=2, cavity_levels=2, qubit_levels=[2, 2])

    measurement = MeasurementProtocol(measurement_times=[-2.0, -1.0, 0.0, 1.0, 2.0])

    initial_state = InitialStateConfig(state_type=InitialStateType.SINGLE_PHOTON)
    noise_config = NoiseConfiguration(
        depolarizing=[0.0, 0.0], dephasing=[0.0, 0.0], relaxation=[0.0, 0.0]
    )

    return ExperimentalParameters(
        physical_constants=physical_constants,
        system_dims=system_dims,
        measurement=measurement,
        initial_state=initial_state,
        noise_config=noise_config,
    )


@pytest.fixture
def two_qubit_trainable():
    """Create trainable parameters with 4 rotation angles."""
    trainable = TrainableParameters()
    trainable.add_rotation_angles(
        names=["theta1_q1", "theta1_q2", "theta2_q1", "theta2_q2"],
        initial_values=[0.0, 0.0, 0.0, 0.0],
    )
    return trainable


@pytest.fixture
def experiment(two_qubit_params, two_qubit_trainable):
    """Create two-qubit experiment instance."""
    return TwoQubitExperiment(two_qubit_params, two_qubit_trainable)


def test_initialization(experiment):
    """Test that experiment initializes correctly."""
    assert experiment.experimental_params.n_qubits == 2
    assert experiment.operators is not None


def test_apply_rotation(experiment):
    """Test unified rotation method."""
    rho0 = experiment.get_initial_state()

    # Rotate qubit 0
    rho1 = experiment.apply_rotation(rho0, np.pi / 4, qubit=0)
    assert abs(rho1.tr() - 1.0) < 1e-10

    # Rotate qubit 1
    rho2 = experiment.apply_rotation(rho0, np.pi / 3, qubit=1)
    assert abs(rho2.tr() - 1.0) < 1e-10


def test_prob_single_qubit(experiment):
    """Test probability measurement for single qubits."""
    rho = experiment.get_initial_state()

    # Measure qubit 0
    p0_q0 = experiment.prob(rho, qubits=[0], state="0")
    p1_q0 = experiment.prob(rho, qubits=[0], state="1")
    assert abs(p0_q0 + p1_q0 - 1.0) < 1e-10

    # Measure qubit 1
    p0_q1 = experiment.prob(rho, qubits=[1], state="0")
    p1_q1 = experiment.prob(rho, qubits=[1], state="1")
    assert abs(p0_q1 + p1_q1 - 1.0) < 1e-10


def test_prob_joint(experiment):
    """Test joint probability measurement."""
    rho = experiment.get_initial_state()

    p00 = experiment.prob(rho, qubits=[0, 1], state="00")
    p01 = experiment.prob(rho, qubits=[0, 1], state="01")
    p10 = experiment.prob(rho, qubits=[0, 1], state="10")
    p11 = experiment.prob(rho, qubits=[0, 1], state="11")

    total = p00 + p01 + p10 + p11
    assert abs(total - 1.0) < 1e-10


def test_measure_all_states(experiment):
    """Test convenience method for all joint states."""
    rho = experiment.get_initial_state()
    probs = experiment.measure_all_states(rho)

    assert "00" in probs and "01" in probs and "10" in probs and "11" in probs
    total = sum(probs.values())
    assert abs(total - 1.0) < 1e-10


def test_simulation_runs(experiment):
    """Test that simulation completes without errors."""
    rho0 = experiment.get_initial_state()
    solver = experiment.get_solver_with_interaction()
    measurements = np.linspace(-2, 2, 5)

    prob = experiment.simulation(
        solver=solver,
        rho=rho0,
        theta1_q1=0.0,
        theta2_q1=0.0,
        theta1_q2=0.0,
        theta2_q2=0.0,
        measurements=measurements,
    )

    # Allow for small numerical errors near 0
    assert -1e-10 <= float(prob) <= 1.0


def test_run_simulation_completes(experiment):
    """Test that run_simulation completes and returns callback."""
    callback = experiment.run_simulation(batch_size=1)

    assert callback is not None
    # History contains multiple keys (epochs, prob_with, prob_without, contrast, etc.)
    assert "epochs" in callback.history
    assert len(callback.history["epochs"]) == 1
    assert "prob_with" in callback.history
    assert len(callback.history["prob_with"]) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
