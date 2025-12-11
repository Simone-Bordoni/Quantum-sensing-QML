"""
Test qubit-qubit interaction functionality
"""

import numpy as np
import pytest

from qsopt.core.experiment import TwoQubitExperiment
from qsopt.core.experimental_parameters import (
    ExperimentalParameters,
    InitialStateConfig,
    InitialStateType,
    InteractionType,
    MeasurementProtocol,
    NoiseConfiguration,
    PhysicalConstants,
    QubitInteraction,
    SystemDimensions,
)
from qsopt.core.trainable_parameters import TrainableParameters


def test_interaction_type_enum():
    """Test that InteractionType enum has the expected values."""
    assert InteractionType.ZZ.value == "sz-sz"
    assert InteractionType.XX.value == "sx-sx"
    assert InteractionType.YY.value == "sy-sy"


def test_qubit_interaction_creation():
    """Test creation of QubitInteraction objects."""
    # Default interaction
    interaction = QubitInteraction()
    assert interaction.qubit_indices == (0, 1)
    assert interaction.chi == 0.0
    assert interaction.interaction_type == InteractionType.ZZ

    # Custom interaction
    interaction = QubitInteraction(
        qubit_indices=(0, 1), chi=5.0, interaction_type=InteractionType.XX
    )
    assert interaction.qubit_indices == (0, 1)
    assert interaction.chi == 5.0
    assert interaction.interaction_type == InteractionType.XX


def test_qubit_interaction_canonical_ordering():
    """Test that qubit indices are canonically ordered."""
    # Indices should be reordered: (1, 0) -> (0, 1)
    interaction = QubitInteraction(qubit_indices=(1, 0), chi=3.0)
    assert interaction.qubit_indices == (0, 1)


def test_qubit_interaction_validation():
    """Test validation of QubitInteraction parameters."""
    # Should raise error for same qubit indices
    with pytest.raises(ValueError, match="must refer to different qubits"):
        QubitInteraction(qubit_indices=(0, 0))

    # Should raise error for negative indices
    with pytest.raises(ValueError, match="must be non-negative"):
        QubitInteraction(qubit_indices=(-1, 0))

    # Should raise error for wrong number of indices
    with pytest.raises(ValueError, match="exactly 2 indices"):
        QubitInteraction(qubit_indices=(0, 1, 2))  # type: ignore


def test_physical_constants_default_interactions():
    """Test default qubit interactions in PhysicalConstants."""
    # Single qubit - no interactions by default
    pc_single = PhysicalConstants(n_qubits=1, chi=10.0)
    assert pc_single.qubit_interactions == []

    # Two qubits - no interactions by default (empty list)
    pc_two = PhysicalConstants(n_qubits=2, chi=[10.0, 10.0])
    assert pc_two.qubit_interactions == []


def test_physical_constants_custom_interactions():
    """Test custom qubit interactions in PhysicalConstants."""
    # Create custom interactions
    interactions = [
        QubitInteraction(qubit_indices=(0, 1), chi=5.0, interaction_type=InteractionType.XX),
        QubitInteraction(qubit_indices=(0, 1), chi=3.0, interaction_type=InteractionType.YY),
    ]

    pc = PhysicalConstants(n_qubits=2, chi=[10.0, 10.0], qubit_interactions=interactions)

    assert len(pc.qubit_interactions) == 2
    assert pc.qubit_interactions[0].interaction_type == InteractionType.XX
    assert pc.qubit_interactions[0].chi == 5.0
    assert pc.qubit_interactions[1].interaction_type == InteractionType.YY
    assert pc.qubit_interactions[1].chi == 3.0


def test_physical_constants_interaction_validation():
    """Test validation of qubit interactions."""
    # Should raise error if interaction involves invalid qubit index
    interactions = [QubitInteraction(qubit_indices=(0, 2), chi=5.0)]  # Qubit 2 doesn't exist

    with pytest.raises(ValueError, match="only 2 qubits in system"):
        PhysicalConstants(n_qubits=2, chi=[10.0, 10.0], qubit_interactions=interactions)


def test_two_qubit_experiment_with_zz_interaction():
    """Test two-qubit experiment with ZZ interaction."""
    # Create physical constants with ZZ interaction
    interactions = [
        QubitInteraction(qubit_indices=(0, 1), chi=2.0, interaction_type=InteractionType.ZZ)
    ]

    physical_constants = PhysicalConstants(
        n_qubits=2,
        chi=[10.0, 10.0],
        photon_cavity_coupling=10.0,
        inverse_pulse_width=1.0,
        qubit_interactions=interactions,
    )

    system_dims = SystemDimensions(field_levels=2, cavity_levels=2, qubit_levels=2)

    measurement = MeasurementProtocol(measurement_times=[-2.0, 2.0])

    initial_state = InitialStateConfig(state_type=InitialStateType.SINGLE_PHOTON)
    noise_config = NoiseConfiguration(
        depolarizing=[0.0, 0.0], dephasing=[0.0, 0.0], relaxation=[0.0, 0.0]
    )

    exp_params = ExperimentalParameters(
        physical_constants=physical_constants,
        system_dims=system_dims,
        measurement=measurement,
        initial_state=initial_state,
        noise_config=noise_config,
    )

    trainable_params = TrainableParameters()
    trainable_params.add_rotation_angles(
        names=["theta1_q1", "theta2_q1", "theta1_q2", "theta2_q2"],
        initial_values=[0.0, 0.0, 0.0, 0.0],
    )

    # Create experiment - should not raise error
    exp = TwoQubitExperiment(exp_params, trainable_params)

    # Check that Hamiltonian was created
    assert exp.hamiltonians is not None
    assert "total" in exp.hamiltonians

    # Verify operators are available
    assert exp.operators is not None
    assert "sigma_z1" in exp.operators
    assert "sigma_z2" in exp.operators


def test_two_qubit_experiment_with_xx_yy_interaction():
    """Test two-qubit experiment with XX and YY interactions."""
    # Create physical constants with mixed interactions
    interactions = [
        QubitInteraction(qubit_indices=(0, 1), chi=3.0, interaction_type=InteractionType.XX),
        QubitInteraction(qubit_indices=(0, 1), chi=2.0, interaction_type=InteractionType.YY),
    ]

    physical_constants = PhysicalConstants(
        n_qubits=2,
        chi=[10.0, 10.0],
        photon_cavity_coupling=10.0,
        inverse_pulse_width=1.0,
        qubit_interactions=interactions,
    )

    system_dims = SystemDimensions(field_levels=2, cavity_levels=2, qubit_levels=2)

    measurement = MeasurementProtocol(measurement_times=[-2.0, 2.0])

    initial_state = InitialStateConfig(state_type=InitialStateType.SINGLE_PHOTON)
    noise_config = NoiseConfiguration(
        depolarizing=[0.0, 0.0], dephasing=[0.0, 0.0], relaxation=[0.0, 0.0]
    )

    exp_params = ExperimentalParameters(
        physical_constants=physical_constants,
        system_dims=system_dims,
        measurement=measurement,
        initial_state=initial_state,
        noise_config=noise_config,
    )

    trainable_params = TrainableParameters()
    trainable_params.add_rotation_angles(
        names=["theta1_q1", "theta2_q1", "theta1_q2", "theta2_q2"],
        initial_values=[np.pi / 2, 0.0, np.pi / 2, 0.0],
    )

    # Create experiment - should not raise error
    exp = TwoQubitExperiment(exp_params, trainable_params)

    # Verify operators are available for XX and YY interactions
    assert exp.operators is not None
    assert "sigma_x1" in exp.operators
    assert "sigma_x2" in exp.operators
    assert "sigma_y1" in exp.operators
    assert "sigma_y2" in exp.operators


def test_experimental_parameters_repr_with_interactions():
    """Test that __repr__ correctly displays qubit interactions."""
    interactions = [
        QubitInteraction(qubit_indices=(0, 1), chi=5.0, interaction_type=InteractionType.ZZ),
        QubitInteraction(qubit_indices=(0, 1), chi=3.0, interaction_type=InteractionType.XX),
    ]

    physical_constants = PhysicalConstants(
        n_qubits=2, chi=[10.0, 10.0], qubit_interactions=interactions
    )

    exp_params = ExperimentalParameters(physical_constants=physical_constants)

    repr_str = repr(exp_params)

    # Check that interactions are displayed
    assert "Qubit interactions:" in repr_str
    assert "2 interaction(s)" in repr_str
    assert "sz-sz" in repr_str
    assert "sx-sx" in repr_str
    assert "χ=5.0000" in repr_str
    assert "χ=3.0000" in repr_str


def test_single_qubit_experiment_ignores_interactions():
    """Test that single qubit experiment works with no interactions."""
    from qsopt.core.experiment import SingleQubitExperiment

    physical_constants = PhysicalConstants(
        n_qubits=1, chi=10.0, photon_cavity_coupling=10.0, inverse_pulse_width=1.0
    )

    system_dims = SystemDimensions(field_levels=2, cavity_levels=2, qubit_levels=2)

    measurement = MeasurementProtocol(measurement_times=[-2.0, 2.0])

    initial_state = InitialStateConfig(state_type=InitialStateType.SINGLE_PHOTON)
    noise_config = NoiseConfiguration(depolarizing=0.0, dephasing=0.0, relaxation=0.0)

    exp_params = ExperimentalParameters(
        physical_constants=physical_constants,
        system_dims=system_dims,
        measurement=measurement,
        initial_state=initial_state,
        noise_config=noise_config,
    )

    trainable_params = TrainableParameters()
    trainable_params.add_rotation_angles(names=["theta1", "theta2"], initial_values=[0.0, 0.0])

    # Create experiment - should work fine with no interactions
    exp = SingleQubitExperiment(exp_params, trainable_params)

    # Verify experiment was created successfully
    assert exp.hamiltonians is not None
    assert exp.operators is not None

    # Verify no interaction terms (empty list for single qubit)
    assert exp.experimental_params.physical_constants.qubit_interactions == []
