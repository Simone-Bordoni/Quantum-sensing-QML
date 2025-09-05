"""
Pytest configuration for qsopt tests.
"""

import numpy as np
import pytest

# Set numpy print options for consistent test output
np.set_printoptions(precision=6, suppress=True)


# Configure pytest fixtures if needed
@pytest.fixture
def default_experimental_parameters():
    """Fixture providing default experimental parameters for testing."""
    from qsopt.core.experimental_parameters import ExperimentalParameters

    return ExperimentalParameters()


@pytest.fixture
def custom_experimental_parameters():
    """Fixture providing custom experimental parameters for testing."""
    from qsopt.core.experimental_parameters import (ExperimentalParameters,
                                                    InitialStateConfig,
                                                    InitialStateType,
                                                    MeasurementProtocol,
                                                    NoiseConfiguration,
                                                    PhysicalConstants,
                                                    SystemDimensions)

    constants = PhysicalConstants(chi=1.0, photon_cavity_coupling=0.5, inverse_pulse_width=0.1)

    dims = SystemDimensions(cavity_levels=2, qubit_levels=2, field_levels=2)

    measurement = MeasurementProtocol(measurement_times=[-5.0, 5.0])

    initial_state = InitialStateConfig(state_type=InitialStateType.SINGLE_PHOTON)

    noise_config = NoiseConfiguration(depolarizing=0.01, dephasing=0.01, relaxation=0.01)

    return ExperimentalParameters(
        physical_constants=constants,
        system_dims=dims,
        measurement=measurement,
        initial_state=initial_state,
        noise_config=noise_config,
    )
