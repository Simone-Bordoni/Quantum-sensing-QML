"""
Comprehensive tests for experimental_parameters module.

This module tests all aspects of the ExperimentalParameters class including:
- Initialization with default and custom parameters
- Validation of parameter constraints
- Backward compatibility properties
- Error handling and edge cases
- Property setters and getters
- String representations
"""

import numpy as np
import pytest

from qsopt.core.experimental_parameters import (ExperimentalParameters,
                                                InitialStateConfig,
                                                InitialStateType,
                                                MeasurementProtocol,
                                                NoiseConfiguration,
                                                PhysicalConstants,
                                                SystemDimensions)


class TestPhysicalConstants:
    """Test the PhysicalConstants dataclass."""

    def test_default_initialization(self):
        """Test default initialization of PhysicalConstants."""
        constants = PhysicalConstants()
        assert constants.n_qubits == 1
        assert constants.chi == [0.5]  # Now a list
        assert constants.photon_cavity_coupling == 1.0
        assert constants.inverse_pulse_width == 0.1

    def test_custom_initialization(self):
        """Test custom initialization of PhysicalConstants."""
        constants = PhysicalConstants(chi=1.0, photon_cavity_coupling=2.0, inverse_pulse_width=0.2)
        assert constants.chi == [1.0]  # Now a list
        assert constants.photon_cavity_coupling == 2.0
        assert constants.inverse_pulse_width == 0.2


class TestSystemDimensions:
    """Test the SystemDimensions dataclass."""

    def test_default_initialization(self):
        """Test default initialization of SystemDimensions."""
        dims = SystemDimensions()
        assert dims.cavity_levels == 2
        assert dims.qubit_levels == 2
        assert dims.field_levels == 2

    def test_custom_initialization(self):
        """Test custom initialization of SystemDimensions."""
        dims = SystemDimensions(cavity_levels=8, qubit_levels=2, field_levels=10)
        assert dims.cavity_levels == 8
        assert dims.qubit_levels == 2
        assert dims.field_levels == 10


class TestMeasurementProtocol:
    """Test the MeasurementProtocol dataclass."""

    def test_default_initialization(self):
        """Test default initialization of MeasurementProtocol."""
        protocol = MeasurementProtocol()
        assert protocol.measurement_times is None  # Default uses interval mode
        assert protocol.initial_time == -5.0
        assert protocol.final_time == 5.0
        assert protocol.time_interval == 1.0

    def test_custom_initialization(self):
        """Test custom initialization of MeasurementProtocol."""
        times = [-10.0, -5.0, 0.0, 5.0, 10.0]
        protocol = MeasurementProtocol(measurement_times=times)
        assert protocol.measurement_times == times


class TestInitialStateConfig:
    """Test the InitialStateConfig dataclass."""

    def test_default_initialization(self):
        """Test default initialization of InitialStateConfig."""
        config = InitialStateConfig()
        assert config.state_type == InitialStateType.SINGLE_PHOTON
        assert config.coherent_alpha is None
        assert config.thermal_n_bar is None
        assert config.custom_amplitudes is None

    def test_coherent_state_config(self):
        """Test configuration for coherent states."""
        config = InitialStateConfig(
            state_type=InitialStateType.COHERENT, coherent_alpha=1.0 + 0.5j
        )
        assert config.state_type == InitialStateType.COHERENT
        assert config.coherent_alpha == 1.0 + 0.5j

    def test_thermal_state_config(self):
        """Test configuration for thermal states."""
        config = InitialStateConfig(state_type=InitialStateType.THERMAL, thermal_n_bar=2.5)
        assert config.state_type == InitialStateType.THERMAL
        assert config.thermal_n_bar == 2.5

    def test_custom_state_config(self):
        """Test configuration for custom states."""
        amplitudes = {(0, 0, 0): 0.7 + 0.0j, (1, 0, 0): 0.3 + 0.0j}
        config = InitialStateConfig(
            state_type=InitialStateType.CUSTOM, custom_amplitudes=amplitudes
        )
        assert config.state_type == InitialStateType.CUSTOM
        assert config.custom_amplitudes == amplitudes


class TestNoiseConfiguration:
    """Test the NoiseConfiguration dataclass."""

    def test_default_initialization(self):
        """Test default initialization of NoiseConfiguration."""
        config = NoiseConfiguration()
        assert config.depolarizing == 0.0
        assert config.dephasing == 0.0
        assert config.relaxation == 0.0
        assert config.custom_operators is None

    def test_custom_initialization(self):
        """Test custom initialization of NoiseConfiguration."""
        config = NoiseConfiguration(depolarizing=0.1, dephasing=0.05, relaxation=0.02)
        assert config.depolarizing == 0.1
        assert config.dephasing == 0.05
        assert config.relaxation == 0.02


class TestExperimentalParameters:
    """Test the ExperimentalParameters class."""

    def test_default_initialization(self):
        """Test default initialization with all default parameters."""
        params = ExperimentalParameters()

        # Check that all components are initialized
        assert isinstance(params.physical_constants, PhysicalConstants)
        assert isinstance(params.system_dims, SystemDimensions)
        assert isinstance(params.measurement, MeasurementProtocol)
        assert isinstance(params.noise_config, NoiseConfiguration)
        assert isinstance(params.initial_state, InitialStateConfig)

        # Check that measurement times are computed
        assert params._measurement_times_list is not None
        assert isinstance(params.measurement_times, np.ndarray)

    def test_custom_initialization(self):
        """Test initialization with custom parameters."""
        constants = PhysicalConstants(chi=1.0, photon_cavity_coupling=2.0)
        dims = SystemDimensions(cavity_levels=8, qubit_levels=2, field_levels=10)
        measurement = MeasurementProtocol(measurement_times=[-10.0, -5.0, 0.0, 5.0, 10.0])

        params = ExperimentalParameters(
            physical_constants=constants, system_dims=dims, measurement=measurement
        )

        assert params.physical_constants.chi == [1.0]  # Now a list
        assert params.physical_constants.photon_cavity_coupling == 2.0
        assert params.system_dims.cavity_levels == 8
        assert params.system_dims.qubit_levels == [2]  # Now a list
        assert params.system_dims.field_levels == 10
        assert len(params.measurement.measurement_times) == 5

    def test_measurement_times_computation(self):
        """Test that measurement times are correctly computed."""
        constants = PhysicalConstants(inverse_pulse_width=0.2)
        measurement = MeasurementProtocol(measurement_times=[-5.0, 0.0, 5.0])

        params = ExperimentalParameters(physical_constants=constants, measurement=measurement)

        # Times are stored and returned as absolute values (no normalization)
        expected_times = np.array([-5.0, 0.0, 5.0])
        assert params._measurement_times_list is not None
        np.testing.assert_array_almost_equal(params.measurement_times, expected_times)

    def test_update_measurement_times(self):
        """Test that measurement times are updated when parameters change."""
        params = ExperimentalParameters()
        assert params._measurement_times_list is not None
        original_times = params.measurement_times.copy()

        # Change the time interval to get different measurement times
        params.time_interval = 2.0
        
        # Times should be different now (fewer measurements with larger interval)
        assert params._measurement_times_list is not None
        assert not np.array_equal(params.measurement_times, original_times)

    # ==================== VALIDATION TESTS ====================

    def test_validation_cavity_levels_too_low(self):
        """Test validation error when cavity levels < 2."""
        dims = SystemDimensions(cavity_levels=1)

        with pytest.raises(ValueError, match="Cavity levels \\(cavity_levels\\) must be >= 2"):
            ExperimentalParameters(system_dims=dims)

    def test_validation_field_levels_too_low(self):
        """Test validation error when field levels < 2."""
        dims = SystemDimensions(field_levels=1)

        with pytest.raises(
            ValueError, match="External field levels \\(field_levels\\) must be >= 2"
        ):
            ExperimentalParameters(system_dims=dims)

    def test_validation_qubit_levels_too_low(self):
        """Test validation error when qubit levels < 2."""
        dims = SystemDimensions(qubit_levels=1)

        with pytest.raises(ValueError, match="Qubit 0 levels must be >= 2"):
            ExperimentalParameters(system_dims=dims)

    def test_validation_chi_non_positive(self):
        """Test validation error when chi <= 0."""
        constants = PhysicalConstants(chi=0.0)

        with pytest.raises(ValueError, match="Dispersive coupling \\(chi\\) for qubit 0 must be > 0"):
            ExperimentalParameters(physical_constants=constants)

    def test_validation_photon_cavity_coupling_non_positive(self):
        """Test validation error when photon_cavity_coupling <= 0."""
        constants = PhysicalConstants(photon_cavity_coupling=-1.0)

        with pytest.raises(
            ValueError, match="Photon-cavity coupling \\(photon_cavity_coupling\\) must be > 0"
        ):
            ExperimentalParameters(physical_constants=constants)

    def test_validation_inverse_pulse_width_non_positive(self):
        """Test validation error when inverse_pulse_width <= 0."""
        constants = PhysicalConstants(inverse_pulse_width=0.0)

        with pytest.raises(
            ValueError, match="Pulse width parameter \\(inverse_pulse_width\\) must be > 0"
        ):
            ExperimentalParameters(physical_constants=constants)

    def test_validation_negative_depolarizing_rate(self):
        """Test validation error when depolarizing rate < 0."""
        noise = NoiseConfiguration(depolarizing=-0.1)

        with pytest.raises(ValueError, match="Depolarization rate for qubit 0 must be >= 0"):
            ExperimentalParameters(noise_config=noise)

    def test_validation_negative_dephasing_rate(self):
        """Test validation error when dephasing rate < 0."""
        noise = NoiseConfiguration(dephasing=-0.1)

        with pytest.raises(ValueError, match="Dephasing rate for qubit 0 must be >= 0"):
            ExperimentalParameters(noise_config=noise)

    def test_validation_negative_relaxation_rate(self):
        """Test validation error when relaxation rate < 0."""
        noise = NoiseConfiguration(relaxation=-0.1)

        with pytest.raises(ValueError, match="Relaxation rate for qubit 0 must be >= 0"):
            ExperimentalParameters(noise_config=noise)

    def test_validation_insufficient_measurement_times(self):
        """Test validation error when fewer than 2 measurement times."""
        measurement = MeasurementProtocol(measurement_times=[0.0])

        with pytest.raises(ValueError, match="At least two measurement times must be specified"):
            ExperimentalParameters(measurement=measurement)

    def test_validation_unsorted_measurement_times(self):
        """Test validation error when measurement times are not sorted."""
        measurement = MeasurementProtocol(measurement_times=[5.0, -5.0, 0.0])

        with pytest.raises(
            ValueError, match="Measurement times must be in ascending order"
        ):
            ExperimentalParameters(measurement=measurement)

    def test_primary_properties_cavity_levels(self):
        """Test primary cavity_levels property."""
        params = ExperimentalParameters()

        # Test getter
        assert params.cavity_levels == params.system_dims.cavity_levels

        # Test setter
        params.cavity_levels = 8
        assert params.system_dims.cavity_levels == 8
        assert params.cavity_levels == 8

    def test_primary_properties_qubit_levels(self):
        """Test primary qubit_levels property."""
        params = ExperimentalParameters()

        # Test getter
        assert params.qubit_levels == params.system_dims.qubit_levels

        # Test setter
        params.qubit_levels = 3
        assert params.system_dims.qubit_levels == [3]  # Now a list
        assert params.qubit_levels == [3]  # Now a list

    def test_primary_properties_field_levels(self):
        """Test primary field_levels property."""
        params = ExperimentalParameters()

        # Test getter
        assert params.field_levels == params.system_dims.field_levels

        # Test setter
        params.field_levels = 15
        assert params.system_dims.field_levels == 15
        assert params.field_levels == 15

    def test_primary_properties_chi(self):
        """Test primary chi property."""
        params = ExperimentalParameters()

        # Test getter
        assert params.chi == params.physical_constants.chi

        # Test setter
        params.chi = 1.5
        assert params.physical_constants.chi == [1.5]  # Now a list
        assert params.chi == [1.5]  # Now a list

    def test_primary_properties_photon_cavity_coupling(self):
        """Test primary photon_cavity_coupling property."""
        params = ExperimentalParameters()

        # Test getter
        assert params.photon_cavity_coupling == params.physical_constants.photon_cavity_coupling

        # Test setter
        params.photon_cavity_coupling = 2.5
        assert params.physical_constants.photon_cavity_coupling == 2.5
        assert params.photon_cavity_coupling == 2.5

    def test_primary_properties_inverse_pulse_width(self):
        """Test primary inverse_pulse_width property."""
        params = ExperimentalParameters()

        # Test getter
        assert params.inverse_pulse_width == params.physical_constants.inverse_pulse_width

        # Test setter
        params.inverse_pulse_width = 0.3
        assert params.physical_constants.inverse_pulse_width == 0.3
        assert params.inverse_pulse_width == 0.3

    def test_primary_properties_measurement_times(self):
        """Test primary measurement_times property."""
        params = ExperimentalParameters()

        # Test getter - should return computed measurement times (interval mode)
        times = params.measurement_times
        assert isinstance(times, np.ndarray)
        # In interval mode, times are computed from initial_time, final_time, time_interval
        assert len(times) >= 2

        # Test setter
        new_times = np.array([-10.0, -5.0, 0.0, 5.0, 10.0])
        params.measurement_times = new_times

        # Check that the measurement protocol was updated correctly (no normalization)
        expected_protocol_times = list(new_times)
        assert params.measurement.measurement_times == expected_protocol_times

        # Check that measurement_times was updated
        assert params._measurement_times_list is not None
        np.testing.assert_array_almost_equal(params.measurement_times, new_times)


if __name__ == "__main__":
    # Run tests if executed directly
    pytest.main([__file__, "-v"])
