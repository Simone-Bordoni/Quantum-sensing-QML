"""
Tests for Experiment Loader Utility
===================================

Test suite for loading experiments from JSON reports.

.. deprecated::
    These tests are for deprecated TrainableParameters functionality.
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from qsopt.core.experimental_parameters import (
    ExperimentalParameters,
    InitialStateConfig,
    MeasurementProtocol,
    NoiseConfiguration,
    PhysicalConstants,
    SystemDimensions,
)
# from qsopt.core.trainable_parameters import TrainableParameters
from qsopt.utils import load_experiment_from_report


pytestmark = pytest.mark.skip(reason="TrainableParameters has been removed, tests need refactoring")


class TestExperimentLoader:
    """Test experiment loader functionality."""

    @pytest.fixture
    def sample_report_explicit_times(self):
        """Create a sample report with explicit measurement times."""
        return {
            "experiment_type": "SingleQubitExperiment",
            "version": "1.0",
            "experimental_parameters": {
                "physical_constants": {
                    "chi": 0.5,
                    "photon_cavity_coupling": 1.0,
                    "inverse_pulse_width": 0.1,
                },
                "system_dimensions": {"cavity_levels": 10, "qubit_levels": 2, "field_levels": 2},
                "measurement_protocol": {
                    "mode": "explicit",
                    "measurement_times": [0.0, 1.0, 2.0, 3.0, 4.0],
                    "initial_time_uncertainty": 0.0,
                },
                "initial_state": {
                    "state_type": "coherent",
                    "coherent_alpha": 2.0,
                    "coherent_alpha_phase": 0.0,
                    "thermal_n_bar": None,
                    "has_custom_amplitudes": False,
                },
                "noise_configuration": {
                    "depolarizing": 0.01,
                    "dephasing": 0.005,
                    "relaxation": 0.002,
                },
            },
            "trainable_parameters": {
                "num_parameters": 3,
                "num_trainable": 2,
                "parameters": [
                    {
                        "index": 0,
                        "name": "theta1",
                        "type": "rotation_angle",
                        "value": 1.57,
                        "trainable": True,
                    },
                    {
                        "index": 1,
                        "name": "theta2",
                        "type": "rotation_angle",
                        "value": 3.14,
                        "trainable": False,
                    },
                    {
                        "index": 2,
                        "name": "custom_param",
                        "type": "custom",
                        "value": 0.5,
                        "trainable": True,
                    },
                ],
            },
        }

    @pytest.fixture
    def sample_report_interval_times(self):
        """Create a sample report with interval-based measurement times."""
        return {
            "experiment_type": "SingleQubitExperiment",
            "version": "1.0",
            "experimental_parameters": {
                "physical_constants": {
                    "chi": 0.5,
                    "photon_cavity_coupling": 1.0,
                    "inverse_pulse_width": 0.1,
                },
                "system_dimensions": {"cavity_levels": 10, "qubit_levels": 2, "field_levels": 2},
                "measurement_protocol": {
                    "mode": "interval",
                    "measurement_times": None,
                    "initial_time": 0.0,
                    "final_time": 10.0,
                    "time_interval": 0.1,
                    "initial_time_uncertainty": 0.0,
                },
                "initial_state": {
                    "state_type": "coherent",
                    "coherent_alpha": 2.0,
                    "coherent_alpha_phase": 0.0,
                    "thermal_n_bar": None,
                    "has_custom_amplitudes": False,
                },
                "noise_configuration": {
                    "depolarizing": 0.01,
                    "dephasing": 0.005,
                    "relaxation": 0.002,
                },
            },
            "trainable_parameters": {
                "num_parameters": 2,
                "num_trainable": 2,
                "parameters": [
                    {
                        "index": 0,
                        "name": "theta1",
                        "type": "rotation_angle",
                        "value": 0.0,
                        "trainable": True,
                    },
                    {
                        "index": 1,
                        "name": "theta2",
                        "type": "rotation_angle",
                        "value": 1.57,
                        "trainable": True,
                    },
                ],
            },
        }

    def test_load_explicit_times_report(self, sample_report_explicit_times):
        """Test loading report with explicit measurement times."""
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(sample_report_explicit_times, f)
            temp_path = f.name

        try:
            # Load experiment
            exp_params, trainable_params, metadata = load_experiment_from_report(temp_path)

            # Check experimental parameters
            assert isinstance(exp_params, ExperimentalParameters)
            assert exp_params.physical_constants.chi == [0.5]  # Now a list
            assert exp_params.physical_constants.photon_cavity_coupling == 1.0
            assert exp_params.system_dims.cavity_levels == 10
            assert exp_params.system_dims.qubit_levels == [2]  # Now a list

            # Check measurement protocol
            assert exp_params.measurement.measurement_times is not None
            assert len(exp_params.measurement.measurement_times) == 5
            assert np.allclose(exp_params.measurement.measurement_times, [0.0, 1.0, 2.0, 3.0, 4.0])

            # Check initial state
            assert exp_params.initial_state.state_type.value == "coherent"
            assert exp_params.initial_state.coherent_alpha == 2.0

            # Check noise config (now stored as lists)
            assert (
                exp_params.noise_config.depolarizing == [0.01]
                or exp_params.noise_config.depolarizing == 0.01
            )
            assert (
                exp_params.noise_config.dephasing == [0.005]
                or exp_params.noise_config.dephasing == 0.005
            )

            # Check trainable parameters
            assert isinstance(trainable_params, TrainableParameters)
            assert len(trainable_params) == 3

            # Check trainable flags
            assert trainable_params.parameters[0].trainable is True
            assert trainable_params.parameters[1].trainable is False
            # Custom parameters are now forced to be non-trainable
            assert trainable_params.parameters[2].trainable is False

            # Check parameter values
            vector = trainable_params.get_parameter_vector()
            assert np.isclose(vector[0], 1.57)
            assert np.isclose(vector[1], 3.14)
            assert np.isclose(vector[2], 0.5)

            # Check metadata
            assert metadata["experiment_type"] == "SingleQubitExperiment"
            assert metadata["version"] == "1.0"

        finally:
            Path(temp_path).unlink()

    def test_load_interval_times_report(self, sample_report_interval_times):
        """Test loading report with interval-based measurement times."""
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(sample_report_interval_times, f)
            temp_path = f.name

        try:
            # Load experiment
            exp_params, trainable_params, metadata = load_experiment_from_report(temp_path)

            # Check measurement protocol
            # Use the property which computes times if not explicitly set
            times = exp_params.measurement_times
            assert times is not None
            assert len(times) == 101
            assert np.isclose(times[0], 0.0)
            assert np.isclose(times[-1], 10.0)

            # Check trainable parameters
            assert len(trainable_params) == 2
            assert all(p.trainable for p in trainable_params.parameters)

        finally:
            Path(temp_path).unlink()

    def test_get_trainable_indices(self, sample_report_explicit_times):
        """Test that trainable indices can be retrieved correctly."""
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(sample_report_explicit_times, f)
            temp_path = f.name

        try:
            exp_params, trainable_params, metadata = load_experiment_from_report(temp_path)

            # Check trainable indices
            # Custom parameters are now forced to be non-trainable, so only theta1 is trainable
            trainable_indices = trainable_params.get_trainable_indices()
            assert trainable_indices == [0]  # Only theta1 (custom_param is forced to non-trainable)

            # Check trainable mask
            mask = trainable_params.get_trainable_mask()
            # Custom parameter is now forced to be non-trainable
            expected = np.array([True, False, False])
            assert np.array_equal(mask, expected)

        finally:
            Path(temp_path).unlink()

    def test_missing_file(self):
        """Test error handling for missing file."""
        with pytest.raises(FileNotFoundError):
            load_experiment_from_report("nonexistent_file.json")

    def test_invalid_json(self):
        """Test error handling for invalid JSON."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("not valid json{")
            temp_path = f.name

        try:
            with pytest.raises(json.JSONDecodeError):
                load_experiment_from_report(temp_path)
        finally:
            Path(temp_path).unlink()

    def test_missing_sections(self):
        """Test error handling for missing required sections."""
        incomplete_report = {
            "experiment_type": "SingleQubitExperiment",
            "version": "1.0",
            # Missing experimental_parameters and trainable_parameters
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(incomplete_report, f)
            temp_path = f.name

        try:
            # The loader will work with defaults when sections are missing
            # This is actually desired behavior for backward compatibility
            exp_params, trainable_params, metadata = load_experiment_from_report(temp_path)

            # Should get default experimental parameters
            assert isinstance(exp_params, ExperimentalParameters)

            # Should get empty trainable parameters
            assert isinstance(trainable_params, TrainableParameters)
            assert len(trainable_params) == 0
        finally:
            Path(temp_path).unlink()
