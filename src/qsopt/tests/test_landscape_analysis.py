"""
Tests for landscape analysis utilities.
"""

import math
from typing import cast

import numpy as np
import pytest

from qsopt.core.experimental_parameters import ExperimentalParameters, InitialStateType
from qsopt.utils import compute_theta1_theta2_landscape


def create_test_experiment():
    """Create minimal experimental parameters for testing."""
    exp_params = ExperimentalParameters()

    # Minimal physical constants
    gm = 0.03 * 2 * np.pi
    sigma = 0.1 * gm
    chi = 0.5 * gm

    exp_params.photon_cavity_coupling = gm
    exp_params.inverse_pulse_width = sigma
    exp_params.chi = chi

    # Minimal dimensions
    exp_params.cavity_levels = 2
    exp_params.qubit_levels = 2
    exp_params.field_levels = 2

    # No noise for fast testing
    exp_params.noise_config.relaxation = 0.0
    exp_params.noise_config.dephasing = 0.0
    exp_params.noise_config.depolarizing = 0.0

    # Measurement times
    initial_time = -5.0 / sigma
    final_time = 5.0 / sigma
    time_interval = 10 / sigma

    exp_params.measurement.initial_time = initial_time
    exp_params.measurement.final_time = final_time
    exp_params.measurement.time_interval = time_interval

    # Initial state
    exp_params.initial_state.state_type = InitialStateType.SINGLE_PHOTON

    return exp_params


def test_initial_time_uncertainty_max_interval_spec():
    """Ensure string specification for initial time uncertainty is resolved."""
    exp_params = create_test_experiment()
    exp_params.initial_time_uncertainty = "max_interval"

    base_times = exp_params.measurement_times
    expected_interval = float(np.max(np.diff(base_times)))
    assert np.isclose(exp_params.initial_time_uncertainty, expected_interval)

    # Changing the measurement interval should update the resolved uncertainty
    exp_params.time_interval = exp_params.time_interval / 2
    updated_times = exp_params.measurement_times
    updated_expected = float(np.max(np.diff(updated_times)))
    assert np.isclose(exp_params.initial_time_uncertainty, updated_expected)

    # Batch sampling should stay within the resolved uncertainty bounds
    exp_params.seed = 42
    batch_times = exp_params.get_measurement_times_with_uncertainty(batch_size=2)
    max_deviation = np.max(np.abs(batch_times - updated_times[np.newaxis, :]))
    assert max_deviation <= exp_params.initial_time_uncertainty + 1e-12


@pytest.mark.skip(reason="compute_theta1_theta2_landscape requires TrainableParameters which has been removed")
def test_compute_theta1_theta2_landscape_structure():
    """Test that landscape computation returns correct structure."""
    exp_params = create_test_experiment()

    # Use small resolution for fast testing
    resolution = 3
    data = compute_theta1_theta2_landscape(
        exp_params,
        resolution=resolution,
        center_theta1=np.pi / 2,
        center_theta2=-np.pi / 2,
        param_range=np.pi / 8,
        verbose=False,
    )

    # Check returned dictionary has expected keys
    assert "theta1_vals" in data
    assert "theta2_vals" in data
    assert "contrast_map" in data
    assert "detection_map" in data
    assert "center_theta1" in data
    assert "center_theta2" in data

    # Check array shapes
    theta1_vals = cast(np.ndarray, data["theta1_vals"])
    theta2_vals = cast(np.ndarray, data["theta2_vals"])
    contrast_map = cast(np.ndarray, data["contrast_map"])
    detection_map = cast(np.ndarray, data["detection_map"])
    assert theta1_vals.shape == (resolution,)
    assert theta2_vals.shape == (resolution,)
    assert contrast_map.shape == (resolution, resolution)
    assert detection_map.shape == (resolution, resolution)

    # Check center values
    assert data["center_theta1"] == np.pi / 2
    assert data["center_theta2"] == -np.pi / 2


@pytest.mark.skip(reason="compute_theta1_theta2_landscape requires TrainableParameters which has been removed")
def test_compute_theta1_theta2_landscape_values():
    """Test that landscape computation produces valid values."""
    exp_params = create_test_experiment()

    resolution = 3
    data = compute_theta1_theta2_landscape(exp_params, resolution=resolution, verbose=False)

    contrast_map = cast(np.ndarray, data["contrast_map"])
    detection_map = cast(np.ndarray, data["detection_map"])
    theta1_vals = cast(np.ndarray, data["theta1_vals"])
    theta2_vals = cast(np.ndarray, data["theta2_vals"])

    # Check contrast map values are in valid range [0, 1]
    assert np.all(contrast_map >= 0.0)
    assert np.all(contrast_map <= 1.0)

    # Check detection map values are in valid range [0, 1]
    assert np.all(detection_map >= 0.0)
    assert np.all(detection_map <= 1.0)

    # Check parameter values span the expected range
    center_theta1 = data["center_theta1"]
    center_theta2 = data["center_theta2"]
    param_range = np.pi / 6  # default

    assert np.isclose(theta1_vals[0], center_theta1 - param_range, rtol=1e-5)
    assert np.isclose(theta1_vals[-1], center_theta1 + param_range, rtol=1e-5)
    assert np.isclose(theta2_vals[0], center_theta2 - param_range, rtol=1e-5)
    assert np.isclose(theta2_vals[-1], center_theta2 + param_range, rtol=1e-5)


@pytest.mark.skip(reason="compute_theta1_theta2_landscape requires TrainableParameters which has been removed")
def test_compute_theta1_theta2_landscape_custom_range():
    """Test landscape computation with custom parameter range."""
    exp_params = create_test_experiment()

    resolution = 3
    custom_range = np.pi / 4
    data = compute_theta1_theta2_landscape(
        exp_params,
        resolution=resolution,
        center_theta1=0.0,
        center_theta2=0.0,
        param_range=custom_range,
        verbose=False,
    )

    # Check parameter values span the custom range
    theta1_vals = cast(np.ndarray, data["theta1_vals"])
    theta2_vals = cast(np.ndarray, data["theta2_vals"])
    assert np.isclose(theta1_vals[0], -custom_range, rtol=1e-5)
    assert np.isclose(theta1_vals[-1], custom_range, rtol=1e-5)
    assert np.isclose(theta2_vals[0], -custom_range, rtol=1e-5)
    assert np.isclose(theta2_vals[-1], custom_range, rtol=1e-5)


@pytest.mark.skip(reason="compute_theta1_theta2_landscape requires TrainableParameters which has been removed")
def test_compute_theta1_theta2_landscape_with_batch():
    """Test landscape computation with batch averaging."""
    exp_params = create_test_experiment()

    # Add uncertainty
    exp_params.measurement.initial_time_uncertainty = 0.1

    resolution = 3
    batch_size = 5
    data = compute_theta1_theta2_landscape(
        exp_params, resolution=resolution, batch_size=batch_size, verbose=False
    )

    # Check structure
    assert "contrast_map" in data
    contrast_map = cast(np.ndarray, data["contrast_map"])
    assert contrast_map.shape == (resolution, resolution)

    # Values should still be in valid range
    assert np.all(contrast_map >= 0.0)
    assert np.all(contrast_map <= 1.0)


def test_compute_time_interval_landscape_continuous():
    """Test time interval landscape computation in continuous mode."""
    from qsopt.utils import compute_time_interval_landscape

    exp_params = create_test_experiment()

    resolution = 5
    data = compute_time_interval_landscape(
        exp_params,
        theta1=np.pi / 2,
        theta2=-np.pi / 2,
        resolution=resolution,
        mode="continuous",
        verbose=False,
    )

    # Check returned dictionary has expected keys
    assert "interval_vals" in data
    assert "contrast_vals" in data
    assert "detection_with" in data
    assert "detection_without" in data
    assert "n_measurements" in data
    assert "theta1" in data
    assert "theta2" in data
    assert "mode" in data

    interval_vals = cast(np.ndarray, data["interval_vals"])
    contrast_vals = cast(np.ndarray, data["contrast_vals"])
    detection_with = cast(np.ndarray, data["detection_with"])
    detection_without = cast(np.ndarray, data["detection_without"])
    n_measurements = cast(np.ndarray, data["n_measurements"])

    assert interval_vals.shape == (resolution,)
    assert contrast_vals.shape == (resolution,)
    assert detection_with.shape == (resolution,)
    assert detection_without.shape == (resolution,)
    assert n_measurements.shape == (resolution,)

    # Check mode
    assert data["mode"] == "continuous"

    # Check values are in valid ranges
    assert np.all(contrast_vals >= 0.0)
    assert np.all(contrast_vals <= 1.0)
    assert np.all(detection_with >= 0.0)
    assert np.all(detection_with <= 1.0)
    assert np.all(detection_without >= 0.0)
    assert np.all(detection_without <= 1.0)


def test_compute_time_interval_landscape_discrete():
    """Test time interval landscape computation in discrete mode."""
    from qsopt.utils import compute_time_interval_landscape

    exp_params = create_test_experiment()

    resolution = 5
    data = compute_time_interval_landscape(
        exp_params,
        theta1=np.pi / 2,
        theta2=-np.pi / 2,
        resolution=resolution,
        mode="discrete",
        verbose=False,
    )

    assert data["mode"] == "discrete"

    interval_vals = cast(np.ndarray, data["interval_vals"])
    assert interval_vals.shape == (resolution,)

    total_time = exp_params.measurement.final_time - exp_params.measurement.initial_time
    expected_fractions = np.arange(1, resolution + 1)
    expected_intervals = total_time / expected_fractions
    # Intervals are now sorted in ascending order
    expected_intervals = np.sort(expected_intervals)

    np.testing.assert_allclose(interval_vals, expected_intervals, rtol=1e-5)


def test_compute_time_interval_landscape_discrete_with_bounds():
    """Discrete mode should honor explicit min/max bounds."""
    from qsopt.utils import compute_time_interval_landscape

    exp_params = create_test_experiment()

    total_time = exp_params.measurement.final_time - exp_params.measurement.initial_time
    max_interval = total_time / 3.7
    min_interval = total_time / 10.5

    resolution = 5
    data = compute_time_interval_landscape(
        exp_params,
        theta1=np.pi / 2,
        theta2=-np.pi / 2,
        resolution=resolution,
        mode="discrete",
        verbose=False,
        min_interval=min_interval,
        max_interval=max_interval,
    )

    assert data["mode"] == "discrete"

    intervals = cast(np.ndarray, data["interval_vals"])
    assert intervals.shape == (resolution,)
    assert np.all(intervals <= max_interval + 1e-8)
    assert np.all(intervals >= min_interval - 1e-8)
    # Intervals are sorted in ascending order, so diffs should be >= 0
    assert np.all(np.diff(intervals) >= -1e-12)

    # First interval should be the smallest (not largest anymore)
    expected_first = total_time / math.floor(total_time / min_interval)
    assert np.isclose(intervals[0], expected_first)


def test_compute_time_interval_landscape_with_batch():
    """Test time interval landscape with batch averaging."""
    from qsopt.utils import compute_time_interval_landscape

    exp_params = create_test_experiment()
    exp_params.measurement.initial_time_uncertainty = 0.1

    resolution = 5
    batch_size = 3
    data = compute_time_interval_landscape(
        exp_params,
        theta1=np.pi / 2,
        theta2=-np.pi / 2,
        resolution=resolution,
        batch_size=batch_size,
        verbose=False,
    )

    # Check batch_size is stored
    assert data["batch_size"] == batch_size

    # Values should still be valid
    contrast_vals = cast(np.ndarray, data["contrast_vals"])
    assert np.all(contrast_vals >= 0.0)
    assert np.all(contrast_vals <= 1.0)


def test_compute_time_interval_landscape_invalid_mode():
    """Test that invalid mode raises error."""
    from qsopt.utils import compute_time_interval_landscape

    exp_params = create_test_experiment()

    with pytest.raises(ValueError, match="mode must be"):
        compute_time_interval_landscape(
            exp_params, theta1=np.pi / 2, theta2=-np.pi / 2, mode="invalid_mode", verbose=False
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
