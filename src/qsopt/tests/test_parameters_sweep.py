"""
Test parameter sweep functionality
"""

import numpy as np
import pytest

from qsopt.core.experiment import SingleQubitExperiment, TwoQubitExperiment
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
from qsopt.utils.parameters_sweep import SweepResults, compute_chi_gamma_sweep
from qsopt.utils.visualization import plot_sweep_results


@pytest.fixture
def single_qubit_experiment():
    """Create a basic single-qubit experiment for testing."""
    physical_constants = PhysicalConstants(
        n_qubits=1, chi=[10.0], photon_cavity_coupling=10.0, inverse_pulse_width=1.0
    )

    system_dims = SystemDimensions(field_levels=2, cavity_levels=2, qubit_levels=2)

    measurement = MeasurementProtocol(measurement_times=[-2.0, -1.0, 0.0, 1.0, 2.0])

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

    return SingleQubitExperiment(exp_params, trainable_params)


@pytest.fixture
def two_qubit_experiment():
    """Create a basic two-qubit experiment for testing."""
    physical_constants = PhysicalConstants(
        n_qubits=2, chi=[10.0, 10.0], photon_cavity_coupling=10.0, inverse_pulse_width=1.0
    )

    system_dims = SystemDimensions(field_levels=2, cavity_levels=2, qubit_levels=[2, 2])

    measurement = MeasurementProtocol(measurement_times=[-2.0, -1.0, 0.0, 1.0, 2.0])

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

    return TwoQubitExperiment(exp_params, trainable_params)


def test_single_qubit_sweep_method_exists(single_qubit_experiment):
    """Test that sweep_chi_gamma method exists on SingleQubitExperiment."""
    assert hasattr(single_qubit_experiment, "sweep_chi_gamma")


def test_two_qubit_sweep_method_exists(two_qubit_experiment):
    """Test that sweep_chi_gamma method exists on TwoQubitExperiment."""
    assert hasattr(two_qubit_experiment, "sweep_chi_gamma")


def test_single_qubit_sweep_small(single_qubit_experiment):
    """Test chi-gamma sweep on single qubit with minimal grid."""
    results = single_qubit_experiment.sweep_chi_gamma(
        chi_interval=[5.0, 15.0],
        gamma_interval=[5.0, 15.0],
        resolution_chi=3,
        resolution_gamma=3,
        batch_size=1,
        verbose=False,
    )

    # Check result structure - should be SweepResults object
    assert isinstance(results, SweepResults)
    assert results.param1_name == "gamma"
    assert results.param2_name == "chi"
    assert "contrast_map" in results.results
    assert "detection_map" in results.results
    assert "detection_without_map" in results.results

    # Check array shapes
    assert len(results.param1_vals) == 3  # gamma values
    assert len(results.param2_vals) == 3  # chi values
    assert results.results["contrast_map"].shape == (3, 3)
    assert results.results["detection_map"].shape == (3, 3)
    assert results.results["detection_without_map"].shape == (3, 3)

    # Check values are reasonable
    assert np.all(results.results["contrast_map"] >= -1.0)
    assert np.all(results.results["contrast_map"] <= 1.0)
    assert np.all(results.results["detection_map"] >= -1e-10)  # Allow for numerical precision
    assert np.all(results.results["detection_map"] <= 1.0)

    # Check scale parameters
    assert results.param1_scale == "linear"
    assert results.param2_scale == "linear"


def test_two_qubit_sweep_small(two_qubit_experiment):
    """Test chi-gamma sweep on two qubits with minimal grid."""
    results = two_qubit_experiment.sweep_chi_gamma(
        chi_interval=[5.0, 15.0],
        gamma_interval=[5.0, 15.0],
        resolution_chi=3,
        resolution_gamma=3,
        batch_size=1,
        verbose=False,
    )

    # Check result structure - should be SweepResults object
    assert isinstance(results, SweepResults)
    assert "contrast_map" in results.results
    assert "detection_map" in results.results
    assert "detection_without_map" in results.results

    # Check scale parameters
    assert results.param1_scale == "linear"
    assert results.param2_scale == "linear"

    # Check array shapes
    assert len(results.param1_vals) == 3
    assert len(results.param2_vals) == 3
    assert results.results["contrast_map"].shape == (3, 3)

    # Check values are reasonable
    assert np.all(results.results["contrast_map"] >= -1.0)
    assert np.all(results.results["contrast_map"] <= 1.0)


def test_compute_chi_gamma_sweep_function(single_qubit_experiment):
    """Test the standalone compute_chi_gamma_sweep function."""
    results = compute_chi_gamma_sweep(
        single_qubit_experiment,
        chi_interval=[5.0, 15.0],
        gamma_interval=[5.0, 15.0],
        resolution_chi=2,
        resolution_gamma=2,
        verbose=False,
    )

    # Results should be a SweepResults object
    assert isinstance(results, SweepResults)
    assert len(results.param1_vals) == 2  # gamma values
    assert len(results.param2_vals) == 2  # chi values
    assert "contrast_map" in results.results
    assert results.results["contrast_map"].shape == (2, 2)
    assert results.param1_scale in ["linear", "log"]
    assert results.param2_scale in ["linear", "log"]


def test_plot_sweep_results(single_qubit_experiment):
    """Test that plotting function runs without error."""
    results = single_qubit_experiment.sweep_chi_gamma(
        chi_interval=[5.0, 15.0],
        gamma_interval=[5.0, 15.0],
        resolution_chi=3,
        resolution_gamma=3,
        batch_size=1,
        verbose=False,
    )

    # Should not raise an error
    fig = plot_sweep_results(results)
    assert fig is not None


def test_sweep_with_log_scale(single_qubit_experiment):
    """Test chi-gamma sweep with logarithmic scale."""
    results = single_qubit_experiment.sweep_chi_gamma(
        chi_interval=[1.0, 100.0],
        gamma_interval=[1.0, 100.0],
        resolution_chi=3,
        resolution_gamma=3,
        chi_scale="log",
        gamma_scale="log",
        batch_size=1,
        verbose=False,
    )

    # Results should be SweepResults
    assert isinstance(results, SweepResults)

    # Check that scale is log
    assert results.param1_scale == "log"  # gamma
    assert results.param2_scale == "log"  # chi

    # Check that values are logarithmically spaced
    chi_vals = results.param2_vals
    gamma_vals = results.param1_vals

    # In log space, differences between consecutive log values should be approximately equal
    log_chi_diffs = np.diff(np.log10(chi_vals))
    log_gamma_diffs = np.diff(np.log10(gamma_vals))

    # Check that log spacing is uniform
    assert np.allclose(log_chi_diffs, log_chi_diffs[0], rtol=1e-5)
    assert np.allclose(log_gamma_diffs, log_gamma_diffs[0], rtol=1e-5)


def test_plot_selective_displays(single_qubit_experiment):
    """Test that plotting function works with selective display options."""
    results = single_qubit_experiment.sweep_chi_gamma(
        chi_interval=[5.0, 15.0],
        gamma_interval=[5.0, 15.0],
        resolution_chi=3,
        resolution_gamma=3,
        batch_size=1,
        verbose=False,
    )

    # Test showing only contrast
    fig1 = plot_sweep_results(results, results_to_plot=["contrast_map"])
    assert fig1 is not None

    # Test showing two plots
    fig2 = plot_sweep_results(results, results_to_plot=["contrast_map", "detection_map"])
    assert fig2 is not None
