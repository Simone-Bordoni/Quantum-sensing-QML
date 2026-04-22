"""
Test parameter sweep functionality
"""

import numpy as np
import pytest

from qsopt.core.experiment import Experiment
from qsopt.core.experimental_parameters import (
    ExperimentalParameters,
    InitialStateConfig,
    InitialStateType,
    MeasurementProtocol,
    NoiseConfiguration,
    PhysicalSetup,
    SystemDimensions,
)
from qsopt.core.circuit import QuantumCircuit, create_ry_circuit_layer
from qsopt.utils.results import SweepResults
from qsopt.utils.visualization import plot_sweep_results


@pytest.fixture
def single_qubit_experiment():
    """Create a basic single-qubit experiment for testing."""
    physical_setup = PhysicalSetup(
        n_qubits=1, chi=[10.0], photon_cavity_coupling=10.0, inverse_pulse_width=1.0
    )

    system_dims = SystemDimensions(field_levels=2, cavity_levels=2, qubit_levels=2)

    measurement = MeasurementProtocol(measurement_times=[-2.0, -1.0, 0.0, 1.0, 2.0])

    initial_state = InitialStateConfig(state_type=InitialStateType.SINGLE_PHOTON)
    noise_config = NoiseConfiguration(depolarizing=0.0, dephasing=0.0, relaxation=0.0)

    exp_params = ExperimentalParameters(
        physical_setup=physical_setup,
        system_dims=system_dims,
        measurement=measurement,
        initial_state=initial_state,
        noise_config=noise_config,
    )

    initial_circuit = create_ry_circuit_layer(n_qubits=1, theta_values=[0.0])
    final_circuit = QuantumCircuit(n_qubits=1)

    return Experiment(exp_params, initial_circuit=initial_circuit, final_circuit=final_circuit)


@pytest.fixture
def two_qubit_experiment():
    """Create a basic two-qubit experiment for testing."""
    physical_setup = PhysicalSetup(
        n_qubits=2, chi=[10.0, 10.0], photon_cavity_coupling=10.0, inverse_pulse_width=1.0
    )

    system_dims = SystemDimensions(field_levels=2, cavity_levels=2, qubit_levels=[2, 2])

    measurement = MeasurementProtocol(measurement_times=[-2.0, -1.0, 0.0, 1.0, 2.0])

    initial_state = InitialStateConfig(state_type=InitialStateType.SINGLE_PHOTON)
    noise_config = NoiseConfiguration(
        depolarizing=[0.0, 0.0], dephasing=[0.0, 0.0], relaxation=[0.0, 0.0]
    )

    exp_params = ExperimentalParameters(
        physical_setup=physical_setup,
        system_dims=system_dims,
        measurement=measurement,
        initial_state=initial_state,
        noise_config=noise_config,
    )

    initial_circuit = create_ry_circuit_layer(n_qubits=2, theta_values=[0.0, 0.0])
    final_circuit = QuantumCircuit(n_qubits=2)

    return Experiment(exp_params, initial_circuit=initial_circuit, final_circuit=final_circuit)


def test_single_qubit_sweep_method_exists(single_qubit_experiment):
    """Test that sweep_chi_gamma method exists on Experiment."""
    assert hasattr(single_qubit_experiment, "sweep_chi_gamma")


def test_two_qubit_sweep_method_exists(two_qubit_experiment):
    """Test that sweep_chi_gamma method exists on Experiment."""
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
    assert "metric_map" in results.results
    assert "detection_map" in results.results
    assert "detection_without_map" in results.results

    # Check array shapes
    assert len(results.param1_vals) == 3  # gamma values
    assert len(results.param2_vals) == 3  # chi values
    assert results.results["metric_map"].shape == (3, 3)
    assert results.results["detection_map"].shape == (3, 3)
    assert results.results["detection_without_map"].shape == (3, 3)

    # Check values are reasonable
    assert np.all(results.results["metric_map"] >= -1.0)
    assert np.all(results.results["metric_map"] <= 1.0)
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
    assert "metric_map" in results.results
    assert "detection_map" in results.results
    assert "detection_without_map" in results.results

    # Check scale parameters
    assert results.param1_scale == "linear"
    assert results.param2_scale == "linear"

    # Check array shapes
    assert len(results.param1_vals) == 3
    assert len(results.param2_vals) == 3
    assert results.results["metric_map"].shape == (3, 3)

    # Check values are reasonable
    assert np.all(results.results["metric_map"] >= -1.0)
    assert np.all(results.results["metric_map"] <= 1.0)


def test_chi_gamma_sweep_alternate_call(single_qubit_experiment):
    """Test chi-gamma sweep with different parameters."""
    results = single_qubit_experiment.sweep_chi_gamma(
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
    assert "metric_map" in results.results
    assert results.results["metric_map"].shape == (2, 2)
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

    # Test showing only metric
    fig1 = plot_sweep_results(results, results_to_plot=["metric_map"])
    assert fig1 is not None

    # Test showing two plots
    fig2 = plot_sweep_results(results, results_to_plot=["metric_map", "detection_map"])
    assert fig2 is not None
