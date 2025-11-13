"""
Test chi-lambda sweep functionality
"""

import pytest
import numpy as np
from qsopt.core.experimental_parameters import (
    ExperimentalParameters,
    PhysicalConstants,
    SystemDimensions,
    MeasurementProtocol,
    InitialStateConfig,
    InitialStateType,
    NoiseConfiguration
)
from qsopt.core.trainable_parameters import TrainableParameters
from qsopt.core.experiment import SingleQubitExperiment, TwoQubitExperiment
from qsopt.utils.chi_lambda_sweep import compute_chi_lambda_sweep
from qsopt.utils.visualization import plot_chi_lambda_sweep


@pytest.fixture
def single_qubit_experiment():
    """Create a basic single-qubit experiment for testing."""
    physical_constants = PhysicalConstants(
        n_qubits=1,
        chi=[10.0],
        photon_cavity_coupling=10.0,
        inverse_pulse_width=1.0
    )
    
    system_dims = SystemDimensions(
        field_levels=2,
        cavity_levels=2,
        qubit_levels=2
    )
    
    measurement = MeasurementProtocol(
        measurement_times=[-2.0, -1.0, 0.0, 1.0, 2.0]
    )
    
    initial_state = InitialStateConfig(state_type=InitialStateType.SINGLE_PHOTON)
    noise_config = NoiseConfiguration(depolarizing=0.0, dephasing=0.0, relaxation=0.0)
    
    exp_params = ExperimentalParameters(
        physical_constants=physical_constants,
        system_dims=system_dims,
        measurement=measurement,
        initial_state=initial_state,
        noise_config=noise_config
    )
    
    trainable_params = TrainableParameters()
    trainable_params.add_rotation_angles(
        names=["theta1", "theta2"],
        initial_values=[0.0, 0.0]
    )
    
    return SingleQubitExperiment(exp_params, trainable_params)


@pytest.fixture
def two_qubit_experiment():
    """Create a basic two-qubit experiment for testing."""
    physical_constants = PhysicalConstants(
        n_qubits=2,
        chi=[10.0, 10.0],
        photon_cavity_coupling=10.0,
        inverse_pulse_width=1.0
    )
    
    system_dims = SystemDimensions(
        field_levels=2,
        cavity_levels=2,
        qubit_levels=[2, 2]
    )
    
    measurement = MeasurementProtocol(
        measurement_times=[-2.0, -1.0, 0.0, 1.0, 2.0]
    )
    
    initial_state = InitialStateConfig(state_type=InitialStateType.SINGLE_PHOTON)
    noise_config = NoiseConfiguration(
        depolarizing=[0.0, 0.0],
        dephasing=[0.0, 0.0],
        relaxation=[0.0, 0.0]
    )
    
    exp_params = ExperimentalParameters(
        physical_constants=physical_constants,
        system_dims=system_dims,
        measurement=measurement,
        initial_state=initial_state,
        noise_config=noise_config
    )
    
    trainable_params = TrainableParameters()
    trainable_params.add_rotation_angles(
        names=["theta1_q1", "theta2_q1", "theta1_q2", "theta2_q2"],
        initial_values=[0.0, 0.0, 0.0, 0.0]
    )
    
    return TwoQubitExperiment(exp_params, trainable_params)


def test_single_qubit_sweep_method_exists(single_qubit_experiment):
    """Test that sweep_chi_lambda method exists on SingleQubitExperiment."""
    assert hasattr(single_qubit_experiment, 'sweep_chi_lambda')


def test_two_qubit_sweep_method_exists(two_qubit_experiment):
    """Test that sweep_chi_lambda method exists on TwoQubitExperiment."""
    assert hasattr(two_qubit_experiment, 'sweep_chi_lambda')


def test_single_qubit_sweep_small(single_qubit_experiment):
    """Test chi-lambda sweep on single qubit with minimal grid."""
    results = single_qubit_experiment.sweep_chi_lambda(
        chi_interval=[5.0, 15.0],
        lambda_interval=[5.0, 15.0],
        resolution_chi=3,
        resolution_lambda=3,
        batch_size=1,
        verbose=False
    )
    
    # Check result structure
    assert 'chi_vals' in results
    assert 'lambda_vals' in results
    assert 'contrast_map' in results
    assert 'detection_map' in results
    assert 'detection_without_map' in results
    assert 'chi_scale' in results
    assert 'lambda_scale' in results
    
    # Check array shapes
    assert len(results['chi_vals']) == 3
    assert len(results['lambda_vals']) == 3
    assert results['contrast_map'].shape == (3, 3)
    assert results['detection_map'].shape == (3, 3)
    assert results['detection_without_map'].shape == (3, 3)
    
    # Check values are reasonable
    assert np.all(results['contrast_map'] >= -1.0)
    assert np.all(results['contrast_map'] <= 1.0)
    assert np.all(results['detection_map'] >= -1e-10)  # Allow for numerical precision
    assert np.all(results['detection_map'] <= 1.0)
    
    # Check scale parameters
    assert results['chi_scale'] == 'linear'
    assert results['lambda_scale'] == 'linear'


def test_two_qubit_sweep_small(two_qubit_experiment):
    """Test chi-lambda sweep on two qubits with minimal grid."""
    results = two_qubit_experiment.sweep_chi_lambda(
        chi_interval=[5.0, 15.0],
        lambda_interval=[5.0, 15.0],
        resolution_chi=3,
        resolution_lambda=3,
        batch_size=1,
        verbose=False
    )
    
    # Check result structure
    assert 'chi_vals' in results
    assert 'lambda_vals' in results
    assert 'contrast_map' in results
    assert 'detection_map' in results
    assert 'detection_without_map' in results
    assert 'chi_scale' in results
    assert 'lambda_scale' in results
    
    # Check array shapes
    assert len(results['chi_vals']) == 3
    assert len(results['lambda_vals']) == 3
    assert results['contrast_map'].shape == (3, 3)
    
    # Check values are reasonable
    assert np.all(results['contrast_map'] >= -1.0)
    assert np.all(results['contrast_map'] <= 1.0)


def test_compute_chi_lambda_sweep_function(single_qubit_experiment):
    """Test the standalone compute_chi_lambda_sweep function."""
    results = compute_chi_lambda_sweep(
        single_qubit_experiment,
        chi_interval=[5.0, 15.0],
        lambda_interval=[5.0, 15.0],
        resolution_chi=2,
        resolution_lambda=2,
        verbose=False
    )
    
    assert 'chi_vals' in results
    assert 'lambda_vals' in results
    assert results['contrast_map'].shape == (2, 2)
    assert 'chi_scale' in results
    assert 'lambda_scale' in results


def test_plot_chi_lambda_sweep(single_qubit_experiment):
    """Test that plotting function runs without error."""
    results = single_qubit_experiment.sweep_chi_lambda(
        chi_interval=[5.0, 15.0],
        lambda_interval=[5.0, 15.0],
        resolution_chi=3,
        resolution_lambda=3,
        batch_size=1,
        verbose=False
    )
    
    # Should not raise an error
    fig = plot_chi_lambda_sweep(results)
    assert fig is not None


def test_sweep_with_log_scale(single_qubit_experiment):
    """Test chi-lambda sweep with logarithmic scale."""
    results = single_qubit_experiment.sweep_chi_lambda(
        chi_interval=[1.0, 100.0],
        lambda_interval=[1.0, 100.0],
        resolution_chi=3,
        resolution_lambda=3,
        chi_scale='log',
        lambda_scale='log',
        batch_size=1,
        verbose=False
    )
    
    # Check that scale parameters are correctly stored
    assert results['chi_scale'] == 'log'
    assert results['lambda_scale'] == 'log'
    
    # Check that log spacing produces expected values
    chi_vals = results['chi_vals']
    lambda_vals = results['lambda_vals']
    
    # In log space, differences between consecutive log values should be approximately equal
    log_chi_diffs = np.diff(np.log10(chi_vals))
    log_lambda_diffs = np.diff(np.log10(lambda_vals))
    
    # Check that log spacing is uniform
    assert np.allclose(log_chi_diffs, log_chi_diffs[0], rtol=1e-5)
    assert np.allclose(log_lambda_diffs, log_lambda_diffs[0], rtol=1e-5)


def test_plot_selective_displays(single_qubit_experiment):
    """Test that plotting function works with selective display options."""
    results = single_qubit_experiment.sweep_chi_lambda(
        chi_interval=[5.0, 15.0],
        lambda_interval=[5.0, 15.0],
        resolution_chi=3,
        resolution_lambda=3,
        batch_size=1,
        verbose=False
    )
    
    # Test showing only contrast
    fig1 = plot_chi_lambda_sweep(
        results,
        show_contrast=True,
        show_detection_with=False,
        show_detection_without=False
    )
    assert fig1 is not None
    assert len(fig1.axes) == 2  # 1 plot + 1 colorbar
    
    # Test showing only detection with photon
    fig2 = plot_chi_lambda_sweep(
        results,
        show_contrast=False,
        show_detection_with=True,
        show_detection_without=False
    )
    assert fig2 is not None
    assert len(fig2.axes) == 2
    
    # Test showing two plots
    fig3 = plot_chi_lambda_sweep(
        results,
        show_contrast=True,
        show_detection_with=True,
        show_detection_without=False
    )
    assert fig3 is not None
    assert len(fig3.axes) == 4  # 2 plots + 2 colorbars
    
    # Test showing all three plots
    fig4 = plot_chi_lambda_sweep(
        results,
        show_contrast=True,
        show_detection_with=True,
        show_detection_without=True
    )
    assert fig4 is not None
    assert len(fig4.axes) == 6  # 3 plots + 3 colorbars
    
    # Test that error is raised when no plots are enabled
    with pytest.raises(ValueError, match="At least one plot type must be enabled"):
        plot_chi_lambda_sweep(
            results,
            show_contrast=False,
            show_detection_with=False,
            show_detection_without=False
        )
