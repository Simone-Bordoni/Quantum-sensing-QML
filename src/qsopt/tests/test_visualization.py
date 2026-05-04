"""
Tests for visualization utilities.
"""

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")  # Use non-interactive backend for testing
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from qsopt.core.experimental_parameters import ExperimentalParameters, InitialStateType
from qsopt.utils import plot_parameter_landscape


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

    # No noise
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


def create_test_landscape_data():
    """Create minimal landscape data for testing."""
    resolution = 5
    theta1_vals = np.linspace(0, np.pi, resolution)
    theta2_vals = np.linspace(-np.pi, 0, resolution)

    # Create simple test patterns
    metric_map = np.random.rand(resolution, resolution) * 0.5 + 0.3  # [0.3, 0.8]
    detection_map = np.random.rand(resolution, resolution) * 0.4 + 0.4  # [0.4, 0.8]

    return {
        "theta1_vals": theta1_vals,
        "theta2_vals": theta2_vals,
        "metric_map": metric_map,
        "detection_map": detection_map,
        "center_theta1": np.pi / 2,
        "center_theta2": -np.pi / 2,
    }


def test_plot_parameter_landscape_creates_figure():
    """Test that plot_parameter_landscape creates a figure."""
    exp_params = create_test_experiment()
    landscape_data = create_test_landscape_data()

    # Create plot without saving
    fig = plot_parameter_landscape(landscape_data, exp_params, save_path=None)

    # Check figure was created
    assert fig is not None
    assert isinstance(fig, Figure)

    # Check figure has two subplots (axes)
    assert len(fig.axes) == 4  # 2 main axes + 2 colorbars

    plt.close(fig)


def test_plot_parameter_landscape_with_save():
    """Test that plot_parameter_landscape can save to file."""
    import tempfile
    from pathlib import Path

    exp_params = create_test_experiment()
    landscape_data = create_test_landscape_data()

    # Create temporary file path
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = str(Path(tmpdir) / "test_landscape.png")

        # Create and save plot
        fig = plot_parameter_landscape(landscape_data, exp_params, save_path=save_path)

        # Check file was created
        assert Path(save_path).exists()

        plt.close(fig)


def test_plot_parameter_landscape_system_info():
    """Test that system info box contains expected information."""
    exp_params = create_test_experiment()
    landscape_data = create_test_landscape_data()

    fig = plot_parameter_landscape(landscape_data, exp_params, save_path=None)

    # The system info is added as a text artist to the figure
    # Check that text was added (not easy to verify exact content)
    assert fig is not None

    plt.close(fig)


def test_plot_parameter_landscape_custom_dpi():
    """Test that custom DPI setting works."""
    import tempfile
    from pathlib import Path

    exp_params = create_test_experiment()
    landscape_data = create_test_landscape_data()

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = str(Path(tmpdir) / "test_landscape_high_dpi.png")

        # Create plot with custom DPI
        fig = plot_parameter_landscape(landscape_data, exp_params, save_path=save_path, dpi=150)

        assert Path(save_path).exists()
        plt.close(fig)


def test_plot_time_interval_landscape_creates_figure():
    """Test that plot_time_interval_landscape creates a figure."""
    from qsopt.utils import plot_time_interval_landscape

    exp_params = create_test_experiment()

    # Create test time interval data
    resolution = 5
    interval_vals = np.linspace(0.1, 5.0, resolution)
    landscape_data = {
        "interval_vals": interval_vals,
        "metric_vals": np.random.rand(resolution) * 0.5 + 0.3,
        "detection_with": np.random.rand(resolution) * 0.4 + 0.4,
        "detection_without": np.random.rand(resolution) * 0.3 + 0.2,
        "n_measurements": np.array([10, 8, 6, 4, 2]),
        "theta1": np.pi / 2,
        "theta2": -np.pi / 2,
        "mode": "continuous",
        "batch_size": 1,
        "initial_time_uncertainty": 0.0,
    }

    # Create plot without saving (default hides measurement count subplot)
    fig = plot_time_interval_landscape(landscape_data, exp_params, save_path=None)

    # Check figure was created
    assert fig is not None
    assert isinstance(fig, Figure)

    # Default configuration renders two subplots (metric + probabilities)
    assert len(fig.axes) == 2

    plt.close(fig)


def test_plot_time_interval_landscape_with_save():
    """Test that plot_time_interval_landscape can save to file."""
    import tempfile
    from pathlib import Path

    from qsopt.utils import plot_time_interval_landscape

    exp_params = create_test_experiment()

    # Create test time interval data
    resolution = 5
    interval_vals = np.linspace(0.1, 5.0, resolution)
    landscape_data = {
        "interval_vals": interval_vals,
        "metric_vals": np.random.rand(resolution) * 0.5 + 0.3,
        "detection_with": np.random.rand(resolution) * 0.4 + 0.4,
        "detection_without": np.random.rand(resolution) * 0.3 + 0.2,
        "n_measurements": np.array([10, 8, 6, 4, 2]),
        "theta1": np.pi / 2,
        "theta2": -np.pi / 2,
        "mode": "continuous",
        "batch_size": 10,
        "initial_time_uncertainty": 0.1,
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = str(Path(tmpdir) / "test_time_interval.png")

        # Create and save plot
        fig = plot_time_interval_landscape(landscape_data, exp_params, save_path=save_path)

        # Check file was created
        assert Path(save_path).exists()

        plt.close(fig)


def test_plot_time_interval_landscape_with_measurement_count():
    """Check optional measurement count subplot appears when requested."""
    from qsopt.utils import plot_time_interval_landscape

    exp_params = create_test_experiment()

    resolution = 4
    interval_vals = np.linspace(0.2, 4.0, resolution)
    landscape_data = {
        "interval_vals": interval_vals,
        "metric_vals": np.linspace(0.3, 0.6, resolution),
        "detection_with": np.linspace(0.4, 0.7, resolution),
        "detection_without": np.linspace(0.2, 0.5, resolution),
        "n_measurements": np.array([12, 9, 6, 3]),
        "theta1": np.pi / 2,
        "theta2": -np.pi / 2,
        "mode": "continuous",
        "batch_size": 1,
        "initial_time_uncertainty": 0.0,
    }

    fig = plot_time_interval_landscape(
        landscape_data, exp_params, save_path=None, show_measurement_count=True
    )

    assert len(fig.axes) == 3
    plt.close(fig)


def test_plot_optimization_dashboard_basic():
    """Test basic optimization dashboard creation."""
    from qsopt import OptimizationCallback
    from qsopt.utils import plot_optimization_dashboard

    # Create mock optimization callback
    callback = OptimizationCallback(save_every=1, save_best=True)

    # Add some history
    for i in range(5):
        callback(
            trainable_params_initial=[0.5 + i * 0.1],
            trainable_params_final=[1.0 + i * 0.1],
            detection_with=0.6 + i * 0.05,
            detection_without=0.3,
            metric=0.3 + i * 0.05,
        )

    # Create dashboard
    fig = plot_optimization_dashboard(callback, save_path=None)

    assert fig is not None
    assert isinstance(fig, Figure)
    plt.close(fig)


def test_plot_optimization_dashboard_with_reference():
    """Test optimization dashboard with reference callback."""
    from qsopt import OptimizationCallback
    from qsopt.utils import plot_optimization_dashboard

    # Create optimization callback
    opt_callback = OptimizationCallback(save_every=1, save_best=True)
    for i in range(3):
        opt_callback(
            trainable_params_initial=[0.5 + i * 0.1],
            trainable_params_final=[],
            detection_with=0.7, detection_without=0.3, metric=0.4
        )

    # Create reference callback
    ref_callback = OptimizationCallback(save_every=1, save_best=False)
    ref_callback(
        trainable_params_initial=[1.5],
        trainable_params_final=[],
        detection_with=0.8, detection_without=0.2, metric=0.6
    )

    # Create dashboard with reference
    fig = plot_optimization_dashboard(opt_callback, reference_callback=ref_callback)

    assert fig is not None
    plt.close(fig)


def test_plot_optimization_dashboard_selective_plots():
    """Test optimization dashboard with selective plot types."""
    from qsopt import OptimizationCallback
    from qsopt.utils import plot_optimization_dashboard

    callback = OptimizationCallback(save_every=1, save_best=True)
    callback(
        trainable_params_initial=[1.0],
        trainable_params_final=[0.5],
        detection_with=0.7, detection_without=0.3, metric=0.4
    )

    # Test with only metric plot
    fig1 = plot_optimization_dashboard(
        callback,
        show_metric=True,
        show_gradients=False,
        show_parameters=False,
        show_trajectory=False,
        show_detection_measures=False,
    )
    assert fig1 is not None
    plt.close(fig1)

    # Test with only probabilities plot
    fig2 = plot_optimization_dashboard(
        callback,
        show_metric=False,
        show_gradients=False,
        show_parameters=False,
        show_trajectory=False,
        show_detection_measures=True,
    )
    assert fig2 is not None
    plt.close(fig2)


def test_plot_optimization_dashboard_no_plots_raises():
    """Test that dashboard raises error when no plots enabled."""
    from qsopt import OptimizationCallback
    from qsopt.utils import plot_optimization_dashboard

    callback = OptimizationCallback(save_every=1, save_best=True)
    callback(
        trainable_params_initial=[1.0],
        trainable_params_final=[],
        detection_with=0.7, detection_without=0.3, metric=0.4
    )

    # Should raise ValueError when all plots disabled
    with pytest.raises(ValueError, match="At least one plot type must be enabled"):
        plot_optimization_dashboard(
            callback,
            show_metric=False,
            show_gradients=False,
            show_parameters=False,
            show_trajectory=False,
            show_detection_measures=False,
        )


def test_plot_optimization_dashboard_confusion_summary_panel_enabled_with_data():
    """Test confusion matrix summary panel renders when toggle is on and callback has data."""
    from qsopt import OptimizationCallback
    from qsopt.utils import plot_optimization_dashboard

    callback = OptimizationCallback(save_every=1, save_best=True)
    callback(
        trainable_params_initial=[1.0],
        trainable_params_final=[0.5],
        detection_with=0.7,
        detection_without=0.3,
        metric=0.4,
    )
    callback.set_measurement_protocol(
        with_photon={"0": 0.2, "1": 0.8},
        without_photon={"0": 0.7, "1": 0.3},
    )

    fig = plot_optimization_dashboard(
        callback,
        show_metric=False,
        show_gradients=False,
        show_parameters=False,
        show_trajectory=False,
        show_detection_measures=True,
        show_confusion_matrix_summary=True,
    )

    assert fig is not None
    assert isinstance(fig, Figure)
    # Detection plot + confusion matrix + its colorbar
    assert len(fig.axes) == 3
    plt.close(fig)


def test_plot_optimization_dashboard_confusion_summary_panel_skipped_without_data():
    """Toggle on should not render summary panel when callback has no confusion/protocol data."""
    from qsopt import OptimizationCallback
    from qsopt.utils import plot_optimization_dashboard

    callback = OptimizationCallback(save_every=1, save_best=True)
    callback(
        trainable_params_initial=[1.0],
        trainable_params_final=[0.5],
        detection_with=0.7,
        detection_without=0.3,
        metric=0.4,
    )

    fig = plot_optimization_dashboard(
        callback,
        show_metric=False,
        show_gradients=False,
        show_parameters=False,
        show_trajectory=False,
        show_detection_measures=True,
        show_confusion_matrix_summary=True,
    )

    assert fig is not None
    assert isinstance(fig, Figure)
    # Only detection plot should be present
    assert len(fig.axes) == 1
    plt.close(fig)


def test_plot_time_evolution_with_cavity_population():
    """Test plot_time_evolution with cavity population enabled."""
    from qsopt.utils import plot_time_evolution
    from qsopt.utils.results import TimeEvolutionResults

    # Create sample time evolution data
    times = np.linspace(-5, 5, 100)
    prob_0 = np.exp(-(times**2))
    prob_1 = 1 - prob_0
    pulse_shape = np.exp(-(times**2))
    cavity_population = 0.1 * np.exp(-(times**2))  # Peak at t=0
    measurement_times = [-5.0, 5.0]

    results = TimeEvolutionResults(
        times=times,
        probabilities={"prob_0": prob_0, "prob_1": prob_1},
        pulse_shape=pulse_shape,
        measurement_times=measurement_times,
        cavity_population=cavity_population,
    )

    # Test with cavity population
    fig = plot_time_evolution(results, show_cavity_population=True)
    assert fig is not None
    assert isinstance(fig, Figure)
    plt.close(fig)

    # Test without cavity population (default)
    fig2 = plot_time_evolution(results, show_cavity_population=False)
    assert fig2 is not None
    plt.close(fig2)


def test_plot_time_evolution_two_qubit_with_cavity():
    """Test plot_time_evolution for two-qubit system with cavity population."""
    from qsopt.utils import plot_time_evolution
    from qsopt.utils.results import TimeEvolutionResults

    # Create sample two-qubit time evolution data
    times = np.linspace(-5, 5, 100)
    prob_00 = 0.5 * np.exp(-(times**2))
    prob_01 = 0.2 * np.ones_like(times)
    prob_10 = 0.2 * np.ones_like(times)
    prob_11 = 1 - prob_00 - prob_01 - prob_10
    cavity_population = 0.15 * np.exp(-(times**2))

    results = TimeEvolutionResults(
        times=times,
        probabilities={
            "prob_00": prob_00,
            "prob_01": prob_01,
            "prob_10": prob_10,
            "prob_11": prob_11,
        },
        pulse_shape=np.exp(-(times**2)),
        measurement_times=[-5.0, 5.0],
        cavity_population=cavity_population,
    )

    # Test with cavity population
    fig = plot_time_evolution(results, show_cavity_population=True)
    assert fig is not None
    assert isinstance(fig, Figure)
    plt.close(fig)


def test_plot_time_evolution_with_field_population():
    """Test plot_time_evolution with external field population enabled."""
    from qsopt.utils import plot_time_evolution
    from qsopt.utils.results import TimeEvolutionResults

    # Create sample time evolution data with both populations
    times = np.linspace(-5, 5, 100)
    prob_0 = np.exp(-(times**2))
    prob_1 = 1 - prob_0
    pulse_shape = np.exp(-(times**2))
    cavity_population = 0.1 * np.exp(-(times**2))
    field_population = 0.8 * np.exp(-(times**2))  # Field population higher than cavity
    measurement_times = [-5.0, 5.0]

    results = TimeEvolutionResults(
        times=times,
        probabilities={"prob_0": prob_0, "prob_1": prob_1},
        pulse_shape=pulse_shape,
        measurement_times=measurement_times,
        cavity_population=cavity_population,
        field_population=field_population,
    )

    # Test with field population only
    fig = plot_time_evolution(results, show_field_population=True)
    assert fig is not None
    assert isinstance(fig, Figure)
    plt.close(fig)

    # Test with both cavity and field populations
    fig2 = plot_time_evolution(
        results, show_cavity_population=True, show_field_population=True
    )
    assert fig2 is not None
    plt.close(fig2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
