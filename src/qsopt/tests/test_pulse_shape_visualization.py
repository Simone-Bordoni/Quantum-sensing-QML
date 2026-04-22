"""
Tests for pulse shape visualization functionality.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib import colors as mcolors
from matplotlib.figure import Figure

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from qsopt.core.experiment.quantum_utils import u0
from qsopt.core.experimental_parameters import (
    ExperimentalParameters,
    InitialStateConfig,
    InitialStateType,
    MeasurementProtocol,
    PhysicalSetup,
    SystemDimensions,
)
from qsopt.utils.visualization import plot_pulse_shape_with_measurements


class TestU0Function:
    """Test the u0 Gaussian pulse function."""

    def test_u0_single_value(self):
        """Test u0 with a single time value."""
        sigma = 0.1
        t = 0.0
        result = u0(t, sigma=sigma)

        # At t=0, u0 should be 1.0
        assert abs(float(result) - 1.0) < 1e-10

    def test_u0_array_values(self):
        """Test u0 with array of time values."""
        sigma = 0.1
        t_vals = np.array([-1.0, 0.0, 1.0])
        result = u0(t_vals, sigma=sigma)

        # Check shape
        assert result.shape == t_vals.shape

        # Check that value at t=0 is maximum
        assert float(result[1]) == pytest.approx(1.0, abs=1e-10)

        # Check symmetry
        assert abs(float(result[0]) - float(result[2])) < 1e-10

    def test_u0_gaussian_decay(self):
        """Test that u0 decays as expected for Gaussian."""
        sigma = 0.1
        t = 5.0
        result = u0(t, sigma=sigma)

        # At t=5.0, u0 should be exp(-(0.1*5)^2) = exp(-0.25)
        expected = np.exp(-0.25)
        assert abs(float(result) - expected) < 1e-10


class TestPulseShapeVisualization:
    """Test the pulse shape visualization function."""

    @pytest.fixture
    def exp_params(self):
        """Create test experimental parameters."""
        physical_setup = PhysicalSetup(
            chi=0.01, photon_cavity_coupling=0.1, inverse_pulse_width=0.1
        )
        system_dims = SystemDimensions(cavity_levels=2, qubit_levels=2, field_levels=2)
        measurement = MeasurementProtocol(
            initial_time=-5.0, final_time=5.0, time_interval=1.0, initial_time_uncertainty=0.0
        )
        initial_state = InitialStateConfig(
            state_type=InitialStateType.COHERENT, coherent_alpha=1.0 + 0.0j
        )

        return ExperimentalParameters(
            physical_setup=physical_setup,
            system_dims=system_dims,
            measurement=measurement,
            initial_state=initial_state,
        )

    def test_plot_pulse_shape_creates_figure(self, exp_params):
        """Test that plot_pulse_shape_with_measurements creates a figure."""
        fig = plot_pulse_shape_with_measurements(exp_params)

        assert fig is not None
        assert isinstance(fig, Figure)

        # Check that figure has axes
        axes = fig.get_axes()
        assert len(axes) > 0

        plt.close(fig)

    def test_plot_pulse_shape_with_save(self, exp_params, tmp_path):
        """Test that pulse shape plot can be saved to file."""
        save_path = tmp_path / "test_pulse_shape.png"

        fig = plot_pulse_shape_with_measurements(exp_params, save_path=str(save_path))

        assert save_path.exists()
        assert save_path.stat().st_size > 0

        plt.close(fig)

    def test_plot_pulse_shape_with_batch(self, exp_params):
        """Ensure batch visualization draws multiple colored measurement sets."""
        batch_size = 3
        exp_params.measurement.initial_time_uncertainty = 0.0
        fig = plot_pulse_shape_with_measurements(exp_params, batch_size=batch_size)

        ax = fig.axes[0]
        measurement_lines = [line for line in ax.get_lines() if line.get_linestyle() == "--"]
        expected_lines = len(exp_params.measurement_times) * batch_size
        assert len(measurement_lines) == expected_lines

        unique_colors = {
            tuple(np.round(mcolors.to_rgb(line.get_color()), decimals=5))
            for line in measurement_lines
        }
        assert len(unique_colors) == batch_size

        plt.close(fig)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
