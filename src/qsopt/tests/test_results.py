"""
Tests for results data structures.
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from qsopt.utils.results import TimeEvolutionResults, load_results, save_results


class TestTimeEvolutionResults:
    """Test suite for TimeEvolutionResults."""

    def test_initialization(self):
        """Test basic initialization."""
        times = np.linspace(0, 10, 100)
        probs = {"prob_0": np.random.rand(100), "prob_1": np.random.rand(100)}

        result = TimeEvolutionResults(times=times, probabilities=probs)

        assert np.array_equal(result.times, times)
        assert result.probabilities == probs
        assert result.pulse_shape is None
        assert result.measurement_times is None

    def test_with_pulse_and_measurements(self):
        """Test initialization with pulse and measurement times."""
        times = np.linspace(-5, 5, 50)
        probs = {"prob_0": np.random.rand(50)}
        pulse = np.random.rand(50)
        meas_times = [-5.0, 0.0, 5.0]

        result = TimeEvolutionResults(
            times=times, probabilities=probs, pulse_shape=pulse, measurement_times=meas_times
        )

        assert result.pulse_shape is not None
        assert result.measurement_times is not None
        assert len(result.measurement_times) == 3

    def test_str_representation(self):
        """Test string representation."""
        times = np.linspace(0, 10, 10)
        probs = {"prob_0": np.random.rand(10), "prob_1": np.random.rand(10)}

        result = TimeEvolutionResults(times=times, probabilities=probs)
        str_repr = str(result)

        assert "TimeEvolutionResults" in str_repr
        assert "Single-qubit" in str_repr
        assert "prob_0" in str_repr

    def test_two_qubit_detection(self):
        """Test that two-qubit system is detected."""
        times = np.linspace(0, 5, 20)
        probs = {
            "prob_00": np.random.rand(20),
            "prob_01": np.random.rand(20),
            "prob_10": np.random.rand(20),
            "prob_11": np.random.rand(20),
        }

        result = TimeEvolutionResults(times=times, probabilities=probs)
        str_repr = str(result)

        assert "Two-qubit" in str_repr

    def test_metadata(self):
        """Test metadata storage."""
        times = np.linspace(0, 10, 10)
        probs = {"prob_0": np.random.rand(10)}
        metadata = {"chi": 0.5, "gamma": 0.03, "description": "Test experiment"}

        result = TimeEvolutionResults(times=times, probabilities=probs, metadata=metadata)

        assert result.metadata["chi"] == 0.5
        assert result.metadata["gamma"] == 0.03

    def test_repr(self):
        """Test repr representation."""
        times = np.linspace(0, 10, 10)
        probs = {"prob_0": np.random.rand(10), "prob_1": np.random.rand(10)}

        result = TimeEvolutionResults(times=times, probabilities=probs)
        repr_str = repr(result)

        assert "TimeEvolutionResults" in repr_str
        assert "10 points" in repr_str


class TestSaveLoad:
    """Test suite for save/load functionality."""

    def test_save_load_time_evolution(self):
        """Test saving and loading TimeEvolutionResults."""
        times = np.linspace(0, 10, 50)
        probs = {"prob_0": np.random.rand(50), "prob_1": np.random.rand(50)}
        pulse = np.random.rand(50)

        original = TimeEvolutionResults(times=times, probabilities=probs, pulse_shape=pulse)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "evolution.npz"
            save_results(original, str(filepath))

            assert filepath.exists()

            loaded = load_results(str(filepath))

            assert isinstance(loaded, TimeEvolutionResults)
            assert np.array_equal(loaded.times, original.times)
            assert np.array_equal(loaded.probabilities["prob_0"], original.probabilities["prob_0"])

    def test_save_with_metadata(self):
        """Test saving and loading with metadata."""
        times = np.linspace(0, 5, 10)
        probs = {"prob_0": np.random.rand(10)}
        metadata = {"chi": 0.5, "test": "value"}

        original = TimeEvolutionResults(times=times, probabilities=probs, metadata=metadata)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "with_metadata.npz"
            save_results(original, str(filepath))

            loaded = load_results(str(filepath))

            assert loaded.metadata["chi"] == 0.5
            assert loaded.metadata["test"] == "value"

    def test_save_with_measurement_times(self):
        """Test saving with measurement times."""
        times = np.linspace(0, 10, 20)
        probs = {"prob_0": np.random.rand(20)}
        meas_times = [0.0, 5.0, 10.0]

        original = TimeEvolutionResults(
            times=times, probabilities=probs, measurement_times=meas_times
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "with_meas_times.npz"
            save_results(original, str(filepath))

            loaded = load_results(str(filepath))

            assert loaded.measurement_times is not None
            assert len(loaded.measurement_times) == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
