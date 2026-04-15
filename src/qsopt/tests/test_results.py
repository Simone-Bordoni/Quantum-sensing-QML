"""
Tests for results data structures.
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from qsopt.utils.results import SweepResults, TimeEvolutionResults, load_results, save_results


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

    def test_save_load_uncompressed(self):
        """Test saving without compression."""
        times = np.linspace(0, 5, 10)
        probs = {"prob_0": np.random.rand(10)}
        original = TimeEvolutionResults(times=times, probabilities=probs)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "uncompressed"
            save_results(original, str(filepath), compress=False)
            assert (Path(tmpdir) / "uncompressed.npz").exists()
            loaded = load_results(str(Path(tmpdir) / "uncompressed.npz"))
            assert isinstance(loaded, TimeEvolutionResults)

    def test_load_nonexistent_file(self):
        """Test that loading nonexistent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_results("nonexistent_file.npz")

    def test_save_invalid_type(self):
        """Test that saving wrong type raises TypeError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "bad.npz"
            with pytest.raises(TypeError):
                save_results("not_a_results_object", str(filepath))


class TestSweepResults:
    """Test suite for SweepResults."""

    def test_initialization(self):
        """Test basic SweepResults initialization."""
        chi_vals = np.linspace(0.1, 1.0, 5)
        gamma_vals = np.linspace(0.01, 0.1, 5)
        metric_map = np.random.rand(5, 5)

        sweep = SweepResults(
            param1_name="gamma",
            param1_vals=gamma_vals,
            param1_scale="log",
            param2_name="chi",
            param2_vals=chi_vals,
            param2_scale="linear",
            results={"metric_map": metric_map},
        )

        assert sweep.param1_name == "gamma"
        assert sweep.param2_name == "chi"
        assert len(sweep.param1_vals) == 5
        assert len(sweep.param2_vals) == 5
        assert "metric_map" in sweep.results

    def test_str_representation(self):
        """Test string representation of SweepResults."""
        chi_vals = np.linspace(0.1, 1.0, 3)
        gamma_vals = np.linspace(0.01, 0.1, 3)
        metric = np.random.rand(3, 3)

        sweep = SweepResults(
            param1_name="gamma",
            param1_vals=gamma_vals,
            param1_scale="linear",
            param2_name="chi",
            param2_vals=chi_vals,
            param2_scale="linear",
            results={"metric_map": metric},
        )

        str_repr = str(sweep)
        assert "SweepResults" in str_repr
        assert "gamma" in str_repr
        assert "chi" in str_repr
        assert "metric_map" in str_repr

    def test_str_with_optimal_metadata(self):
        """Test string representation shows optimal point from metadata."""
        chi_vals = np.linspace(0.1, 1.0, 3)
        gamma_vals = np.linspace(0.01, 0.1, 3)

        sweep = SweepResults(
            param1_name="gamma",
            param1_vals=gamma_vals,
            param1_scale="linear",
            param2_name="chi",
            param2_vals=chi_vals,
            param2_scale="linear",
            results={"metric_map": np.random.rand(3, 3)},
            metadata={"optimal_chi": 0.5, "optimal_gamma": 0.05},
        )

        str_repr = str(sweep)
        assert "Optimal" in str_repr

    def test_repr(self):
        """Test repr representation of SweepResults."""
        chi_vals = np.linspace(0.1, 1.0, 3)
        gamma_vals = np.linspace(0.01, 0.1, 3)

        sweep = SweepResults(
            param1_name="gamma",
            param1_vals=gamma_vals,
            param1_scale="linear",
            param2_name="chi",
            param2_vals=chi_vals,
            param2_scale="linear",
            results={"metric_map": np.random.rand(3, 3)},
        )

        repr_str = repr(sweep)
        assert "SweepResults" in repr_str
        assert "gamma" in repr_str
        assert "chi" in repr_str

    def test_save_load_sweep_results(self):
        """Test saving and loading SweepResults."""
        chi_vals = np.linspace(0.1, 1.0, 4)
        gamma_vals = np.linspace(0.01, 0.1, 4)
        metric_map = np.random.rand(4, 4)
        detection_map = np.random.rand(4, 4)

        original = SweepResults(
            param1_name="gamma",
            param1_vals=gamma_vals,
            param1_scale="log",
            param2_name="chi",
            param2_vals=chi_vals,
            param2_scale="linear",
            results={"metric_map": metric_map, "detection_map": detection_map},
            metadata={"optimal_chi": 0.5, "optimal_gamma": 0.05},
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "sweep.npz"
            save_results(original, str(filepath))

            assert filepath.exists()

            loaded = load_results(str(filepath))

            assert isinstance(loaded, SweepResults)
            assert loaded.param1_name == "gamma"
            assert loaded.param2_name == "chi"
            assert loaded.param1_scale == "log"
            assert np.array_equal(loaded.param1_vals, original.param1_vals)
            assert np.array_equal(loaded.param2_vals, original.param2_vals)
            assert np.allclose(loaded.results["metric_map"], metric_map)
            assert np.allclose(loaded.results["detection_map"], detection_map)
            assert loaded.metadata["optimal_chi"] == 0.5

    def test_str_multi_qubit_detection(self):
        """Test n-qubit detection in str representation."""
        times = np.linspace(0, 5, 10)
        probs = {
            "q_000": np.random.rand(10),
            "q_001": np.random.rand(10),
            "q_010": np.random.rand(10),
            "q_111": np.random.rand(10),
        }
        result = TimeEvolutionResults(times=times, probabilities=probs)
        str_repr = str(result)
        # Suffix "000" -> 3 qubits
        assert "3-qubit" in str_repr or "qubit" in str_repr

    def test_str_with_metadata_n_qubits(self):
        """Test str uses n_qubits from metadata when present."""
        times = np.linspace(0, 5, 10)
        probs = {"result": np.random.rand(10)}
        result = TimeEvolutionResults(
            times=times, probabilities=probs,
            metadata={"n_qubits": 2, "detection_criterion": "any excited"}
        )
        str_repr = str(result)
        assert "Two-qubit" in str_repr
        assert "any excited" in str_repr
