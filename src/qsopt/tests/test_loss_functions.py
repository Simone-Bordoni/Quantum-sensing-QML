"""
Tests for loss functions and detection probability definitions.

Tests for DetectionMetric class and associated detection criteria.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from qsopt.core.loss_functions import DetectionMetric, std_metric, std_batching


class TestDetectionMetricInit:
    """Test DetectionMetric initialization with various criteria."""

    def test_default_any_excited(self):
        """Test default 'any excited' criterion for 1 qubit."""
        dm = DetectionMetric(n_qubits=1)
        assert dm.n_qubits == 1
        assert "any excited" in dm.detection_name
        states = dm.detection_states
        assert isinstance(states, list)
        assert "1" in states  # single qubit: |1⟩ is excited

    def test_any_excited_2qubits(self):
        """Test 'any excited' for 2 qubits."""
        dm = DetectionMetric(n_qubits=2)
        states = dm.detection_states
        # Should include 01, 10, 11 - all except 00
        assert "01" in states
        assert "10" in states
        assert "11" in states
        assert "00" not in states

    def test_min_excited_default(self):
        """Test 'min excited' with default param (1 excitation)."""
        dm = DetectionMetric(n_qubits=2, detection_criterion="min excited")
        states = dm.detection_states
        # Should include all states with at least 1 excitation
        assert "01" in states
        assert "10" in states
        assert "11" in states
        assert "00" not in states

    def test_min_excited_custom_param(self):
        """Test 'min excited' with custom detection_param."""
        dm = DetectionMetric(n_qubits=2, detection_criterion="min excited", detection_param=2)
        states = dm.detection_states
        # Only |11⟩ has 2 excitations for 2-qubit system
        assert "11" in states
        assert "01" not in states
        assert "10" not in states

    def test_excited_qubits_default(self):
        """Test 'excited qubits' with default param (qubit 0)."""
        dm = DetectionMetric(n_qubits=2, detection_criterion="excited qubits")
        states = dm.detection_states
        # States where qubit 0 is excited: 10, 11
        assert "10" in states
        assert "11" in states
        assert "01" not in states

    def test_excited_qubits_custom(self):
        """Test 'excited qubits' with specific qubit list."""
        dm = DetectionMetric(n_qubits=2, detection_criterion="excited qubits", detection_param=[1])
        states = dm.detection_states
        # States where qubit 1 is excited: 01, 11
        assert "01" in states
        assert "11" in states
        assert "10" not in states

    def test_custom_states(self):
        """Test 'custom states' criterion."""
        dm = DetectionMetric(
            n_qubits=2, detection_criterion="custom states", detection_param=["11"]
        )
        assert dm.detection_states == ["11"]

    def test_min_fidelity(self):
        """Test 'min fidelity' criterion changes aggregation logic."""
        dm = DetectionMetric(n_qubits=1, detection_criterion="min fidelity")
        # Should use list aggregation
        assert dm.multiple_measurement_name == "list aggregation"
        assert dm.batching_name == "fidelity batching"

    def test_max_trace_distance(self):
        """Test 'max trace distance' criterion."""
        dm = DetectionMetric(n_qubits=1, detection_criterion="max trace distance")
        assert dm.multiple_measurement_name == "list aggregation"
        assert dm.batching_name == "trace distance batching"

    def test_max_distance(self):
        """Test 'max distance' criterion."""
        dm = DetectionMetric(n_qubits=2, detection_criterion="max distance")
        assert dm.detection_name == "max distance"
        assert dm.multiple_measurement_name == "list aggregation"

    def test_invalid_criterion_raises(self):
        """Test that invalid criterion raises ValueError."""
        with pytest.raises(ValueError, match="criterion"):
            DetectionMetric(n_qubits=1, detection_criterion="invalid_criterion")

    def test_custom_metric(self):
        """Test custom metric function."""
        custom_metric = lambda x, y: x + y
        dm = DetectionMetric(n_qubits=1, metric=custom_metric, metric_name="sum")
        assert dm.custom_metric is True
        assert dm.metric_name == "sum"

    def test_custom_multiple_measurement_logic(self):
        """Test custom multiple measurement logic."""
        custom_logic = (jnp.array(0.0), lambda x, y: x + y, lambda x: x)
        dm = DetectionMetric(n_qubits=1, multiple_measurement_logic=custom_logic)
        assert dm.custom_multiple_measurement_logic is True

    def test_protocol_name_custom(self):
        """Test custom protocol name."""
        dm = DetectionMetric(n_qubits=1, protocol_name="my_protocol")
        assert dm.protocol_name == "my_protocol"

    def test_protocol_name_auto(self):
        """Test automatically generated protocol name."""
        dm = DetectionMetric(n_qubits=1)
        assert "any excited" in dm.protocol_name
        assert "contrast" in dm.protocol_name

    def test_repr(self):
        """Test string representation."""
        dm = DetectionMetric(n_qubits=2)
        repr_str = repr(dm)
        assert "DetectionMetric" in repr_str
        assert "any excited" in repr_str


class TestDetectionMetricCall:
    """Test DetectionMetric __call__ method."""

    def test_call_contrast(self):
        """Test calling metric computes contrast."""
        dm = DetectionMetric(n_qubits=1)
        # Default metric: -(p_with - p_without) = -(0.7 - 0.3) = -0.4
        result = dm(0.7, 0.3)
        # std_metric returns negative contrast
        assert float(result) == pytest.approx(-0.4, abs=1e-6)

    def test_call_zero_contrast(self):
        """Test contrast is zero when both probabilities equal."""
        dm = DetectionMetric(n_qubits=1)
        result = dm(0.5, 0.5)
        assert float(result) == pytest.approx(0.0, abs=1e-6)


class TestStdFunctions:
    """Test standalone std metric/batching functions."""

    def test_std_metric(self):
        """Test std_metric returns negative contrast."""
        result = std_metric(0.8, 0.3)
        assert float(result) == pytest.approx(-0.5, abs=1e-6)

    def test_std_batching(self):
        """Test std_batching aggregates over batch."""
        detect_with = [0.6, 0.8]
        detect_without = [0.3, 0.4]
        p_with, p_without, contrast = std_batching(detect_with, detect_without)
        assert float(p_with) == pytest.approx(0.7, abs=1e-5)
        assert float(p_without) == pytest.approx(0.35, abs=1e-5)
        assert float(contrast) == pytest.approx(0.35, abs=1e-5)

    def test_prob_initializer(self):
        """Test default prob_initializer is 1."""
        dm = DetectionMetric(n_qubits=1)
        assert float(dm.prob_initializer) == 1.0

    def test_measurement_aggregation_multiplies(self):
        """Test default measurement_aggregation multiplies probabilities."""
        dm = DetectionMetric(n_qubits=1)
        result = dm.measurement_aggregation(jnp.array(0.8), jnp.array(0.9))
        assert float(result) == pytest.approx(0.72, abs=1e-6)

    def test_post_aggregation_complements(self):
        """Test default post_aggregation returns 1 - x."""
        dm = DetectionMetric(n_qubits=1)
        result = dm.post_aggregation(jnp.array(0.3))
        assert float(result) == pytest.approx(0.7, abs=1e-6)


class TestDetectionMetricValidationErrors:
    """Test validation errors in DetectionMetric."""

    def test_excited_qubits_non_list_param(self):
        """Test excited_qubits requires list param."""
        with pytest.raises(ValueError):
            DetectionMetric(n_qubits=2, detection_criterion="excited qubits", detection_param=1)

    def test_custom_states_non_string_param(self):
        """Test custom_states requires list of strings."""
        with pytest.raises(ValueError):
            DetectionMetric(n_qubits=2, detection_criterion="custom states", detection_param=[1, 2])


@pytest.mark.skip(reason="DetectionFromProbabilities has been removed")
class TestDetectionFromProbabilities:
    """Legacy test suite - skipped since DetectionFromProbabilities has been removed."""

    def test_placeholder(self):
        pass


@pytest.mark.skip(reason="Predefined detection functions have been removed")
class TestPredefinedDetectionCriteria:
    """Legacy test suite - skipped since predefined detection functions have been removed."""

    @pytest.fixture
    def sample_probs(self):
        """Sample probability distribution for testing."""
        return {"p00": 0.1, "p01": 0.2, "p10": 0.3, "p11": 0.4}

    def test_default_detection(self, sample_probs):
        """Test default detection criterion (1 - P(00))."""
        detector = DetectionFromProbabilities()
        result = detector(sample_probs)
        assert jnp.isclose(result, 0.9)
        assert detector.name == "1-P(00)"

    def test_custom_detection_11(self, sample_probs):
        """Test custom detection for |11⟩ only."""

        def detect_11(probs):
            return probs["p11"]

        detector = DetectionFromProbabilities(detect_11, name="P(11)")
        result = detector(sample_probs)
        assert jnp.isclose(result, 0.4)
        assert detector.name == "P(11)"

    def test_custom_detection_qubit2(self, sample_probs):
        """Test custom detection for qubit 2 in |1⟩."""

        def detect_q2(probs):
            return probs["p01"] + probs["p11"]

        detector = DetectionFromProbabilities(detect_q2, name="P(q2=1)")
        result = detector(sample_probs)
        assert jnp.isclose(result, 0.6)

    def test_custom_detection_qubit1(self, sample_probs):
        """Test custom detection for qubit 1 in |1⟩."""

        def detect_q1(probs):
            return probs["p10"] + probs["p11"]

        detector = DetectionFromProbabilities(detect_q1, name="P(q1=1)")
        result = detector(sample_probs)
        assert jnp.isclose(result, 0.7)

    def test_compute_contrast_positive_difference(self):
        """Test contrast computation with positive difference."""
        contrast = DetectionFromProbabilities.compute_contrast(0.8, 0.2)
        assert jnp.isclose(contrast, 0.6)

    def test_compute_contrast_negative_difference(self):
        """Test contrast computation with negative difference."""
        contrast = DetectionFromProbabilities.compute_contrast(0.2, 0.8)
        assert jnp.isclose(contrast, 0.6)

    def test_compute_contrast_zero(self):
        """Test contrast computation with identical probabilities."""
        contrast = DetectionFromProbabilities.compute_contrast(0.5, 0.5)
        assert jnp.isclose(contrast, 0.0)

    def test_compute_contrast_maximum(self):
        """Test contrast computation with maximum difference."""
        contrast = DetectionFromProbabilities.compute_contrast(1.0, 0.0)
        assert jnp.isclose(contrast, 1.0)

    def test_repr(self):
        """Test string representation."""
        detector = DetectionFromProbabilities(name="custom_criterion")
        repr_str = repr(detector)
        assert "DetectionFromProbabilities" in repr_str
        assert "custom_criterion" in repr_str

    def test_jax_compatibility(self, sample_probs):
        """Test that detection functions are JAX-compatible."""
        detector = DetectionFromProbabilities()

        # Convert to JAX arrays
        jax_probs = {k: jnp.array(v) for k, v in sample_probs.items()}
        result = detector(jax_probs)

        # Should return JAX array
        assert isinstance(result, jnp.ndarray)

    def test_edge_case_all_00(self):
        """Test with all probability in |00⟩."""
        probs = {"p00": 1.0, "p01": 0.0, "p10": 0.0, "p11": 0.0}
        detector = DetectionFromProbabilities()
        result = detector(probs)
        assert jnp.isclose(result, 0.0)

    def test_edge_case_all_11(self):
        """Test with all probability in |11⟩."""
        probs = {"p00": 0.0, "p01": 0.0, "p10": 0.0, "p11": 1.0}
        detector = DetectionFromProbabilities()
        result = detector(probs)
        assert jnp.isclose(result, 1.0)


@pytest.mark.skip(reason="predefined detection functions have been removed")
class TestPredefinedDetectionCriteria:
    """Test suite for predefined detection functions."""

    @pytest.fixture
    def sample_probs(self):
        """Sample probability distribution for testing."""
        return {"p00": 0.1, "p01": 0.2, "p10": 0.3, "p11": 0.4}

    def test_detection_not_00(self, sample_probs):
        """Test detection_not_00 function."""
        result = detection_not_00(sample_probs)
        expected = 1.0 - 0.1
        assert jnp.isclose(result, expected)

    def test_detection_11(self, sample_probs):
        """Test detection_11 function."""
        result = detection_11(sample_probs)
        assert jnp.isclose(result, 0.4)

    def test_detection_qubit1(self, sample_probs):
        """Test detection_qubit1 function."""
        result = detection_qubit1(sample_probs)
        expected = 0.3 + 0.4  # p10 + p11
        assert jnp.isclose(result, expected)

    def test_detection_qubit2(self, sample_probs):
        """Test detection_qubit2 function."""
        result = detection_qubit2(sample_probs)
        expected = 0.2 + 0.4  # p01 + p11
        assert jnp.isclose(result, expected)

    def test_detection_any_excited(self, sample_probs):
        """Test detection_any_excited function."""
        result = detection_any_excited(sample_probs)
        expected = 1.0 - 0.1
        assert jnp.isclose(result, expected)

    def test_detection_both_excited(self, sample_probs):
        """Test detection_both_excited function."""
        result = detection_both_excited(sample_probs)
        assert jnp.isclose(result, 0.4)

    def test_detection_xor(self, sample_probs):
        """Test detection_xor function."""
        result = detection_xor(sample_probs)
        expected = 0.2 + 0.3  # p01 + p10
        assert jnp.isclose(result, expected)

    def test_all_criteria_with_uniform_distribution(self):
        """Test all criteria with uniform distribution."""
        uniform_probs = {"p00": 0.25, "p01": 0.25, "p10": 0.25, "p11": 0.25}

        assert jnp.isclose(detection_not_00(uniform_probs), 0.75)
        assert jnp.isclose(detection_11(uniform_probs), 0.25)
        assert jnp.isclose(detection_qubit1(uniform_probs), 0.5)
        assert jnp.isclose(detection_qubit2(uniform_probs), 0.5)
        assert jnp.isclose(detection_any_excited(uniform_probs), 0.75)
        assert jnp.isclose(detection_both_excited(uniform_probs), 0.25)
        assert jnp.isclose(detection_xor(uniform_probs), 0.5)

    def test_all_criteria_with_extremes(self):
        """Test all criteria with extreme probability distributions."""
        # All in |00⟩
        all_00 = {"p00": 1.0, "p01": 0.0, "p10": 0.0, "p11": 0.0}
        assert jnp.isclose(detection_not_00(all_00), 0.0)
        assert jnp.isclose(detection_xor(all_00), 0.0)

        # All in |11⟩
        all_11 = {"p00": 0.0, "p01": 0.0, "p10": 0.0, "p11": 1.0}
        assert jnp.isclose(detection_not_00(all_11), 1.0)
        assert jnp.isclose(detection_11(all_11), 1.0)
        assert jnp.isclose(detection_xor(all_11), 0.0)

    def test_predefined_criteria_equivalences(self, sample_probs):
        """Test that equivalent criteria produce same results."""
        # detection_not_00 and detection_any_excited should be equivalent
        assert jnp.isclose(detection_not_00(sample_probs), detection_any_excited(sample_probs))

        # detection_11 and detection_both_excited should be equivalent
        assert jnp.isclose(detection_11(sample_probs), detection_both_excited(sample_probs))


@pytest.mark.skip(reason="DetectionFromProbabilities has been removed")
class TestContrastMetrics:
    """Test suite for contrast computation."""

    def test_contrast_symmetry(self):
        """Test that contrast is symmetric."""
        contrast1 = DetectionFromProbabilities.compute_contrast(0.7, 0.3)
        contrast2 = DetectionFromProbabilities.compute_contrast(0.3, 0.7)
        assert jnp.isclose(contrast1, contrast2)

    def test_contrast_bounds(self):
        """Test that contrast is always between 0 and 1."""
        test_cases = [
            (0.0, 0.0),
            (0.5, 0.5),
            (1.0, 1.0),  # Same values
            (0.0, 1.0),
            (1.0, 0.0),  # Maximum difference
            (0.3, 0.7),
            (0.2, 0.9),  # Random values
        ]

        for p_with, p_without in test_cases:
            contrast = DetectionFromProbabilities.compute_contrast(p_with, p_without)
            assert 0.0 <= contrast <= 1.0

    def test_contrast_linearity(self):
        """Test linear relationship of contrast."""
        # If we scale both probabilities by same factor, contrast should scale too
        p1, p2 = 0.8, 0.2
        contrast1 = DetectionFromProbabilities.compute_contrast(p1, p2)

        # Scale both by 0.5
        contrast2 = DetectionFromProbabilities.compute_contrast(p1 * 0.5, p2 * 0.5)

        # Contrast should also be scaled (since it's absolute difference)
        assert jnp.isclose(contrast1 * 0.5, contrast2)

    def test_contrast_with_jax_arrays(self):
        """Test contrast computation with JAX arrays."""
        p_with = jnp.array(0.8)
        p_without = jnp.array(0.2)
        contrast = DetectionFromProbabilities.compute_contrast(p_with, p_without)

        assert isinstance(contrast, jnp.ndarray)
        assert jnp.isclose(contrast, 0.6)


@pytest.mark.skip(reason="DetectionFromProbabilities has been removed")
class TestIntegrationScenarios:
    """Integration tests for realistic sensing scenarios."""

    def test_ideal_sensing_scenario(self):
        """Test scenario with perfect distinguishability."""
        # With photon: high probability of detection
        probs_with = {"p00": 0.0, "p01": 0.0, "p10": 0.0, "p11": 1.0}

        # Without photon: no detection
        probs_without = {"p00": 1.0, "p01": 0.0, "p10": 0.0, "p11": 0.0}

        detector = DetectionFromProbabilities()

        p_with = detector(probs_with)
        p_without = detector(probs_without)
        contrast = detector.compute_contrast(p_with, p_without)

        assert jnp.isclose(p_with, 1.0)
        assert jnp.isclose(p_without, 0.0)
        assert jnp.isclose(contrast, 1.0)

    def test_no_sensing_scenario(self):
        """Test scenario with no distinguishability."""
        # Same distribution with and without photon
        probs = {"p00": 0.4, "p01": 0.2, "p10": 0.2, "p11": 0.2}

        detector = DetectionFromProbabilities()

        p_with = detector(probs)
        p_without = detector(probs)
        contrast = detector.compute_contrast(p_with, p_without)

        assert jnp.isclose(contrast, 0.0)

    def test_realistic_sensing_scenario(self):
        """Test realistic scenario with partial distinguishability."""
        # With photon: some excitation
        probs_with = {"p00": 0.2, "p01": 0.3, "p10": 0.3, "p11": 0.2}

        # Without photon: mostly ground state
        probs_without = {"p00": 0.7, "p01": 0.1, "p10": 0.1, "p11": 0.1}

        detector = DetectionFromProbabilities()

        p_with = detector(probs_with)
        p_without = detector(probs_without)
        contrast = detector.compute_contrast(p_with, p_without)

        assert 0.0 < contrast < 1.0
        assert p_with > p_without

    def test_multiple_detection_criteria_comparison(self):
        """Test comparing different detection criteria on same data."""
        probs = {"p00": 0.1, "p01": 0.2, "p10": 0.3, "p11": 0.4}

        # Create detectors with different criteria
        detectors = [
            DetectionFromProbabilities(detection_not_00, "not_00"),
            DetectionFromProbabilities(detection_11, "only_11"),
            DetectionFromProbabilities(detection_qubit1, "q1_excited"),
            DetectionFromProbabilities(detection_qubit2, "q2_excited"),
        ]

        results = [det(probs) for det in detectors]

        # All should return valid probabilities
        for result in results:
            assert 0.0 <= result <= 1.0

        # Check specific values
        assert jnp.isclose(results[0], 0.9)  # not_00
        assert jnp.isclose(results[1], 0.4)  # only_11
        assert jnp.isclose(results[2], 0.7)  # q1_excited
        assert jnp.isclose(results[3], 0.6)  # q2_excited
