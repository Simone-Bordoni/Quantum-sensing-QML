"""
Tests for loss functions and detection probability definitions.

This module tests the DetectionFromProbabilities class and predefined
detection criteria for quantum sensing experiments.
"""

import jax.numpy as jnp
import pytest

from qsopt.core.loss_functions import (
    DetectionFromProbabilities,
    detection_11,
    detection_any_excited,
    detection_both_excited,
    detection_not_00,
    detection_qubit1,
    detection_qubit2,
    detection_xor,
)


class TestDetectionFromProbabilities:
    """Test suite for DetectionFromProbabilities class."""

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
