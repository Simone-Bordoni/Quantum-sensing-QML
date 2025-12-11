"""
Loss functions and detection probability definitions for quantum sensing experiments.

This module provides utilities for defining custom detection criteria
and computing contrast metrics from measurement probabilities.
"""

from typing import Callable, Dict

import jax.numpy as jnp
from jax import jit


class DetectionFromProbabilities:
    """
    Define custom detection probability criteria from measurement outcome probabilities.

    This class allows flexible definition of what constitutes a "photon detected" event
    based on the final state probabilities of a two-qubit measurement. Common examples:
    - 1 - P(00): Any outcome except |00⟩
    - P(11): Only the |11⟩ outcome
    - P(01) + P(11): Qubit 2 in state |1⟩
    - P(10) + P(11): Qubit 1 in state |1⟩

    All operations are JAX-compatible for gradient-based optimization.

    Parameters
    ----------
    detection_fn : Callable[[Dict[str, float]], float], optional
        Custom function that takes a dictionary with keys 'p00', 'p01', 'p10', 'p11'
        and returns the detection probability. If None, defaults to 1 - P(00).
    name : str, optional
        Name describing the detection criterion (for logging/plotting).

    Examples
    --------
    >>> # Default: detect anything except |00⟩
    >>> detector = DetectionFromProbabilities()
    >>> probs = {'p00': 0.1, 'p01': 0.2, 'p10': 0.3, 'p11': 0.4}
    >>> detector(probs)
    0.9

    >>> # Custom: detect only |11⟩
    >>> def detect_11(probs):
    ...     return probs['p11']
    >>> detector = DetectionFromProbabilities(detect_11, name="P(11)")
    >>> detector(probs)
    0.4

    >>> # Custom: detect qubit 2 in |1⟩
    >>> def detect_q2(probs):
    ...     return probs['p01'] + probs['p11']
    >>> detector = DetectionFromProbabilities(detect_q2, name="P(q2=1)")
    >>> detector(probs)
    0.6
    """

    def __init__(
        self, detection_fn: Callable[[Dict[str, float]], float] = None, name: str = "1-P(00)"
    ):
        """Initialize the detection probability calculator."""
        if detection_fn is None:
            # Default: detect anything except |00⟩
            @jit
            def default_detection(probs):
                return 1.0 - probs["p00"]

            self.detection_fn = default_detection
        else:
            self.detection_fn = jit(detection_fn)

        self.name = name

    def __call__(self, probabilities: Dict[str, float]) -> float:
        """
        Compute detection probability from measurement outcome probabilities.

        Parameters
        ----------
        probabilities : Dict[str, float]
            Dictionary containing 'p00', 'p01', 'p10', 'p11' keys with
            the respective measurement outcome probabilities.

        Returns
        -------
        float
            Detection probability according to the defined criterion.
        """
        return self.detection_fn(probabilities)

    @staticmethod
    @jit
    def compute_contrast(p_with_photon: float, p_without_photon: float) -> float:
        """
        Compute sensing contrast from detection probabilities.

        The contrast quantifies how well we can distinguish between the presence
        and absence of a photon in the input cavity.

        Parameters
        ----------
        p_with_photon : float
            Detection probability when input photon is present.
        p_without_photon : float
            Detection probability when input photon is absent.

        Returns
        -------
        float
            Contrast value, defined as |P(detect|with) - P(detect|without)|.

        Notes
        -----
        The contrast ranges from 0 (no distinguishability) to 1 (perfect distinguishability).
        """
        return jnp.abs(p_with_photon - p_without_photon)

    def __repr__(self) -> str:
        """String representation of the detector."""
        return f"DetectionFromProbabilities(criterion='{self.name}')"


# Predefined detection criteria for common use cases
def detection_not_00(probs: Dict[str, float]) -> float:
    """Detect any outcome except |00⟩: 1 - P(00)."""
    return 1.0 - probs["p00"]


def detection_11(probs: Dict[str, float]) -> float:
    """Detect only |11⟩ outcome: P(11)."""
    return probs["p11"]


def detection_qubit1(probs: Dict[str, float]) -> float:
    """Detect qubit 1 in |1⟩: P(10) + P(11)."""
    return probs["p10"] + probs["p11"]


def detection_qubit2(probs: Dict[str, float]) -> float:
    """Detect qubit 2 in |1⟩: P(01) + P(11)."""
    return probs["p01"] + probs["p11"]


def detection_any_excited(probs: Dict[str, float]) -> float:
    """Detect at least one qubit excited: P(01) + P(10) + P(11) = 1 - P(00)."""
    return 1.0 - probs["p00"]


def detection_both_excited(probs: Dict[str, float]) -> float:
    """Detect both qubits excited: P(11)."""
    return probs["p11"]


def detection_xor(probs: Dict[str, float]) -> float:
    """Detect exactly one qubit excited (XOR): P(01) + P(10)."""
    return probs["p01"] + probs["p10"]
