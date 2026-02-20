"""
Loss functions and detection probability definitions for quantum sensing experiments.

This module provides utilities for defining custom detection criteria
and computing contrast metrics from measurement probabilities.
"""

from typing import Callable, Dict, Union, List, Optional

import jax.numpy as jnp
from jax import jit


class DetectionMetric:
    """
    Define custom detection probability criteria from measurement outcome probabilities.

    This class allows flexible definition of what constitutes a "photon detected" event
    based on the final state probabilities of an n-qubit measurement. Common examples:
    - 1 - P(00...0): Any outcome except |00...0⟩
    - P(11...1): Only the |11...1⟩ outcome
    - P(00..1...0) + P(10...1...0) + ... + P(11...1...1): Specific qubit in state |1⟩

    All operations are JAX-compatible for gradient-based optimization.

    Parameters
    ----------
    detection_fn : Callable[[Dict[str, float]], float], optional
        Custom function that takes a dictionary with keys 'p00...0', 'p10...0', 'p01...0', ... , 'p11...1'
        and returns the detection probability. If None, defaults to 1 - P(00).
    name : str, optional
        Name describing the detection criterion (for logging/plotting).
    n_qubits : int, optional
        Number of qubits, default to 2

    Examples (for 2 qubits)
    --------
    >>> # Default: detect anything except |00⟩
    >>> detector = DetectionMetric()
    >>> probs = {'p00': 0.1, 'p01': 0.2, 'p10': 0.3, 'p11': 0.4}
    >>> detector(probs)
    0.9

    >>> # Custom: detect only |11⟩
    >>> def detect_11(probs):
    ...     return probs['p11']
    >>> detector = DetectionMetric(detect_11, name="P(11)")
    >>> detector(probs)
    0.4

    >>> # Custom: detect qubit 2 in |1⟩
    >>> def detect_q2(probs):
    ...     return probs['p01'] + probs['p11']
    >>> detector = DetectionMetric(detect_q2, name="P(q2=1)")
    >>> detector(probs)
    0.6
    """

    def __init__(
        self, metric: Optional[Callable[float, float]] = None, name: str = "any excited", \
            n_qubits: int=2, detection_param: Optional[Union[int, List[str], List[int]]] = None
    ):
        """Initialize the detection probability calculator."""

        self.detection, self.required_states = self.std_detection(name, n_qubits, detection_param)
        
        if metric is None:
            self.metric = self.std_metric
        else:
            self.metric = jit(metric)


    def __call__(self, probabilities: Dict[str, float]) -> float:
        """
        Compute detection probability from measurement outcome probabilities.

        Parameters
        ----------
        probabilities : Dict[str, float]
            Dictionary containing '00...0', '10...0', '01...0', ... , '11...1' keys with
            the respective measurement outcome probabilities.

        Returns
        -------
        float
            Detection probability according to the defined criterion.
        """
        return self.metric(self.detection(probabilities))
    
    def std_metric(x: float)-> float:
        return x

    def std_detection(self, name, n_qubits, detection_param):
        """Predefined detection criterias for common use cases:
        
            - any excited: detects if there is any excitation

            - set excitations: detects if there are more than a set number of excitations

            - qubit list: detects if one or more of the qubits in a list are excited
            
            - state list: detects states that belong to a list of states
        """
        if name == 'any excited':
            
            all_0_state = 'P_all0'
            @jit
            def any_excited(probs: Dict[str, float]) -> float:
                """Detect any outcome except the ground state |00...0⟩: 1 - P(0)."""
                return 1.0 - probs[all_0_state]
            return any_excited, []

        elif name == 'set excitations':
            
            if detection_param < n_qubits//2:
                states = [format(i, f'0{n_qubits}b') \
                    for i in range(2**n_qubits) \
                    if sum(list(map(int,format(i, f'0{n_qubits}b')))) < detection_param]
                @jit
                def set_excitations(probs: Dict[str, float]) -> float:
                    """Detect a set number of excitations or more."""
                    return 1-sum([probs[state] for state in states])

            elif detection_param >= n_qubits//2:
                states = [format(i, f'0{n_qubits}b') \
                    for i in range(2**n_qubits) \
                    if sum(list(map(int,format(i, f'0{n_qubits}b')))) >= detection_param]
                @jit
                def set_excitations(probs: Dict[str, float]) -> float:
                    """Detect a set number of excitations or more."""
                    return sum([probs[state] for state in states])
        
            return set_excitations, states

        elif name == 'qubit list':

            if not all([isinstance(state, int) for state in states]):
                raise ValueError(f"Qubit list detection expects detection_param to be a list of int qubit indexes")

            states = [format(i, f'0{n_qubits}b') \
                    for i in range(2**n_qubits) \
                    if any([list(map(int,format(i, f'0{n_qubits}b')))[j]==1 for j in detection_param]) \
                    ]
            @jit
            def set_qubit(probs: Dict[str, float]) -> float:
                f"""Detect any excitation of the following qubits: {detection_param}"""
                return sum([probs[state] for state in states])

            return set_qubits, states

        elif name == 'state list':

            if not all([isinstance(state, str) for state in detection_param]):
                raise ValueError(f"State list detection expects detection_param to be a list of string states")

            states = detection_param
            @jit
            def set_states(probs: Dict[str, float]) -> float:
                f"""Detect any of the following states: {detection_param}"""
                return sum([probs[state] for state in states])

            return set_states, states


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



