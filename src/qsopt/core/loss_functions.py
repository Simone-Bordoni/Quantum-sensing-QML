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
    metric : Callable[[float,float], float], optional
        Custom function that takes probabilities of detection with and without photon and derives a loss.
        If None, defaults to lambda x,y: -(x-y)
    n_qubits : int, optional
        Number of qubits, defaults to 2
    detection_criterion : str, optional
        Critirion of detection, each criterion uses differently detection_param:

            - "any excited" (default): detects if there is any excitation. Corresponds to
                Doesn't take any parameter, detection_param default None

            - "min excited": detects if there are more than a set number of excitations.
                Takes int number of excitation as parameter, detection_param default 2

            - "excited qubits": detects if one or more of the qubits in a list are excited
                Takes List[int] list of qubit indexes, detection_param default [0]

            - "custom states": detects states that belong to a list of states
                Takes List[str] list of state keys, detection_param default ['00...0']

    detection_param : Union[int, List[str], List[int]], optional
        Parameter for the detection criterion, defaults to None
    name : str, optional
        Name used for logging/plotting, defaults to "1 - P_all0"


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
        self, metric: Optional[Callable[[float,float], float]] = None, n_qubits: int=2, \
            detection_criterion: str = "any excited", detection_param: Optional[Union[int, List[str], List[int]]] = None,
            metric_name: Optional[str] = 'custom metric'
    ):
        """Initialize the detection probability calculator."""

        self.detection_states, self.detection_name = self.std_detection(detection_criterion, detection_param, n_qubits)

        if metric is None:
            self.metric = std_metric
            metric_name = 'contrast'
        else:
            self.metric = jit(metric)
        self.metric_name = metric_name


    def __call__(self, p_with_photon: float, p_without_photon: float) -> float:
        """
        Compute loss from detection probability.

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
        return self.metric(p_with_photon,p_without_photon)

    def std_detection(self, criterion, detection_param, n_qubits):
        """Predefined detection criterion for common use cases:

            - any excited: detects if there is any excitation.
                detection_param: None

            - min excited: detects if there are more than a set number of excitations
                detection_param: int, number of excitations

            - excited qubits: detects if one or more of the qubits in a list are excited
                detection_param: List[int], list of qubit indexes

            - custom states: detects states that belong to a list of states
                detection_param: List[str], list of state keys
        """
        if criterion == 'any excited': #DEFAULT, corresponds to 'min excited' with detection_param=1

            non_0_states = [format(i, f'0{n_qubits}b') for i in range(1,2**n_qubits)]
            return non_0_states, criterion

        elif criterion == 'min excited':

            if detection_param is None:
                detection_param = 1
            elif (not isinstance(detection_param,int) and (0 < detection_param < n_qubits)):
                raise ValueError(f"min excited detection expects detection_param to be an int between 1 and {n_qubits-1}.\n\
                                   Value given: {detection_param}\n\
                                   Type given: {type(detection_param)}")

            states = [format(i, f'0{n_qubits}b') \
                for i in range(2**n_qubits) \
                if sum(list(map(int,format(i, f'0{n_qubits}b')))) >= detection_param]

            name = f'min {detection_param} excited'

            return states, name

        elif criterion == 'excited qubits':

            if detection_param is None:
                detection_param = [0]
            elif not isinstance(detection_param,list):
                raise ValueError("excited qubits detection expects detection_param to be a list")
            elif not all((0 <= qubit < n_qubits) if isinstance(qubit, int) else False for qubit in detection_param):
                raise ValueError(f"excited qubits detection expects elements of detection_param to be int qubit indexes between 0 and {n_qubits-1}.\n   \
                                   The following elements were given: {detection_param}\n\
                                   There are elements of types: {set([type(i) for i in detection_param])}")

            states = [format(i, f'0{n_qubits}b') \
                    for i in range(2**n_qubits) \
                    if any([list(map(int,format(i, f'0{n_qubits}b')))[j]==1 for j in detection_param]) \
                    ]

            name = f'excited qubits: {detection_param}'

            return states, name

        elif criterion == 'custom states':
            if detection_param is None:
                detection_param = [format(0, f'0{n_qubits}b')]
            elif not all([isinstance(state, str) for state in detection_param]):
                raise ValueError(f"custom states detection expects detection_param to be a list of string states")

            all_possible_states = set([format(i, f'0{n_qubits}b') for i in range(n_qubits)])
            invalid_states = set(detection_param) - all_possible_states
            if len(invalid_states) != 0:
                raise ValueError(f"{len(invalid_states)} invalid states given: {invalid_states}")

            states = detection_param

            name = criterion

            return states, name

    def __repr__(self) -> str:
        """String representation of the detector."""
        return f"DetectionMetric:\n\
        criterion='{self.detection_name}'\n\
        metric='{self.metric_name}'"

@staticmethod
@jit
def std_metric(p_with_photon: float, p_without_photon: float)-> float:
    contrast = p_with_photon - p_without_photon
    return -contrast
