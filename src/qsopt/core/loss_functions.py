"""
Loss functions and detection probability definitions for quantum sensing experiments.

This module provides utilities for defining custom detection criteria
and computing detection metrics from measurement probabilities.
"""

from typing import Callable, Dict, Union, List, Optional, Tuple, TypeAlias, TypeVar

import jax
import numpy as np
import qutip as qt
from qutip.core.data.extract import extract
import jax.numpy as jnp
from jax import Array, jit
import qutip_jax

T = TypeVar("T")
Aggregator: TypeAlias = Tuple[
    Optional[T],
    Optional[Callable[[T, T], T]],
    Optional[Callable[[T], T]],
]


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
    n_qubits : int
    metric : Callable[[float,float], float], optional
        Custom function that takes detection measures with and without photon and derives a metric value (loss).
        If None, defaults to lambda x,y: -(x-y)
        Number of qubits, defaults to 2
    detection_criterion : str, optional
        Critirion of detection, each criterion uses differently detection_param:

            - 'any excited' (default): detects if there is any excitation. Corresponds to
                Doesn't take any parameter, detection_param default None

            - 'min excited': detects if there are more than a set number of excitations.
                Takes int number of excitation as parameter, detection_param default 2

            - 'excited qubits': detects if one or more of the qubits in a list are excited
                Takes List[int] list of qubit indexes, detection_param default [0]

            - 'custom states': detects states that belong to a list of states
                Takes List[str] list of state keys, detection_param default ['00...0']

            - 'min fidelity': doesn't detect and evolves the mixture of states, minimizes the fidelity 
                Doesn't take any parameter, detection_param default None

            - 'max trace distance': doesn't detect and evolves the mixture of states, maximizes the trace distance
                Doesn't take any parameter, detection_param default None
            
            - 'max computational distance': doesn't detect and maximizes the distance between interaction and 
            non interaction measurements (on the computational basis) for all the states
                detection_param: Tuple[Callable[[array, array], array], float] distance function and hardness. default is squared Euclidean distance with hardness 0.9

    detection_param : Union[int, List[str], List[int], Tuple[Callable[[array, array], array], float]], optional
        Parameter for the detection criterion, defaults to None
    multiple_measurement_logic: Tuple[type,Callable[[type,type], type], optional
        Protocol that aggregates detection measures from multiple measurements. Contains an initialization value, an aggregator function and a post-aggregation function. 
        If None, defaults to (jnp.array(1),lambda x,y: x*y, lambda x: 1-x)
    batching_logic: Callable[...,Tuple[float]], optional
        Protocol that aggregates detection measures from different batches. Takes as input the list of detection measures for the batches and outputs the aggregated detection measure.
        If None, defaults to average over batch.
    protocol_name: str, optional
    metric_name : str, optional
    multiple_measurement_name: str, optional
    batching_name: str, optional


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
        self,  n_qubits: int, metric: Optional[Callable[[float,float], float]] = None, \
            detection_criterion: str = "any excited", detection_param: Optional[Union[int, List[str], List[int], Tuple[Callable[[array, array], array], float]]] = None, \
            multiple_measurement_logic: Optional[Union[Aggregator[Array], Aggregator[list]]] = None, \
            batching_logic: Optional[Callable[...,Tuple[float]]] = None, \
            protocol_name: Optional[str] = None, \
            metric_name: Optional[str] = 'custom metric', \
            multiple_measurement_name: Optional[str] = 'custom multiple measurement logic', \
            batching_name: Optional[str] = 'custom batching'
    ):
        """Initialize the detection metric protocol."""

        self.n_qubits = n_qubits
        self.detection_criterion = detection_criterion
        self.detection_param = detection_param

        # define the multiple measurement logic. default: (jnp.array(1),lambda x,y: x*y, lambda x: 1-x)
        if multiple_measurement_logic is None:
            self.custom_multiple_measurement_logic = False
            self.aggregate_init = jnp.array(1)
            self.measurement_aggregation = lambda x,y: x*y
            self.post_aggregation = lambda x: 1-x
            self.multiple_measurement_name = 'std probability aggregation'
        else:
            self.custom_multiple_measurement_logic = True
            self.aggregate_init = jnp.array(1) if multiple_measurement_logic[0] is None else multiple_measurement_logic[0]
            self.measurement_aggregation = lambda x,y: x*y if multiple_measurement_logic[1] is None else multiple_measurement_logic[1]
            self.post_aggregation = lambda x: 1-x if multiple_measurement_logic[2] is None else multiple_measurement_logic[2]
            self.multiple_measurement_name = multiple_measurement_name

        # define the metric function. default: contrast = lambda x,y: -(x-y)
        if metric is None:
            self.custom_metric = False
            self.metric = std_metric
            metric_name = 'contrast'
        else:
            self.custom_metric = True
            self.metric = jit(metric)

        self.metric_name = metric_name
        
        # define batching logic. default: average over batch
        if batching_logic is None:
            self.custom_batch = False
            self.batching_logic = std_batching
            self.batching_name = 'average batching'
        else:
            self.custom_batch = True
            self.batching_logic = batching_logic
            self.batching_name = batching_name

        # define detection condition
        self.detection_states, self.detection_name = self.build_detection(detection_criterion, detection_param)

        if protocol_name is not None:
            self.protocol_name = protocol_name
        else:
            self.protocol_name = self.detection_name + ' with ' + self.metric_name

        


    def __call__(self, measure_with_photon: float, measure_without_photon: float) -> float:
        """
        Compute loss from detection probability.

        Parameters
        ----------
        measure_with_photon : float
            Measurement outcome when the photon is present.
        measure_without_photon : float
            Measurement outcome when the photon is absent.

        Returns
        -------
        float
            Detection probability according to the defined criterion.
        """
        return self.metric(measure_with_photon, measure_without_photon)

    def build_detection(self, criterion, detection_param):
        """Possible criterions:

            - 'any excited': detects if there is any excitation.
                detection_param: None

            - 'min excited': detects if there are more than a set number of excitations
                detection_param: int, number of excitations

            - 'excited qubits': detects if one or more of the qubits in a list are excited
                detection_param: List[int], list of qubit indexes

            - 'custom states': detects states that belong to a list of states
                detection_param: List[str], list of state keys

            - 'min fidelity': doesn't detect and evolves the mixture of states, minimizes the fidelity
                detection_param: None

            - 'max trace distance': doesn't detect and evolves the mixture of states, maximizes the trace distance
                detection_param: None
            
            - 'max computational distance': maximizes the distance between interaction and 
            non interaction measurements (on the computational basis) for all the states
                detection_param: Tuple[Callable[[array, array], array], float] distance function and hardness. default is squared Euclidean distance with hardness 0.9
        
        """
        if criterion == 'any excited': #DEFAULT, corresponds to 'min excited' with detection_param=1

            non_0_states = [format(i, f'0{self.n_qubits}b') for i in range(1,2**self.n_qubits)]
            return non_0_states, criterion

        elif criterion == 'min excited':

            if detection_param is None:
                detection_param = 1
            elif (not isinstance(detection_param,int) and (0 < detection_param < self.n_qubits)):
                raise ValueError(f"min excited detection expects detection_param to be an int between 1 and {self.n_qubits-1}.\n\
                                   Value given: {detection_param}\n\
                                   Type given: {type(detection_param)}")

            states = [format(i, f'0{self.n_qubits}b') \
                for i in range(2**self.n_qubits) \
                if sum(list(map(int,format(i, f'0{self.n_qubits}b')))) >= detection_param]

            name = f'min {detection_param} excited'

            return states, name

        elif criterion == 'excited qubits':

            if detection_param is None:
                detection_param = [0]
            elif not isinstance(detection_param,list):
                raise ValueError("excited qubits detection expects detection_param to be a list")
            elif not all((0 <= qubit < self.n_qubits) if isinstance(qubit, int) else False for qubit in detection_param):
                raise ValueError(f"excited qubits detection expects elements of detection_param to be int qubit indexes between 0 and {self.n_qubits-1}.\n   \
                                   The following elements were given: {detection_param}\n\
                                   There are elements of types: {set([type(i) for i in detection_param])}")

            states = [format(i, f'0{self.n_qubits}b') \
                    for i in range(2**self.n_qubits) \
                    if any([list(map(int,format(i, f'0{self.n_qubits}b')))[j]==1 for j in detection_param]) \
                    ]

            name = f'excited qubits: {detection_param}'

            return states, name

        elif criterion == 'custom states':
            if detection_param is None:
                detection_param = [format(0, f'0{self.n_qubits}b')]
            elif not all([isinstance(state, str) for state in detection_param]):
                raise ValueError("custom states detection expects detection_param to be a list of string states")

            all_possible_states = set([format(i, f'0{self.n_qubits}b') for i in range(2**self.n_qubits)])
            invalid_states = set(detection_param) - all_possible_states
            if len(invalid_states) != 0:
                raise ValueError(f"{len(invalid_states)} invalid states given: {invalid_states}")

            states = detection_param

            name = criterion

            return states, name
        
        elif criterion == 'min fidelity':
            # this criterion must measure couples of rho from the interacting and non interacting simulations,
            # measurement_aggregation and aggregate_init are updated to handle probability lists
            if not self.custom_multiple_measurement_logic:
                self.aggregate_init = []
                self.measurement_aggregation = list_aggregation
                self.post_aggregation = lambda x: x
                self.multiple_measurement_name = 'list aggregation'

            # batching logic is updated to
            if not self.custom_batch:             
                self.batching_logic = fidelity_batching
                self.batching_name = 'fidelity batching'

            # metric name is updated
            if not self.custom_metric:      
                self.metric_name = 'fidelity'
                self.metric = lambda x,y: x-y

            return 'all states', 'no detection'


        elif criterion == 'max trace distance':
            # this criterion must measure couples of rho from the interacting and non interacting simulations,
            # measurement_aggregation and aggregate_init are updated to handle probability lists
            if not self.custom_multiple_measurement_logic:
                self.aggregate_init = []
                self.measurement_aggregation = list_aggregation
                self.post_aggregation = lambda x: x
                self.multiple_measurement_name = 'list aggregation'

            # batching logic is updated to   
            if not self.custom_batch:          
                self.batching_logic = trace_distance_batching
                self.batching_name = 'trace distance batching'

            # metric name is updated
            if not self.custom_metric:      
                self.metric_name = 'trace distance'

            return 'all states', 'no detection'
        
        elif criterion == 'max computational distance':
            # this criterion must measure separetly all states, 
            # compute the distance (defined by the metric) between the interacting and non interacting probabilities in each state and then average over states,
            # measurement_aggregation and aggregate_init are updated to handle probability lists
            if not self.custom_multiple_measurement_logic:
                self.aggregate_init = []
                self.measurement_aggregation = list_aggregation
                self.post_aggregation = lambda x: x
                self.multiple_measurement_name = 'list aggregation'

            if detection_param is None: # default distance is squared Euclidean and default hardness is 0.9
                distance_metric = lambda x, y: jnp.power(x - y, 2)
                hardness = 0.9 
            else:
                x,y = detection_param
                if x and not callable(x):
                    raise ValueError(
                        "Invalid detection_param for criterion 'max computational distance': expected a tuple with"
                        "first element a callable that takes two arrays and returns an array of the same shape."
                    )

                probe_with = jnp.asarray(np.random.rand(100))
                probe_without = jnp.asarray(np.random.rand(100))
                try:
                    probe_distance = x(probe_with, probe_without)
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        "Invalid detection_param for criterion 'max computational distance': the callable must accept "
                        "two arrays (interacting and non interacting detection measures) and return "
                        "an array of the same shape."
                    ) from exc

                if jnp.shape(probe_distance) != jnp.shape(probe_with):
                    raise ValueError(
                        "Invalid detection_param for criterion 'max computational distance':"
                        "the callable's return value must have the same shape as inputs."
                    )
                if y and not (isinstance(y, (int, float)) and 0 < y < 1):
                    raise ValueError(
                        "Invalid detection_param for criterion 'max computational distance': expected a tuple with"
                        "second element a float between 0 and 1 representing the hardness of the detection."
                    )
                if x is None:
                    distance_metric = lambda x, y: jnp.power(x - y, 2)
                elif y is None:
                    hardness = 0.9
                
            (distance_metric, hardness) = detection_param
            
            # batching logic is updated to             
            @staticmethod
            @jit
            def max_dist_batching(detect_with_batch: List[jnp.array],detect_without_batch: List[jnp.array])\
                -> Tuple[float, float, float]:
                detect_with = jnp.array(detect_with_batch)
                detect_without = jnp.array(detect_without_batch)
                # Shape: (batch_size, n_measurements, n_states)
                # Sum over states (axis=-1), then average over batch and measurements
                distance = distance_metric(detect_with, detect_without) - 2*hardness*distance_metric(detect_with, jnp.zeros_like(detect_with)) - 2*hardness*distance_metric(detect_without, jnp.zeros_like(detect_without))
                average_dist = jnp.mean(jnp.sum(distance, axis=-1))/2

                return average_dist, 0
            
            if not self.custom_batch:
                self.batching_logic = max_dist_batching
                self.batching_name = 'max computational distance batching'

            # metric name is updated
            if not self.custom_metric:      
                self.metric_name = 'computational distance'

            return 'all states', criterion

        else:
            raise ValueError(f"criterion was given the value '{criterion}'\n\
            criterion must be a string of the following:\n\
            - 'any excited': detects if there is any excitation.\n\
                detection_param: None\n\n\
            - 'min excited': detects if there are more than a set number of excitations\n\
                detection_param: int, number of excitations\n\n\
            - 'excited qubits': detects if one or more of the qubits in a list are excited\n\
                detection_param: List[int], list of qubit indexes\n\n\
            - 'custom states': detects states that belong to a list of states\n\
                detection_param: List[str], list of state keys\n\n\
            - 'min fidelity': computes the fidelity between the interacting and non interacting states \n\
                detection_param: None\n\n\
            - 'max trace distance': computes the trace distance between the interacting and non interacting states \n\
                detection_param: None\n\n\
            - 'max computational distance': maximizes the distance between interaction and \n\
                non interaction measurements (on the computational basis) for all the states \n\
                detection_param: Tuple[Callable[[array, array], array], float] distance function and hardness. default is squared Euclidean distance with hardness 0.9" \
            )

    def __repr__(self) -> str:
        """String representation of the detector."""
        return f"\nDetectionMetric: {self.protocol_name}\n\
detection criterion:\n\
  '{self.detection_name}'\n\
metric:\n\
  '{self.metric_name}'\n\
batching logic:\n\
  '{self.batching_name}'\n\
multiple measurement logic:\n\
  '{self.multiple_measurement_name}'\n"
    

# Support function for different metrics

@staticmethod
@jit
def std_metric(p_with_photon: float, p_without_photon: float)-> float:
    contrast = p_with_photon - p_without_photon
    return contrast

@staticmethod
@jit
def std_batching(detect_with_batch: List[float],detect_without_batch: List[float]):
    # Average over batch
    detect_with = jnp.mean(jnp.array(detect_with_batch))
    detect_without = jnp.mean(jnp.array(detect_without_batch))
    return detect_with, detect_without

@staticmethod
@jit
def fidelity_batching(detect_with_batch: List[qt.Qobj],detect_without_batch: List[qt.Qobj])\
    -> Tuple[float, float, float]:

    n_subsystems = len(detect_with_batch[0][0].dims[0])
    n = range(n_subsystems)
    detect_with = [item.ptrace(n[2:]) for sublist in detect_with_batch for item in sublist]
    detect_without = [item.ptrace(n[2:]) for sublist in detect_without_batch for item in sublist]

    fidelity_list = [fidelity(extract(rho_with.data, "JaxArray"), extract(rho_without.data, "JaxArray")) for rho_with, rho_without in zip(detect_with, detect_without)]

    total_fidelity = jnp.mean(jnp.array(fidelity_list))

    #The first output is 0 and the second one is the fidelity so that the fidelity is minimized
    return 1, total_fidelity

@staticmethod
@jit
def trace_distance_batching(detect_with_batch: List[qt.Qobj],detect_without_batch: List[qt.Qobj])\
    -> Tuple[float, float, float]:

    n_subsystems = len(detect_with_batch[0][0].dims[0])
    n = range(n_subsystems)
    detect_with = [item.ptrace(n[2:]) for sublist in detect_with_batch for item in sublist]
    detect_without = [item.ptrace(n[2:]) for sublist in detect_without_batch for item in sublist]
    
    trace_distance_list = [trace_distance(extract(rho_with.data, "JaxArray"), extract(rho_without.data, "JaxArray")) for rho_with, rho_without in zip(detect_with, detect_without)]
    
    total_trace_distance = jnp.mean(jnp.array(trace_distance_list))
    return total_trace_distance, 0

@staticmethod
@jit
def list_aggregation(tot: list, new: list)\
    -> list:
    return tot + new

@staticmethod
@jit
def trace_distance(rho, sigma):
    delta = rho - sigma
    # Singular values of delta
    s = jnp.linalg.eigvalsh(delta)
    return 0.5 * jnp.sum(jnp.abs(s))

@jit
def _hermitian_part(mat):
    return 0.5 * (mat + mat.conj().T)

@jit
def _trace_normalize_density(mat, eps=1e-12):
    mat_h = _hermitian_part(mat)
    trace_val = jnp.real(jnp.trace(mat_h))
    return mat_h / (trace_val + eps)

@staticmethod
@jit
def sqrtm_psd(mat):
    # Branchless PSD matrix square root via Newton-Schulz iterations.
    mat_h = _hermitian_part(mat)
    d = mat_h.shape[0]
    eye = jnp.eye(d, dtype=mat_h.dtype)
    eps = 1e-8

    # Small diagonal regularization keeps the iterate away from singular points.
    mat_h = mat_h + eps * eye
    scale = jnp.real(jnp.trace(mat_h)) + eps

    y = mat_h / scale
    z = eye

    def body(_, yz):
        y, z = yz
        t = 0.5 * (3.0 * eye - z @ y)
        return (y @ t, t @ z)

    y, _ = jax.lax.fori_loop(0, 20, body, (y, z))
    return y * jnp.sqrt(scale)

@staticmethod
@jit
def fidelity(rho, sigma):
    # Normalize inputs so numerical trace drift does not bias fidelity outside [0, 1].
    rho_n = _trace_normalize_density(rho)
    sigma_n = _trace_normalize_density(sigma)

    sqrt_rho = sqrtm_psd(rho_n)
    inner = _hermitian_part(sqrt_rho @ sigma_n @ sqrt_rho)
    sqrt_inner = sqrtm_psd(inner)
    trace_sqrt_inner = jnp.trace(sqrt_inner)

    return jnp.real(trace_sqrt_inner * jnp.conj(trace_sqrt_inner))

def is_valid_density_matrix(rho):
    rho_e = extract(rho.data, "JaxArray")
    rho = jnp.array(rho.full())
    print(f'difference extraction: {(rho-rho_e).max()}')
    hermitian = jnp.allclose(rho, rho.conj().T)
    eigvals = jnp.linalg.eigvalsh(rho)
    positive = jnp.all(eigvals >= -1e-10)  # tolerance
    trace_one = jnp.isclose(jnp.trace(rho), 1.0)
    return hermitian, positive, trace_one
