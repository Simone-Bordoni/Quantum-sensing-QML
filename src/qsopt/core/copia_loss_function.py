"""
Loss functions and detection probability definitions for quantum sensing experiments.

This module provides utilities for defining custom detection criteria
and computing detection metrics from measurement probabilities.
"""

from typing import Callable, Dict, Union, List, Optional, Tuple, TypeAlias, TypeVar
import types

import jax
import numpy as np
import qutip as qt
from qutip.core.data.extract import extract
import jax.numpy as jnp
from jax import Array, jit
import qutip_jax
import random
import warnings

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
        Number of qubits in the system.
    metric : Callable[[float,float], float], optional
        Custom function that takes detection measures with and without photon and derives a metric value (loss).
        If None, defaults to std_metric (contrast: x - y).
    detection_criterion : str, optional
        Criterion of detection, each criterion uses differently detection_param:

            - 'any excited' (default): detects if there is any excitation.
                Takes no parameter, detection_param default None

            - 'min excited': detects if there are more than a set number of excitations.
                Takes int number of excitations as parameter, detection_param default 1

            - 'excited qubits': detects if one or more of the qubits in a list are excited
                Takes List[int] list of qubit indexes, detection_param default [0]

            - 'custom states': detects states that belong to a list of states
                Takes List[str] list of state keys (e.g., ['00', '11']), detection_param default None (all-zeros state)

            - 'min fidelity': evolves the mixture of states, minimizes the fidelity between with/without photon
                Takes no parameter, detection_param default None

            - 'max trace distance': evolves the mixture of states, maximizes the trace distance between with/without photon
                Takes no parameter, detection_param default None
            
            - 'max computational distance': maximizes the orthogonality between interaction and 
            non-interaction measurements (on the computational basis) for all states
                Takes optional Tuple[float, float] (inverse_pow_coefficient, pow_exp), default (4, 2)

    detection_param : Union[int, List[str], List[int], Tuple[Callable[[Array, Array], Array], float]], optional
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
    >>> # Default: detect anything except |00⟩ with contrast metric
    >>> detector = DetectionMetric(n_qubits=2)
    >>> # Simulate measurement: 90% excited state detected with photon, 30% without
    >>> contrast = detector(measure_with_photon=0.9, measure_without_photon=0.3)
    >>> contrast
    0.6

    >>> # Custom: detect only |11⟩ state
    >>> detector = DetectionMetric(n_qubits=2, detection_criterion='custom states', detection_param=['11'])
    >>> detector(measure_with_photon=0.4, measure_without_photon=0.1)
    0.3

    >>> # Minimize fidelity between with/without photon density matrices
    >>> detector = DetectionMetric(n_qubits=2, detection_criterion='min fidelity')
    """
 

    def __init__(
        self,  \
            n_qubits: int, \
            detection_criterion: str = "any excited", \
            detection_param: Optional[Union[int, List[str], List[int], Tuple[Callable[[Array, Array], Array], float]]] = None, \
            metric: Optional[Callable[[float,float], float]] = None, \
            multiple_measurement_logic: Optional[Union[Aggregator[Array], Aggregator[list]]] = None, \
            name: Optional[str] = None, \
    ):
        """Initialize the detection metric."""

        self.n_qubits = n_qubits
        self.n_subsystems = n_qubits + 2

        # create the detection metric initializer:
        # we need the projectors which are built in the experiment, so the callable 
        # is built in the experiment initialization and not here.
        initialize, self.name = self.init_detection(detection_criterion, \
                                                        detection_param, \
                                                        metric, \
                                                        multiple_measurement_logic \
                                                        )
        
        self.initialize = types.MethodType(initialize, self)
        
        # Overwrite protocol name if provided
        if name is not None:
            self.name = name        


    def __call__(self, list_rho_a: List[qt.Qobj], list_rho_b: List[qt.Qobj], epoch_fraction: float = 1) -> Tuple[float, Tuple[float, float]]:
        """
        Compute loss from detection probability.

        Parameters
        ----------
        list_rho_a : List[qt.Qobj]
            List of density matrices from simulation a.
        list_rho_b : List[qt.Qobj]
            List of density matrices from simulation b.
        epoch_fraction : Optional[float]
            Fraction of the epoch for which to compute the detection metric.

        Returns
        -------
        float
            Detection metric value computed according to the defined criterion and metric.
        tuple of floats
            (detection measure with photon, detection measure without photon) if applicable.
        """
        return self.callable_detection(list_rho_a, list_rho_b, epoch_fraction)

    def init_detection(self, criterion, parameter, metric, multi_measurement_logic \
                        ) -> Tuple[Callable[[List[qt.Qobj], List[qt.Qobj]], Tuple[float, Tuple[float, float]]], str]:
        """
        Build detection metric callable and return corresponding states and criterion name.

        Parameters
        ----------
        criterion : str
            Detection criterion type
        parameter : 
            Parameters to customize the criterion
        metric : callable
            Function to compute the detection metric
        measurement_logic : tuple
            Logic for aggregating measurements

        Returns
        -------
        callable
            Function that computes the detection metric from lists of density matrices.
        string
            detection_name describing the criterion and parameters used, for logging and visualization purposes.
        
        Criterion types:

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
            
            - 'max computational distance': maximizes the metric between measurements (on the computational basis) for all the states.
            Default metric is weighted squared distance.
                detection_param: None
            
            - 'custom matrix distance': maximizes a custom metric between matrices.
                detection_param: Callable[[Array, Array], float], a function that takes as input the two density matrices and outputs a distance measure to be maximized.
        
        """
        state_detection = ['any excited', 'min excited', 'control qubits', 'excited qubits', 'custom states']
        matrix_distance = ['min fidelity', 'max trace distance', 'max computational distance', 'custom matrix distance']

        number = (int, float, jax.Array)

        if multi_measurement_logic is not None:
            if not isinstance(multi_measurement_logic[0], number):
                raise ValueError(f"multiple_measurement_logic expects the first element of the tuple to be a float (the initialization value for the aggregation). Value given: {multi_measurement_logic[0]}")
            elif not callable(multi_measurement_logic[1]):
                raise ValueError(f"multiple_measurement_logic expects the second element of the tuple to be a callable (x,y)->z (the aggregation function for multiple measurements). Value given: {multi_measurement_logic[1]}")
            elif not callable(multi_measurement_logic[2]):
                raise ValueError(f"multiple_measurement_logic expects the third element of the tuple to be a callable (x)->z (the post-aggregation function for multiple measurements). Value given: {multi_measurement_logic[2]}")
            else:
                error_msg = f"multiple_measurement_logic contains invalid elements:\n\
                            2nd element: The aggregation function must be able to take as input two floats and outputs a float.\n\
                            3rd element: The post-aggregation function must be able to take as input one float and output a float."
                try:
                    for i in range(100):
                        test_agg = multi_measurement_logic[0]
                        for j in range(10):
                            test_agg = multi_measurement_logic[1](test_agg, random.random())
                        test_post_agg = multi_measurement_logic[2](test_agg)
                        if not isinstance(test_post_agg, number):
                            raise ValueError(error_msg + f"\nError from test:\naggregation function output: {test_agg}, post-aggregation function output: {test_post_agg}")
                except Exception as e:
                    raise ValueError(error_msg + f"\nError from test: {e}")
                
                aggregate_init = multi_measurement_logic[0]
                measurement_aggregation = multi_measurement_logic[1]
                post_aggregation = multi_measurement_logic[2]
                custom_meas_aggr = True

        if criterion in state_detection:
            
            if multi_measurement_logic is None:
                aggregate_init = jnp.array(1)
                measurement_aggregation = lambda x,y: x*(1-y)
                post_aggregation = lambda x: 1-x
                custom_meas_aggr = False
            
           
            if metric is None:
                metric = std_states_metric
                custom_metric = False
            elif not callable(metric):
                raise ValueError(f"metric expects a callable (x,y)->z. Value given: {metric}")
            else:
                error_msg = f"metric function must be able to take as input two floats and output a float."
                try:
                    for i in range(100):
                        test_metric = metric(random.random(), random.random())
                        if not isinstance(test_metric, number):
                            raise ValueError(error_msg + f"\nError from test:\nmetric output: {test_metric}")
                except Exception as e:
                    raise ValueError(error_msg + f"\nError from test: {e}")
                
                metric = jit(metric)
                custom_metric = True

            if criterion == 'any excited':

                detection_states = [format(i, f'0{self.n_qubits}b') for i in range(1,2**self.n_qubits)]
                detection_name = "at least 1 excitation"

                if parameter is not None:
                    warnings.warn(f"'any excited' detection criterion does not take any parameter. Value given: {parameter}. The parameter will be ignored.")
            
            elif criterion == 'min excited':

                if parameter is None:
                    raise ValueError("'min excited' detection expects detection_param to be a number of excitation, int between 1 and n_qubits-1. Value given: None")
                elif not isinstance(parameter, int) or not (0 < parameter < self.n_qubits):
                    raise ValueError(f"'min excited' detection expects detection_param to be a number of excitations, int between 1 and {self.n_qubits-1}. Value given: {parameter}")
                
                detection_states = [format(i, f'0{self.n_qubits}b') \
                    for i in range(2**self.n_qubits) \
                    if sum(list(map(int,format(i, f'0{self.n_qubits}b')))) >= parameter]
                detection_name = f'at least {parameter} excitation'

            elif criterion in ['control qubits', 'excited qubits']:

                if parameter is None:
                    raise ValueError(f"'control qubits' detection expects detection_param to be an List of unique qubit indexes, ints between 0 and n_qubits-1.\nValue given: None")
                elif not isinstance(parameter, list) or not all(isinstance(i, int) for i in parameter) or not all([0 <= i < self.n_qubits for i in parameter]):
                    raise ValueError(f"'control qubits' detection expects detection_param to be an List of unique qubit indexes, ints between 0 and n_qubits-1.\nValue given: {parameter}")
                elif parameter != list(set(parameter)):
                    warnings.warn("detection_param for 'control qubits' detection has non unique elements. The additional elements will be ignored.")
                    parameter = set(parameter)

                detection_states = [format(i, f'0{self.n_qubits}b') \
                    for i in range(2**self.n_qubits) if any([format(i, f'0{self.n_qubits}b')[j] == '1' for j in parameter])]
                detection_name = f"control qubits ({','.join(map(str, parameter))})"

            elif criterion == 'custom states':
                if parameter is None:
                    raise ValueError("custom states detection expects detection_param to be a list of str states to detect. Value given: None")
                elif not isinstance(parameter, list) or not all(isinstance(state, str) for state in parameter):
                    raise ValueError(f"custom states detection expects detection_param to be a list of str states to detect. Value given: {parameter}")
                elif not all(len(state) == self.n_qubits for state in parameter):
                    raise ValueError(f"custom states detection expects detection_param to be a list of str states of length equal to n_qubits={self.n_qubits}. Value given: {parameter}")
                elif not all(set(state) <= {'0','1'} for state in parameter):
                    raise ValueError(f"custom states detection expects detection_param to be a list of str states containing only '0' and '1' characters. Value given: {parameter}")
                elif len(parameter) == 0:
                    raise ValueError("custom states detection expects detection_param to be a non-empty list of str states to detect. Value given: empty list")
                elif parameter != list(set(parameter)):
                    warnings.warn("detection_param for 'custom states' detection has non unique elements. The additional elements will be ignored.")
                    parameter = set(parameter)

                detection_states = parameter
                detection_name = f"custom states: {parameter}"
            
            
            def build_detection(self, p_all):
                """Return a builder that, given projectors, creates and assigns the
                jitted detection callable on this instance.

                This function is returned by the outer `build_detection` factory so
                the experiment can call it when projectors are available.
                """

                detection_projectors = [p_all[i] for i in range(2**self.n_qubits)
                                        if format(i, f'0{self.n_qubits}b') in detection_states]
                
                print(f"DEBUG loss_functions:\n\nDetection projectors built for states: {detection_states}\nProjectors: {detection_projectors}")

                def callable_detection(list_rho_1: List[qt.Qobj], list_rho_2: List[qt.Qobj], epoch_fraction: float)\
                     -> Tuple[float, Tuple[float, float]]:
                    
                    detection1_tot = aggregate_init
                    detection2_tot = aggregate_init

                    for rho_1, rho_2 in zip(list_rho_1, list_rho_2):
                        p1 = sum([jnp.real((projector * rho_1 * projector).tr()) for projector in detection_projectors])
                        p2 = sum([jnp.real((projector * rho_2 * projector).tr()) for projector in detection_projectors])

                        detection1_tot = measurement_aggregation(detection1_tot, p1)
                        detection2_tot = measurement_aggregation(detection2_tot, p2)

                    detection1_tot = post_aggregation(detection1_tot)
                    detection2_tot = post_aggregation(detection2_tot)

                    metric_value = metric(detection1_tot, detection2_tot)

                    return metric_value, (detection1_tot, detection2_tot, metric_value)

                self.callable_detection = jax.jit(callable_detection)

            if custom_meas_aggr and not custom_metric:
                detection_name += f'\nwith custom measurement aggregation'
            elif custom_metric and not custom_meas_aggr:
                detection_name += f'\nwith custom metric'
            elif custom_metric and custom_meas_aggr:
                detection_name += f'\nwith custom metric and measurement aggregation'

            # return the projector-builder and the detection info so callers can
            # create/assign the real detection callable after projectors exist
            return build_detection, detection_name
                
        elif criterion in matrix_distance:

            if multi_measurement_logic is None:
                aggregate_init = 0
                measurement_aggregation = lambda x,y: x + y * jnp.sqrt(y * y + 1e-12) # = lambda x,y: x + y**3
                post_aggregation = lambda x: x
                custom_meas_aggr = False
           
            if criterion in ['min fidelity','max trace distance']:

                if metric is not None:
                    warnings.warn(f"'{criterion}' detection criterion does uses fidelity and doesn't allow for a custom metric.\n\
                                   The custom metric will be ignored. Value given: {metric}")
                if parameter is not None:
                    warnings.warn(f"'{criterion}' detection criterion doesn't take a detection_param.\n\
                                   The detection_param will be ignored. Value given: {parameter}")
                
                trace_subsys = range(self.n_subsystems - self.n_qubits, self.n_subsystems)

                if criterion == 'min fidelity':

                    matrix_distance = lambda rho_with, rho_without:\
                        1-fidelity(extract(rho_with.ptrace(trace_subsys).data, "JaxArray"),\
                                    extract(rho_without.ptrace(trace_subsys).data, "JaxArray"))
                    
                    detection_name = 'minimize fidelity'

                elif criterion == 'max trace distance':

                    matrix_distance = lambda rho_with, rho_without:\
                        trace_distance(extract(rho_with.ptrace(trace_subsys).data, "JaxArray"),\
                                        extract(rho_without.ptrace(trace_subsys).data, "JaxArray"))
                    
                    detection_name = 'maximize trace distance'

                def build_detection(self, p_all):
                    """Return a builder that, given projectors, creates and assigns the
                    jitted detection callable on this instance.

                    This function is returned by the outer `build_detection` factory so
                    the experiment can call it when projectors are available.
                    """

                    def callable_detection(list_rho_1: List[qt.Qobj], list_rho_2: List[qt.Qobj], epoch_fraction: float)\
                         -> Tuple[float, Tuple[float, float]]:
                        # Implementation for fidelity and trace distance

                        metric_tot = aggregate_init

                        for rho1,rho2 in zip(list_rho_1, list_rho_2):
                            metric_tot = measurement_aggregation(metric_tot, matrix_distance(rho1, rho2))
                        
                        metric_tot = post_aggregation(metric_tot)
                        
                        return metric_tot, (0,0, metric_value)

                    self.callable_detection = jax.jit(callable_detection)

            elif criterion == 'max computational distance':

                if metric is None:
                    metric = lambda x,y,f: -4*x*y + (1-f)*(x**2 + y**2)
                    custom_metric = False
                elif not callable(metric):
                    raise ValueError(f"metric expects a callable (x,y,f)->z. Where f is the epoch fraction.\n\
                                     Value given: {metric}")
                else:
                    error_msg = f"metric function must be able to take as input three floats and output a float.\n\
                        The three inputs are 1 for each compared matrix, the other is the epoch fraction\n\
                        and can be used to vary the metric over time, must be included even if it's not used."
                    try:
                        for i in range(100):
                            test_metric = metric(random.random(), random.random(), random.random())
                            if not isinstance(test_metric, number):
                                raise ValueError(error_msg + f"\nError from test:\nmetric output: {test_metric}")
                    except Exception as e:
                        raise ValueError(error_msg + f"\nError from test: {e}")
                    
                    custom_metric = True  

                metric = jit(metric)              

                if parameter is None:
                    parameter = lambda x,y: metric(x,y,1)
                elif not callable(metric):
                    raise ValueError(f"max computational distance detection expects detection_param to be a callable (x,y)->z (validation metric). Value given: {parameter}")
                else:
                    error_msg = f"max computational distance detection expects detection_param to be a callable (x,y)->z (validation metric).\n\
                        The function must be able to take as input two floats and output a float.\n\
                        The inputs are 1 for each compared matrix."
                    try:
                        for i in range(100):
                            test_validation = parameter(random.random(), random.random())
                            if not isinstance(test_validation, number):
                                raise ValueError(error_msg + f"\nError from test:\nvalidation output: {test_validation}")
                    except Exception as e:
                        raise ValueError(error_msg + f"\nError from test: {e}")
                    
                validation = jit(parameter)
                

                def build_detection(self, p_all):
                    """Return a builder that, given projectors, creates and assigns the
                    jitted detection callable on this instance.

                    This function is returned by the outer `build_detection` factory so
                    the experiment can call it when projectors are available.
                    """

                    def callable_detection(list_rho_1: List[qt.Qobj], list_rho_2: List[qt.Qobj], epoch_fraction: float)\
                     -> Tuple[float, Tuple[float, float]]:
                        # Implementation for computational distance
                        
                        metric_tot = aggregate_init
                        validation_tot = aggregate_init

                        for rho1,rho2 in zip(list_rho_1, list_rho_2):
                            p_with = jnp.array([jnp.real((projector * rho1 * projector).tr()) for projector in p_all])
                            p_without = jnp.array([jnp.real((projector * rho2 * projector).tr()) for projector in p_all])
                            metric_tot = measurement_aggregation(metric_tot, jnp.sum(metric(p_with, p_without, epoch_fraction)))
                            validation_tot = measurement_aggregation(validation_tot, jnp.sum(validation(p_with, p_without)))
                            
                        metric_tot = post_aggregation(metric_tot)
                        validation_tot = post_aggregation(validation_tot)

                        return metric_tot, (0,0, validation_tot)

                    self.callable_detection = jax.jit(callable_detection)

                detection_name = 'maximize computational distance'
                if custom_metric:
                    detection_name += ' with custom metric'

            elif criterion == 'custom matrix distance':
            
                if metric is None:
                    raise ValueError(f"'custom matrix distance' detection criterion expects a custom metric function\n\
                                     that takes as input two density matrices and the epoch fraction\n\
                                     then outputs a distance measure to be maximized.\n\
                                     Value given: None")
                elif not callable(metric):
                    raise ValueError(f"'custom matrix distance' detection criterion expects a callable custom metric\n\
                                     that takes as input two density matrices and the epoch fraction\n\
                                     then outputs a distance measure to be maximized.\n\
                                     Value given: {metric}")
                else:
                    try:
                        for i in range(100):
                            test_rho_1 = qt.rand_dm(self.n_qubits+2)
                            test_rho_2 = qt.rand_dm(self.n_qubits+2)
                            test_metric = metric(test_rho_1, test_rho_2, random.random())
                            if not isinstance(test_metric, number):
                                raise ValueError(f"Custom metric output must be a float. Output obtained during testing: {test_metric}")
                    except Exception as e:
                        raise ValueError(f"Custom metric must be a callable that takes as input two density matrices and the epoch fraction\n\
                                         then outputs a float.\n\
                                         Error from testing: {e}")                

                if parameter is None:
                    parameter = lambda x,y: metric(x,y,1)
                elif not callable(metric):
                    raise ValueError(f"custom matrix distance detection expects detection_param to be a callable validation metric\n\
                                     that takes as input two density matrices then outputs a distance measure to be maximized.\n\
                                     Value given: {metric}")
                else:
                    try:
                        for i in range(100):
                            test_rho_1 = qt.rand_dm(self.n_qubits+2)
                            test_rho_2 = qt.rand_dm(self.n_qubits+2)
                            test_validation = parameter(test_rho_1, test_rho_2)
                            if not isinstance(test_validation, number):
                                raise ValueError(f"detection_param output must be a float. Output obtained during testing: {test_validation}")
                    except Exception as e:
                        raise ValueError(f"custom matrix distance expects detection_param to be a callable\n\
                                         that takes as input two density matrices then outputs a float.\n\
                                         Error from testing: {e}")
                    
                validation = jit(parameter)

                def build_detection(self, p_all):
                    """Return a builder that, given projectors, creates and assigns the
                    jitted detection callable on this instance.

                    This function is returned by the outer `build_detection` factory so
                    the experiment can call it when projectors are available.
                    """

                    def callable_detection(list_rho_1: List[qt.Qobj], list_rho_2: List[qt.Qobj], epoch_fraction: float)\
                     -> Tuple[float, Tuple[float, float]]:
                        # Implementation for custom matrix distance

                        metric_tot = aggregate_init
                        validation_tot = aggregate_init

                        for rho1,rho2 in zip(list_rho_1, list_rho_2):
                            metric_tot = measurement_aggregation(metric_tot, metric(rho1, rho2, epoch_fraction))
                            validation_tot = measurement_aggregation(validation_tot, validation(rho1, rho2))

                        metric_tot = post_aggregation(metric_tot)
                        validation_tot = post_aggregation(validation_tot)

                        return metric_tot, (0,0, validation_tot)

                    self.callable_detection = jax.jit(callable_detection)

                detection_name = 'maximize custom matrix distance'

            if custom_meas_aggr:
                detection_name += f'\nwith custom measurement aggregation'

            # return the projector-builder and the detection info so callers can
            # create/assign the real detection callable after projectors exist
            return build_detection, detection_name



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
            - 'max computational distance': maximizes a distance between interaction and \n\
                non interaction measurements (on the computational basis) for all the states.\n\
                Can use a custom metric to redefine the distance\n\
                detection_param: None \n\n\
            - 'custom matrix distance': maximizes a custom metric between matrices.\n\
                detection_param: None"
            )

    def __repr__(self) -> str:
        """String representation of the detector."""
        return f"\nDetectionMetric:\n{self.name}\n"
    

# Support function for different metrics

@jit
def std_states_metric(p1: float, p2: float)-> float:
    distance = p1 - p2
    return distance

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
