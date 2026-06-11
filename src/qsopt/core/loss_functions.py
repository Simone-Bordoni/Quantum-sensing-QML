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
import copy

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
    n_cavities : int
        Number of cavities in the system.
    n_fields : int
        Number of fields in the system.
    n_qubits : int
        Number of qubits in the system.
    config_names: List[str]
        List of configuration names for the states 
    metric : Callable[[float,float], float], optional
        Custom function that takes detection measures with and without photon and derives a metric value (loss).
        If None, defaults to std_metric (contrast: x - y).
    detection_criterion : str, optional
        Criterion of detection, each criterion uses differently detection_param:

            - 'any excited' (works only for 2-configuration systems): detects when there is any excitation.
                First configuration is assigned to the all-zeros state, the second configuration is assigned to all the other states.
                Takes bool parameter, detection_param default False, if True, the all-zeros state is assigned to the second configuration

            - 'num excited': different configurations are assigned to different numbers of excitations.
                Takes Dict[str, int] dictionary mapping configuration names to number of excitations as parameter,
                detection_param default None (cardinality of the config_name list is used)

            - 'control qubits': different configurations are assigned to specific lists of qubits.
                Majority vote is applied over the qubits to assign intermidiate states to one of the configurations.
                Takes Dict[str, List[int]] dictionary mapping configuration names to List[int] list of qubit indexes as parameter.
                detection_param default None (cardonality of the config_name list is used, each configuration is assigned one qubit)

            - 'custom states': different configurations are assigned to specific lists of states.
                Takes Dict[str, List[str]] dictionary mapping configuration names to List[str] list of state keys (e.g., ['00', '11']) as parameter.
                detection_param default None (each state counting up in binary is assigned to a configuration cardinally)

            - 'min fidelity': evolves the mixture of states, minimizes the fidelity between with/without photon
                Takes no parameter, detection_param default None

            - 'max trace distance': evolves the mixture of states, maximizes the trace distance between with/without photon
                Takes no parameter, detection_param default None
            
            - 'max computational distance': maximizes the orthogonality between interaction and 
                non-interaction measurements (on the computational basis) for all states
                Takes optional Tuple[float, float] (inverse_pow_coefficient, pow_exp), default (4, 2)
            
            - 'custom matrix distance': maximizes a custom metric between the with/without photon density matrices.
                Takes Callable[[Array, Array], float] a function that takes as input the two density matrices 
                and outputs a distance measure to be maximized.
                detection_param default None

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
            n_cavities: int, \
            n_fields: int, \
            n_qubits: int, \
            config_names: List[str], \
            detection_criterion: str = "num excited", \
            detection_param: Optional[Union[int, List[str], List[int], Tuple[Callable[[Array, Array], Array], float]]] = None, \
            metric: Optional[Callable[[float,float], float]] = None, \
            multiple_measurement_logic: Optional[Union[Aggregator[Array], Aggregator[list]]] = None, \
            config_aggregation_strength: float = -1.0, \
            name: Optional[str] = None, \
    ):
        """Initialize the detection metric."""

        self.n_qubits = n_qubits
        self.n_subsystems = n_cavities + n_fields + n_qubits
        self.config_names = config_names
        self.detection_criterion = detection_criterion

        # Exponent p of the power mean used to combine the per-pair metrics into a single
        # objective (see callable_detection): mean = ((1/N) * sum_i m_i^p)^(1/p). p < 0 weights
        # the worst-separated configuration pair most heavily (a soft minimum, the default),
        # which pushes the optimizer to separate every pair rather than only the easy ones.
        # p == 0 is degenerate (forbidden) and p >= 1 rewards already-good pairs (discouraged).
        if not isinstance(config_aggregation_strength, (int, float)):
            raise TypeError(f"config_aggregation_strength must be a float, usually non zero values between -2 and 1. Value given: {config_aggregation_strength}")
        if config_aggregation_strength >= 1.0:
            warnings.warn(f"config_aggregation_strength should be less than 1, otherwise the solutions will be clustered togheter.\
                           Value given: {config_aggregation_strength}.")
        elif config_aggregation_strength == 0.0:
            raise ValueError("config_aggregation_strength cannot be 0, if zero is desired, use a small value instead.")

        self.config_aggregation_strength = config_aggregation_strength

        if len(set(self.config_names)) != len(self.config_names):
            raise ValueError(f"config_names must have unique values. Value given: {self.config_names}")
        # All unordered pairs of configurations. Discrimination is scored pairwise (how well
        # each pair of configurations can be told apart) and then aggregated, so the general
        # multi-configuration problem reduces to a set of two-configuration comparisons.
        self.config_pairs = [
                    (config1, config2)
                    for i, config1 in enumerate(self.config_names[:-1])
                    for config2 in self.config_names[i + 1 :]]

        # Build the detection metric in two stages. `init_detection` resolves the criterion
        # and validation/aggregation logic now, but the actual jitted callable needs the
        # measurement projectors P_all, which only exist once the experiment has built its
        # operators. So `init_detection` returns a *builder* (`initialize`) that the
        # experiment calls later (via `self.initialize(P_all)`) to assign `callable_detection`.
        initialize, self.name = self.init_detection(detection_criterion, \
                                                        detection_param, \
                                                        metric, \
                                                        multiple_measurement_logic \
                                                        )
        
        self.initialize = types.MethodType(initialize, self)
        
        # Overwrite protocol name if provided
        if name is not None:
            self.name = name        


    def __call__(self, rho_dict: Dict[str, List[qt.Qobj]], epoch_fraction: float=1.0)\
                     -> Tuple[float, Tuple[Dict[str, float], float]]:
        """
        Compute loss from detection probability.

        Parameters
        ----------
        rho_lists : Dict[str, List[qt.Qobj]]
            Dictionary of density matrices from different simulations.
        epoch_fraction : Optional[float]
            Fraction of the epoch for which to compute the detection metric.

        Returns
        -------
        float
            Detection metric value computed according to the defined criterion and metric.
        tuple of dict and float
            detection measures for different simulations if applicable (dict), and validation metric (float).
        """

        metric, (detection_dict, validation) = self.callable_detection(rho_dict, epoch_fraction)
                
        return metric, (detection_dict, validation)
    
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
            - 'any excited' (works only for 2-configuration systems): detects when there is any excitation.
                First configuration is assigned to the all-zeros state, the second configuration is assigned to all the other states.
                Takes bool parameter, detection_param default False, if True, the all-zeros state is assigned to the second configuration

            - 'num excited': different configurations are assigned to different numbers of excitations.
                Takes Dict[str, int] dictionary mapping configuration names to number of excitations as parameter,
                detection_param default None (cardinality of the config_name list is used)

            - 'control qubits': different configurations are assigned to specific lists of qubits.
                Majority vote is applied over the qubits to assign intermidiate states to one of the configurations.
                Takes Dict[str, List[int]] dictionary mapping configuration names to List[int] list of qubit indexes as parameter.
                detection_param default None (cardonality of the config_name list is used, each configuration is assigned one qubit)

            - 'custom states': different configurations are assigned to specific lists of states.
                Takes Dict[str, List[str]] dictionary mapping configuration names to List[str] list of state keys (e.g., ['00', '11']) as parameter.
                detection_param default None (each configuration is assigned an equal number of states counting up in binary following configuration cardinality)

            - 'min fidelity': evolves the mixture of states, minimizes the fidelity between with/without photon
                Takes no parameter, detection_param default None

            - 'max trace distance': evolves the mixture of states, maximizes the trace distance between with/without photon
                Takes no parameter, detection_param default None
            
            - 'max computational distance': maximizes the orthogonality between interaction and 
                non-interaction measurements (on the computational basis) for all states
                Takes optional Tuple[float, float] (inverse_pow_coefficient, pow_exp), default (4, 2)
            
            - 'custom matrix distance': maximizes a custom metric between the with/without photon density matrices.
                Takes Callable[[Array, Array], float] a function that takes as input the two density matrices 
                and outputs a distance measure to be maximized.
                detection_param default None
        """
        state_detection = ['any excited', 'num excited', 'control qubits', 'control states', 'custom states']
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

            # detection_states maps each configuration name to the list of computational-basis
            # states (binary strings) assigned to it by the chosen criterion below.
            detection_states = {}

            # Default multi-measurement aggregation for the probability-based criteria: treat
            # each measurement's value y as a per-shot "miss" probability (1 - y) and accumulate
            # the running product, so the final post-aggregation 1 - prod_i(1 - y_i) is the
            # probability of detecting in at least one of the sequential measurements.
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

                if len(self.config_names) != 2:
                    raise ValueError(f"'any excited' detection criterion only works for 2-configuration systems. Value given: {len(self.config_names)} configurations.")
                
                if parameter is False or parameter is None:
                    detection_states[self.config_names[0]] = [format(0, f'0{self.n_qubits}b')]
                    detection_states[self.config_names[1]] = [format(i, f'0{self.n_qubits}b') for i in range(1,2**self.n_qubits)]
                else:
                    detection_states[self.config_names[1]] = [format(0, f'0{self.n_qubits}b')]
                    detection_states[self.config_names[0]] = [format(i, f'0{self.n_qubits}b') for i in range(1,2**self.n_qubits)]
                detection_name = "at least 1 excitation"

            elif criterion == 'num excited':

                # Assign to each configuration the states with a given total number of excited
                # qubits (Hamming weight). Default: configuration i <- states with i excitations.
                if parameter is None:
                    parameter = {config_name: i for i, config_name in enumerate(self.config_names)}
                elif not isinstance(parameter, dict)\
                     or not all(isinstance(v, int) for v in parameter.values())\
                     or not all(k in self.config_names for k in parameter.keys())\
                     or not all([0 <= v <= self.n_qubits for v in parameter.values()])\
                     or len(parameter) == 0:
                    raise ValueError(f"'num excited' detection expects detection_param to be a non empty dictionary mapping configuration names to numbers of excitations,\
                                     with int values between 0 and {self.n_qubits-1}. Value given: {parameter}")
                
                if set(parameter.keys()) != set(self.config_names):
                    raise ValueError(f"'num excited' detection expects detection_param to be a dictionary mapping ALL configuration names to numbers of excitations,\
                                     with int values between 0 and {self.n_qubits-1}. Value given: {parameter}")

                
                # For each configuration, collect every basis state whose bit-sum (number of
                # excited qubits) equals that configuration's target excitation count.
                detection_states = {config_name: [format(i, f'0{self.n_qubits}b') \
                    for i in range(2**self.n_qubits) \
                    if sum(list(map(int,format(i, f'0{self.n_qubits}b')))) == parameter[config_name]] \
                    for config_name in self.config_names}
                
                detection_name = f'excitation number'

            elif criterion in ['control qubits', 'qubit indexes']:
                
                if parameter is None:
                    parameter = {config_name: [i] for i, config_name in enumerate(self.config_names)}
                    
                if not isinstance(parameter, dict)\
                     or not all(isinstance(v, list) for v in parameter.values())\
                     or not all(isinstance(i, int) for v in parameter.values() for i in v)\
                     or not all([0 <= i < self.n_qubits for v in parameter.values() for i in v])\
                     or not all(k in self.config_names for k in parameter.keys())\
                     or len(parameter) == 0:
                    raise ValueError(f"'control qubits' detection expects detection_param to be a non empty dictionary mapping\
                                     configuration names to lists of qubit indexes, with ints between 0 and n_qubits-1.\n\
                                     Value given: {parameter}")
                
                if any(sorted(parameter[config_name]) != sorted(set(parameter[config_name])) for config_name in parameter):
                    warnings.warn("detection_param for 'control qubits' detection has non unique elements. The additional elements will be ignored.")
                    parameter = {config_name: list(set(states)) for config_name, states in parameter.items()}

                # Each configuration "owns" a set of control qubits. For every basis state, the
                # configuration whose control qubits are most excited (majority vote) claims it,
                # so intermediate states are routed to the closest configuration.
                for i in range(2**self.n_qubits):
                    state = format(i, f'0{self.n_qubits}b')
                    state_as_list = list(map(int, state))
                    config_votes = {config_name: sum(state_as_list[j] for j in parameter[config_name]) for config_name in self.config_names}
                    config_max_vote = max(config_votes, key=config_votes.get)
                    detection_states[config_max_vote] = detection_states.get(config_max_vote, []) + [state]
                    
                detection_name = f"control qubits"

            elif criterion in ['custom states', 'control states']:
                if parameter is None:
                    states_per_config = 2**self.n_qubits // len(self.config_names)
                    parameter = {name: [format(j, f'0{self.n_qubits}b') for j in range(i * states_per_config, (i + 1) * states_per_config)] for i, name in enumerate(self.config_names)} 

                all_states = [format(i, f'0{self.n_qubits}b') for i in range(2**self.n_qubits)]

                if not isinstance(parameter, dict)\
                     or not all(isinstance(v, list) for v in parameter.values())\
                     or not all(isinstance(state, str) for v in parameter.values() for state in v)\
                     or not all([state in all_states for v in parameter.values() for state in v])\
                     or not all(k in self.config_names for k in parameter.keys())\
                     or len(parameter) == 0:
                    raise ValueError(f"'custom states' detection expects detection_param to be a non empty dictionary mapping\
                                     configuration names to lists of valid qubit states, e.g. '000' or '101' for 3 qubits.\n\
                                     Value given: {parameter}")
                
                if any(sorted(parameter[config_name]) != sorted(set(parameter[config_name])) for config_name in parameter):
                    warnings.warn("detection_param for 'custom states' detection has non unique elements. The additional elements will be ignored.")
                    parameter = {config_name: list(set(states)) for config_name, states in parameter.items()}

                detection_states = {config_name: parameter[config_name] for config_name in self.config_names}
                detection_name = f"control states"
            
            
            def build_detection(self, p_all):
                """Return a builder that, given projectors, creates and assigns the
                jitted detection callable on this instance.

                This function is returned by the outer `build_detection` factory so
                the experiment can call it when projectors are available.
                """
                # For each configuration, sum the computational-basis projectors of the states
                # assigned to it -> one projector onto that configuration's measurement subspace.
                projectors = {config_name: sum([p_all[i] for i in range(2**self.n_qubits)\
                                                        if format(i, f'0{self.n_qubits}b') in states])
                                        for config_name, states in detection_states.items()}
                n_config_pairs = len(self.config_pairs)

                print(f"DEBUG loss_functions:\n\nDetection projectors built for states: {detection_states}\nProjectors: {projectors}")

                def callable_detection(rho_dict: Dict[str, List[qt.Qobj]], epoch_fraction: float)\
                     -> Tuple[float, Tuple[Dict[str, float], float]]:
                    
                    metric_tot = aggregate_init
                    detection_tot = {config_name: aggregate_init for config_name in self.config_names}

                    # Loop over the sequential measurements in the protocol, aggregating as we go.
                    for meas in range(len(rho_dict[self.config_names[0]])):

                        temp_metric = 0
                        # Probability that each configuration's own evolved state lands in its
                        # assigned subspace: P = Tr(Pi * rho * Pi).
                        detection_temp = {config_name: jnp.real((projector * rho_dict[config_name][meas] * projector).tr()) for config_name, projector in projectors.items()}

                        # Score each configuration pair, then combine via the power mean of
                        # exponent config_aggregation_strength (negative -> emphasise worst pair).
                        for config1, config2 in self.config_pairs:
                            temp_metric += pow(metric(detection_temp[config1], detection_temp[config2]), self.config_aggregation_strength)

                        temp_metric = (temp_metric/n_config_pairs) ** (1/self.config_aggregation_strength)

                        # Fold this measurement into the running aggregates.
                        metric_tot = measurement_aggregation(metric_tot, temp_metric)
                        detection_tot = {config_name: measurement_aggregation(detection_tot[config_name], detection_temp[config_name]) for config_name in self.config_names}

                    metric_tot = post_aggregation(metric_tot)
                    detection_tot = {config_name: post_aggregation(detection_tot[config_name]) for config_name in self.config_names}

                    return metric_tot, (detection_tot, metric_tot)

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

            # Default multi-measurement aggregation for the matrix-distance criteria: a
            # log-sum-exp over measurements (accumulate sum_i exp(y_i), then take the log). This
            # is a smooth maximum that rewards the single best-separating measurement while
            # remaining differentiable.
            if multi_measurement_logic is None:
                aggregate_init = 0
                measurement_aggregation = lambda x,y:  x + jnp.exp(y)
                post_aggregation = lambda x: jnp.log(x)
                custom_meas_aggr = False
           
            if criterion in ['min fidelity','max trace distance']:

                if metric is not None:
                    warnings.warn(f"'{criterion}' detection criterion does uses fidelity and doesn't allow for a custom metric.\n\
                                   The custom metric will be ignored. Value given: {metric}")
                if parameter is not None:
                    warnings.warn(f"'{criterion}' detection criterion doesn't take a detection_param.\n\
                                   The detection_param will be ignored. Value given: {parameter}")
                
                # These criteria compare the reduced qubit states directly, so we trace out the
                # cavity and field modes (the qubits occupy the last n_qubits subsystems).
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

                    
                    n_config_pairs = len(self.config_pairs)
                    detection_tot = {config_name: 0 for config_name in self.config_names}

                    def callable_detection(rho_dict: Dict[str, List[qt.Qobj]], epoch_fraction: float)\
                     -> Tuple[float, Tuple[Dict[str, float], float]]:
                        # Implementation for fidelity and trace distance
                        
                        metric_tot = aggregate_init

                        for meas in range(len(rho_dict[self.config_names[0]])):

                            temp_metric = 0

                            for config1, config2 in self.config_pairs:

                                temp_metric += pow(matrix_distance(rho_dict[config1][meas], rho_dict[config2][meas]), self.config_aggregation_strength)

                            temp_metric = (temp_metric/n_config_pairs) ** (1/self.config_aggregation_strength)   

                            metric_tot = measurement_aggregation(metric_tot, temp_metric)
                        
                        metric_tot = post_aggregation(metric_tot)
                        
                        return metric_tot, (detection_tot, metric_tot)

                    self.callable_detection = jax.jit(callable_detection)

            elif criterion == 'max computational distance':

                if metric is None:
                    tan_h = lambda f: 1-(jnp.tanh(7*(f-0.5))+1)/2
                    metric = lambda x,y,f: -4*x*y + tan_h(f)*(x**2 + y**2)
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
                elif not callable(parameter):
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
                    
                    n_config_pairs = len(self.config_pairs)
                    detection_tot = {config_name: 0 for config_name in self.config_names}

                    def callable_detection(rho_dict: Dict[str, List[qt.Qobj]], epoch_fraction: float)\
                     -> Tuple[float, Tuple[Dict[str, float], float]]:
                        # Implementation for computational distance
                        
                        metric_tot = aggregate_init
                        validation_tot = aggregate_init

                        for meas in range(len(rho_dict[self.config_names[0]])):

                            temp_metric = 0
                            temp_validation = 0
                            p = {config_name: jnp.array([jnp.real((projector * rho_dict[config_name][meas] * projector).tr()) for projector in p_all]) for config_name in self.config_names}
    
                            for config1, config2 in self.config_pairs:
                                temp_metric += pow(jnp.sum(metric(p[config1], p[config2], epoch_fraction)), self.config_aggregation_strength)
                                temp_validation += pow(jnp.sum(validation(p[config1], p[config2])), self.config_aggregation_strength)
                                
                            temp_metric = (temp_metric/n_config_pairs) ** (1/self.config_aggregation_strength)
                            temp_validation = (temp_validation/n_config_pairs) ** (1/self.config_aggregation_strength)

                            metric_tot = measurement_aggregation(metric_tot, temp_metric)
                            validation_tot = measurement_aggregation(validation_tot, temp_validation)
                            
                        metric_tot = post_aggregation(metric_tot)
                        validation_tot = post_aggregation(validation_tot)

                        return metric_tot, (detection_tot, validation_tot)

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
                elif not callable(parameter):
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

                    n_config_pairs = len(self.config_pairs)
                    detection_tot = {config_name: 0 for config_name in self.config_names}

                    def callable_detection(rho_dict: Dict[str, List[qt.Qobj]], epoch_fraction: float)\
                     -> Tuple[float, Tuple[Dict[str, float], float]]:
                        # Implementation for custom matrix distance

                        metric_tot = aggregate_init
                        validation_tot = aggregate_init

                        for meas in range(len(rho_dict[self.config_names[0]])):

                            temp_metric = 0
                            temp_validation = 0

                            for config1, config2 in self.config_pairs:
                                temp_metric += pow(metric(rho_dict[config1][meas], rho_dict[config2][meas], epoch_fraction), self.config_aggregation_strength)
                                temp_validation += pow(validation(rho_dict[config1][meas], rho_dict[config2][meas]), self.config_aggregation_strength)
                                
                            temp_metric = (temp_metric/n_config_pairs) ** (1/self.config_aggregation_strength)
                            temp_validation = (temp_validation/n_config_pairs) ** (1/self.config_aggregation_strength)

                            metric_tot = measurement_aggregation(metric_tot, temp_metric)
                            validation_tot = measurement_aggregation(validation_tot, temp_validation)

                        metric_tot = post_aggregation(metric_tot)
                        validation_tot = post_aggregation(validation_tot)

                        return metric_tot, (detection_tot, validation_tot)

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
    distance = abs(p1 - p2)
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

