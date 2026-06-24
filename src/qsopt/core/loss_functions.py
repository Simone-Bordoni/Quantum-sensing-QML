"""
Loss functions and detection probability definitions for quantum sensing experiments.

This module provides utilities for defining custom detection criteria
and computing detection metrics from measurement probabilities.
"""

from typing import Callable, Dict, Union, List, Optional, Tuple, TypeAlias
import types

import jax
import numpy as np
import qutip as qt
from qutip.core.data.extract import extract
import jax.numpy as jnp
from jax import Array, jit
from jax.scipy.special import logsumexp
import qutip_jax
import random
import warnings
import copy

# 3-tuple protocol for aggregating a sequence of per-measurement values into a single scalar:
#   (initial_value,
#    aggregation_fn(acc, step_value, epoch_fraction) -> acc,
#    post_aggregation_fn(acc, epoch_fraction) -> result)
# epoch_fraction (0 at the first step, ->1 at the last) lets the aggregation anneal over training.
Aggregator: TypeAlias = Tuple[
    Optional[Union[float, Array]],
    Optional[Callable[[Union[float, Array], Union[float, Array], float], Union[float, Array]]],
    Optional[Callable[[Union[float, Array], float], Union[float, Array]]],
]

_AGG_EPS = 1e-7


def _anneal_beta(f):
    """Inverse temperature for soft-min/soft-max over measurements: ramps 0.1 -> 10 over training
    (soft ~mean early, hard ~max/min late). log10(beta) = 2*f**0.63 - 1."""
    return 10.0 ** (2.0 * f ** 0.63 - 1.0)


class MeasurementAggregator:
    """Combine per-measurement scores ``y_k`` with window weights ``w_k`` into one scalar.

    The fold accumulates ``S = Σ w_k·transform(y_k, f)`` and the result is ``post(S / normalizer, f)``.
    The caller (``DetectionMetric``, via its ``perturbation_type``) supplies the normalizer:
      - transient  -> normalizer = 1   (more in-window measurements raise the score; an event
                      localized in time only needs to be caught once).
      - persistent -> normalizer = Σw  (count-invariant detection *rate* for an always-present
                      perturbation).
    ``Σw`` is constant over an optimization, so it is summed once and passed in, not re-accumulated.
    A weight-0 measurement never changes the result (validated at build time, including for custom
    aggregators; see :meth:`validate`).

    transform(value, f) -> per-measurement contribution added to the weighted sum.
    post(arg, f)        -> final transform of the normalized accumulator.
    """

    # name -> (transform, post). Normalization (transient/persistent) is applied by the caller.
    _PRESETS = {
        "softmax": (lambda y, f: jnp.exp(_anneal_beta(f) * y),  lambda s, f: jnp.log(s) / _anneal_beta(f)),
        "softmin": (lambda y, f: jnp.exp(-_anneal_beta(f) * y), lambda s, f: -jnp.log(s) / _anneal_beta(f)),
        "average": (lambda y, f: y,                             lambda s, f: s),
        "OR":      (lambda y, f: jnp.log1p(_AGG_EPS - y),       lambda s, f: 1.0 - jnp.exp(s)),
        "AND":     (lambda y, f: jnp.log(y + _AGG_EPS),         lambda s, f: jnp.exp(s)),
    }

    def __init__(self, transform, post, init=0.0, name="custom"):
        self.transform = transform
        self.post = post
        self.init = init
        self.name = name

    @classmethod
    def from_preset(cls, preset):
        """Build a built-in aggregator by name: 'softmax', 'softmin', 'average', 'OR', 'AND'."""
        if preset not in cls._PRESETS:
            raise ValueError(f"Unknown aggregator preset {preset!r}; choose from {sorted(cls._PRESETS)}")
        transform, post = cls._PRESETS[preset]
        return cls(transform, post, name=preset)

    def init_acc(self):
        return self.init

    def step(self, acc, value, weight, epoch_fraction):
        return acc + weight * self.transform(value, epoch_fraction)

    def result(self, acc, normalizer, epoch_fraction):
        return self.post(acc / normalizer, epoch_fraction)

    def validate(self, n_trials=100):
        """Check over random configs that a weight-0 measurement (any value, incl. the y=1
        boundary) leaves the result unchanged and finite. Raises on failure or transform/post error."""
        for _ in range(n_trials):
            f = random.random()
            n = random.randint(2, 5)
            ys = [random.random() for _ in range(n)]
            ws = [random.random() for _ in range(n)]
            try:
                acc = self.init_acc()
                for y, w in zip(ys, ws):
                    acc = self.step(acc, y, w, f)
                wsum = sum(ws)
                r1 = self.result(acc, wsum, f)
                # appended weight-0 measurement: a random value and the y=1 boundary (catches 0*inf)
                extras = [(y_extra, self.result(self.step(acc, y_extra, 0.0, f), wsum, f))
                          for y_extra in (random.random(), 1.0)]
            except Exception as e:
                raise ValueError(f"Aggregator {self.name!r} raised an error while aggregating measurements: {e}") from e
            if not bool(jnp.isfinite(r1)):
                raise ValueError(
                    f"Aggregator {self.name!r} produced a non-finite result ({float(r1)}) on valid inputs.")
            for y_extra, r2 in extras:
                if not bool(jnp.allclose(r1, r2)):
                    raise ValueError(
                        f"Aggregator {self.name!r} does not ignore weight-0 measurements "
                        f"(adding a weight-0 measurement with value {y_extra:.3g} changed the result "
                        f"{float(r1):.6g} -> {float(r2):.6g}).")
        return self


def _resolve_aggregator(measurement_aggregator, default_preset):
    """None | preset name (str) | MeasurementAggregator -> a validated MeasurementAggregator."""
    if measurement_aggregator is None:
        agg = MeasurementAggregator.from_preset(default_preset)
    elif isinstance(measurement_aggregator, str):
        agg = MeasurementAggregator.from_preset(measurement_aggregator)
    elif isinstance(measurement_aggregator, MeasurementAggregator):
        agg = measurement_aggregator
    else:
        raise TypeError(f"measurement_aggregator must be None, a preset name, or a MeasurementAggregator; got {type(measurement_aggregator)}")
    return agg.validate()


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
    metric : Callable, optional
        Optional metric whose signature depends on detection_criterion:
        - state-detection criteria: a per-configuration transform metric(detection_probability, epoch_fraction)
          applied to each configuration's true-positive detection probability before the soft-min
          aggregation (default: identity, i.e. maximise every configuration's detection);
        - 'max computational distance' / 'custom matrix distance': a pairwise metric(x, y, epoch_fraction)
          (see detection_criterion);
        - 'min fidelity' / 'max trace distance': ignored.
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
    multiple_measurement_logic: Optional[Aggregator]
        Protocol that aggregates detection measures over the sequential measurements: a 3-tuple of
        (initialization value, aggregation function (acc, value, epoch_fraction)->acc,
        post-aggregation function (acc, epoch_fraction)->result). epoch_fraction lets the aggregation
        anneal over training. If None, defaults to an OR over measurements for the state-detection
        criteria and a temperature-annealed soft-max for the matrix-distance criteria.
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
            perturbation_type: str, \
            detection_criterion: str = "num excited", \
            detection_param: Optional[Union[int, List[str], List[int], Tuple[Callable[[Array, Array], Array], float]]] = None, \
            metric: Optional[Callable[Union[[float,float],[float,float,float]], float]] = None, \
            measurement_aggregator: Optional[Union[str, MeasurementAggregator]] = None, \
            config_aggregation_strength: float = 0.3, \
            name: Optional[str] = None, \
    ):
        """Initialize the detection metric."""

        if perturbation_type not in ("transient", "persistent"):
            raise ValueError(f"perturbation_type must be 'transient' or 'persistent', got {perturbation_type!r}")
        self.perturbation_type = perturbation_type

        self.n_qubits = n_qubits
        self.n_subsystems = n_cavities + n_fields + n_qubits
        self.config_names = config_names
        self.detection_criterion = detection_criterion

        # Pairs are combined with a log-sum-exp soft-minimum (see callable_detection), with inverse
        # temperature beta = config_aggregation_strength > 0: larger beta emphasises the worst-
        # separated pair (beta -> inf: hard min, beta -> 0: mean), and unlike a power mean it accepts
        # negative metrics. 0 is forbidden (degenerate mean / divide-by-zero); a negative value flips
        # it into a soft-maximum that separates only one configuration from the rest.
        if not isinstance(config_aggregation_strength, (int, float)):
            raise TypeError(f"config_aggregation_strength must be a positive float\nValue given: {config_aggregation_strength}")
        if config_aggregation_strength < 0:
            warnings.warn(f"config_aggregation_strength should be positive, otherwise only 1 configuration will be well separated from the others.\
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
                                                        measurement_aggregator \
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

    def init_detection(self, criterion, parameter, metric, measurement_aggregator \
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

        if criterion in state_detection:

            # Default aggregator 'OR' for state-detection criteria
            aggregator = _resolve_aggregator(measurement_aggregator, "OR")

            # Per-config transform applied to each configuration's true-positive probability.
            if metric is None:
                metric = std_states_metric
                custom_metric = False
            elif not callable(metric):
                raise ValueError(f"metric expects a callable (p, epoch_fraction)->z applied to each configuration's detection probability. Value given: {metric}")
            else:
                _validate_callable(
                    metric, lambda: (random.random(), random.random()),
                    "metric function must take a configuration's detection probability and the epoch fraction (two floats) and output a float.")
                metric = jit(metric)
                custom_metric = True

            # Resolve which computational-basis states each configuration owns, then build the
            # (diagonal) projector-based detection callable around them.
            detection_states, detection_name = self._resolve_state_detection_states(criterion, parameter)
            build_detection = _make_state_detection_builder(detection_states, metric, aggregator)

            detection_name += f'\n{aggregator.name} aggregation, {self.perturbation_type}'
            if custom_metric:
                detection_name += ', custom metric'

            # return the projector-builder and the detection info so callers can
            # create/assign the real detection callable after projectors exist
            return build_detection, detection_name

        elif criterion in matrix_distance:

            # default aggregator for matrix-distance criteria
            aggregator = _resolve_aggregator(measurement_aggregator, "softmax")

            if criterion in ['min fidelity', 'max trace distance']:

                if metric is not None:
                    warnings.warn(f"'{criterion}' detection criterion does uses fidelity and doesn't allow for a custom metric.\n\
                                   The custom metric will be ignored. Value given: {metric}")
                if parameter is not None:
                    warnings.warn(f"'{criterion}' detection criterion doesn't take a detection_param.\n\
                                   The detection_param will be ignored. Value given: {parameter}")

                # These criteria compare the reduced qubit states directly, so we trace out the cavity
                # and field modes (the qubits occupy the last n_qubits subsystems). The partial trace is
                # done once per configuration (in `prepare`) and reused across pairs.
                trace_subsys = range(self.n_subsystems - self.n_qubits, self.n_subsystems)
                make_prepare = lambda p_all: (lambda rho: extract(rho.ptrace(trace_subsys).data, "JaxArray"))

                if criterion == 'min fidelity':
                    pair_metric = lambda rho_with, rho_without, f: 1 - fidelity(rho_with, rho_without)
                    detection_name = 'minimize fidelity'
                else:
                    pair_metric = lambda rho_with, rho_without, f: trace_distance(rho_with, rho_without)
                    detection_name = 'maximize trace distance'

                # No separate validation metric: the reported validation equals the training metric.
                build_detection = _make_matrix_distance_builder(make_prepare, pair_metric, None, aggregator)

            elif criterion == 'max computational distance':

                if metric is None:
                    # Total-variation distance: 0.5*|x - y| per state (summed to 0.5*||x-y||_1).
                    # In [0,1], peakedness-invariant (0 for identical distributions), = the
                    # single-measurement distinguishability. f is accepted but unused.
                    metric = lambda x,y,f: 0.5*abs(x-y)
                    custom_metric = False
                elif not callable(metric):
                    raise ValueError(f"metric expects a callable (x,y,f)->z. Where f is the epoch fraction.\n\
                                     Value given: {metric}")
                else:
                    _validate_callable(
                        metric, lambda: (random.random(), random.random(), random.random()),
                        "metric function must be able to take as input three floats and output a float.\n\
                        The three inputs are 1 for each compared matrix, the other is the epoch fraction\n\
                        and can be used to vary the metric over time, must be included even if it's not used.")
                    custom_metric = True

                metric = jit(metric)

                if parameter is None:
                    parameter = lambda x,y: metric(x,y,1)
                elif not callable(parameter):
                    raise ValueError(f"max computational distance detection expects detection_param to be a callable (x,y)->z (validation metric). Value given: {parameter}")
                else:
                    _validate_callable(
                        parameter, lambda: (random.random(), random.random()),
                        "max computational distance detection expects detection_param to be a callable (x,y)->z (validation metric).\n\
                        The function must be able to take as input two floats and output a float.\n\
                        The inputs are 1 for each compared matrix.")
                validation = jit(parameter)

                # Each P_all[i] = I_cavities ⊗ I_fields ⊗ |q_i><q_i| is diagonal, so the outcome
                # probability Tr(P_all[i] * rho) = <diag(P_all[i]), diag(rho)>. Stack the projector
                # diagonals into one (n_outcomes, D) matrix so the whole probability vector is a single
                # matrix-vector product against diag(rho) (no partial trace, computed once per config).
                def make_prepare(p_all):
                    # .diag() returns the diagonal directly and is format-agnostic (P_all is
                    # stored as qutip's Dia format, which extract(..., "JaxArray") cannot handle).
                    # The projectors are Hermitian so their diagonals are real-valued, but qutip
                    # returns them as complex128; cast to real here, otherwise the probability
                    # vector (and hence the whole metric) is silently promoted to complex.
                    proj_diag_matrix = jnp.real(jnp.stack([jnp.asarray(projector.diag()) for projector in p_all]))
                    return lambda rho: proj_diag_matrix @ jnp.real(jnp.diag(extract(rho.data, "JaxArray")))

                pair_metric = lambda pa, pb, f: jnp.sum(metric(pa, pb, f))
                pair_validation = lambda pa, pb: jnp.sum(validation(pa, pb))
                build_detection = _make_matrix_distance_builder(make_prepare, pair_metric, pair_validation, aggregator)

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
                    _validate_callable(
                        metric, lambda: (qt.rand_dm(self.n_qubits + 2), qt.rand_dm(self.n_qubits + 2), random.random()),
                        "Custom metric must be a callable that takes as input two density matrices and the epoch fraction\n\
                                         then outputs a float.")

                if parameter is None:
                    parameter = lambda x,y: metric(x,y,1)
                elif not callable(parameter):
                    raise ValueError(f"custom matrix distance detection expects detection_param to be a callable validation metric\n\
                                     that takes as input two density matrices then outputs a distance measure to be maximized.\n\
                                     Value given: {metric}")
                else:
                    _validate_callable(
                        parameter, lambda: (qt.rand_dm(self.n_qubits + 2), qt.rand_dm(self.n_qubits + 2)),
                        "custom matrix distance expects detection_param to be a callable\n\
                                         that takes as input two density matrices then outputs a float.")
                validation = jit(parameter)

                # Custom metrics consume the density matrices directly, so prepare is the identity.
                make_prepare = lambda p_all: (lambda rho: rho)
                pair_metric = lambda rho1, rho2, f: metric(rho1, rho2, f)
                pair_validation = lambda rho1, rho2: validation(rho1, rho2)
                build_detection = _make_matrix_distance_builder(make_prepare, pair_metric, pair_validation, aggregator)

                detection_name = 'maximize custom matrix distance'

            detection_name += f'\n{aggregator.name} aggregation, {self.perturbation_type}'

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

    def _resolve_state_detection_states(self, criterion, parameter):
        """Map each configuration name to the computational-basis states assigned to it.

        Returns (detection_states, detection_name) for the state-detection criteria. The
        actual (diagonal) projectors are built later by _make_state_detection_builder, once
        the measurement projectors P_all are available.
        """
        # detection_states maps each configuration name to the list of computational-basis
        # states (binary strings) assigned to it by the chosen criterion below.
        detection_states = {}

        if criterion == 'any excited':

            if len(self.config_names) != 2:
                raise ValueError(f"'any excited' detection criterion only works for 2-configuration systems. Value given: {len(self.config_names)} configurations.")

            if parameter is False or parameter is None:
                detection_states[self.config_names[0]] = [format(0, f'0{self.n_qubits}b')]
                detection_states[self.config_names[1]] = [format(i, f'0{self.n_qubits}b') for i in range(1,2**self.n_qubits)]
            elif parameter is True:
                detection_states[self.config_names[1]] = [format(0, f'0{self.n_qubits}b')]
                detection_states[self.config_names[0]] = [format(i, f'0{self.n_qubits}b') for i in range(1,2**self.n_qubits)]
            else:
                raise ValueError(f"'any excited' detection expects detection_param to be a bool.\nValue given:{parameter}")
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
                 or sorted(parameter.keys()) != sorted(self.config_names)\
                 or sorted(set(parameter.values())) != sorted(parameter.values()):

                raise ValueError(f"'num excited' detection expects detection_param to be a non empty dictionary mapping all configuration names to unique numbers of excitations,\
                                 with int values between 0 and {self.n_qubits}. Value given: {parameter}")

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
                 or sorted(parameter.keys()) != sorted(self.config_names)\
                 or any([len(value) == 0 for value in parameter.values()]):
                raise ValueError(f"'control qubits' detection expects detection_param to be a dictionary mapping\
                                 all configuration names to non empty lists of qubit indexes, with ints between 0 and n_qubits-1.\n\
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

            if sorted(detection_states.keys()) != sorted(self.config_names):
                raise ValueError(f"'control qubits' detection got detection_param: {parameter}\n\
                    The following configurations weren't assigned any states because their reference qubits were all included in at least another configuration's reference qubits:\n\
                    {set(self.config_names)-set(detection_states.keys())}")

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
                 or sorted(parameter.keys()) != sorted(self.config_names)\
                 or any([len(value) == 0 for value in parameter.values()]):
                raise ValueError(f"'custom states' detection expects detection_param to be a dictionary mapping\
                                 configuration names to non empty lists of valid qubit states, e.g. '000' or '101' for 3 qubits.\n\
                                 Value given: {parameter}")

            if any(sorted(parameter[config_name]) != sorted(set(parameter[config_name])) for config_name in parameter):
                warnings.warn("detection_param for 'custom states' detection has non unique elements. The additional elements will be ignored.")
                parameter = {config_name: list(set(states)) for config_name, states in parameter.items()}

            # Unlike the partitioning criteria ('num excited'/'control qubits'), custom states are
            # user-supplied and may assign the same state to several configurations. This is allowed
            # (it can be intentional), but a shared state counts toward every configuration claiming
            # it and blurs their discrimination, so warn rather than silently accept it.
            assigned_states = [state for states in parameter.values() for state in states]
            overlapping = sorted({state for state in assigned_states if assigned_states.count(state) > 1})
            if overlapping:
                warnings.warn(
                    f"detection_param for 'custom states' detection assigns the same state(s) {overlapping} "
                    f"to multiple configurations. Overlapping detection regions are kept (this may be "
                    f"intentional) but reduce the discriminability between those configurations."
                )

            detection_states = {config_name: parameter[config_name] for config_name in self.config_names}
            detection_name = f"control states"

        return detection_states, detection_name

    def __repr__(self) -> str:
        """String representation of the detector."""
        return f"\nDetectionMetric:\n{self.name}\n"


# Detection-builder helpers (all run once at setup; nothing here is inside the jitted hot path)
number = (int, float, jax.Array)


def _validate_callable(fn, args_factory, error_msg):
    """Sanity-check a user-supplied callable on random inputs (runs once, at setup).

    Calls ``fn(*args_factory())`` 100 times and verifies the output is a scalar number,
    raising a descriptive ValueError otherwise. ``args_factory`` returns a fresh argument
    tuple on each call.
    """
    for _ in range(100):
        try:
            out = fn(*args_factory())
        except Exception as e:
            raise ValueError(f"{error_msg}\nError from test: {e}")
        if not isinstance(out, number):
            raise ValueError(f"{error_msg}\nError from test:\noutput: {out}")


def _resolve_measurement_aggregation(multi_measurement_logic, default):
    """Validate a multiple_measurement_logic 3-tuple, or fall back to ``default``.

    Returns (aggregate_init, measurement_aggregation, post_aggregation, is_custom), where
    is_custom is True iff a user-supplied tuple was used. The aggregation folds each
    measurement's value into a running accumulator, then post-processes it (see DetectionMetric).
    """
    if multi_measurement_logic is None:
        init, agg, post = default
        return init, agg, post, False

    if not isinstance(multi_measurement_logic[0], number):
        raise ValueError(f"multiple_measurement_logic expects the first element of the tuple to be a float (the initialization value for the aggregation). Value given: {multi_measurement_logic[0]}")
    if not callable(multi_measurement_logic[1]):
        raise ValueError(f"multiple_measurement_logic expects the second element of the tuple to be a callable (acc, value, epoch_fraction)->acc (the aggregation function for multiple measurements). Value given: {multi_measurement_logic[1]}")
    if not callable(multi_measurement_logic[2]):
        raise ValueError(f"multiple_measurement_logic expects the third element of the tuple to be a callable (acc, epoch_fraction)->result (the post-aggregation function for multiple measurements). Value given: {multi_measurement_logic[2]}")

    error_msg = f"multiple_measurement_logic contains invalid elements:\n\
                            2nd element: The aggregation function must take (accumulator, value, epoch_fraction) (three floats) and output a float.\n\
                            3rd element: The post-aggregation function must take (accumulator, epoch_fraction) (two floats) and output a float."
    try:
        for i in range(100):
            test_agg = multi_measurement_logic[0]
            for j in range(10):
                test_agg = multi_measurement_logic[1](test_agg, random.random(), random.random())
            test_post_agg = multi_measurement_logic[2](test_agg, random.random())
            if not isinstance(test_post_agg, number):
                raise ValueError(error_msg + f"\nError from test:\naggregation function output: {test_agg}, post-aggregation function output: {test_post_agg}")
    except Exception as e:
        raise ValueError(error_msg + f"\nError from test: {e}")

    return multi_measurement_logic[0], multi_measurement_logic[1], multi_measurement_logic[2], True


def _make_state_detection_builder(detection_states, metric, aggregator):
    """Return the projector-builder for the state-detection criteria.

    The returned ``build_detection(self, p_all)`` precomputes each configuration's diagonal
    projector (once, when projectors exist) and assigns the jitted ``callable_detection`` onto
    the instance. See DetectionMetric.init_detection for the two-stage build.
    """

    def build_detection(self, p_all):
        config_names = self.config_names

        # For each configuration, sum the single-state projectors P_all[i] of the basis states
        # assigned to it -> one projector onto that configuration's measurement subspace. Every
        # P_all[i] = I_cavities ⊗ I_fields ⊗ |q_i><q_i| is diagonal, so each config projector is
        # diagonal too: extract its diagonal once (length = full Hilbert dim) and reuse it every
        # call. Tr(P_config * rho * P_config) = Tr(P_config * rho) is then a single dot product
        # against diag(rho) — no partial trace, no matrix products.
        projectors = {name: sum([p_all[i] for i in range(2**self.n_qubits)
                                 if format(i, f'0{self.n_qubits}b') in states])
                      for name, states in detection_states.items()}
        # .diag() returns the diagonal directly and is format-agnostic (P_all is stored as
        # qutip's Dia format, which extract(..., "JaxArray") cannot handle). The projectors are
        # Hermitian so their diagonals are real-valued, but qutip returns them as complex128;
        # cast to real here, otherwise the detection probability (jnp.dot below) and hence the
        # whole metric is silently promoted to complex (float() on the result then raises).
        proj_diags = {name: jnp.real(jnp.asarray(projector.diag()))
                      for name, projector in projectors.items()}

        # Call-invariant values, hoisted out of the jitted callable_detection (run once here).
        n_configs = len(config_names)
        inv_beta = 1.0 / self.config_aggregation_strength
        neg_beta = -self.config_aggregation_strength
        log_n_configs = jnp.log(n_configs)
        config_names_0 = config_names[0]

        def callable_detection(rho_dict: Dict[str, List[qt.Qobj]], epoch_fraction: float,
                               weights: Array, normalizer: Union[float, Array])\
                -> Tuple[float, Tuple[Dict[str, float], float]]:

            metric_tot = aggregator.init_acc()
            detection_tot = {name: aggregator.init_acc() for name in config_names}

            # Loop over the sequential measurements in the protocol, aggregating as we go.
            for meas in range(len(rho_dict[config_names_0])):

                # Tr(P_config * rho) = <diag(P_config), diag(rho)> since P_config is diagonal.
                detection_temp = {name: jnp.dot(proj_diags[name], jnp.real(jnp.diag(extract(rho_dict[name][meas].data, "JaxArray"))))
                                  for name in config_names}

                # Apply the per-config metric to each true-positive probability, then combine with
                # the soft-min so the optimizer lifts EVERY configuration's detection (emphasising
                # the worst-detected one).
                config_metrics = jnp.stack([metric(detection_temp[name], epoch_fraction) for name in config_names])
                temp_metric = -inv_beta * (logsumexp(neg_beta * config_metrics) - log_n_configs)

                # Fold this measurement into the running aggregates, weighted by its window weight.
                w = weights[meas]
                metric_tot = aggregator.step(metric_tot, temp_metric, w, epoch_fraction)
                detection_tot = {name: aggregator.step(detection_tot[name], detection_temp[name], w, epoch_fraction) for name in config_names}

            metric_tot = aggregator.result(metric_tot, normalizer, epoch_fraction)
            detection_tot = {name: aggregator.result(detection_tot[name], normalizer, epoch_fraction) for name in config_names}

            return metric_tot, (detection_tot, metric_tot)

        self.callable_detection = jax.jit(callable_detection)

    return build_detection


def _make_matrix_distance_builder(make_prepare, pair_metric, pair_validation, aggregator):
    """Return the projector-builder shared by all matrix-distance criteria.

    Each criterion follows the same per-measurement shape: every configuration's density
    matrix is mapped through ``prepare`` (once per config — e.g. a partial trace or a
    population vector), every configuration *pair* is scored with
    ``pair_metric(prepared_a, prepared_b, epoch_fraction)``, the pair scores are combined with
    the log-sum-exp soft-min, and the per-measurement results are aggregated.

    Parameters
    ----------
    make_prepare : Callable[[list], Callable]
        Given the measurement projectors ``p_all``, returns the per-config ``prepare`` (which
        may depend on them, e.g. for the computational-distance population vector).
    pair_metric : Callable[[X, X, float], float]
        Training score for a configuration pair, on the prepared representations.
    pair_validation : Optional[Callable[[X, X], float]]
        Separate validation score. If None, no extra computation is done and the reported
        validation equals the training metric.
    aggregator : MeasurementAggregator
        Combines the window-weighted per-measurement scores into the final metric.
    """
    separate_validation = pair_validation is not None

    def build_detection(self, p_all):
        # Call-invariant values, hoisted out of the jitted callable_detection (run once here).
        config_names = self.config_names
        config_pairs = self.config_pairs
        config_names_0 = config_names[0]
        n_config_pairs = len(config_pairs)
        inv_beta = 1.0 / self.config_aggregation_strength
        neg_beta = -self.config_aggregation_strength
        log_n_config_pairs = jnp.log(n_config_pairs)
        detection_tot = {name: 0 for name in config_names}
        prepare = make_prepare(p_all)

        def callable_detection(rho_dict: Dict[str, List[qt.Qobj]], epoch_fraction: float)\
                -> Tuple[float, Tuple[Dict[str, float], float]]:

            metric_tot = aggregate_init
            validation_tot = aggregate_init

            for meas in range(len(rho_dict[config_names_0])):

                # Per-config preprocessing, computed once and reused across every pair.
                prepared = {name: prepare(rho_dict[name][meas]) for name in config_names}

                metric_pairs = jnp.stack([pair_metric(prepared[config1], prepared[config2], epoch_fraction)
                                          for config1, config2 in config_pairs])
                temp_metric = -inv_beta * (logsumexp(neg_beta * metric_pairs) - log_n_config_pairs)
                metric_tot = measurement_aggregation(metric_tot, temp_metric, epoch_fraction)

                if separate_validation:
                    # Validation is aggregated at f=1.0 (the fully hardened soft-max) regardless of
                    # the training epoch, so the reported validation is comparable across epochs.
                    validation_pairs = jnp.stack([pair_validation(prepared[config1], prepared[config2])
                                                  for config1, config2 in config_pairs])
                    temp_validation = -inv_beta * (logsumexp(neg_beta * validation_pairs) - log_n_config_pairs)
                    validation_tot = measurement_aggregation(validation_tot, temp_validation, 1.0)

            metric_tot = post_aggregation(metric_tot, epoch_fraction)
            validation_tot = post_aggregation(validation_tot, 1.0) if separate_validation else metric_tot

            return metric_tot, (detection_tot, validation_tot)

        self.callable_detection = jax.jit(callable_detection)

    return build_detection


# Support function for different metrics

@jit
def std_states_metric(p: float, epoch_fraction: float) -> float:
    # Default per-config transform: maximise each configuration's true-positive detection
    # probability directly (identity); epoch_fraction is accepted for signature consistency.
    return p

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
