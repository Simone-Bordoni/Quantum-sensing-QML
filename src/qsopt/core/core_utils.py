"""
Shared math/function utilities for quantum sensing optimization.

Small reusable functions (annealing schedules, etc.) parameterized by the epoch
fraction f in [0, 1] (0 at the first step, ->1 at the last), plus the sweep-axis
helpers used by :meth:`Experiment.sweep` (kept out of the class so only the main
methods live in ``experiment.py``).
"""

from math import prod
from typing import Any, Dict, List, Optional, Tuple, Union
import jax
import jax.numpy as jnp
import numpy as np
import qutip as qt
from jax import Array
from jax.scipy.special import erfc

from qsopt.core.experiment.quantum_utils import PROMOTABLE_TYPES, build_hamiltonians


@jax.jit
def gaussian_modulation(t, **kwargs):
    """
    Time-dependent coupling function for input cavity transparency.

    Args:
        t: float or JAX array, time variable
        **kwargs: Dictionary containing 'sigma' parameter (pulse bandwidth)

    Returns:
        JAX array: Normalized coupling strength g(t)
    """
    sigma = kwargs.get("sigma", 0.1)
    dx = sigma * t
    coupling = jnp.sqrt(2 * sigma / jnp.sqrt(jnp.pi) * jnp.exp(-(dx**2)) / erfc(dx))
    return jnp.array(coupling, float)


def annealing_weight(
    epoch_fraction: Union[float, Array], v: float = 0.6, span: float = 99.0
) -> Union[float, Array]:
    """Smooth decreasing annealing weight g(f) in [0, 1], f = epoch_fraction.

    Logit-space tanh sigmoid, symmetric about f=0.5:
        f:   0 ------ 0.2 ---- 0.5 ---- 0.8 ------ 1
        g:   1        0.99     0.5      0.01       0

    g(f) = (tanh(-(k/2) ln(f/(1-f))) + 1) / 2,   k = ln(span) / ln((1+v)/(1-v))
      - v    : width of the transition band   (v=0.6 -> f in [0.2, 0.8])
      - span : odds g/(1-g) at the band edges (span=99 -> g = 0.99 / 0.01 there)

    Used to anneal the ODE-solver tolerance from loose (early) to tight (late).
    """
    f = jnp.clip(epoch_fraction, 1e-7, 1.0 - 1e-7)
    k = jnp.log(span) / jnp.log((1.0 + v) / (1.0 - v + 1e-16))
    a = -(k / 2.0) * jnp.log(f / (1.0 - f))
    return (jnp.tanh(a) + 1.0) / 2.0

def state_probs_to_dict(probs: np.ndarray, config_names: Optional[List[str]] = None) -> Dict[str, Dict[str, float]]:
    """
    Convert a (n_configs, n_states) probability array to the native per-config state-probability dict.

    Used both to build the state-probability dicts stored on the callback and to accept array input in
    :func:`make_confusion_matrix`.

    Args:
        - ``probs`` (np.ndarray): Array of shape (n_configs, n_states) of per-config state probabilities.
        - ``config_names`` (Optional[List[str]]): Row names; defaults to 'config_1', 'config_2', ... .

    Returns:
        - ``state_probabilities`` (Dict[str, Dict[str, float]]): Per config, a map of binary state
          string to probability.

    Raises:
        ValueError: If ``config_names`` is given but its length differs from the number of config rows.
    """

    n_configs, n_states = probs.shape
    if config_names is None:
        config_names = [f"config_{i + 1}" for i in range(n_configs)]
    elif len(config_names) != n_configs:
        raise ValueError(f"config_names has {len(config_names)} entries but probs has {n_configs} config rows.")

    # n_states = 2**n_qubits computational basis states -> binary string labels.
    n_qubits = n_states.bit_length() - 1
    state_labels = [format(i, f"0{n_qubits}b") for i in range(n_states)]

    return {name: dict(zip(state_labels, probs[c].tolist())) for c, name in enumerate(config_names)}


def make_confusion_matrix(
    state_probabilities: Union[Dict[str, Dict[str, float]], np.ndarray],
    config_names: Optional[List[str]] = None,
    states_map: Optional[Dict[str, List[str]]] = None,
) -> Tuple[Dict[Tuple[str, str], float], Dict[str, List[str]]]:
    """
    Build a soft confusion matrix (and a state->config map) from per-config state probabilities.

    Each basis state is classified to a configuration and the confusion cell (A, B) then accumulates
    config A's probability mass over the states classified as B. With no ``states_map`` the classifier
    is argmax over configs and the resulting map is returned; with a ``states_map`` that fixed map is
    used instead (no re-classification). States are binary strings, e.g. '000'..'111' for 3 qubits.

    Args:
        - ``state_probabilities`` (Union[Dict[str, Dict[str, float]], np.ndarray]): Either the native
          per-config map of binary state string to probability, or a (n_configs, n_states) array.
        - ``config_names`` (Optional[List[str]]): Names for the array rows; defaults to
          'config_1', 'config_2', ... . Ignored when a dict is passed.
        - ``states_map`` (Optional[Dict[str, List[str]]]): Fixed config -> owned states map to classify
          by; if omitted, states are classified by argmax and a fresh map is built.

    Returns:
        - ``confusion_matrix`` (Dict[Tuple[str, str], float]): Soft confusion matrix mapping
          (true, predicted) configuration pairs to the accumulated probability mass.
        - ``states_map`` (Dict[str, List[str]]): Config -> owned states map (the passed one, or the
          argmax map that was built).
    """

    # Accept a (n_configs, n_states) array; convert to the native dict and run one code path.
    if isinstance(state_probabilities, np.ndarray):
        state_probabilities = state_probs_to_dict(state_probabilities, config_names)

    config_names = list(state_probabilities.keys())

    # Classify by argmax (building a fresh map) unless a fixed map is supplied.
    build_map = states_map is None
    if build_map:
        states_map = {name: [] for name in config_names}
    else:
        # Invert the fixed map to look up each state's predicted config.
        state_to_config = {state: name for name, states in states_map.items() for state in states}

    # (true, predicted) -> accumulated probability mass.
    confusion_matrix = {(true, pred): 0.0 for true in config_names for pred in config_names}

    for state in state_probabilities[config_names[0]].keys():
        prob_state = {name: probs.get(state, 0.0) for name, probs in state_probabilities.items()}
        if build_map:
            predicted = max(prob_state, key=prob_state.get)
            states_map[predicted].append(state)
        else:
            predicted = state_to_config[state]
        # Add each config's probability for this state to its (config, predicted) cell.
        for name in config_names:
            confusion_matrix[(name, predicted)] += prob_state.get(name, 0.0)

    return confusion_matrix, states_map

def confusion_quality(confusion_matrix: Dict[Tuple[str, str], float]) -> float:
    """
    Score a confusion matrix by the geometric mean of its diagonal: ``(prod(c_ii))^(1/n)``.

    Penalises imbalance: a single poorly classified configuration (c_ii near 0) collapses the score,
    so it rewards matrices that discriminate every configuration well.

    Args:
        - ``confusion_matrix`` (Dict[Tuple[str, str], float]): Soft confusion matrix mapping
          (true, predicted) configuration pairs to the accumulated probability mass.

    Returns:
        - ``quality`` (float): Geometric mean of the diagonal (correct-classification) entries.
    """

    # Diagonal entries c_ii = correct-classification mass per configuration.
    diagonal = [value for (true, pred), value in confusion_matrix.items() if true == pred]

    return pow(prod(diagonal), 1 / len(diagonal))


# ---------------------------- generic N-dimensional sweep helpers ----------------------------


def sweep_key_types(exp) -> Dict[str, Any]:
    """Map every global-arg key of ``exp`` to its interaction type."""
    types: Dict[str, Any] = {}

    def add(interactions, prefix):
        for inter in interactions:
            params = inter.parameters if isinstance(inter.parameters, dict) else {}
            for p in params:
                types[f"{prefix}{inter._interaction_context()}__{p}"] = inter.interaction_type

    add(exp.experimental_params.interactions, "BaseModel_")
    for cfg in exp.experimental_params.configuration_set:
        add(cfg.interactions, f"Conf:{cfg.name}_")
    return types


def resolve_sweep_keys(exp, name: str) -> List[str]:
    """A full global-arg key, or a short parameter name matching every key ending in __<name>."""
    if name in exp.global_args:
        return [name]
    matches = [k for k in exp.global_args if k.endswith(f"__{name}")]
    if not matches:
        params = sorted({k.split("__")[-1] for k in exp.global_args})
        raise ValueError(f"Unknown sweep parameter {name!r}; available: {params} (or pass a full global-arg key)")
    return matches


def is_baked(exp, key: str) -> bool:
    """Check whether a parameter must be rebuilt to sweep, or only feeds a coefficient.

    Builds the operators (Hamiltonian and Lindblad) at two different values of ``key``, then
    evaluates both at the *same* probe value. If ``key`` is baked into a matrix the two builds
    differ even at identical args (rebuild needed); if it only feeds a coefficient both read the
    probe value and match (args-sweepable). Probing the Lindblad too catches a rate baked into a
    collapse operator but absent from H (e.g. kappa in ``L = sqrt(kappa)*a``).

    Args:
        exp (Experiment): the experiment providing the operators and parameters to build from.
        key (str): global-args key of the parameter to test.
    Returns:
        bool: True if a solver rebuild is needed to sweep it, False if it only feeds a coefficient.
    """
    # build at two values, but compare both evaluated at the SAME probe value (1.0)
    args = (exp.operators, exp.experimental_params, exp.n_cavities, exp.n_fields, exp.n_qubits)
    H1, L1, global_args = build_hamiltonians(*args, overrides={key: 1.0})
    H2, L2, _ = build_hamiltonians(*args, overrides={key: 2.0})
    probe_args = dict(global_args)
    probe_args[key] = 1.0

    def matrix(operator, time):
        # QobjEvo -> evaluate at the probe args; a plain Qobj is time-independent
        evaluated = operator if isinstance(operator, qt.Qobj) else operator(time, probe_args)
        return np.asarray(evaluated.full())

    # keys of H1/L1 are 'base' plus each configuration name
    # Hamiltonian per configuration, at two times to catch time-dependent coefficients
    for configuration in H1:
        for time in (0.0, 1.0):
            if not np.allclose(matrix(H1[configuration], time), matrix(H2[configuration], time)):
                return True
    # each configuration's Lindblad operators, pairwise
    for configuration in L1:
        for operator_1, operator_2 in zip(L1[configuration], L2[configuration]):
            for time in (0.0, 1.0):
                if not np.allclose(matrix(operator_1, time), matrix(operator_2, time)):
                    return True
    return False


def classify_sweep_axis(exp, name: str, key_types: Dict[str, Any]):
    """Return (lane, keys). Lanes: 'measurement' (time_interval), 'promote' (baked + promotable
    -> args-coefficient), 'rebuild' (baked + non-promotable), 'coeff' (already a coefficient)."""
    if name == "time_interval":
        return "measurement", []
    keys = resolve_sweep_keys(exp, name)
    if {key_types[k] for k in keys} <= PROMOTABLE_TYPES:
        return "promote", keys
    return ("rebuild" if any(is_baked(exp, k) for k in keys) else "coeff"), keys


def adaptive_map(fn, grid, verbose):
    """Run ``fn`` over ``grid``'s leading axis, starting fully parallel (batch_size = all) and
    halving the batch on GPU OOM down to sequential (batch_size = 1). No GPU -> just runs."""
    grid = jnp.asarray(grid)
    bs = int(grid.shape[0])
    while True:
        try:
            out = jax.lax.map(fn, grid, batch_size=bs)
            jax.block_until_ready(out)
            return out
        except Exception as e:
            msg = str(e).lower()
            if bs <= 1 or ("resource_exhausted" not in msg and "out of memory" not in msg):
                raise
            bs = max(1, bs // 2)
            if verbose:
                print(f"  GPU OOM -> retrying with batch_size={bs}")
