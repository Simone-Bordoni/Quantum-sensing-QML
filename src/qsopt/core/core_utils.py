"""
Shared math/function utilities for quantum sensing optimization.

Small reusable functions (annealing schedules, etc.) parameterized by the epoch
fraction f in [0, 1] (0 at the first step, ->1 at the last), plus the sweep-axis
helpers used by :meth:`Experiment.sweep` (kept out of the class so only the main
methods live in ``experiment.py``).
"""

import itertools
import warnings
from math import prod
from typing import Any, Dict, List, Optional, Tuple, Union
import jax
import jax.numpy as jnp
import numpy as np
import qutip as qt
from jax import Array
from jax.scipy.special import erfc

from qsopt.core.experiment.quantum_utils import PROMOTABLE_TYPES, _build_hamiltonian_term, build_hamiltonians


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


# ---------------- branching detection-map derivation (joint, correlation-aware) ----------------
# These consume the per-config leaf dicts from Experiment.branching_simulation ({path: path prob})
# and pick the single deployable state->config map. The objective is joint over the measurement train,
# unlike make_confusion_matrix's per-measurement marginals: transient classifies each whole train (with a
# 'mixed' category), persistent averages per-measurement rates. Only "contested" states are searched.


def branching_paths(n_qubits: int, n_measurements: int) -> List[tuple]:
    """All measurement-outcome paths of a branching simulation (the keys of the leaf-probability dict).

    Enumerates every length-``n_measurements`` sequence of computational basis outcomes (each a per-qubit
    bit-string) in the order the branch split produces (first measurement most significant), so zipping
    this with the flat leaf-probability array yields ``{path: probability}``.

    Args:
        - ``n_qubits`` (int): number of qubits (basis states are the ``2**n_qubits`` bit-strings).
        - ``n_measurements`` (int): number of measurements M in the train.
    Returns:
        - ``paths`` (List[tuple]): the ``(2**n_qubits)**M`` outcome-path tuples, e.g. ``('01', '11')``.
    """
    labels = [format(k, f"0{n_qubits}b") for k in range(2 ** n_qubits)]
    return list(itertools.product(labels, repeat=n_measurements))


def _leaf_state_counts(leaf_data: Dict[tuple, float], n_states: int) -> Tuple[np.ndarray, np.ndarray]:
    """Collapse a config's leaves into distinct per-state count vectors and their summed probability.

    Args:
        - ``leaf_data`` (Dict[tuple, float]): outcome history (bit-string per measurement) -> path prob.
        - ``n_states`` (int): number of computational basis states (2**n_qubits).
    Returns:
        - ``unique_counts`` (np.ndarray): (U, n_states) distinct occurrence-count vectors.
        - ``weights`` (np.ndarray): (U,) total path probability carrying each count vector.
    """
    histories = list(leaf_data.keys())
    probs = np.array([leaf_data[h] for h in histories], dtype=float)
    # count how many measurements landed on each basis state (bit-string -> index via int(s, 2))
    counts = np.array([np.bincount([int(s, 2) for s in h], minlength=n_states) for h in histories], dtype=int)
    # many leaves share a count vector; dedupe and sum their probability onto the representative
    unique_counts, inverse = np.unique(counts, axis=0, return_inverse=True)
    weights = np.zeros(len(unique_counts))
    np.add.at(weights, inverse.ravel(), probs)
    return unique_counts, weights


def _balanced_geomean(values: List[float]) -> float:
    """Geometric mean of per-config scores; 0 if any is 0 so one undetected config collapses the score."""
    values = np.asarray(values, dtype=float)
    if values.size == 0 or np.any(values <= 0):
        return 0.0
    return float(np.exp(np.mean(np.log(values))))


def _state_production(leaf_data_per_config: Dict[str, Dict[tuple, float]], config_names: List[str],
                      n_states: int) -> np.ndarray:
    """Map-independent P(config produces basis state at least once), for each (config, state).

    Args:
        - ``leaf_data_per_config`` (Dict[str, Dict[tuple, float]]): per config, its branching leaf dict.
        - ``config_names`` (List[str]): configuration names (row order of the result).
        - ``n_states`` (int): number of computational basis states.
    Returns:
        - ``production`` (np.ndarray): (n_configs, n_states) at-least-once production probability.
    """
    production = np.zeros((len(config_names), n_states))
    for c, name in enumerate(config_names):
        for history, prob in leaf_data_per_config[name].items():
            # count a state once per leaf however often it recurs (at-least-once)
            for state in set(int(s, 2) for s in history):
                production[c, state] += prob
    return production


def _contested_states(production: np.ndarray, ground_idx: int,
                      threshold: float) -> Tuple[List[int], np.ndarray]:
    """Split basis states into contested (must be searched) and fixed-to-their-producer.

    A state is contested when >=2 configs produce it above ``threshold``; otherwise it is fixed to its
    sole above-threshold producer (or ground if none), optimal up to the ignored sub-threshold mass.

    Args:
        - ``production`` (np.ndarray): (n_configs, n_states) from :func:`_state_production`.
        - ``ground_idx`` (int): index of the ground configuration.
        - ``threshold`` (float): production probability above which a config counts as a producer.
    Returns:
        - ``contested`` (List[int]): basis-state indices to brute-force.
        - ``fixed_assign`` (np.ndarray): (n_states,) forced config index, -1 on contested slots.
    """
    n_configs, n_states = production.shape
    contested: List[int] = []
    fixed_assign = np.full(n_states, -1)
    for state in range(n_states):
        producers = np.where(production[:, state] >= threshold)[0]
        if len(producers) >= 2:
            contested.append(state)                  # genuine trade-off -> search it
        elif len(producers) == 1:
            fixed_assign[state] = int(producers[0])  # single producer owns it
        else:
            fixed_assign[state] = ground_idx         # produced by nobody meaningfully -> "no detection"
    return contested, fixed_assign


def _score_map(assign: np.ndarray, per_config: Dict[str, Tuple[np.ndarray, np.ndarray]],
               config_names: List[str], ground_idx: int, transient: bool
               ) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, Dict[str, float]]]:
    """Score one candidate map into per-config true-detection, false-signal and row-stochastic confusion rows.

    Each row (true config) is a probability distribution over the predicted categories, summing to 1.
    Transient classifies each whole measurement train by the distinct non-ground configs it detected:
    none -> ground; exactly one -> that config; two or more -> ``'mixed'``. Persistent instead classifies
    every measurement independently and averages the per-measurement rates over the train. ground is scored
    like any other config (its diagonal enters the balance); a wrong-config classification is a false
    signal, while landing on ground is a missed detection (not a false signal).

    Args:
        - ``assign`` (np.ndarray): (n_states,) config index each basis state maps to.
        - ``per_config`` (Dict[str, Tuple[np.ndarray, np.ndarray]]): name -> (unique_counts, weights).
        - ``config_names`` (List[str]): configuration names.
        - ``ground_idx`` (int): ground configuration index.
        - ``transient`` (bool): True -> per-train classification (with 'mixed'), False -> per-measurement rate.
    Returns:
        - ``true_detection`` (Dict[str, float]): per config, its diagonal (correct-classification probability/rate).
        - ``false_signal`` (Dict[str, float]): per config, mass classified as a wrong, non-ground category.
        - ``confusion_rows`` (Dict[str, Dict[str, float]]): per true config, {predicted category: mass}.
    """
    n_configs = len(config_names)
    ground_name = config_names[ground_idx]
    onehot = np.eye(n_configs)[assign]  # (n_states, n_configs): basis state -> owning config
    nonground = [j for j in range(n_configs) if j != ground_idx]

    true_detection, false_signal, confusion_rows = {}, {}, {}
    for name in config_names:
        unique_counts, weights = per_config[name]  # (U, n_states), (U,)
        if transient:
            # detects[u, j] = leaf u measured at least one state the map assigns to config j
            present = unique_counts > 0  # (U, n_states)
            detects = (present[:, :, None] & (onehot > 0)[None, :, :]).any(axis=1)  # (U, n_configs)
            n_nonground = detects[:, nonground].sum(axis=1)  # distinct non-ground configs the train detected
            row = {pred: 0.0 for pred in config_names}
            row["mixed"] = float(weights[n_nonground >= 2].sum())       # detected two or more non-ground configs
            row[ground_name] = float(weights[n_nonground == 0].sum())   # detected only ground
            single = n_nonground == 1
            for j in nonground:                                         # detected exactly this one non-ground config
                row[config_names[j]] = float(weights[single & detects[:, j]].sum())
        else:
            # independent measurements: mean over the train of the per-config classification rate.
            # every train has M measurements, so M is the row-sum of any count vector.
            n_measurements = int(unique_counts.sum(axis=1)[0])
            rate = (weights @ (unique_counts @ onehot)) / n_measurements  # (n_configs,)
            row = {config_names[j]: float(rate[j]) for j in range(n_configs)}
        confusion_rows[name] = row
        true_detection[name] = row[name]  # diagonal: config correctly classified as itself
        # false signal: mass sent to a wrong, non-ground category (another config or 'mixed')
        false_signal[name] = sum(v for pred, v in row.items() if pred != name and pred != ground_name)
    return true_detection, false_signal, confusion_rows


def derive_detection_map(leaf_data_per_config: Dict[str, Dict[tuple, float]], config_names: List[str],
                         ground: str, perturbation_type: str, n_states: int,
                         false_signal_weight: float = 1.0, false_signal_constraint: Optional[float] = None,
                         contested_threshold: float = 1e-3, max_search: int = 500_000) -> Dict[str, Any]:
    """Derive the single deployable state->config map from branching (correlation-aware) leaf data.

    Every basis state maps to one config; maps are ranked by a balanced geometric mean over all configs
    (ground included) of their diagonal correct-classification -- transient: each measurement train is
    classified by the distinct non-ground configs it detected (none -> ground, one -> that config, two or
    more -> 'mixed'); persistent: the per-measurement classification rate -- while false signals (mass sent
    to a wrong, non-ground category) are penalized by ``false_signal_weight`` or hard-constrained by
    ``false_signal_constraint``. Only contested states (produced by >=2 configs above
    ``contested_threshold``) are brute-forced; the rest are fixed to their sole producer.

    Args:
        - ``leaf_data_per_config`` (Dict[str, Dict[tuple, float]]): per config, the branching leaf dict
          ``{history: path_prob}`` from :meth:`Experiment.branching_simulation`.
        - ``config_names`` (List[str]): configuration names.
        - ``ground`` (str): ground (no-detection) configuration name.
        - ``perturbation_type`` (str): 'transient' (at-least-once) or 'persistent' (rate).
        - ``n_states`` (int): number of computational basis states (2**n_qubits).
        - ``false_signal_weight`` (float): lambda subtracted as ``lambda * mean_false`` when unconstrained.
        - ``false_signal_constraint`` (Optional[float]): if set, only maps whose worst per-config false
          signal is <= this are eligible; among them balanced true detection is maximized.
        - ``contested_threshold`` (float): production probability above which a config counts as a producer
          of a state (0.0 searches every jointly-produced state, i.e. exact).
        - ``max_search`` (int): raise if the contested brute force exceeds this many candidate maps.
    Returns:
        - ``result`` (Dict[str, Any]): ``states_map`` (config -> owned bit-string labels),
          ``confusion_matrix`` ((true, pred) -> probability or rate), per-config ``true_detection`` and
          ``false_signal``, the winner's ``true_score``/``mean_false``/``max_false``, and the
          ``n_contested``/``n_maps_searched`` search sizes.
    """
    config_names = list(config_names)
    n_configs = len(config_names)
    ground_idx = config_names.index(ground)
    transient = perturbation_type == "transient"

    # setup (once): per-config count vectors, state production, and the contested/fixed split
    per_config = {name: _leaf_state_counts(leaf_data_per_config[name], n_states) for name in config_names}
    production = _state_production(leaf_data_per_config, config_names, n_states)
    contested, fixed_assign = _contested_states(production, ground_idx, contested_threshold)

    n_maps = n_configs ** len(contested)
    if n_maps > max_search:
        raise NotImplementedError(
            f"{n_maps} candidate maps over {len(contested)} contested states exceeds max_search="
            f"{max_search}. Raise max_search / contested_threshold, or add a greedy fallback.")

    # search: only the contested states vary; fixed ones stay at their producer
    best = None
    for combo in itertools.product(range(n_configs), repeat=len(contested)):
        assign = fixed_assign.copy()
        assign[contested] = combo
        true_detection, false_signal, rows = _score_map(
            assign, per_config, config_names, ground_idx, transient)
        true_score = _balanced_geomean(list(true_detection.values()))
        mean_false = float(np.mean(list(false_signal.values())))
        max_false = float(np.max(list(false_signal.values())))
        # constrained: feasible-first then most detection; unconstrained: penalized detection
        if false_signal_constraint is not None:
            feasible = max_false <= false_signal_constraint
            key = (feasible, true_score if feasible else -max_false)
        else:
            key = (True, true_score - false_signal_weight * mean_false)
        if best is None or key > best[0]:
            best = (key, assign, true_detection, false_signal, rows, true_score, mean_false, max_false)

    _, assign, true_detection, false_signal, rows, true_score, mean_false, max_false = best
    if true_score == 0.0:
        warnings.warn("Best detection map has balanced true-detection 0: a config is never correctly "
                      "detected (false-signal constraint too tight, or configs indistinguishable here).")

    # build the deployable map (config -> owned bit-string states) and the confusion matrix
    n_qubits = n_states.bit_length() - 1
    labels = [format(i, f"0{n_qubits}b") for i in range(n_states)]
    states_map = {name: [] for name in config_names}
    for state, c in enumerate(assign):
        states_map[config_names[c]].append(labels[state])
    # rows are true configs; columns are every config plus a 'mixed' prediction column (transient only).
    pred_categories = list(config_names) + (["mixed"] if transient else [])
    confusion_matrix = {(t, pred): float(rows[t].get(pred, 0.0)) for t in config_names for pred in pred_categories}

    return dict(states_map=states_map, confusion_matrix=confusion_matrix, true_detection=true_detection,
                false_signal=false_signal, true_score=true_score, mean_false=mean_false, max_false=max_false,
                n_contested=len(contested), n_maps_searched=n_maps)


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


def collapse_unsafe_keys(exp) -> set:
    """Global keys that must NOT be promoted because they sit on a Lindblad collapse-operator term
    that already coexists with another coefficient-bearing term.

    qutip-jax bug: when a single collapse operator SUMS two coefficient-bearing terms on DIFFERENT
    operators (``c1*A + c2*B`` with ``A != B``) it computes ``L†L`` wrong (drops the cross term
    ``A†B``), so the master equation stops preserving trace and the solve blows up. Coefficients
    multiplied into one term are fine, and so is one coefficient term (a pulse ``g(t)`` or one promoted
    parameter) plus constant parts -- that is what baking gives. Each collapse contribution here is a
    distinct operator, so promoting a parameter onto a collapse term is only allowed when no OTHER term
    of that same collapse operator is time-modulated or itself promotable (that other term would be the
    second summand). ``gamma`` stays promotable because it multiplies into the already-modulated pulse
    term rather than adding a new summand. Everything flagged here is swept via the rebuild lane (baked
    back into a constant operator) instead. See ``_interaction_terms`` for the matching build-time guard.

    Args:
        exp (Experiment): experiment providing the interactions and operators.
    Returns:
        set[str]: global-args keys that are unsafe to promote.
    """
    unsafe: set = set()

    def scan(interactions, prefix):
        for inter in interactions:
            l_terms = [t for t in _build_hamiltonian_term(inter, exp.operators, exp.n_cavities, exp.n_fields)
                       if t.kind == 'L']
            n_modulated = sum(1 for t in l_terms if t.modulated)
            for term in l_terms:
                other_modulated = n_modulated - (1 if term.modulated else 0)
                other_promotable = any(other is not term and other.params for other in l_terms)
                if other_modulated >= 1 or (not term.modulated and other_promotable):
                    for name in term.params:
                        unsafe.add(f"{prefix}{inter._interaction_context()}__{name}")

    scan(exp.experimental_params.interactions, "BaseModel_")
    for cfg in exp.experimental_params.configuration_set:
        scan(cfg.interactions, f"Conf:{cfg.name}_")
    return unsafe


def classify_sweep_axis(exp, name: str, key_types: Dict[str, Any]):
    """Return (lane, keys). Lanes: 'measurement' (time_interval), 'promote' (baked + promotable
    -> args-coefficient), 'rebuild' (baked + non-promotable), 'coeff' (already a coefficient)."""
    if name == "time_interval":
        return "measurement", []
    keys = resolve_sweep_keys(exp, name)
    # a parameter on a multi-coefficient Lindblad collapse operator must be baked, not promoted
    if any(k in collapse_unsafe_keys(exp) for k in keys):
        return "rebuild", keys
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
