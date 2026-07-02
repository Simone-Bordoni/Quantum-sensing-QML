"""
Shared math/function utilities for quantum sensing optimization.

Small reusable functions (annealing schedules, etc.) parameterized by the epoch
fraction f in [0, 1] (0 at the first step, ->1 at the last), plus the sweep-axis
helpers used by :meth:`Experiment.sweep` (kept out of the class so only the main
methods live in ``experiment.py``).
"""

from typing import Any, Dict, List, Union
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
