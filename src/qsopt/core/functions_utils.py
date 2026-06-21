"""
Shared math/function utilities for quantum sensing optimization.

Small reusable functions (annealing schedules, etc.) parameterized by the epoch
fraction f in [0, 1] (0 at the first step, ->1 at the last).
"""

from typing import Union

import jax.numpy as jnp
from jax import Array


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
