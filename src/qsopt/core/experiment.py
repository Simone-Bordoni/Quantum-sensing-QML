"""
Quantum Sensing Experiment Class
================================

Main experiment class that orchestrates quantum sensing protocols with configurable
parameters, noise models, and optimization strategies.
"""

import warnings
from typing import Any, Dict, List

import jax.numpy as jnp
import numpy as np
import qutip as qt
from jax.scipy.special import erfc

# Suppress Diffrax complex dtype warning
warnings.filterwarnings("ignore", message="Complex dtype support in Diffrax is a work in progress*")

class SingleQubitExperiment:
    """A class representing a single qubit photon detection experiment."""
    
    