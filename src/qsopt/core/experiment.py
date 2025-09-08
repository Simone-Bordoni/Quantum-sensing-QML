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
from qsopt.core.experimental_parameters import ExperimentalParameters
from qsopt.core.trainable_parameters import TrainableParameters

# Suppress Diffrax complex dtype warning
warnings.filterwarnings("ignore", message="Complex dtype support in Diffrax is a work in progress*")

class SingleQubitExperiment:
    """A class representing a single qubit photon detection experiment."""
    
    def __init__(self, experimental_params: ExperimentalParameters, trainable_params: TrainableParameters):
        self.experimental_params = experimental_params
        self.trainable_params = trainable_params

    def __post_init__(self):
        """Post-initialization to set up operators and hamiltonian."""
        self._generate_operators()
        self._generate_hamiltonian()

    def _generate_operators(self):
        """Generate the necessary operators for the experiment."""
        pass

    def _generate_hamiltonian(self):
        """Generate the Hamiltonian for the experiment."""
        pass
