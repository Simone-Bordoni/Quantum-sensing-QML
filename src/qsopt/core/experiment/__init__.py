"""
Experiment Module
=================

This module contains experiment classes for quantum sensing protocols.

Components:
- base: Abstract base class defining the experiment interface
- n_qubit_experiment: n qubit sensing experiment (in development)
- quantum_utils: Utility functions for quantum operations
"""

from .base import Experiment
from .n_qubit_experiment import NQubitExperiment

__all__ = [
    "Experiment",
    "NQubitExperiment",
]
