"""
Core module for quantum sensing experiments
===========================================

This module contains the fundamental classes for setting up and running
quantum sensing experiments.
"""

from .callback import OptimizationCallback
from .experiment import Experiment
from .experiment.quantum_utils import (
    generate_initial_state,
    generate_system_operators,
)
from .experimental_parameters import (
    ExperimentalParameters,
    MeasurementProtocol,
    NoiseModel,
    PhysicalModel,
    SystemConfiguration,
    SubsystemState,
    State,
)
# TODO: TrainableParameters has been removed in favor of circuit-based parameter management
# from .trainable_parameters import (
#     Parameter,
#     ParameterConstraints,
#     ParameterType,
#     TrainableParameters,
# )
