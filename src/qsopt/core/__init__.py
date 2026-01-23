"""
Core module for quantum sensing experiments
===========================================

This module contains the fundamental classes for setting up and running
quantum sensing experiments.
"""

from .callback import OptimizationCallback
from .experiment import Experiment, SingleQubitExperiment, TwoQubitExperiment
from .experiment.quantum_utils import (
    apply_single_qubit_rotation,
    create_measurement_projector,
    generate_initial_state,
    generate_single_qubit_operators,
    generate_two_qubit_operators,
    measure_qubit_probability,
    project_and_measure,
)
from .experimental_parameters import (
    ExperimentalParameters,
    InitialStateConfig,
    MeasurementProtocol,
    NoiseConfiguration,
    PhysicalConstants,
    SystemDimensions,
)
# TODO: TrainableParameters has been removed in favor of circuit-based parameter management
# from .trainable_parameters import (
#     Parameter,
#     ParameterConstraints,
#     ParameterType,
#     TrainableParameters,
# )
