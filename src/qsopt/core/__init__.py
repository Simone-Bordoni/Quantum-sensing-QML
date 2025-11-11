"""
Core module for quantum sensing experiments
===========================================

This module contains the fundamental classes for setting up and running
quantum sensing experiments.
"""

from .experimental_parameters import (ExperimentalParameters,
                                      InitialStateConfig, MeasurementProtocol,
                                      NoiseConfiguration, PhysicalConstants,
                                      SystemDimensions)
from .trainable_parameters import (Parameter, ParameterConstraints, 
                                   ParameterType, TrainableParameters)
from .experiment import Experiment, SingleQubitExperiment, TwoQubitExperiment
from .callback import OptimizationCallback
from .experiment.quantum_utils import (
    generate_single_qubit_operators,
    generate_two_qubit_operators,
    generate_initial_state,
    apply_single_qubit_rotation,
    create_measurement_projector,
    project_and_measure,
    measure_qubit_probability
)

