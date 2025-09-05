"""
Core module for quantum sensing experiments
===========================================

This module contains the fundamental classes for setting up and running
quantum sensing experiments.
"""

from .experiment import Experiment
from .experimental_parameters import (ExperimentalParameters,
                                      InitialStateConfig, MeasurementProtocol,
                                      NoiseConfiguration, PhysicalConstants,
                                      SystemDimensions)
from .trainable_parameters import (OptimizationConfig, ParameterConstraints,
                                   ParameterGroup, ParameterType,
                                   TrainableParameters)
