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
from .trainable_parameters import (ParameterConstraints,
                                   ParameterGroup, ParameterType,
                                   TrainableParameters)
