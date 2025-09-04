"""
Core module for quantum sensing experiments
===========================================

This module contains the fundamental classes for setting up and running
quantum sensing experiments.
"""

from .experiment import Experiment
from .trainable_parameters import TrainableParameters, ParameterGroup, ParameterConstraints, OptimizationConfig, ParameterType
from .experimental_parameters import ExperimentalParameters, PhysicalConstants, SystemDimensions, MeasurementProtocol, NoiseConfiguration, InitialStateConfig, NoiseConfiguration


