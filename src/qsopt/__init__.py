"""
Quantum Sensing Optimization Library (qsopt)
============================================

A specialized library for parameter optimization in quantum sensing experiments
using QuTiP-JAX backend for automatic differentiation.
"""

__version__ = "0.1.0"
__author__ = "Simone Bordoni, Nathan Gargioni"
__email__ = "simone.bordoni@uniroma1.it"

# Core experimental framework
from .core.experiment import Experiment
from .core.trainable_parameters import TrainableParameters, ParameterConstraints, OptimizationConfig, ParameterType
from .core.experimental_parameters import ExperimentalParameters, PhysicalConstants, SystemDimensions, MeasurementProtocol, InitialStateConfig, InitialStateType, NoiseConfiguration
# Optimization components
from .optimization.optimizer import Optimizer, OptimizationConfig as OptimizerConfig, OptimizationResult, OptimizerType, create_optimizer

