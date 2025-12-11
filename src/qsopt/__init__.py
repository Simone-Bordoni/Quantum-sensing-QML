"""
Quantum Sensing Optimization Library (qsopt)
============================================

A specialized library for parameter optimization in quantum sensing experiments
using QuTiP-JAX backend for automatic differentiation.
"""

__version__ = "0.1.0"
__author__ = "Simone Bordoni, Nathan Gargioni"
__email__ = "simone.bordoni@uniroma1.it"

from .core.callback import OptimizationCallback
from .core.experiment import Experiment, SingleQubitExperiment, TwoQubitExperiment

# Core experimental framework
from .core.experimental_parameters import (
    ExperimentalParameters,
    InitialStateConfig,
    InitialStateType,
    MeasurementProtocol,
    NoiseConfiguration,
    PhysicalConstants,
    SystemDimensions,
)
from .core.trainable_parameters import ParameterConstraints, ParameterType, TrainableParameters

# Visualization utilities
from .utils.visualization import (
    plot_contrast_evolution,
    plot_optimization_dashboard,
    plot_parameter_trajectory,
)
