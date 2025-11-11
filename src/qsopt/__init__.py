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
from .core.experimental_parameters import (ExperimentalParameters,
                                           InitialStateConfig,
                                           InitialStateType,
                                           MeasurementProtocol,
                                           NoiseConfiguration,
                                           PhysicalConstants, SystemDimensions)
from .core.trainable_parameters import (ParameterConstraints,
                                        ParameterType, TrainableParameters)
from .core.experiment import Experiment, SingleQubitExperiment, TwoQubitExperiment
from .core.callback import OptimizationCallback

# Visualization utilities
from .utils.visualization import (plot_optimization_dashboard,
                                  plot_contrast_evolution,
                                  plot_parameter_trajectory)
