"""
Quantum Sensing Optimization Library (qsopt)
============================================

A specialized library for parameter optimization in quantum sensing experiments
using QuTiP-JAX backend for automatic differentiation.
"""

__version__ = "0.1.0"
__author__ = "Simone Bordoni, Nathan Campioni"
__email__ = "simone.bordoni@uniroma1.it"

from .core.callback import OptimizationCallback
from .core.experiment import Experiment
from .core.loss_functions import DetectionMetric

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
# TrainableParameters has been removed in favor of circuit-based parameter management

from .utils.visualization import (
    plot_contrast_evolution,
    plot_optimization_dashboard,
    plot_parameter_trajectory,
)
