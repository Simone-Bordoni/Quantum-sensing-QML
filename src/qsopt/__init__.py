"""
Quantum Sensing Optimization Library (qsopt)
============================================

A specialized library for parameter optimization in quantum sensing experiments
using QuTiP-JAX backend for automatic differentiation.
"""

__version__ = "0.1.0"
__author__ = "Simone Bordoni, Nathan Campioni"
__email__ = "simone.bordoni@uniroma1.it, nathan.campioni@gmail.com"

from .core.callback import OptimizationCallback
from .core.circuit import QuantumCircuit, create_ry_circuit
from .core.experiment import Experiment
from .core.gates import CNOTGate, CZGate, HadamardGate, RXGate, RYGate, RZGate
from .core.loss_functions import DetectionMetric

# Core experimental framework
from .core.experimental_parameters import (
    ExperimentalParameters,
    State,
    InteractionType,
    TimeProtocol,
    NoiseModel,
    PhysicalModel,
    Interaction,
    SystemConfiguration,
    SubsystemState,
)
# TrainableParameters has been removed in favor of circuit-based parameter management

from .utils.visualization import (
    plot_metric_evolution,
    plot_optimization_dashboard,
    plot_parameter_trajectory,
)
