"""
Utility modules for quantum sensing optimization.

This module provides visualization and helper utilities for:
- Optimization visualization and dashboards
- Parameter landscape analysis
- Experiment data loading
"""

from .visualization import (
    plot_optimization_dashboard,
    plot_contrast_evolution,
    plot_parameter_trajectory,
    plot_parameter_landscape,
    plot_time_interval_landscape,
    plot_chi_gamma_sweep,
    plot_chi_lambda_sweep,  # Backward compatibility alias
    plot_time_evolution,
    plot_two_qubit_probabilities,
    plot_single_probability_map,
    plot_qubit_time_evolution
)
from .experiment_loader import load_experiment_from_report
from .landscape_analysis import (
    compute_theta1_theta2_landscape,
    compute_time_interval_landscape
)
from .chi_lambda_sweep import compute_chi_gamma_sweep, compute_chi_lambda_sweep

__all__ = [
    'plot_optimization_dashboard',
    'plot_contrast_evolution',
    'plot_parameter_trajectory',
    'plot_parameter_landscape',
    'plot_time_interval_landscape',
    'plot_chi_gamma_sweep',
    'plot_chi_lambda_sweep',  # Backward compatibility
    'plot_time_evolution',
    'plot_two_qubit_probabilities',
    'plot_single_probability_map',
    'plot_qubit_time_evolution',
    'load_experiment_from_report',
    'compute_theta1_theta2_landscape',
    'compute_time_interval_landscape',
    'compute_chi_gamma_sweep',
    'compute_chi_lambda_sweep'  # Backward compatibility
]
