"""
Utility modules for quantum sensing optimization.

This module provides visualization and helper utilities for:
- Optimization visualization and dashboards
- Parameter landscape analysis
- Experiment data loading

Note: Parameter sweep functions (compute_chi_gamma_sweep, etc.) have been
removed. Use the Experiment class methods instead:
    - experiment.sweep_chi_gamma()
"""

from .experiment_loader import load_experiment_from_report
from .landscape_analysis import compute_theta1_theta2_landscape, compute_time_interval_landscape
from .results import SweepResults, TimeEvolutionResults, load_results, save_results
from .visualization import (
    plot_metric_evolution,
    plot_optimization_dashboard,
    plot_parameter_landscape,
    plot_parameter_trajectory,
    plot_sweep_results,
    plot_time_evolution,
    plot_time_interval_landscape,
)

__all__ = [
    "plot_optimization_dashboard",
    "plot_metric_evolution",
    "plot_parameter_trajectory",
    "plot_parameter_landscape",
    "plot_time_interval_landscape",
    "plot_sweep_results",
    "plot_time_evolution",
    "load_experiment_from_report",
    "compute_theta1_theta2_landscape",
    "compute_time_interval_landscape",
    "TimeEvolutionResults",
    "SweepResults",
    "save_results",
    "load_results",
]
