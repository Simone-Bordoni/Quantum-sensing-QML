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
from .results import SweepResults, TimeEvolutionResults, load_results, save_results
from .visualization import (
    interactive_sweep,
    plot_metric_evolution,
    plot_optimization_dashboard,
    plot_parameter_trajectory,
    plot_sweep_corner,
    plot_sweep_results,
    plot_time_evolution,
)

__all__ = [
    "plot_optimization_dashboard",
    "plot_metric_evolution",
    "plot_parameter_trajectory",
    "plot_sweep_results",
    "plot_sweep_corner",
    "interactive_sweep",
    "plot_time_evolution",
    "load_experiment_from_report",
    "TimeEvolutionResults",
    "SweepResults",
    "save_results",
    "load_results",
]
