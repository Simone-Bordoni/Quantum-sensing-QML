"""
Quantum Utils Module (Compatibility Layer)
==========================================

This module provides backward compatibility for imports.
All quantum utility functions have been moved to the experiment submodule.

Deprecated:
    Direct imports from this module are deprecated. Use:
    - from qsopt.core.experiment.quantum_utils import ...
"""

# Re-export from new location for backward compatibility
from .experiment.quantum_utils import (
    gu,
    u0,
    generate_single_qubit_operators,
    generate_two_qubit_operators,
    generate_initial_state,
    apply_single_qubit_rotation,
    create_measurement_projector,
    project_and_measure,
    measure_qubit_probability
)

__all__ = [
    'gu',
    'u0',
    'generate_single_qubit_operators',
    'generate_two_qubit_operators',
    'generate_initial_state',
    'apply_single_qubit_rotation',
    'create_measurement_projector',
    'project_and_measure',
    'measure_qubit_probability'
]
