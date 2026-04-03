"""
Parameter Landscape Analysis Utilities
=======================================

This module provides functions for computing parameter space landscapes
for quantum sensing optimization.

.. deprecated::
    Most landscape functions have been moved to the Experiment class as methods.
    compute_theta1_theta2_landscape depends on TrainableParameters which is being removed.

Functions:
    compute_theta1_theta2_landscape: Compute 2D landscape over rotation parameters (deprecated)
    
Note:
    compute_time_interval_landscape has been moved to Experiment.compute_time_interval_landscape()
"""

import math
import time
import warnings
from typing import TYPE_CHECKING, Any, Dict, Optional, Union

import numpy as np

if TYPE_CHECKING:
    from qsopt.core.experiment import Experiment

from qsopt.core.experimental_parameters import ExperimentalParameters
# TODO: Remove TrainableParameters dependency - refactor to use circuits
# from qsopt.core.trainable_parameters import TrainableParameters

def compute_theta1_theta2_landscape(
    exp_params: ExperimentalParameters,
    resolution: int = 25,
    center_theta1: float = np.pi / 2,
    center_theta2: float = -np.pi / 2,
    param_range: float = np.pi / 6,
    batch_size: int = 1,
    verbose: bool = True,
) -> Dict[str, Union[np.ndarray, float]]:
    """
    Compute parameter landscape for θ₁, θ₂ rotation strategy.

    .. deprecated::
        This function depends on TrainableParameters which is being removed.
        It will be refactored to work with circuit-based parameters in a future release.

    This function evaluates the sensing contrast and detection probability
    across a 2D grid of rotation parameters (θ₁, θ₂). Each point represents
    a quantum sensing simulation with different rotation angles applied
    before and after time evolution.

    The workflow for each parameter point:
        1. Set θ₁ and θ₂ rotation angles
        2. Run quantum simulation with and without photon interaction
        3. Calculate sensing contrast (difference in detection probabilities)
        4. Store results in 2D arrays

    Args:
        exp_params: Configured experimental parameters including physical
            constants, noise configuration, and measurement protocol.
        resolution: Number of points per dimension in the parameter grid.
            Total evaluations = resolution². Default: 25.
        center_theta1: Center value for θ₁ in radians. Default: π/2 (90°).
        center_theta2: Center value for θ₂ in radians. Default: -π/2 (-90°).
        param_range: Range around center values in radians (±param_range).
            Default: π/6 (±30°).
        batch_size: Number of random realizations to average over for
            measurement uncertainty. Default: 1.
        verbose: If True, print progress information. Default: True.

    Returns:
        Dictionary containing:
            - 'theta1_vals': Array of θ₁ values evaluated (length=resolution)
            - 'theta2_vals': Array of θ₂ values evaluated (length=resolution)
            - 'contrast_map': 2D array of sensing contrast values
              (shape: resolution × resolution)
            - 'detection_map': 2D array of detection probability values
              with photon interaction (shape: resolution × resolution)
            - 'center_theta1': Center θ₁ value used
            - 'center_theta2': Center θ₂ value used

    Example:
        >>> from qsopt.core.experimental_parameters import ExperimentalParameters
        >>> from qsopt.utils import compute_theta1_theta2_landscape
        >>>
        >>> # Configure experiment
        >>> exp_params = ExperimentalParameters()
        >>> exp_params.measurement.initial_time = -5.0
        >>> exp_params.measurement.final_time = 5.0
        >>> exp_params.measurement.time_interval = 2.5
        >>>
        >>> # Compute landscape
        >>> results = compute_theta1_theta2_landscape(
        ...     exp_params,
        ...     resolution=10,
        ...     param_range=np.pi/6
        ... )
        >>>
        >>> # Analyze results
        >>> max_idx = np.unravel_index(
        ...     np.argmax(results['contrast_map']),
        ...     results['contrast_map'].shape
        ... )
        >>> print(f"Maximum contrast: {results['contrast_map'][max_idx]:.6f}")

    Notes:
        - Computation time scales as O(resolution²)
        - Each point requires a full quantum dynamics simulation
        - For resolution=25: expect 10-30 minutes (system dependent)
        - For resolution=50: expect 1-2 hours
        - Results are stored in row-major order: contrast_map[j, i]
          corresponds to (theta1_vals[i], theta2_vals[j])

    See Also:
        plot_parameter_landscape: Visualize the computed landscape
    """
    warnings.warn(
        "compute_theta1_theta2_landscape() depends on TrainableParameters which is deprecated. "
        "This function will be refactored in a future release.",
        DeprecationWarning,
        stacklevel=2
    )

    raise NotImplementedError(
        "This function requires TrainableParameters which has been removed. "
        "Please use circuit-based parameter management instead."
    )


def compute_time_interval_landscape(
    experiment: Union["Experiment", ExperimentalParameters],
    resolution: int = 50,
    mode: str = "continuous",
    batch_size: int = 1,
    verbose: bool = True,
    min_interval: Optional[float] = None,
    max_interval: Optional[float] = None,
    theta1: Optional[float] = None,
    theta2: Optional[float] = None,
) -> Dict[str, Union[np.ndarray, float, str, int]]:
    """
    Compute contrast landscape vs measurement time interval.

    .. deprecated::
        This function has been moved to the Experiment class as a method.
        Use ``experiment.compute_time_interval_landscape()`` instead.

    This function is maintained for backward compatibility but will be removed
    in a future release. Please update your code to use the Experiment method.

    Args:
        experiment: Experiment instance (or ExperimentalParameters for backward compat).
        resolution: Number of time interval values to evaluate. Default: 50.
        mode: Computation mode - either 'continuous' or 'discrete'.
        batch_size: Number of random realizations to average over. Default: 1.
        verbose: Print progress information. Default: True.
        min_interval: Minimum interval to consider. Default: auto-determined.
        max_interval: Maximum interval to consider. Default: total_time.
        theta1: Initial rotation angle (for ExperimentalParameters backward compat).
        theta2: Final rotation angle (for ExperimentalParameters backward compat).

    Returns:
        Dictionary containing interval sweep results.

    See Also:
        :meth:`qsopt.core.experiment.Experiment.compute_time_interval_landscape`:
            The new method location.
    """
    import numpy as _np
    warnings.warn(
        "compute_time_interval_landscape() has been moved to Experiment.compute_time_interval_landscape(). "
        "This standalone function is deprecated and will be removed in a future release.",
        DeprecationWarning,
        stacklevel=2
    )

    # Backward compatibility: accept ExperimentalParameters and build an Experiment
    if isinstance(experiment, ExperimentalParameters):
        from qsopt.core.experiment import Experiment
        from qsopt.core.circuit import create_ry_circuit_layer
        import numpy as _np2

        _theta1 = theta1 if theta1 is not None else _np2.pi / 2
        _theta2 = theta2 if theta2 is not None else -_np2.pi / 2
        n_qubits = experiment.n_qubits
        initial_circuit = create_ry_circuit_layer(n_qubits=n_qubits, theta_values=[_theta1] * n_qubits)
        final_circuit = create_ry_circuit_layer(n_qubits=n_qubits, theta_values=[_theta2] * n_qubits)
        experiment = Experiment(experiment, initial_circuit=initial_circuit, final_circuit=final_circuit)

    result = experiment.compute_time_interval_landscape(
        resolution=resolution,
        mode=mode,
        batch_size=batch_size,
        verbose=verbose,
        min_interval=min_interval,
        max_interval=max_interval,
    )

    # Add theta1/theta2 to the result for backward compat
    if theta1 is not None:
        result["theta1"] = theta1
    if theta2 is not None:
        result["theta2"] = theta2

    return result
