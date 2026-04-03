"""
Experiment Loader Utilities
===========================

Utilities for loading and reconstructing experiment configurations from saved reports.

.. deprecated::
    These functions depend on TrainableParameters which is being removed.
    They will be refactored to work with circuit-based parameters in a future release.
"""

import json
import warnings
from pathlib import Path
from typing import Any, Dict, Tuple

from ..core.callback import OptimizationCallback
from ..core.experimental_parameters import (
    ExperimentalParameters,
    InitialStateConfig,
    InitialStateType,
    MeasurementProtocol,
    NoiseConfiguration,
    PhysicalConstants,
    SystemDimensions,
)
# TODO: Remove TrainableParameters dependency - refactor to use circuits
# from ..core.trainable_parameters import ParameterType, TrainableParameters


def load_experiment_from_report(
    json_path: str,
) -> Tuple[ExperimentalParameters, Any, Dict[str, Any]]:
    """
    Load and reconstruct experiment configuration from a JSON report file.

    .. deprecated::
        This function depends on TrainableParameters which is being removed.
        It will be refactored to work with circuit-based parameters in a future release.

    This function loads the experiment report and reconstructs the ExperimentalParameters
    and trainable parameters from the saved configuration.

    Args:
        json_path: Path to the JSON report file

    Returns:
        Tuple containing:
            - ExperimentalParameters: Reconstructed experimental parameters
            - None: Trainable parameters (deprecated)
            - Dict: Additional metadata including callback information

    Example:
        >>> # This function is deprecated
        >>> # from qsopt.utils import load_experiment_from_report
        >>> #
        >>> # # Load experiment configuration
        >>> # exp_params, trainable_params, metadata = load_experiment_from_report('results/report.json')
    """
    warnings.warn(
        "load_experiment_from_report() depends on TrainableParameters which is deprecated. "
        "This function will be refactored in a future release.",
        DeprecationWarning,
        stacklevel=2
    )

    raise NotImplementedError(
        "This function requires TrainableParameters which has been removed. "
        "Please use circuit-based parameter management instead."
    )
