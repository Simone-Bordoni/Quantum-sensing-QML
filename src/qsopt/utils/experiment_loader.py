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

    # Extract data from report
    exp_params_dict = report.get("experimental_parameters", {})
    trainable_params_dict = report.get("trainable_parameters", {})
    callback_info = report.get("callback_info")

    # Reconstruct PhysicalConstants
    phys_const_dict = exp_params_dict.get("physical_constants", {})
    physical_constants = PhysicalConstants(
        chi=phys_const_dict.get("chi", 0.5),
        photon_cavity_coupling=phys_const_dict.get("photon_cavity_coupling", 1.0),
        inverse_pulse_width=phys_const_dict.get("inverse_pulse_width", 0.1),
    )

    # Reconstruct SystemDimensions
    sys_dims_dict = exp_params_dict.get("system_dimensions", {})
    system_dims = SystemDimensions(
        cavity_levels=sys_dims_dict.get("cavity_levels", 2),
        qubit_levels=sys_dims_dict.get("qubit_levels", 2),
        field_levels=sys_dims_dict.get("field_levels", 2),
    )

    # Reconstruct MeasurementProtocol
    meas_protocol_dict = exp_params_dict.get("measurement_protocol", {})
    mode = meas_protocol_dict.get("mode", "interval")

    if mode == "explicit":
        # Use explicit measurement times
        measurement = MeasurementProtocol(
            measurement_times=meas_protocol_dict.get("measurement_times"),
            initial_time_uncertainty=meas_protocol_dict.get("initial_time_uncertainty", 0.0),
        )
    else:
        # Use interval-based protocol
        measurement = MeasurementProtocol(
            measurement_times=None,
            initial_time=meas_protocol_dict.get("initial_time", -5.0),
            final_time=meas_protocol_dict.get("final_time", 5.0),
            time_interval=meas_protocol_dict.get("time_interval", 1.0),
            initial_time_uncertainty=meas_protocol_dict.get("initial_time_uncertainty", 0.0),
        )

    # Reconstruct InitialStateConfig
    initial_state_dict = exp_params_dict.get("initial_state", {})
    state_type_str = initial_state_dict.get("state_type", "single_photon")
    state_type = InitialStateType(state_type_str)

    # Handle complex coherent_alpha reconstruction
    coherent_alpha = None
    if initial_state_dict.get("coherent_alpha") is not None:
        amplitude = initial_state_dict["coherent_alpha"]
        phase = initial_state_dict.get("coherent_alpha_phase", 0.0)
        import numpy as np

        coherent_alpha = amplitude * np.exp(1j * phase)

    initial_state = InitialStateConfig(
        state_type=state_type,
        coherent_alpha=coherent_alpha,
        thermal_n_bar=initial_state_dict.get("thermal_n_bar"),
        custom_amplitudes=None,  # Custom amplitudes cannot be serialized
    )

    # Reconstruct NoiseConfiguration
    noise_dict = exp_params_dict.get("noise_configuration", {})
    noise_config = NoiseConfiguration(
        depolarizing=noise_dict.get("depolarizing", 0.0),
        dephasing=noise_dict.get("dephasing", 0.0),
        relaxation=noise_dict.get("relaxation", 0.0),
    )

    # Create ExperimentalParameters
    experimental_params = ExperimentalParameters(
        physical_constants=physical_constants,
        system_dims=system_dims,
        measurement=measurement,
        initial_state=initial_state,
        noise_config=noise_config,
    )

    # Reconstruct TrainableParameters
    trainable_params = TrainableParameters()

    # Add parameters from the saved configuration
    params_list = trainable_params_dict.get("parameters", [])
    for param_dict in params_list:
        param_name = param_dict["name"]
        param_value = param_dict["value"]
        param_trainable = param_dict.get("trainable", True)
        param_type_str = param_dict["type"]

        if param_type_str == "rotation_angle":
            trainable_params.add_rotation_angles(
                names=param_name, initial_values=param_value, trainable=param_trainable
            )
        elif param_type_str == "custom":
            trainable_params.add_custom_parameters(
                names=param_name, initial_values=param_value, trainable=param_trainable
            )
        # Note: measurement_time type is not yet implemented

    # Prepare metadata
    metadata = {
        "experiment_type": report.get("experiment_type"),
        "version": report.get("version"),
        "callback_info": callback_info,
    }

    # Load callback data if available
    if callback_info is not None and "callback_data_path" in callback_info:
        callback_path = callback_info["callback_data_path"]
        if Path(callback_path).exists():
            callback_data = OptimizationCallback.load(callback_path)
            metadata["callback_data"] = callback_data
        else:
            print(f"Warning: Callback data file not found: {callback_path}")

    return experimental_params, trainable_params, metadata
