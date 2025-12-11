"""
Results Data Structures
========================

This module provides standardized data structures for storing quantum sensing
experiment results, including time evolution and parameter sweep data.

Classes:
    TimeEvolutionResults: Container for time evolution data
    SweepResults: Container for parameter sweep data

Functions:
    save_results: Save results to file (npz format for arrays)
    load_results: Load results from file
"""

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np


@dataclass
class TimeEvolutionResults:
    """
    Container for time evolution results.

    This class stores time points, probability trajectories, pulse shape,
    and measurement times for quantum system time evolution. It provides
    a standardized format for time evolution data and enables unified visualization.

    Attributes:
        times: 1D array of time points
        probabilities: Dictionary mapping state names to probability arrays:
            - For single qubit: 'prob_0', 'prob_1'
            - For two qubits: 'prob_00', 'prob_01', 'prob_10', 'prob_11'
        pulse_shape: Optional 1D array of pulse envelope values
        measurement_times: Optional list/array of measurement time points
        metadata: Optional dictionary for additional information (system params, etc.)

    Example:
        >>> evolution = TimeEvolutionResults(
        ...     times=times,
        ...     probabilities={'prob_0': p0_array, 'prob_1': p1_array},
        ...     pulse_shape=pulse,
        ...     measurement_times=[-5.0, 5.0]
        ... )
        >>> print(evolution)
        >>> plot_time_evolution(evolution)
        >>> save_results(evolution, 'evolution_data.npz')
    """

    times: np.ndarray
    probabilities: Dict[str, np.ndarray]
    pulse_shape: Optional[np.ndarray] = None
    measurement_times: Optional[Union[List[float], np.ndarray]] = None
    metadata: Dict[str, Union[float, str, np.ndarray, List]] = field(default_factory=dict)

    def __str__(self) -> str:
        """Return a human-readable string representation of the time evolution results."""
        lines = ["TimeEvolutionResults:"]
        lines.append(f"  Time interval: [{self.times.min():.3g}, {self.times.max():.3g}]")
        lines.append(f"  Number of time points: {len(self.times)}")

        # Detect system type
        prob_keys = sorted(self.probabilities.keys())
        is_two_qubit = any("prob_" in k and len(k) == 7 for k in prob_keys)
        system_type = "Two-qubit" if is_two_qubit else "Single-qubit"
        lines.append(f"  System type: {system_type}")
        lines.append(f"  Available probabilities: {', '.join(prob_keys)}")

        # Pulse and measurements info
        lines.append(
            f"  Pulse shape: {'Available' if self.pulse_shape is not None else 'Not available'}"
        )
        if self.measurement_times is not None:
            meas_times = np.asarray(self.measurement_times)
            lines.append(f"  Measurement times: {list(meas_times)}")
        else:
            lines.append(f"  Measurement times: Not specified")

        return "\n".join(lines)

    def __repr__(self) -> str:
        """Return a detailed string representation for debugging."""
        return (
            f"TimeEvolutionResults(times=array({len(self.times)} points), "
            f"probabilities={list(self.probabilities.keys())}, "
            f"pulse={'available' if self.pulse_shape is not None else 'None'})"
        )


@dataclass
class SweepResults:
    """
    Container for parameter sweep results.

    This class stores the parameters swept, their scales, and all computed results
    (probabilities, contrast, detection maps). It provides a standardized format
    for all sweep functions and enables unified visualization.

    Attributes:
        param1_name: Name of first sweep parameter (e.g., 'chi', 'gamma', 'asymmetry')
        param1_vals: 1D array of first parameter values
        param1_scale: Scale for first parameter ('linear' or 'log')
        param2_name: Name of second sweep parameter
        param2_vals: 1D array of second parameter values
        param2_scale: Scale for second parameter ('linear' or 'log')
        results: Dictionary mapping result names to 2D arrays:
            - For single qubit: 'contrast_map', 'detection_map', 'detection_without_map'
            - For two qubits: 'p00', 'p01', 'p10', 'p11', plus optional contrast/detection maps
        metadata: Optional dictionary for additional information (optimal points, etc.)

    Example:
        >>> sweep = SweepResults(
        ...     param1_name='gamma', param1_vals=gamma_vals, param1_scale='linear',
        ...     param2_name='chi', param2_vals=chi_vals, param2_scale='linear',
        ...     results={'contrast_map': contrast, 'detection_map': detection}
        ... )
        >>> print(sweep)
        >>> plot_sweep_results(sweep)
        >>> save_results(sweep, 'sweep_data.npz')
    """

    param1_name: str
    param1_vals: np.ndarray
    param1_scale: str
    param2_name: str
    param2_vals: np.ndarray
    param2_scale: str
    results: Dict[str, np.ndarray]
    metadata: Dict[str, Union[float, str, np.ndarray, List]] = field(default_factory=dict)

    def __str__(self) -> str:
        """Return a human-readable string representation of the sweep results."""
        lines = ["SweepResults:"]
        lines.append(f"  Parameter 1: {self.param1_name}")
        lines.append(f"    Interval: [{self.param1_vals.min():.3g}, {self.param1_vals.max():.3g}]")
        lines.append(f"    Scale: {self.param1_scale}")
        lines.append(f"    Resolution: {len(self.param1_vals)} points")
        lines.append(f"  Parameter 2: {self.param2_name}")
        lines.append(f"    Interval: [{self.param2_vals.min():.3g}, {self.param2_vals.max():.3g}]")
        lines.append(f"    Scale: {self.param2_scale}")
        lines.append(f"    Resolution: {len(self.param2_vals)} points")
        lines.append(f"  Available results: {', '.join(sorted(self.results.keys()))}")

        # Show optimal point if available in metadata
        if "optimal_chi" in self.metadata and "optimal_gamma" in self.metadata:
            lines.append(
                f"  Optimal point: {self.param2_name}={self.metadata['optimal_chi']:.3g}, "
                f"{self.param1_name}={self.metadata['optimal_gamma']:.3g}"
            )

        return "\n".join(lines)

    def __repr__(self) -> str:
        """Return a detailed string representation for debugging."""
        return (
            f"SweepResults(param1_name='{self.param1_name}', param2_name='{self.param2_name}', "
            f"shape=({len(self.param2_vals)}, {len(self.param1_vals)}), "
            f"results={list(self.results.keys())})"
        )


def save_results(
    results: Union[TimeEvolutionResults, SweepResults],
    filepath: Union[str, Path],
    compress: bool = True,
) -> None:
    """
    Save results to file in npz format.

    Saves all arrays and metadata in numpy's npz format, which efficiently stores
    multiple arrays with compression. Metadata is serialized to JSON string.

    Args:
        results: TimeEvolutionResults or SweepResults object to save
        filepath: Path where to save the file (will add .npz extension if missing)
        compress: Whether to compress the file (default: True)

    Example:
        >>> save_results(sweep_results, 'sweep_chi_gamma.npz')
        >>> save_results(time_evolution, 'evolution.npz', compress=False)
    """
    filepath = Path(filepath)
    if not filepath.suffix:
        filepath = filepath.with_suffix(".npz")

    # Prepare data dictionary
    data = {}

    if isinstance(results, TimeEvolutionResults):
        data["_type"] = np.array(["TimeEvolutionResults"], dtype="U")
        data["times"] = results.times

        # Save probabilities with prefixed keys
        for key, val in results.probabilities.items():
            data[f"prob_{key}"] = val

        if results.pulse_shape is not None:
            data["pulse_shape"] = results.pulse_shape

        if results.measurement_times is not None:
            data["measurement_times"] = np.asarray(results.measurement_times)

        # Save metadata as JSON string
        data["metadata"] = np.array([json.dumps(results.metadata)], dtype="U")

    elif isinstance(results, SweepResults):
        data["_type"] = np.array(["SweepResults"], dtype="U")
        data["param1_name"] = np.array([results.param1_name], dtype="U")
        data["param1_vals"] = results.param1_vals
        data["param1_scale"] = np.array([results.param1_scale], dtype="U")
        data["param2_name"] = np.array([results.param2_name], dtype="U")
        data["param2_vals"] = results.param2_vals
        data["param2_scale"] = np.array([results.param2_scale], dtype="U")

        # Save results with prefixed keys
        for key, val in results.results.items():
            data[f"result_{key}"] = val

        # Save metadata as JSON string
        data["metadata"] = np.array([json.dumps(results.metadata)], dtype="U")

    else:
        raise TypeError(f"Expected TimeEvolutionResults or SweepResults, got {type(results)}")

    # Save with or without compression
    if compress:
        np.savez_compressed(filepath, **data)
    else:
        np.savez(filepath, **data)


def load_results(filepath: Union[str, Path]) -> Union[TimeEvolutionResults, SweepResults]:
    """
    Load results from npz file.

    Loads results previously saved with save_results() and reconstructs the
    appropriate dataclass (TimeEvolutionResults or SweepResults).

    Args:
        filepath: Path to the npz file

    Returns:
        TimeEvolutionResults or SweepResults object

    Example:
        >>> results = load_results('sweep_chi_gamma.npz')
        >>> print(results)
        >>> plot_sweep_results(results)
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    # Load data
    with np.load(filepath, allow_pickle=True) as data:
        result_type = str(data["_type"][0])

        if result_type == "TimeEvolutionResults":
            # Extract times
            times = data["times"]

            # Extract probabilities
            probabilities = {}
            for key in data.files:
                if key.startswith("prob_"):
                    prob_key = key[5:]  # Remove 'prob_' prefix
                    probabilities[prob_key] = data[key]

            # Extract optional fields
            pulse_shape = data["pulse_shape"] if "pulse_shape" in data else None
            measurement_times = data["measurement_times"] if "measurement_times" in data else None

            # Extract metadata
            metadata = json.loads(str(data["metadata"][0])) if "metadata" in data else {}

            return TimeEvolutionResults(
                times=times,
                probabilities=probabilities,
                pulse_shape=pulse_shape,
                measurement_times=measurement_times,
                metadata=metadata,
            )

        elif result_type == "SweepResults":
            # Extract parameters
            param1_name = str(data["param1_name"][0])
            param1_vals = data["param1_vals"]
            param1_scale = str(data["param1_scale"][0])
            param2_name = str(data["param2_name"][0])
            param2_vals = data["param2_vals"]
            param2_scale = str(data["param2_scale"][0])

            # Extract results
            results = {}
            for key in data.files:
                if key.startswith("result_"):
                    result_key = key[7:]  # Remove 'result_' prefix
                    results[result_key] = data[key]

            # Extract metadata
            metadata = json.loads(str(data["metadata"][0])) if "metadata" in data else {}

            return SweepResults(
                param1_name=param1_name,
                param1_vals=param1_vals,
                param1_scale=param1_scale,
                param2_name=param2_name,
                param2_vals=param2_vals,
                param2_scale=param2_scale,
                results=results,
                metadata=metadata,
            )

        else:
            raise ValueError(f"Unknown result type: {result_type}")
