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
        cavity_population: Optional 1D array of cavity population <a†a> values
        field_population: Optional 1D array of external field population <a_in†a_in> values
        metadata: Optional dictionary for additional information (system params, etc.)

    Example:
        >>> evolution = TimeEvolutionResults(
        ...     times=times,
        ...     probabilities={'prob_0': p0_array, 'prob_1': p1_array},
        ...     pulse_shape=pulse,
        ...     measurement_times=[-5.0, 5.0],
        ...     cavity_population=cavity_pop_array
        ... )
        >>> print(evolution)
        >>> plot_time_evolution(evolution, show_cavity_population=True)
        >>> save_results(evolution, 'evolution_data.npz')
    """

    times: np.ndarray
    probabilities: Optional[Dict[str, np.ndarray]]
    pulse_shape: Optional[np.ndarray] = None
    measurement_times: Optional[Union[List[float], np.ndarray]] = None
    cavity_population: Optional[np.ndarray] = None
    field_population: Optional[np.ndarray] = None
    metadata: Dict[str, Union[float, str, np.ndarray, List]] = field(default_factory=dict)

    def __str__(self) -> str:
        """Return a human-readable string representation of the time evolution results."""
        lines = ["TimeEvolutionResults:"]
        lines.append(f"  Time interval: [{self.times.min():.3g}, {self.times.max():.3g}]")
        lines.append(f"  Number of time points: {len(self.times)}")

        # Infer n_qubits from metadata or from probability keys
        n_qubits = self.metadata.get("n_qubits")
        if n_qubits is None and self.probabilities:
            # Infer from the length of the binary suffix of the first probability key
            # e.g. "prob_0" -> 1 qubit, "prob_00" -> 2 qubits
            first_key = next(iter(self.probabilities))
            suffix = first_key.split("_")[-1]
            if suffix.isdigit() and all(c in "01" for c in suffix):
                n_qubits = len(suffix)
            else:
                n_qubits = 1

        detection_criterion = self.metadata.get("detection_criterion", "N/A")

        # Detect system type
        if n_qubits == 1:
            system_type = "Single-qubit"
        elif n_qubits == 2:
            system_type = "Two-qubit"
        else:
            system_type = f"{n_qubits}-qubit"
        lines.append(f"  System type: {system_type}")
        lines.append(f"  Detection Criterion: {detection_criterion}")

        # List probability keys
        if self.probabilities:
            lines.append(f"  Probabilities: {list(self.probabilities.keys())}")

        # Pulse and measurements info
        lines.append(
            f"  Pulse shape: {'Available' if self.pulse_shape is not None else 'Not available'}"
        )
        if self.measurement_times is not None:
            meas_times = list(np.asarray(self.measurement_times))
            lines.append(f"  Measurement times: {[ float(i) for i in meas_times]}")
        else:
            lines.append(f"  Measurement times: Not specified")

        lines.append(
            f"  Cavity population: {'Available' if self.cavity_population is not None else 'Not available'}"
        )
        lines.append(
            f"  Field population: {'Available' if self.field_population is not None else 'Not available'}"
        )

        return "\n".join(lines)

    def __repr__(self) -> str:
        """Return a detailed string representation for debugging."""
        return (
            f"TimeEvolutionResults(times=array({len(self.times)} points), "
            f"DetectionCriterion={self.probabilities.keys()}, "
            f"pulse={'available' if self.pulse_shape is not None else 'None'})"
        )


@dataclass
class SweepResults:
    """
    Container for parameter sweep results.

    This class stores the parameters swept, their scales, and all computed results
    (probabilities, contrast, detection maps). It provides a standardized format
    for all sweep functions and enables unified visualization.

    Axes are stored as parallel lists (one entry per swept parameter).

    Attributes:
        axis_names: List of swept-parameter names, one per axis.
        axis_vals: List of 1D value arrays, one per axis.
        axis_scales: List of scales ('linear' or 'log'), one per axis.
        results: Dict mapping each scalar output name ('metric', 'detection_<config>' per
            configuration, 'validation') to an N-D array of that value over the whole grid.
        metadata: Optional dict for extra info (optimal points, etc.).
    """

    axis_names: List[str]
    axis_vals: List[np.ndarray]
    axis_scales: List[str]
    results: Dict[str, np.ndarray]
    metadata: Dict[str, Union[float, str, np.ndarray, List]] = field(default_factory=dict)

    @property
    def ndim(self) -> int:
        return len(self.axis_names)

    @property
    def shape(self) -> tuple:
        return tuple(len(v) for v in self.axis_vals)

    def __str__(self) -> str:
        """Return a human-readable string representation of the sweep results."""
        lines = ["SweepResults:"]
        for i, (name, vals, scale) in enumerate(zip(self.axis_names, self.axis_vals, self.axis_scales)):
            v = np.asarray(vals)
            lines.append(f"  Axis {i} ({name}): [{v.min():.3g}, {v.max():.3g}], {scale}, {len(v)} points")
        lines.append(f"  Available results: {', '.join(sorted(self.results.keys()))}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        """Return a detailed string representation for debugging."""
        return (
            f"SweepResults(\n"
            f"  axis_names={self.axis_names},\n"
            f"  shape={self.shape},\n"
            f"  results={list(self.results.keys())},\n"
            f"  metadata={list(self.metadata.keys())},\n"
            ")"
        )


def _json_default_serializer(obj):
    """JSON fallback for numpy types in metadata (arrays -> lists, scalars -> Python numbers)."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


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
        data["metadata"] = np.array(
            [json.dumps(results.metadata, default=_json_default_serializer)], dtype="U"
        )

    elif isinstance(results, SweepResults):
        data["_type"] = np.array(["SweepResults"], dtype="U")
        data["axis_names"] = np.array(results.axis_names, dtype="U")
        data["axis_scales"] = np.array(results.axis_scales, dtype="U")
        # axes may have different lengths, so store each one separately
        for i, vals in enumerate(results.axis_vals):
            data[f"axis_vals_{i}"] = np.asarray(vals)

        # Save results with prefixed keys
        for key, val in results.results.items():
            data[f"result_{key}"] = val

        # Save metadata as JSON string
        data["metadata"] = np.array(
            [json.dumps(results.metadata, default=_json_default_serializer)], dtype="U"
        )

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
    with np.load(filepath, allow_pickle=True) as raw:
        data: dict = dict(raw)  # convert NpzFile to dict for type safety
    result_type = str(data["_type"][0])

    if result_type == "TimeEvolutionResults":
        # Extract times
        times = data["times"]

        # Extract probabilities
        probabilities = {}
        for key in data.keys():
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
        # Extract axes (one axis_vals_<i> array per axis)
        axis_names = [str(n) for n in data["axis_names"]]
        axis_scales = [str(s) for s in data["axis_scales"]]
        axis_vals = [data[f"axis_vals_{i}"] for i in range(len(axis_names))]

        # Extract results
        results = {}
        for key in data.keys():
            if key.startswith("result_"):
                result_key = key[7:]  # Remove 'result_' prefix
                results[result_key] = data[key]

        # Extract metadata
        metadata = json.loads(str(data["metadata"][0])) if "metadata" in data else {}

        return SweepResults(
            axis_names=axis_names,
            axis_vals=axis_vals,
            axis_scales=axis_scales,
            results=results,
            metadata=metadata,
        )

    else:
        raise ValueError(f"Unknown result type: {result_type}")
