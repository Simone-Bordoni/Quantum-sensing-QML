"""
Visualization utilities for quantum sensing optimization results.

This module provides functions for creating comprehensive optimization dashboards
that display key data including the detection metric, gradient evolution,
parameter trajectories, and detection measures.
"""

import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from qsopt.core.callback import OptimizationCallback
from qsopt.core.experimental_parameters import ExperimentalParameters
from qsopt.utils.results import SweepResults, TimeEvolutionResults


def plot_optimization_dashboard(
    optimization_callback: OptimizationCallback,
    reference_callback: Optional[OptimizationCallback] = None,
    show_metric: bool = True,
    show_gradients: bool = True,
    show_parameters: bool = True,
    show_trajectory: bool = True,
    show_detection_measures: bool = True,
    show_confusion_matrix_summary: bool = False,
    figsize: Tuple[int, int] = (16, 18),
    save_path: Optional[str] = None,
    dpi: int = 300,
) -> Figure:
    """
    Create a comprehensive optimization dashboard with multiple subplots.

    This function generates a multi-panel visualization showing:
    - Metric evolution (with optional reference benchmark)
    - Gradient magnitude evolution (log scale)
    - Parameter evolution over epochs (initial and final circuit parameters)
    - Detection measures (with and without photon)

    Args:
        optimization_callback: OptimizationCallback from ``optimize_rotations()``
            Contains history of epochs, metric values, detection measures, and parameters
        reference_callback: Optional SimulationCallback from ``run_simulation()``
            If provided, reference values are shown as horizontal benchmark lines
        show_metric: Display detection metric evolution plot when True
        show_gradients: Display gradient magnitude evolution plot when True
        show_parameters: Display parameter evolution plot when True
        show_detection_measures: Display detection measures plot
        show_confusion_matrix_summary: Display confusion matrix and protocol/state summary
            when True and callback values are available
        figsize: Figure size as ``(width, height)`` in inches (default: 16x18)
        save_path: Optional path to save the figure (e.g., ``'dashboard.pdf'``)
            If None, figure is displayed but not saved
        dpi: Resolution for saved figure (default: 300)

    Returns:
        matplotlib Figure object containing the dashboard

    Example:
        >>> # Basic usage with optimization only
        >>> history = experiment.optimize_rotations(theta_init=[1.5, -1.3], num_steps=50)
        >>> fig = plot_optimization_dashboard(history)
        >>>
        >>> # With reference comparison
        >>> results = experiment.run_simulation()
        >>> history = experiment.optimize_rotations(theta_init=[1.5, -1.3], num_steps=50)
        >>> fig = plot_optimization_dashboard(history, reference_callback=results,
        ...                                   save_path='opt_dashboard.pdf')
        >>>
        >>> # Selective plotting
        >>> fig = plot_optimization_dashboard(history,
        ...                                   show_gradients=False,
        ...                                   show_detection_measures=False)
    """
    # Confusion-matrix summary panel is shown only if explicitly requested and the callback carries all
    # three plotted artifacts: confusion_matrix ((true, predicted) -> detection prob/rate) as a heatmap,
    # states_map (config -> classified states) and false_signal (per config) as text summaries.
    has_detection_protocol = bool(getattr(optimization_callback, "states_map", None))
    has_confusion_values = any(
        float(value) > 0.0
        for value in (getattr(optimization_callback, "confusion_matrix", {}) or {}).values()
    )
    has_false_signal = bool(getattr(optimization_callback, "false_signal", None))
    show_confusion_summary_panel = show_confusion_matrix_summary and (
        has_detection_protocol and has_confusion_values and has_false_signal
    )

    # Count active plots to determine layout
    active_plots = [
        show_metric,
        show_gradients,
        show_trajectory,
        show_parameters,
        show_detection_measures,
        show_confusion_summary_panel,
    ]
    n_plots = sum(active_plots)

    if n_plots == 0:
        raise ValueError("At least one plot type must be enabled")

    # Determine grid layout
    if n_plots <= 2:
        n_rows, n_cols = 1, n_plots
    elif n_plots <= 4:
        n_rows, n_cols = 2, 2
    else:
        n_rows, n_cols = 3, 2

    # Extract history data
    history = optimization_callback.get_history()
    epochs = np.array(history["epochs"])
    metric_values = np.array(history["metric"])
    validation_values = np.array(history.get("validation", []), dtype=float)

    # Per-configuration detection measures over epochs. detection_dict is a list (one entry
    # per saved epoch) of {configuration_name: detection_measure}; the configuration set is
    # read from the first populated entry. Missing/None values become NaN so they are simply
    # skipped when plotting (e.g. matrix-distance criteria do not report per-config measures).
    detection_history = history.get("detection_dict", []) or []
    detection_config_names: List[str] = []
    for entry in detection_history:
        if entry:
            detection_config_names = list(entry.keys())
            break
    detection_series = {name: [] for name in detection_config_names}
    for entry in detection_history:
        for name in detection_config_names:
            value = entry.get(name) if entry else None
            detection_series[name].append(float(value) if value is not None else np.nan)
    detection_series = {name: np.array(values, dtype=float) for name, values in detection_series.items()}
    has_detection_series = any(np.any(np.isfinite(values)) for values in detection_series.values())

    # Extract parameter arrays from tuple structure
    param_arrays = []
    param_names = []
    n_initial_params = 0
    n_final_params = 0

    if history["trainable_params"]:
        # trainable_params is now tuple[list, list] of (initial_params, final_params)
        first_params_tuple = history["trainable_params"][0]
        initial_params, final_params = first_params_tuple
        n_initial_params = len(initial_params)
        n_final_params = len(final_params)

        # Generate generic parameter names
        param_names = [f"init_{i}" for i in range(n_initial_params)] + [f"final_{i}" for i in range(n_final_params)]

        # Extract all parameter values (flatten tuple to list)
        for initial_params, final_params in history["trainable_params"]:
            flat_params = [float(p) for p in initial_params] + [float(p) for p in final_params]
            param_arrays.append(flat_params)

    param_arrays = np.array(param_arrays)
    n_total_params = len(param_names)

    # Gradient magnitude per epoch. The callback now stores the true optimization gradients
    # (history["grads"], one flat vector per saved epoch), so use their norm directly. Fall
    # back to a finite-difference estimate from the parameter trajectory when gradients are
    # unavailable (e.g. callbacks loaded without grads, or run_simulation reference outputs).
    grad_norms = np.array(
        [
            np.nan if grad is None else float(np.linalg.norm(np.asarray(grad, dtype=float)))
            for grad in history.get("grads", []) or []
        ]
    )
    if grad_norms.size == 0 or not np.any(np.isfinite(grad_norms)):
        gradients = np.zeros_like(param_arrays)
        if len(param_arrays) > 1:
            # Central differences for interior points
            gradients[1:-1] = (param_arrays[2:] - param_arrays[:-2]) / 2
            # Forward difference for first point
            gradients[0] = param_arrays[1] - param_arrays[0]
            # Backward difference for last point
            gradients[-1] = param_arrays[-1] - param_arrays[-2]
        grad_norms = np.linalg.norm(gradients, axis=1) if len(gradients) > 0 else np.array([])

    # Extract reference values if provided
    reference_metric = None
    reference_validation = None
    reference_detection = None
    reference_params = None

    if reference_callback is not None:
        ref_history = reference_callback.get_history()
        if ref_history["metric"]:
            reference_metric = ref_history["metric"][0]
            ref_validation = ref_history.get("validation", [])
            reference_validation = ref_validation[0] if len(ref_validation) else None
            ref_detection = ref_history.get("detection_dict", [])
            reference_detection = ref_detection[0] if ref_detection and ref_detection[0] else None

        if ref_history["trainable_params"]:
            # Extract reference params from tuple structure
            ref_initial, ref_final = ref_history["trainable_params"][0]
            reference_params = [float(p) for p in ref_initial] + [float(p) for p in ref_final]

    # Create figure
    fig = plt.figure(figsize=figsize)
    axes = []
    plot_idx = 0

    # Plot 1: Metric & Validation Evolution
    if show_metric:
        ax = plt.subplot(n_rows, n_cols, plot_idx + 1)
        axes.append(ax)
        plot_idx += 1

        ax.plot(epochs, metric_values, "g-", linewidth=2, alpha=0.8, label="Metric (training)")

        # The validation objective is tracked alongside the training metric; show it on the
        # same axes so the two can be compared directly.
        if validation_values.size == epochs.size and validation_values.size > 0:
            ax.plot(epochs, validation_values, "b-", linewidth=2, alpha=0.8, label="Validation (training)")

        if reference_metric is not None:
            ax.axhline(
                y=reference_metric,
                color="green",
                linestyle="--",
                linewidth=1.8,
                alpha=0.6,
                label="Metric (reference)",
            )
        if reference_validation is not None:
            ax.axhline(
                y=reference_validation,
                color="blue",
                linestyle="--",
                linewidth=1.8,
                alpha=0.6,
                label="Validation (reference)",
            )

        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel("Objective", fontsize=12)
        ax.set_title("Metric & Validation Evolution", fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

    # Plot 2: Gradient Magnitude Evolution
    if show_gradients and len(grad_norms) > 0:
        ax = plt.subplot(n_rows, n_cols, plot_idx + 1)
        axes.append(ax)
        plot_idx += 1

        ax.semilogy(epochs, grad_norms, "m-", linewidth=2, alpha=0.8)
        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel("Gradient Magnitude", fontsize=12)
        ax.set_title("Gradient Evolution (Log Scale)", fontsize=14)
        ax.grid(True, alpha=0.3)

    # Plot 3: Parameter Evolution
    if show_parameters and len(param_arrays) > 0:
        ax = plt.subplot(n_rows, n_cols, plot_idx + 1)
        axes.append(ax)
        plot_idx += 1

        # Use different color schemes for initial vs final circuit parameters
        colors_init = plt.cm.Blues(np.linspace(0.4, 0.9, max(1, n_initial_params)))  # pylint: disable=no-member
        colors_final = plt.cm.Oranges(np.linspace(0.4, 0.9, max(1, n_final_params)))  # pylint: disable=no-member

        for i, name in enumerate(param_names):
            # Choose color based on whether it's initial or final circuit parameter
            if i < n_initial_params:
                color = colors_init[i] if n_initial_params > 1 else colors_init[0]
                linestyle = '-'
            else:
                color = colors_final[i - n_initial_params] if n_final_params > 1 else colors_final[0]
                linestyle = '--'

            params_deg = param_arrays[:, i] * 180 / np.pi
            ax.plot(epochs, params_deg, linestyle, linewidth=2, label=name, color=color, alpha=0.8)

            # Add reference line if available
            if reference_params is not None and i < len(reference_params):
                ref_deg = reference_params[i] * 180 / np.pi
                ax.axhline(y=ref_deg, color=color, linestyle=":", alpha=0.5, linewidth=1.5)

        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel("Rotation Angle (degrees)", fontsize=12)
        ax.set_title(f"Parameter Evolution ({n_total_params} params: {n_initial_params} init, {n_final_params} final)", fontsize=14)
        ax.legend(fontsize=9 if n_total_params > 4 else 10, ncol=2 if n_total_params > 6 else 1)
        ax.grid(True, alpha=0.3)

    # Plot 4: Detection Measures Evolution (one line per configuration)
    if show_detection_measures:
        ax = plt.subplot(n_rows, n_cols, plot_idx + 1)
        axes.append(ax)
        plot_idx += 1

        if has_detection_series:
            cmap = plt.get_cmap("tab10")
            for i, name in enumerate(detection_config_names):
                color = cmap(i % cmap.N)
                values = detection_series[name]
                mask = np.isfinite(values)
                ax.plot(
                    epochs[mask],
                    values[mask],
                    "-",
                    color=color,
                    linewidth=2,
                    alpha=0.85,
                    label=str(name),
                )

                # Reference benchmark for this configuration (matching colour, dashed).
                if reference_detection is not None and reference_detection.get(name) is not None:
                    ax.axhline(
                        y=float(reference_detection[name]),
                        color=color,
                        linestyle="--",
                        linewidth=1.8,
                        alpha=0.5,
                    )

            ax.set_xlabel("Epoch", fontsize=12)
            ax.set_ylabel("Detection Measure", fontsize=12)
            ax.set_title("Detection Measures Evolution", fontsize=14)
            ax.legend(
                fontsize=9 if len(detection_config_names) > 4 else 10,
                ncol=2 if len(detection_config_names) > 6 else 1,
            )
            ax.grid(True, alpha=0.3)
        else:
            # Matrix-distance criteria (e.g. 'max computational distance') do not report
            # per-configuration detection probabilities, so there is nothing to plot.
            ax.text(
                0.5,
                0.5,
                "No per-configuration\ndetection measures recorded",
                ha="center",
                va="center",
                fontsize=12,
                transform=ax.transAxes,
            )
            ax.set_title("Detection Measures Evolution", fontsize=14)
            ax.set_xticks([])
            ax.set_yticks([])

    # Plot 5: Parameter Trajectory
    if show_trajectory and len(param_arrays) >= 2 and n_total_params >= 2:
        ax = plt.subplot(n_rows, n_cols, plot_idx + 1)
        axes.append(ax)
        plot_idx += 1

        # Use first two parameters by default
        param_indices = (0, 1)
        idx1, idx2 = param_indices
        theta1_deg = param_arrays[:, idx1] * 180 / np.pi
        theta2_deg = param_arrays[:, idx2] * 180 / np.pi

        # Plot trajectory
        scatter = ax.scatter(
            theta1_deg,
            theta2_deg,
            c=epochs,
            cmap="viridis",
            s=50,
            alpha=0.7,
            edgecolors="black",
            linewidth=1,
        )
        ax.plot(theta1_deg, theta2_deg, "k-", alpha=0.4, linewidth=1.5)

        # Mark start and end
        ax.plot(
            theta1_deg[0],
            theta2_deg[0],
            "ro",
            markersize=12,
            label="Start",
            markeredgecolor="black",
            markeredgewidth=2,
        )
        ax.plot(
            theta1_deg[-1],
            theta2_deg[-1],
            "gs",
            markersize=12,
            label="End",
            markeredgecolor="black",
            markeredgewidth=2,
        )

        # Mark reference if available
        if reference_params is not None and len(reference_params) > max(param_indices):
            ref_theta1_deg = reference_params[idx1] * 180 / np.pi
            ref_theta2_deg = reference_params[idx2] * 180 / np.pi
            ax.plot(
                ref_theta1_deg,
                ref_theta2_deg,
                "b^",
                markersize=14,
                label="Reference",
                markeredgecolor="black",
                markeredgewidth=2,
            )

        ax.set_xlabel(f"{param_names[idx1]} (degrees)", fontsize=12)
        ax.set_ylabel(f"{param_names[idx2]} (degrees)", fontsize=12)
        ax.set_title("Optimization Trajectory", fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label("Epoch", fontsize=10)

    # Plot 6: Confusion Matrix + Protocol/Probabilities Summary
    if show_confusion_summary_panel:
        ax = plt.subplot(n_rows, n_cols, plot_idx + 1)
        axes.append(ax)
        plot_idx += 1

        confusion_matrix = getattr(optimization_callback, "confusion_matrix", {}) or {}
        states_map = getattr(optimization_callback, "states_map", {}) or {}
        false_signal = getattr(optimization_callback, "false_signal", None)

        # Rows are the true configurations (from the confusion keys, falling back to the states_map);
        # columns are the predicted categories, which may include prediction-only labels (e.g. 'mixed')
        # that are never a true configuration -> shown as a column only.
        row_names: List[str] = []
        for true_name, _ in confusion_matrix:
            if true_name not in row_names:
                row_names.append(true_name)
        for name in states_map.keys():
            if name not in row_names:
                row_names.append(name)
        col_names = list(row_names)
        for _, pred_name in confusion_matrix:
            if pred_name not in col_names:
                col_names.append(pred_name)

        n_rows, n_cols = len(row_names), len(col_names)
        # confusion_matrix[(true, predicted)] = mass; rows are the true config, columns the predicted one.
        cm_values = np.array(
            [[float(confusion_matrix.get((true, pred), 0.0)) for pred in col_names] for true in row_names],
            dtype=float,
        ).reshape(n_rows, n_cols)

        vmax = max(1.0, float(np.max(cm_values))) if cm_values.size else 1.0
        im = ax.imshow(cm_values, cmap="Blues", vmin=0.0, vmax=vmax, aspect="auto")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        ax.set_xticks(range(n_cols))
        ax.set_xticklabels([f"Pred: {name}" for name in col_names], rotation=20, ha="right")
        ax.set_yticks(range(n_rows))
        ax.set_yticklabels([f"True: {name}" for name in row_names])
        ax.set_title("Confusion Matrix", fontsize=14)

        for i in range(n_rows):
            for j in range(n_cols):
                ax.text(
                    j,
                    i,
                    f"{cm_values[i, j]:.3f}",
                    ha="center",
                    va="center",
                    color="black",
                    fontsize=9,
                    fontweight="bold",
                )

        summary_lines = []
        if has_detection_protocol:
            summary_lines.append("Protocol (states per configuration):")
            for name in row_names:
                if name in states_map:
                    summary_lines.append(f"  {name}: {list(states_map[name])}")

        if has_false_signal and isinstance(false_signal, dict):
            summary_lines.append("False signal (joint, per configuration):")
            for name in row_names:
                if name in false_signal:
                    summary_lines.append(f"  {name}: {float(false_signal[name]):.3f}")

        if summary_lines:
            # Position the summary in figure coordinates to the right of the confusion
            # matrix axis so it won't be occluded by the image or colorbar. Use the
            # axis bounding box to choose a sensible location and fall back to the
            # left side if there is no space on the right.
            bbox = ax.get_position()
            # Place summary to the LEFT of the confusion matrix, slightly lower
            # and a bit less left so it doesn't collide with figure elements.
            x_fig = bbox.x0 + 0.055
            y_fig = bbox.y0 + bbox.height * 0.31
            # Clamp to avoid going off-figure
            x_fig = max(0.02, x_fig)

            fig.text(
                x_fig,
                y_fig,
                "\n".join(summary_lines),
                transform=fig.transFigure,
                va="center",
                ha="right",
                fontsize=9,
                family="monospace",
                bbox=dict(boxstyle="round", facecolor="#f8f9fa", alpha=0.95, pad=0.6),
            )

    # Overall title
    plt.suptitle("Optimization Dashboard", fontsize=18)
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.3)

    # Save if path provided
    if save_path is not None:
        # Create directory if it doesn't exist
        save_path_obj = Path(save_path)
        save_path_obj.parent.mkdir(parents=True, exist_ok=True)

        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")

    return fig


def plot_metric_evolution(
    optimization_callback: OptimizationCallback,
    reference_callback: Optional[OptimizationCallback] = None,
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None,
    dpi: int = 300,
) -> Figure:
    """
    Create a standalone plot of metric evolution.

    Args:
        optimization_callback: OptimizationCallback from ``optimize_rotations()``
        reference_callback: Optional reference output from ``run_simulation()``
        figsize: Figure size as ``(width, height)`` in inches
        save_path: Optional path to save the figure
        dpi: Resolution (dots-per-inch) for saved figure

    Returns:
        matplotlib Figure object
    """
    history = optimization_callback.get_history()
    epochs = np.array(history["epochs"])
    metric_values = np.array(history["metric"])

    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(
        epochs,
        metric_values,
        "g-",
        linewidth=2.5,
        alpha=0.8,
        label="Optimized",
        marker="o",
        markersize=4,
    )

    if reference_callback is not None:
        ref_history = reference_callback.get_history()
        if ref_history["metric"]:
            reference_metric = ref_history["metric"][0]
            ax.axhline(
                y=reference_metric,
                color="red",
                linestyle="--",
                linewidth=2,
                alpha=0.7,
                label="Reference",
            )

    ax.set_xlabel("Epoch", fontsize=14)
    ax.set_ylabel("Metric", fontsize=14)
    ax.set_title("Metric Evolution", fontsize=16)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path is not None:
        # Create directory if it doesn't exist
        save_path_obj = Path(save_path)
        save_path_obj.parent.mkdir(parents=True, exist_ok=True)

        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")

    return fig


def plot_parameter_trajectory(
    optimization_callback: OptimizationCallback,
    reference_callback: Optional[OptimizationCallback] = None,
    param_indices: Tuple[int, int] = (0, 1),
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[str] = None,
    dpi: int = 300,
) -> Figure:
    """
    Create a standalone plot of optimization trajectory in parameter space.

    Args:
        optimization_callback: OptimizationCallback returned by ``optimize_rotations()``
        reference_callback: Optional reference callback from ``run_simulation()``
        param_indices: Pair of parameter indices to display (default: ``(0, 1)``)
        figsize: Figure size as ``(width, height)`` in inches
        save_path: Optional path to save the figure
        dpi: Resolution (dots-per-inch) for saved figure

    Returns:
        matplotlib Figure object
    """
    history = optimization_callback.get_history()
    epochs = np.array(history["epochs"])

    # Extract parameters from tuple structure
    param_arrays = []
    param_names = []
    if history["trainable_params"]:
        # trainable_params is now tuple[list, list] of (initial_params, final_params)
        first_params_tuple = history["trainable_params"][0]
        initial_params, final_params = first_params_tuple
        n_initial = len(initial_params)
        n_final = len(final_params)

        # Generate generic parameter names
        param_names = [f"init_{i}" for i in range(n_initial)] + [f"final_{i}" for i in range(n_final)]

        # Extract all parameter values (flatten tuple to list)
        for initial_params, final_params in history["trainable_params"]:
            flat_params = [float(p) for p in initial_params] + [float(p) for p in final_params]
            param_arrays.append(flat_params)

    param_arrays = np.array(param_arrays)

    if len(param_arrays) == 0 or len(param_names) < 2:
        raise ValueError("Need at least 2 parameters for trajectory plot")

    idx1, idx2 = param_indices
    theta1_deg = param_arrays[:, idx1] * 180 / np.pi
    theta2_deg = param_arrays[:, idx2] * 180 / np.pi

    fig, ax = plt.subplots(figsize=figsize)

    # Plot trajectory
    scatter = ax.scatter(
        theta1_deg,
        theta2_deg,
        c=epochs,
        cmap="viridis",
        s=50,
        alpha=0.7,
        edgecolors="black",
        linewidth=1,
    )
    ax.plot(theta1_deg, theta2_deg, "k-", alpha=0.4, linewidth=1.5)

    # Mark start and end
    ax.plot(
        theta1_deg[0],
        theta2_deg[0],
        "ro",
        markersize=12,
        label="Start",
        markeredgecolor="black",
        markeredgewidth=2,
    )
    ax.plot(
        theta1_deg[-1],
        theta2_deg[-1],
        "gs",
        markersize=12,
        label="End",
        markeredgecolor="black",
        markeredgewidth=2,
    )

    # Mark reference if available
    if reference_callback is not None:
        ref_history = reference_callback.get_history()
        if ref_history["trainable_params"]:
            # Extract reference params from tuple structure
            ref_initial, ref_final = ref_history["trainable_params"][0]
            ref_params = [float(p) for p in ref_initial] + [float(p) for p in ref_final]
            ref_theta1_deg = ref_params[idx1] * 180 / np.pi
            ref_theta2_deg = ref_params[idx2] * 180 / np.pi
            ax.plot(
                ref_theta1_deg,
                ref_theta2_deg,
                "b^",
                markersize=14,
                label="Reference",
                markeredgecolor="black",
                markeredgewidth=2,
            )

    ax.set_xlabel(f"{param_names[idx1]} (degrees)", fontsize=14)
    ax.set_ylabel(f"{param_names[idx2]} (degrees)", fontsize=14)
    ax.set_title("Optimization Trajectory", fontsize=16)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)

    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Epoch", fontsize=12)

    plt.tight_layout()

    if save_path is not None:
        # Create directory if it doesn't exist
        save_path_obj = Path(save_path)
        save_path_obj.parent.mkdir(parents=True, exist_ok=True)

        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"Trajectory plot saved to: {save_path}")

    return fig


def plot_pulse_shape_with_measurements(
    exp_params: "ExperimentalParameters",
    save_path: Optional[str] = None,
    dpi: int = 300,
    batch_size: int = 1,
) -> Figure:
    """
    Plot Gaussian pulse envelope with measurement time markers.

    Creates a visualization showing the temporal shape of the Gaussian input pulse
    along with vertical markers indicating when measurements are performed. This helps
    understand the relationship between the pulse envelope and the measurement protocol.

    Args:
        exp_params: ExperimentalParameters object containing system configuration
        save_path: Optional path to save the figure
        dpi: Resolution for saved figure (default: 300)
        batch_size: Number of measurement realizations to visualize. If > 1,
            measurement times are drawn using distinct colors for each batch
            (default: 1)

    Returns:
        matplotlib.figure.Figure: Figure object containing the plot

    Example:
    >>> from qsopt.core.experimental_parameters import ExperimentalParameters
    >>> exp_params = ExperimentalParameters(...)
    >>> fig = plot_pulse_shape_with_measurements(exp_params)
    >>> fig = plot_pulse_shape_with_measurements(exp_params, batch_size=5)
        >>> # Plot shows pulse shape with measurement markers

    Note:
        - Pulse shape is computed using the u0() function from quantum_utils
        - Measurement times are extracted from exp_params.measurement
        - The plot window extends beyond the measurement range to show pulse decay
    """
    from ..core.experiment.quantum_utils import u0

    # Extract measurement protocol information
    initial_time = exp_params.measurement.initial_time
    final_time = exp_params.measurement.final_time
    interval = exp_params.measurement.time_interval

    # Generate measurement times using ExperimentalParameters helper
    measurement_times = exp_params.get_measurement_times_with_uncertainty(batch_size)
    if measurement_times.ndim == 1:
        measurement_sequences = [measurement_times]
    else:
        measurement_sequences = [measurement_times[i] for i in range(measurement_times.shape[0])]
    n_measurements = len(measurement_sequences[0]) if measurement_sequences else 0

    # Create time array for plotting pulse (extend beyond measurement range)
    time_range = final_time - initial_time
    t_plot = np.linspace(initial_time - 0.3 * time_range, final_time + 0.3 * time_range, 1000)

    # Compute pulse envelope using u0
    sigma = exp_params.inverse_pulse_width
    pulse_vals = np.array([float(u0(t, sigma=sigma)) for t in t_plot])

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot pulse envelope
    ax.plot(t_plot, pulse_vals, "b-", linewidth=2, label="Gaussian pulse envelope")

    # Add vertical lines for measurement times (support multiple realizations)
    cmap = plt.get_cmap("tab10")
    for seq_idx, times in enumerate(measurement_sequences):
        color = cmap(seq_idx % cmap.N)
        label_prefix = "Measurement times" if batch_size == 1 else f"Realization {seq_idx + 1}"
        for line_idx, t_meas in enumerate(times):
            ax.axvline(
                t_meas,
                color=color,
                linestyle="--",
                alpha=0.6,
                linewidth=1.5,
                label=label_prefix if line_idx == 0 else None,
            )

    # Shade the measurement region
    ax.axvspan(initial_time, final_time, alpha=0.1, color="green", label="Measurement window")

    # Formatting
    ax.set_xlabel("Time", fontsize=12)
    ax.set_ylabel("Pulse amplitude |u₀(t)|", fontsize=12)
    ax.set_title("Gaussian Pulse Shape with Measurement Protocol", fontsize=14, pad=15)
    ax.legend(loc="upper right", fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle=":", linewidth=0.5)
    ax.set_ylim(0.0, 1.1)

    # Add system information
    system_info = f"""PULSE AND MEASUREMENT CONFIGURATION

Physical Parameters:
  • Pulse width (σ):        {1/exp_params.inverse_pulse_width:.6f}
  • Uncertainty:    {exp_params.measurement.initial_time_uncertainty:.6f}

Measurement Protocol:
  • Initial time:       {initial_time:.6f}
  • Final time:         {final_time:.6f}
  • Time interval:      {interval:.6f}
  • Total duration:     {time_range:.6f}
    • Number of measurements:  {n_measurements}
    • Batch visualizations:    {len(measurement_sequences)}
"""

    # Add text box with system info
    fig.text(
        0.05,
        0.01,
        system_info,
        fontsize=9,
        family="monospace",
        verticalalignment="bottom",
        bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.7, pad=0.8),
    )

    plt.tight_layout()

    # Save figure if path provided
    if save_path:
        save_path_obj = Path(save_path)
        save_path_obj.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"Pulse shape plot saved to: {save_path}")

    return fig


def plot_time_evolution(
    evolution_data: Optional[Union["TimeEvolutionResults", Dict[str, np.ndarray]]] = None,
    times: Optional[np.ndarray] = None,
    probabilities: Optional[Dict[str, np.ndarray]] = None,
    pulse_shape: Optional[np.ndarray] = None,
    measurement_times: Optional[Union[List[float], np.ndarray]] = None,
    cavity_population: Optional[np.ndarray] = None,
    field_population: Optional[np.ndarray] = None,
    title: Optional[str] = None,
    selected_states: Optional[List[str]] = None,
    show_pulse: bool = True,
    show_measurements: bool = True,
    show_cavity_population: bool = False,
    show_field_population: bool = False,
    normalize_detection: bool = False,
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None,
    dpi: int = 300,
) -> Figure:
    """
    Unified time evolution plotting for single and two-qubit systems.

    This function can be called in three ways:
    1. With TimeEvolutionResults object (recommended, output from time_evolution() method)
    2. With evolution_data dict for backward compatibility
    3. With individual times and probabilities arrays for custom plotting

    Automatically detects single vs two-qubit based on probability keys.
    Allows selective plotting of states and optional pulse shape and measurement markers.

    Args:
        evolution_data: TimeEvolutionResults object or dict from time_evolution() containing:
            - times/['times']: Time points array
            - probabilities/['prob_0', 'prob_1']: For single qubit
            - probabilities/['prob_00', 'prob_01', 'prob_10', 'prob_11']: For two qubits
            - pulse_shape/['pulse_shape']: Optional pulse envelope
            - measurement_times/['measurement_times']: Optional measurement times
            - cavity_population: Optional cavity population array
        times: Time points array (alternative to evolution_data)
        probabilities: Dict with probability arrays (alternative to evolution_data). Keys:
            - Single qubit: 'prob_0', 'prob_1'
            - Two qubits: 'prob_00', 'prob_01', 'prob_10', 'prob_11'
        pulse_shape: Optional pulse envelope array (same length as times)
        measurement_times: Optional list/array of measurement time points for vertical markers
        cavity_population: Optional array of cavity population values <a†a>
        field_population: Optional array of external field population values <a_in†a_in>
        title: Plot title. Auto-generated if None.
        selected_states: Optional list of state keys to plot (e.g., ['prob_00', 'prob_11']).
            If None, all available states are plotted.
        show_pulse: Whether to show pulse shape as filled area. Default: True
        show_measurements: Whether to show measurement time markers. Default: True
        show_cavity_population: Whether to show cavity population <a†a> on the same y-axis. Default: False
        show_field_population: Whether to show field population <a_in†a_in> on the same y-axis. Default: False
        normalize_detection: Whether to min-max normalize detection series to [0, 1] for plotting. Default: False
        figsize: Figure size (width, height) in inches
        save_path: Optional path to save figure
        dpi: Resolution for saved figure

    Returns:
        matplotlib Figure object

    Examples:
        >>> # Using TimeEvolutionResults (recommended)
        >>> results = experiment.time_evolution(t_start=-5, t_end=5)
        >>> print(results)  # Shows available plot options
        >>> fig = plot_time_evolution(results)

        >>> # With cavity population on secondary y-axis
        >>> fig = plot_time_evolution(results, show_cavity_population=True)

        >>> # Using evolution_data dict (backward compatible)
        >>> results_dict = {'times': times, 'prob_0': p0, 'prob_1': p1}
        >>> fig = plot_time_evolution(evolution_data=results_dict)

        >>> # With measurement markers and cavity population
        >>> fig = plot_time_evolution(
        ...     evolution_data=results,
        ...     measurement_times=experiment.experimental_params.measurement.measurement_times,
        ...     show_cavity_population=True,
        ...     title='Single Qubit Evolution'
        ... )

        >>> # Custom plotting with selected states only
        >>> fig = plot_time_evolution(
        ...     times=results['times'],
        ...     probabilities={'prob_00': results['prob_00'], 'prob_11': results['prob_11']},
        ...     pulse_shape=results['pulse_shape'],
        ...     selected_states=['prob_00', 'prob_11'],
        ...     show_pulse=True
        ... )

        >>> # Two-qubit with cavity population, without pulse shape
        >>> fig = plot_time_evolution(
        ...     evolution_data=results_2q,
        ...     show_pulse=False,
        ...     show_measurements=False,
        ...     show_cavity_population=True
        ... )
    """
    # Parse input: TimeEvolutionResults object or individual arguments
    if evolution_data is not None:
        if not isinstance(evolution_data, TimeEvolutionResults):
            raise TypeError(
                "evolution_data must be a TimeEvolutionResults object. "
                "Dict-like access is no longer supported."
            )

        times = evolution_data.times
        probabilities = evolution_data.probabilities
        if pulse_shape is None:
            pulse_shape = evolution_data.pulse_shape
        if measurement_times is None:
            measurement_times = evolution_data.measurement_times
        if cavity_population is None:
            cavity_population = evolution_data.cavity_population
        if field_population is None:
            field_population = evolution_data.field_population
    elif times is None or probabilities is None:
        raise ValueError("Must provide either evolution_data or both times and probabilities")

    # Detect system type based on available probability keys
    available_keys = list(probabilities.keys())
    is_two_qubit = any(
        key in available_keys for key in ["prob_00", "prob_01", "prob_10", "prob_11"]
    )

    # Filter to selected states if specified
    if selected_states is not None:
        probabilities = {k: v for k, v in probabilities.items() if k in selected_states}
        if not probabilities:
            raise ValueError(f"None of selected_states {selected_states} found in probabilities")

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    def _normalize_to_unit_interval(values: np.ndarray) -> np.ndarray:
        values = np.asarray(values)
        v_min = float(np.min(values))
        v_max = float(np.max(values))
        denom = v_max - v_min
        if denom <= 0:
            return np.zeros_like(values)
        return (values - v_min) / denom

    # Plot detection measures (supports legacy probability keys).
    if "detection_measure" in probabilities:
        detection_label = "Detection metric"
        detection_values = probabilities["detection_measure"]
    elif "detection_probability" in probabilities:
        detection_label = "Detection probability"
        detection_values = probabilities["detection_probability"]
    else:
        detection_label = None
        detection_values = None

    if detection_values is not None:
        if normalize_detection:
            detection_values = _normalize_to_unit_interval(detection_values)
            detection_label = f"{detection_label} (normalized to [0, 1])"
        ax.plot(times, detection_values, label=detection_label, linewidth=2, linestyle="-")

    if "nondetection_measure" in probabilities:
        nondetection_values = probabilities["nondetection_measure"]
        nondetection_label = "No-detection metric"
        if normalize_detection:
            nondetection_values = _normalize_to_unit_interval(nondetection_values)
            nondetection_label = f"{nondetection_label} (normalized to [0, 1])"
        ax.plot(times, nondetection_values, label=nondetection_label, linewidth=2, linestyle="--")
    elif "nondetection_probability" in probabilities:
        nondetection_values = probabilities["nondetection_probability"]
        nondetection_label = "No-detection probability"
        if normalize_detection:
            nondetection_values = _normalize_to_unit_interval(nondetection_values)
            nondetection_label = f"{nondetection_label} (normalized to [0, 1])"
        ax.plot(times, nondetection_values, label=nondetection_label, linewidth=2, linestyle="--")

    # Add cavity population on same y-axis if requested
    if show_cavity_population and cavity_population is not None:
        ax.plot(
            times,
            cavity_population,
            color="purple",
            linewidth=2,
            linestyle="-",
            alpha=0.7,
            label=r"Cavity $\langle a^\dagger a \rangle$",
        )

    # Add field population on same y-axis if requested
    if show_field_population and field_population is not None:
        ax.plot(
            times,
            field_population,
            color="brown",
            linewidth=2,
            linestyle="-.",
            alpha=0.7,
            label=r"Field $\langle a_{\mathrm{in}}^\dagger a_{\mathrm{in}} \rangle$",
        )

    # Add pulse shape if provided and requested
    if show_pulse and pulse_shape is not None:
        ax.fill_between(times, 0, pulse_shape, alpha=0.2, label="Pulse shape", color="gray")

    # Add measurement time markers if provided and requested
    if show_measurements and measurement_times is not None:
        measurement_times = np.asarray(measurement_times)

        for i, t_meas in enumerate(measurement_times):
            # Skip if measurement time is outside plot range
            if t_meas < times.min() or t_meas > times.max():
                continue

            ax.axvline(
                t_meas,
                color="red",
                linestyle="--",
                alpha=0.5,
                linewidth=1.5,
                label="Measurement times" if i == 0 else "",
            )

    # Auto-generate title if not provided
    if title is None:
        if is_two_qubit:
            title = "Two-Qubit State Evolution"
        else:
            title = "Single-Qubit State Evolution"

    # Labels and formatting
    ax.set_xlabel("Time", fontsize=12)
    ax.set_ylabel("Population", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    ax.set_ylim(-0.05, 1.05)

    plt.tight_layout()

    # Save if path provided
    if save_path:
        save_path_obj = Path(save_path)
        save_path_obj.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"Time evolution plot saved to: {save_path}")

    return fig


# ---------------------------------------------------------------------------
# Parameter-sweep visualization (N-dimensional SweepResults)
# ---------------------------------------------------------------------------

# Filled/labelled contour levels for probability-like maps (detection measures in [0, 1]).
# 40 uniform fill bands (0.025 step: 0.1 split four ways) so colorbar ticks stay round.
_PROB_FILL_LEVELS = list(np.round(np.linspace(0.0, 1.0, 41), 4))
_PROB_LABEL_LEVELS = [0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 0.99, 1.0]
# Colorbar tick positions for the fixed [0, 1] range: every 0.2.
_PROB_CBAR_TICKS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]


def _pretty_axis(name: str) -> str:
    """Human/greek-friendly axis label.

    Args:
        name: Raw axis name from the sweep.

    Returns:
        str: Label with underscores replaced and common greek names substituted.
    """
    return name.replace("_", " ").replace("gamma", "γ").replace("chi", "χ").replace("Delta", "Δ")


def _axis_index(sweep: "SweepResults", name: str) -> int:
    """Index of a named axis, raising a helpful error if it is unknown.

    Args:
        sweep: The sweep results.
        name: Axis name to look up.

    Returns:
        int: Position of ``name`` in ``sweep.axis_names``.
    """
    if name not in sweep.axis_names:
        raise ValueError(f"Unknown axis '{name}'; available axes: {sweep.axis_names}")
    return sweep.axis_names.index(name)


def _best_index(sweep: "SweepResults") -> List[int]:
    """Grid index of the optimum (metadata ``best_index``, else argmax of the metric map).

    Args:
        sweep: The sweep results.

    Returns:
        List[int]: One index per axis pointing at the best-metric grid point.
    """
    best = sweep.metadata.get("best_index")
    if best is not None:
        return [int(x) for x in best]
    metric = sweep.results.get("metric")
    if metric is None:
        return [0] * sweep.ndim
    return list(int(x) for x in np.unravel_index(int(np.nanargmax(metric)), metric.shape))


def _resolve_fixed_indices(sweep: "SweepResults", fixed: Optional[Dict[str, float]]) -> List[int]:
    """Per-axis indices used to fix hidden axes, defaulting to the optimum.

    Args:
        sweep: The sweep results.
        fixed: Optional {axis_name: value} overrides; each value is snapped to the
            nearest grid point on that axis.

    Returns:
        List[int]: One index per axis (optimum, with the requested overrides applied).
    """
    idx = _best_index(sweep)
    if fixed:
        for name, value in fixed.items():
            k = _axis_index(sweep, name)
            idx[k] = int(np.argmin(np.abs(np.asarray(sweep.axis_vals[k], dtype=float) - value)))
    return idx


def _reduce_to_2d(data: np.ndarray, i: int, j: int, mode: str, fixed_idx: List[int]) -> np.ndarray:
    """Collapse an N-D result array to a 2D map over axes (i, j), indexed [i, j].

    Args:
        data: N-D result array over the whole grid.
        i: Axis mapped to the first (x) output dimension.
        j: Axis mapped to the second (y) output dimension.
        mode: 'slice' fixes hidden axes at ``fixed_idx``; 'max'/'mean' reduce over them.
        fixed_idx: Per-axis indices used when ``mode == 'slice'``.

    Returns:
        np.ndarray: 2D array of shape (len_i, len_j).
    """
    arr = np.moveaxis(data, (i, j), (0, 1))
    others = [k for k in range(data.ndim) if k not in (i, j)]
    if not others:
        return arr
    if mode == "slice":
        return arr[(slice(None), slice(None)) + tuple(int(fixed_idx[k]) for k in others)]
    reducer = np.nanmax if mode == "max" else np.nanmean
    return reducer(arr, axis=tuple(range(2, arr.ndim)))


def _reduce_to_1d(data: np.ndarray, i: int, mode: str, fixed_idx: List[int]) -> np.ndarray:
    """Collapse an N-D result array to a 1D curve over axis i.

    Args:
        data: N-D result array over the whole grid.
        i: Axis kept as the output dimension.
        mode: 'slice' fixes the other axes at ``fixed_idx``; 'max'/'mean' reduce over them.
        fixed_idx: Per-axis indices used when ``mode == 'slice'``.

    Returns:
        np.ndarray: 1D array of length len_i.
    """
    arr = np.moveaxis(data, i, 0)
    others = [k for k in range(data.ndim) if k != i]
    if not others:
        return arr
    if mode == "slice":
        return arr[(slice(None),) + tuple(int(fixed_idx[k]) for k in others)]
    reducer = np.nanmax if mode == "max" else np.nanmean
    return reducer(arr, axis=tuple(range(1, arr.ndim)))


def _detection_config_names(sweep: "SweepResults") -> List[str]:
    """Configuration names that have a detection map in the results.

    Args:
        sweep: The sweep results.

    Returns:
        List[str]: Config names extracted from 'detection_<config>' result keys.
    """
    return [k[len("detection_"):] for k in sweep.results if k.startswith("detection_")]


def _selected_result_keys(sweep: "SweepResults", show_metric: bool, show_validation: bool,
                          show_detection: bool) -> List[str]:
    """Result keys to plot given the metric/validation/detection toggles.

    Args:
        sweep: The sweep results.
        show_metric: Include the 'metric' map if present.
        show_validation: Include the 'validation' map if present.
        show_detection: Include every 'detection_<config>' map.

    Returns:
        List[str]: Ordered result keys to plot.
    """
    keys: List[str] = []
    if show_metric and "metric" in sweep.results:
        keys.append("metric")
    if show_validation and "validation" in sweep.results:
        keys.append("validation")
    if show_detection:
        keys += [f"detection_{c}" for c in _detection_config_names(sweep)]
    return keys


def _result_title(key: str) -> str:
    """Readable panel title for a result key."""
    if key == "metric":
        return "Detection metric"
    if key == "validation":
        return "Validation metric"
    if key.startswith("detection_"):
        return f"Detection: {key[len('detection_'):]}"
    return key


def _result_cmap(key: str) -> str:
    """Colormap for a result key; one shared 'viridis' scale for every quantity."""
    return "viridis"


def _is_probability_like(key: str) -> bool:
    """Whether a result is a probability-like measure bounded to [0, 1]."""
    return key.startswith("detection_")


def _contour_levels(z: np.ndarray, is_probability: bool):
    """Fill and label contour levels appropriate for the data.

    Args:
        z: 2D data being contoured.
        is_probability: Use fixed [0, 1] levels when True; data-aware levels otherwise.

    Returns:
        tuple: (fill_levels, label_levels) arrays/lists for contourf/contour.
    """
    if is_probability:
        return _PROB_FILL_LEVELS, _PROB_LABEL_LEVELS
    if not np.isfinite(z).any():  # no valid data; return a harmless default range
        return np.linspace(0.0, 1.0, 61), np.linspace(0.0, 1.0, 6)
    zmin, zmax = float(np.nanmin(z)), float(np.nanmax(z))
    if np.isclose(zmin, zmax):
        eps = max(1e-6, abs(zmax) * 0.1 + 1e-6)
        zmin, zmax = zmin - eps, zmax + eps
    return np.linspace(zmin, zmax, 61), np.linspace(zmin, zmax, 6)


def _panel_grid(n: int) -> Tuple[int, int]:
    """Square-ish (nrows, ncols) layout for n panels."""
    if n <= 1:
        return 1, 1
    if n == 2:
        return 1, 2
    ncols = int(np.ceil(np.sqrt(n)))
    return int(np.ceil(n / ncols)), ncols


def _save_fig(fig: Figure, save_path: str, dpi: int, what: str) -> None:
    """Save a figure, creating parent directories, and print the location."""
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    print(f"{what} plot saved to: {path}")


def _mark_optimum_star(ax, x: float, y: float, label: Optional[str] = None) -> None:
    """Draw a high-visibility red star marker at (x, y)."""
    ax.scatter([x], [y], marker="*", s=420, c="red", edgecolors="black",
               linewidths=2.2, zorder=10, clip_on=False, label=label)


def _draw_contour_panel(ax, xvals, yvals, z_xy, *, xscale="linear", yscale="linear",
                        xlabel="", ylabel="", title="", cmap="rainbow", is_probability=False,
                        levels=None, label_levels=None, mark_xy=None, mark_label=None,
                        add_colorbar=True, colorbar_label="Value", colorbar_ticks=None):
    """Draw one filled+labelled contour panel of a 2D map indexed [x, y].

    Args:
        ax: Target matplotlib axis.
        xvals: 1D values for the x axis.
        yvals: 1D values for the y axis.
        z_xy: 2D data indexed [x, y] (transposed internally for contouring).
        xscale, yscale: 'linear' or 'log' per axis.
        xlabel, ylabel, title: Panel labels.
        cmap: Colormap name.
        is_probability: Use [0, 1] contour levels when True.
        levels, label_levels: Explicit levels; computed from the data when None.
        mark_xy: Optional (x, y) to mark with an optimum star.
        mark_label: Optional legend label for the marker.
        add_colorbar: Attach a colorbar to this panel.
        colorbar_label: Colorbar label text.
        colorbar_ticks: Explicit colorbar tick positions; automatic when None.

    Returns:
        The QuadContourSet from contourf.
    """
    X, Y = np.meshgrid(np.asarray(xvals), np.asarray(yvals))
    Z = np.asarray(z_xy).T
    # An all-NaN/all-inf map has no valid contour levels; show a placeholder instead of crashing.
    if not np.isfinite(Z).any():
        ax.text(0.5, 0.5, "no finite data", ha="center", va="center", transform=ax.transAxes)
        if xscale == "log":
            ax.set_xscale("log")
        if yscale == "log":
            ax.set_yscale("log")
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        if title:
            ax.set_title(title, fontsize=11)
        return None
    if levels is None:
        levels, label_levels = _contour_levels(Z, is_probability)
    cf = ax.contourf(X, Y, Z, levels=levels, cmap=cmap)
    if add_colorbar:
        cbar = plt.colorbar(cf, ax=ax, fraction=0.046, pad=0.03)
        cbar.set_label(colorbar_label, fontsize=9)
        # Default to the same tick ratios as the fixed [0, 1] range, mapped to the level span.
        ticks = colorbar_ticks if colorbar_ticks is not None else \
            np.linspace(float(levels[0]), float(levels[-1]), len(_PROB_CBAR_TICKS))
        cbar.set_ticks(ticks)
    cs = ax.contour(X, Y, Z, levels=label_levels, colors="k", linewidths=0.2)
    ax.clabel(cs, inline=True, fontsize=7)
    if mark_xy is not None:
        _mark_optimum_star(ax, mark_xy[0], mark_xy[1], mark_label)
        if mark_label:
            ax.legend(loc="upper right", fontsize=8)
    if xscale == "log":
        ax.set_xscale("log")
    if yscale == "log":
        ax.set_yscale("log")
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    if title:
        ax.set_title(title, fontsize=11)
    return cf


def _draw_line_panel(ax, xvals, y, *, xscale="linear", xlabel="", ylabel="", title="",
                     mark_x=None):
    """Draw one 1D curve panel with an optional optimum marker.

    Args:
        ax: Target matplotlib axis.
        xvals: 1D x values.
        y: 1D values to plot.
        xscale: 'linear' or 'log'.
        xlabel, ylabel, title: Panel labels.
        mark_x: Optional x location to mark with a dashed vertical line.
    """
    ax.plot(np.asarray(xvals), np.asarray(y), "-o", ms=3, lw=1.4, color="#1f77b4")
    if mark_x is not None:
        ax.axvline(mark_x, color="#ff2d55", ls="--", lw=1.2, label=f"optimum {mark_x:.3g}")
        ax.legend(loc="best", fontsize=8)
    if xscale == "log":
        ax.set_xscale("log")
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    if title:
        ax.set_title(title, fontsize=11)
    ax.grid(True, alpha=0.3)


def _resolve_quantity_key(sweep: "SweepResults", quantity: str, config: Optional[str]) -> str:
    """Map a quantity selector to a concrete result key.

    Args:
        sweep: The sweep results.
        quantity: 'metric', 'validation', 'detection', or a raw result key.
        config: Detection configuration to use when ``quantity == 'detection'``
            (defaults to the first available configuration).

    Returns:
        str: A key present in ``sweep.results``.
    """
    if quantity in ("metric", "validation"):
        if quantity not in sweep.results:
            raise ValueError(f"'{quantity}' not in results: {list(sweep.results)}")
        return quantity
    if quantity == "detection":
        configs = _detection_config_names(sweep)
        if not configs:
            raise ValueError("No detection results available in this sweep.")
        chosen = config if config is not None else configs[0]
        if chosen not in configs:
            raise ValueError(f"Unknown detection config '{chosen}'; available: {configs}")
        return f"detection_{chosen}"
    if quantity in sweep.results:
        return quantity
    raise ValueError(f"quantity must be 'metric', 'validation' or 'detection' (got {quantity!r}); "
                     f"available results: {list(sweep.results)}")


def plot_sweep_results(
    sweep: "SweepResults",
    display_axes: Optional[Union[str, List[str]]] = None,
    fixed: Optional[Dict[str, float]] = None,
    *,
    show_metric: bool = True,
    show_validation: bool = True,
    show_detection: bool = True,
    fixed_range: bool = True,
    figsize: Optional[Tuple[int, int]] = None,
    mark_optimal: bool = True,
    save_path: Optional[str] = None,
    dpi: int = 300,
) -> Figure:
    """Plot the selected quantities of an N-dimensional sweep over one or two axes.

    One panel is drawn per selected quantity (metric, validation and each detection
    configuration). ``display_axes`` chooses the one or two axes to show; any further
    axis is fixed at ``fixed`` (defaulting to the optimum), i.e. a slice through the
    best point. Use :func:`plot_sweep_corner` for a full multi-axis overview.

    Args:
        sweep: SweepResults from :meth:`Experiment.sweep`.
        display_axes: One or two axis names to plot (default: the first one or two axes).
            One name gives line plots, two names give contour plots.
        fixed: Optional {axis_name: value} for the non-displayed axes; each value snaps
            to the nearest grid point. Defaults to the optimum for every hidden axis.
        show_metric: Plot the 'metric' map (default True).
        show_validation: Plot the 'validation' map (default True).
        show_detection: Plot every 'detection_<config>' map (default True).
        fixed_range: Use a fixed [0, 1] color range for all contour panels (default
            True); when False the color range is fit to each panel's data.
        figsize: Figure size in inches; auto-sized from the panel count when None.
        mark_optimal: Mark the optimum on each panel (default True).
        save_path: Optional path to save the figure.
        dpi: Resolution for the saved figure.

    Returns:
        Figure: The matplotlib figure.

    Example:
        >>> sweep = experiment.sweep({'chi': np.linspace(1, 30, 10),
        ...                           'gamma': np.linspace(1, 60, 10)})
        >>> plot_sweep_results(sweep)                          # metric + validation + detections
        >>> plot_sweep_results(sweep, show_detection=False)    # metric + validation only
        >>> plot_sweep_results(sweep, display_axes='chi')      # 1D slice through the optimum
    """
    keys = _selected_result_keys(sweep, show_metric, show_validation, show_detection)
    if not keys:
        raise ValueError("No results selected; enable at least one of "
                         "show_metric / show_validation / show_detection.")

    # Surface NaNs in the data (e.g. from diverged solves) instead of hiding them behind a plot.
    nan_maps = {k: float(np.isnan(sweep.results[k]).mean()) for k in keys
                if np.isnan(sweep.results[k]).any()}
    if nan_maps:
        warnings.warn("sweep results contain NaN (fraction per map): "
                      + ", ".join(f"{k}={f:.0%}" for k, f in nan_maps.items()))

    if display_axes is None:
        display_axes = sweep.axis_names[: min(2, sweep.ndim)]
    elif isinstance(display_axes, str):
        display_axes = [display_axes]
    display_axes = list(display_axes)
    if not 1 <= len(display_axes) <= 2:
        raise ValueError("display_axes must name one or two axes.")
    disp_idx = [_axis_index(sweep, n) for n in display_axes]
    fixed_idx = _resolve_fixed_indices(sweep, fixed)

    # Note describing where the hidden axes are pinned.
    hidden = [n for n in sweep.axis_names if n not in display_axes]
    fixed_note = ""
    if hidden:
        parts = [f"{_pretty_axis(n)}={sweep.axis_vals[_axis_index(sweep, n)][fixed_idx[_axis_index(sweep, n)]]:.3g}"
                 for n in hidden]
        fixed_note = "fixed at " + ", ".join(parts)

    nrows, ncols = _panel_grid(len(keys))
    if figsize is None:
        figsize = (6.5 * ncols, 5.0 * nrows)
    fig, axes_array = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    axes = axes_array.flatten()

    for ax, key in zip(axes, keys):
        data = sweep.results[key]
        if len(disp_idx) == 1:
            i = disp_idx[0]
            y = _reduce_to_1d(data, i, "slice", fixed_idx)
            mark_x = sweep.axis_vals[i][fixed_idx[i]] if mark_optimal else None
            _draw_line_panel(ax, sweep.axis_vals[i], y, xscale=sweep.axis_scales[i],
                             xlabel=_pretty_axis(sweep.axis_names[i]), ylabel=_result_title(key),
                             title=_result_title(key), mark_x=mark_x)
        else:
            i, j = disp_idx
            z = _reduce_to_2d(data, i, j, "slice", fixed_idx)
            mark_xy = ((sweep.axis_vals[i][fixed_idx[i]], sweep.axis_vals[j][fixed_idx[j]])
                       if mark_optimal else None)
            levels, label_levels = (_PROB_FILL_LEVELS, _PROB_LABEL_LEVELS) if fixed_range else (None, None)
            cbar_ticks = _PROB_CBAR_TICKS if fixed_range else None
            _draw_contour_panel(ax, sweep.axis_vals[i], sweep.axis_vals[j], z,
                                xscale=sweep.axis_scales[i], yscale=sweep.axis_scales[j],
                                xlabel=_pretty_axis(sweep.axis_names[i]),
                                ylabel=_pretty_axis(sweep.axis_names[j]),
                                title=_result_title(key), cmap=_result_cmap(key),
                                is_probability=_is_probability_like(key),
                                levels=levels, label_levels=label_levels, mark_xy=mark_xy,
                                colorbar_label="Detection" if key.startswith("detection_") else "Value",
                                colorbar_ticks=cbar_ticks)

    for ax in axes[len(keys):]:
        ax.axis("off")

    subtitle = []
    if "best_metric" in sweep.metadata:
        subtitle.append(f"best metric {sweep.metadata['best_metric']:.4g}")
    if fixed_note:
        subtitle.append(fixed_note)
    if subtitle:
        fig.suptitle("   |   ".join(subtitle), fontsize=10, y=0.995)
    fig.tight_layout()
    if save_path:
        _save_fig(fig, save_path, dpi, "Sweep results")
    return fig


def plot_sweep_corner(
    sweep: "SweepResults",
    quantity: str = "metric",
    config: Optional[str] = None,
    *,
    fixed_range: bool = True,
    figsize: Optional[Tuple[int, int]] = None,
    mark_optimal: bool = True,
    save_path: Optional[str] = None,
    dpi: int = 300,
) -> Figure:
    """Full N×N corner matrix of one quantity over every pair of sweep axes.

    For each axis pair the upper triangle shows a 2D *slice through the optimum*
    (hidden axes fixed at the best point) and the lower triangle shows a 2D
    *max-projection* (best value over the hidden axes); the diagonal shows the 1D
    max-projection for each axis. All 2D panels share one color scale.

    Args:
        sweep: SweepResults from :meth:`Experiment.sweep` (needs at least 2 axes).
        quantity: Which map to show: 'metric', 'validation' or 'detection'.
        config: Detection configuration when ``quantity == 'detection'`` (defaults to
            the first available configuration).
        fixed_range: Use a fixed [0, 1] color range for all panels (default True);
            when False the color range is fit to the data.
        figsize: Figure size in inches; auto-sized from the axis count when None.
        mark_optimal: Mark the optimum on every panel (default True).
        save_path: Optional path to save the figure.
        dpi: Resolution for the saved figure.

    Returns:
        Figure: The matplotlib figure.

    Example:
        >>> sweep = experiment.sweep({'chi': ..., 'gamma': ..., 'Delta': ..., 'g': ..., 'kappa': ...})
        >>> plot_sweep_corner(sweep, quantity='metric')
        >>> plot_sweep_corner(sweep, quantity='detection', config='with_photon')
    """
    key = _resolve_quantity_key(sweep, quantity, config)
    data = sweep.results[key]
    n = sweep.ndim
    if n < 2:
        raise ValueError("Corner plots need at least 2 sweep axes; use plot_sweep_results instead.")

    fixed_idx = _best_index(sweep)
    is_prob = _is_probability_like(key)
    cmap = _result_cmap(key)
    # Shared color scale so slice and max-projection panels are directly comparable.
    if fixed_range:
        levels, label_levels = _PROB_FILL_LEVELS, _PROB_LABEL_LEVELS
    else:
        levels, label_levels = _contour_levels(data, is_prob)

    if figsize is None:
        figsize = (3.1 * n, 3.0 * n)
    fig, axes = plt.subplots(n, n, figsize=figsize, squeeze=False)

    for r in range(n):
        for c in range(n):
            ax = axes[r][c]
            # Column names label the top row only (not the bottom or the diagonal).
            if r == 0:
                ax.set_title(_pretty_axis(sweep.axis_names[c]), fontsize=9)
            if r == c:
                # Diagonal: 1D max-projection for this axis.
                y = _reduce_to_1d(data, r, "max", fixed_idx)
                mark_x = sweep.axis_vals[r][fixed_idx[r]] if mark_optimal else None
                _draw_line_panel(ax, sweep.axis_vals[r], y, xscale=sweep.axis_scales[r],
                                 mark_x=mark_x)
                # Set the row name after the line panel, which clears the y label.
                if c == 0:
                    ax.set_ylabel(_pretty_axis(sweep.axis_names[r]), fontsize=9)
                continue
            # x = axis c, y = axis r; upper triangle slices, lower triangle max-projects.
            mode = "slice" if r < c else "max"
            z = _reduce_to_2d(data, c, r, mode, fixed_idx)
            mark_xy = ((sweep.axis_vals[c][fixed_idx[c]], sweep.axis_vals[r][fixed_idx[r]])
                       if mark_optimal else None)
            _draw_contour_panel(ax, sweep.axis_vals[c], sweep.axis_vals[r], z,
                                xscale=sweep.axis_scales[c], yscale=sweep.axis_scales[r],
                                cmap=cmap, is_probability=is_prob, levels=levels,
                                label_levels=label_levels, mark_xy=mark_xy, add_colorbar=False)
            if c == 0:
                ax.set_ylabel(_pretty_axis(sweep.axis_names[r]), fontsize=9)

    # Align all row names to a common x so differing tick-label widths don't stagger them.
    fig.align_ylabels(axes[:, 0])

    # One shared colorbar for every 2D panel.
    sm = plt.cm.ScalarMappable(cmap=cmap,
                               norm=plt.Normalize(vmin=float(levels[0]), vmax=float(levels[-1])))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes.ravel().tolist(), fraction=0.02, pad=0.02)
    cbar.set_label(_result_title(key), fontsize=10)
    if fixed_range:
        cbar.set_ticks(_PROB_CBAR_TICKS)

    fig.suptitle(f"Corner sweep — {_result_title(key)}   "
                 f"(upper: slice @ optimum · lower: max-projection · diag: 1D max)", fontsize=11)
    if save_path:
        _save_fig(fig, save_path, dpi, "Corner sweep")
    return fig


def interactive_sweep(
    sweep: "SweepResults",
    quantity: str = "metric",
    config: Optional[str] = None,
):
    """Interactive contour explorer for a sweep (Jupyter; requires ipywidgets).

    Dropdowns choose the quantity and the two displayed axes; a selection slider per
    remaining axis fixes its value; a reduce selector switches the hidden axes between
    slice, max and mean. Sliders are hidden while a display axis or a non-slice reduce
    mode makes them irrelevant.

    Args:
        sweep: SweepResults from :meth:`Experiment.sweep` (needs at least 2 axes).
        quantity: Initial quantity ('metric', 'validation' or 'detection').
        config: Detection configuration used for the initial 'detection' quantity.

    Returns:
        The ipywidgets container being displayed.
    """
    try:
        import ipywidgets as widgets
    except ImportError as exc:
        raise ImportError("interactive_sweep requires ipywidgets: pip install ipywidgets") from exc

    names = sweep.axis_names
    if sweep.ndim < 2:
        raise ValueError("interactive_sweep needs at least 2 sweep axes.")

    quantity_options = [q for q in ("metric", "validation") if q in sweep.results]
    quantity_options += [f"detection:{c}" for c in _detection_config_names(sweep)]
    initial = f"detection:{config}" if (quantity == "detection" and config) else quantity
    if initial not in quantity_options:
        initial = quantity_options[0]

    best = _best_index(sweep)
    q_dd = widgets.Dropdown(options=quantity_options, value=initial, description="quantity")
    x_dd = widgets.Dropdown(options=names, value=names[0], description="x axis")
    y_dd = widgets.Dropdown(options=names, value=names[1], description="y axis")
    reduce_dd = widgets.Dropdown(options=["slice", "max", "mean"], value="slice", description="hidden")
    sliders = {
        n: widgets.SelectionSlider(
            options=[(f"{v:.3g}", k) for k, v in enumerate(sweep.axis_vals[_axis_index(sweep, n)])],
            value=best[_axis_index(sweep, n)], description=n)
        for n in names
    }
    out = widgets.Output()

    def render(*_):
        with out:
            out.clear_output(wait=True)
            xi, yi = _axis_index(sweep, x_dd.value), _axis_index(sweep, y_dd.value)
            if xi == yi:
                print("Choose two different axes for x and y.")
                return
            key = (f"detection_{q_dd.value.split(':', 1)[1]}"
                   if q_dd.value.startswith("detection:") else q_dd.value)
            fixed_idx = list(best)
            for n in names:
                fixed_idx[_axis_index(sweep, n)] = sliders[n].value
            # Only show sliders that actually affect a slice view.
            for n, slider in sliders.items():
                k = _axis_index(sweep, n)
                slider.layout.display = "none" if (k in (xi, yi) or reduce_dd.value != "slice") else ""
            z = _reduce_to_2d(sweep.results[key], xi, yi, reduce_dd.value, fixed_idx)
            fig, ax = plt.subplots(figsize=(7, 5.5))
            mark_xy = (sweep.axis_vals[xi][best[xi]], sweep.axis_vals[yi][best[yi]])
            # Fixed [0, 1] color range so every rendered view is directly comparable.
            _draw_contour_panel(ax, sweep.axis_vals[xi], sweep.axis_vals[yi], z,
                                xscale=sweep.axis_scales[xi], yscale=sweep.axis_scales[yi],
                                xlabel=_pretty_axis(names[xi]), ylabel=_pretty_axis(names[yi]),
                                title=f"{_result_title(key)} ({reduce_dd.value})",
                                cmap=_result_cmap(key), is_probability=_is_probability_like(key),
                                levels=_PROB_FILL_LEVELS, label_levels=_PROB_LABEL_LEVELS,
                                colorbar_ticks=_PROB_CBAR_TICKS, mark_xy=mark_xy)
            plt.show()

    for widget in [q_dd, x_dd, y_dd, reduce_dd, *sliders.values()]:
        widget.observe(render, names="value")
    render()
    ui = widgets.VBox([widgets.HBox([q_dd, reduce_dd]), widgets.HBox([x_dd, y_dd]),
                       widgets.VBox(list(sliders.values())), out])
    # Return the container and let Jupyter display it; calling display() here as well
    # would render two synced copies of the same widget.
    return ui
