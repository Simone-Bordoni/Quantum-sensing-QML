"""
Visualization utilities for quantum sensing optimization results.

This module provides functions for creating comprehensive optimization dashboards
that display key metrics including sensing contrast, gradient evolution, parameter
trajectories, and detection probabilities.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec

from qsopt.core.callback import OptimizationCallback
from qsopt.core.experimental_parameters import ExperimentalParameters
from qsopt.utils.results import SweepResults, TimeEvolutionResults


def plot_optimization_dashboard(
    optimization_callback: OptimizationCallback,
    reference_callback: Optional[OptimizationCallback] = None,
    show_contrast: bool = True,
    show_gradients: bool = True,
    show_parameters: bool = True,
    show_trajectory: bool = True,
    show_probabilities: bool = True,
    figsize: Tuple[int, int] = (16, 18),
    save_path: Optional[str] = None,
    dpi: int = 300,
) -> Figure:
    """
    Create a comprehensive optimization dashboard with multiple subplots.

    This function generates a multi-panel visualization showing:
    - Sensing contrast evolution (with optional reference benchmark)
    - Gradient magnitude evolution (log scale)
    - Parameter evolution over epochs
    - Optimization trajectory in parameter space
    - Detection probabilities (with and without photon)

    Args:
        optimization_callback: OptimizationCallback from ``optimize_rotations()``
            Contains history of epochs, contrast, probabilities, and parameters
        reference_callback: Optional SimulationCallback from ``run_simulation()``
            If provided, reference values are shown as horizontal benchmark lines
        show_contrast: Display sensing contrast evolution plot when True
        show_gradients: Display gradient magnitude evolution plot when True
        show_parameters: Display parameter evolution plot when True
        show_trajectory: Display optimization trajectory in parameter space
        show_probabilities: Display detection probabilities plot
        figsize: Figure size as ``(width, height)`` in inches
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
        ...                                   show_trajectory=False)
    """
    # Count active plots to determine layout
    active_plots = [
        show_contrast,
        show_gradients,
        show_parameters,
        show_trajectory,
        show_probabilities,
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
    contrast = np.array(history["contrast"])
    prob_with = np.array(history["prob_with"])
    prob_without = np.array(history["prob_without"])

    # Extract parameter arrays (assuming rotation angles)
    param_arrays = []
    param_names = []
    if history["trainable_params"]:
        first_params = history["trainable_params"][0]
        angles = first_params.get_rotation_angles()
        param_names = list(angles.keys())

        for tp in history["trainable_params"]:
            angles = tp.get_rotation_angles()
            param_arrays.append([angles[name][0] for name in param_names])

    param_arrays = np.array(param_arrays)

    # Calculate gradients (approximate from contrast differences)
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
    reference_contrast = None
    reference_prob_with = None
    reference_prob_without = None
    reference_params = None

    if reference_callback is not None:
        ref_history = reference_callback.get_history()
        if ref_history["contrast"]:
            reference_contrast = ref_history["contrast"][0]
            reference_prob_with = ref_history["prob_with"][0]
            reference_prob_without = ref_history["prob_without"][0]

        if ref_history["trainable_params"]:
            ref_tp = ref_history["trainable_params"][0]
            ref_angles = ref_tp.get_rotation_angles()
            reference_params = [ref_angles[name][0] for name in param_names]

    # Create figure
    fig = plt.figure(figsize=figsize)
    axes = []
    plot_idx = 0

    # Plot 1: Sensing Contrast Evolution
    if show_contrast:
        ax = plt.subplot(n_rows, n_cols, plot_idx + 1)
        axes.append(ax)
        plot_idx += 1

        ax.plot(epochs, contrast, "g-", linewidth=2, alpha=0.8, label="Optimized")

        if reference_contrast is not None:
            ax.axhline(
                y=reference_contrast,
                color="red",
                linestyle="--",
                linewidth=2,
                alpha=0.7,
                label="Reference",
            )

        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel("Sensing Contrast", fontsize=12)
        ax.set_title("Sensing Contrast Evolution", fontsize=14)
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

        colors = ["r", "b", "g", "orange", "purple", "brown"]

        for i, name in enumerate(param_names):
            color = colors[i % len(colors)]
            params_deg = param_arrays[:, i] * 180 / np.pi
            ax.plot(epochs, params_deg, "-", linewidth=2, label=name, color=color, alpha=0.8)

            # Add reference line if available
            if reference_params is not None:
                ref_deg = reference_params[i] * 180 / np.pi
                ax.axhline(y=ref_deg, color=color, linestyle="--", alpha=0.5, linewidth=1.5)

        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel("Rotation Angle (degrees)", fontsize=12)
        ax.set_title("Parameter Evolution", fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

    # Plot 4: Optimization Trajectory in Parameter Space
    if show_trajectory and len(param_arrays) > 0 and len(param_names) >= 2:
        ax = plt.subplot(n_rows, n_cols, plot_idx + 1)
        axes.append(ax)
        plot_idx += 1

        # Check if this is a two-qubit system (4 parameters: theta1_q1, theta2_q1, theta1_q2, theta2_q2)
        if len(param_names) >= 4:
            # Two-qubit trajectory: plot both qubits' angle evolution
            theta1_q1_deg = param_arrays[:, 0] * 180 / np.pi
            theta2_q1_deg = param_arrays[:, 1] * 180 / np.pi
            theta1_q2_deg = param_arrays[:, 2] * 180 / np.pi
            theta2_q2_deg = param_arrays[:, 3] * 180 / np.pi

            # Plot trajectory for qubit 1
            ax.plot(theta1_q1_deg, theta2_q1_deg, 'o-', linewidth=2, alpha=0.7,
                   label='Qubit 1', color='tab:blue', markersize=4)
            # Mark start and end for qubit 1
            ax.plot(theta1_q1_deg[0], theta2_q1_deg[0], 'o', markersize=10,
                   color='tab:blue', markeredgecolor='black', markeredgewidth=1.5)
            ax.plot(theta1_q1_deg[-1], theta2_q1_deg[-1], 's', markersize=10,
                   color='tab:blue', markeredgecolor='black', markeredgewidth=1.5)

            # Plot trajectory for qubit 2
            ax.plot(theta1_q2_deg, theta2_q2_deg, 'o-', linewidth=2, alpha=0.7,
                   label='Qubit 2', color='tab:orange', markersize=4)
            # Mark start and end for qubit 2
            ax.plot(theta1_q2_deg[0], theta2_q2_deg[0], 'o', markersize=10,
                   color='tab:orange', markeredgecolor='black', markeredgewidth=1.5)
            ax.plot(theta1_q2_deg[-1], theta2_q2_deg[-1], 's', markersize=10,
                   color='tab:orange', markeredgecolor='black', markeredgewidth=1.5)

            ax.set_xlabel("θ₁ (degrees)", fontsize=12)
            ax.set_ylabel("θ₂ (degrees)", fontsize=12)
            ax.set_title("Optimization Trajectory (Both Qubits)", fontsize=14)
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
        else:
            # Single-qubit trajectory: use first two parameters
            theta1_deg = param_arrays[:, 0] * 180 / np.pi
            theta2_deg = param_arrays[:, 1] * 180 / np.pi

            # Plot trajectory with color gradient
            scatter = ax.scatter(
                theta1_deg,
                theta2_deg,
                c=epochs,
                cmap="viridis",
                s=30,
                alpha=0.7,
                edgecolors="black",
                linewidth=0.5,
            )
            ax.plot(theta1_deg, theta2_deg, "k-", alpha=0.3, linewidth=1)

            # Mark start and end points
            ax.plot(
                theta1_deg[0], theta2_deg[0], "ro", markersize=8, label="Start", markeredgecolor="black"
            )
            ax.plot(
                theta1_deg[-1], theta2_deg[-1], "gs", markersize=8, label="End", markeredgecolor="black"
            )

            # Mark reference point if available
            if reference_params is not None:
                ref_theta1_deg = reference_params[0] * 180 / np.pi
                ref_theta2_deg = reference_params[1] * 180 / np.pi
                ax.plot(
                    ref_theta1_deg,
                    ref_theta2_deg,
                    "b^",
                    markersize=10,
                    label="Reference",
                    markeredgecolor="black",
                )

            ax.set_xlabel(f"{param_names[0]} (degrees)", fontsize=12)
            ax.set_ylabel(f"{param_names[1]} (degrees)", fontsize=12)
            ax.set_title("Optimization Trajectory", fontsize=14)
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)

            # Add colorbar
            cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
            cbar.set_label("Epoch", fontsize=10)

    # Plot 5: Detection Probabilities Evolution
    if show_probabilities:
        ax = plt.subplot(n_rows, n_cols, plot_idx + 1)
        axes.append(ax)
        plot_idx += 1

        ax.plot(epochs, prob_with, "g-", linewidth=2, label="With Photon (Optimized)", alpha=0.8)
        ax.plot(
            epochs, prob_without, "r-", linewidth=2, label="Without Photon (Optimized)", alpha=0.8
        )

        # Add reference benchmarks if available
        if reference_prob_with is not None:
            ax.axhline(
                y=reference_prob_with,
                color="green",
                linestyle="--",
                linewidth=2,
                alpha=0.6,
                label="With Photon (Reference)",
            )
        if reference_prob_without is not None:
            ax.axhline(
                y=reference_prob_without,
                color="red",
                linestyle="--",
                linewidth=2,
                alpha=0.6,
                label="Without Photon (Reference)",
            )

        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel("Detection Probability", fontsize=12)
        ax.set_title("Detection Probabilities Evolution", fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

    # Overall title
    plt.suptitle("Optimization Dashboard", fontsize=18)
    plt.tight_layout()

    # Save if path provided
    if save_path is not None:
        # Create directory if it doesn't exist
        save_path_obj = Path(save_path)
        save_path_obj.parent.mkdir(parents=True, exist_ok=True)

        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")

    return fig


def plot_contrast_evolution(
    optimization_callback: OptimizationCallback,
    reference_callback: Optional[OptimizationCallback] = None,
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None,
    dpi: int = 300,
) -> Figure:
    """
    Create a standalone plot of sensing contrast evolution.

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
    contrast = np.array(history["contrast"])

    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(
        epochs,
        contrast,
        "g-",
        linewidth=2.5,
        alpha=0.8,
        label="Optimized",
        marker="o",
        markersize=4,
    )

    if reference_callback is not None:
        ref_history = reference_callback.get_history()
        if ref_history["contrast"]:
            reference_contrast = ref_history["contrast"][0]
            ax.axhline(
                y=reference_contrast,
                color="red",
                linestyle="--",
                linewidth=2,
                alpha=0.7,
                label="Reference",
            )

    ax.set_xlabel("Epoch", fontsize=14)
    ax.set_ylabel("Sensing Contrast", fontsize=14)
    ax.set_title("Sensing Contrast Evolution", fontsize=16)
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

    # Extract parameters
    param_arrays = []
    param_names = []
    if history["trainable_params"]:
        first_params = history["trainable_params"][0]
        angles = first_params.get_rotation_angles()
        param_names = list(angles.keys())

        for tp in history["trainable_params"]:
            angles = tp.get_rotation_angles()
            param_arrays.append([angles[name][0] for name in param_names])

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
            ref_tp = ref_history["trainable_params"][0]
            ref_angles = ref_tp.get_rotation_angles()
            ref_params = [ref_angles[name][0] for name in param_names]
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


def plot_parameter_landscape(
    landscape_data: Dict[str, Union[np.ndarray, float]],
    exp_params: "ExperimentalParameters",
    save_path: Optional[str] = None,
    dpi: int = 300,
) -> Figure:
    """
    Plot parameter space landscape with system information.

    Creates a two-panel visualization showing:
    1. Sensing contrast landscape as a 2D heatmap
    2. Detection probability landscape as a 2D heatmap

    Includes a comprehensive system information box showing:
    - Physical constants (coupling strengths, pulse widths)
    - Noise configuration (relaxation, dephasing, depolarizing)
    - Measurement protocol (timing, intervals)
    - Landscape statistics (ranges, optimal points)

    Args:
        landscape_data: Dictionary from compute_theta1_theta2_landscape() containing:
            - 'theta1_vals': Array of θ₁ values
            - 'theta2_vals': Array of θ₂ values
            - 'contrast_map': 2D array of contrast values
            - 'detection_map': 2D array of detection probabilities
            - 'center_theta1': Center θ₁ value
            - 'center_theta2': Center θ₂ value
        exp_params: ExperimentalParameters instance with system configuration
        save_path: Optional file path to save figure. If None, figure is not saved.
        dpi: Resolution for saved figure. Default: 300.

    Returns:
        matplotlib Figure object

    Example:
        >>> from qsopt.utils import compute_theta1_theta2_landscape, plot_parameter_landscape
        >>> from qsopt.core.experimental_parameters import ExperimentalParameters
        >>>
        >>> exp_params = ExperimentalParameters()
        >>> # Configure exp_params...
        >>>
        >>> landscape = compute_theta1_theta2_landscape(exp_params, resolution=25)
        >>> fig = plot_parameter_landscape(
        ...     landscape,
        ...     exp_params,
        ...     save_path='landscape.png'
        ... )
        >>> plt.show()

    Notes:
        - The figure includes comprehensive system information at the bottom
        - Optimal parameter locations are marked with symbols
        - Color maps: 'viridis' for contrast, 'plasma' for detection
        - Layout is optimized for publication-quality output
    """
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 16))

    # Extract data
    theta1_vals = landscape_data["theta1_vals"]
    theta2_vals = landscape_data["theta2_vals"]
    contrast_map = np.asarray(landscape_data["contrast_map"])
    detection_map = np.asarray(landscape_data["detection_map"])
    center_theta1 = landscape_data["center_theta1"]
    center_theta2 = landscape_data["center_theta2"]

    # Create meshgrid
    P1, P2 = np.meshgrid(theta1_vals, theta2_vals)

    # Convert to degrees for display
    P1_deg = np.degrees(P1)
    P2_deg = np.degrees(P2)
    center_x = np.degrees(center_theta1)
    center_y = np.degrees(center_theta2)

    # Plot 1: Contrast landscape
    im1 = ax1.contourf(P1_deg, P2_deg, contrast_map, levels=30, cmap="viridis")
    ax1.set_xlabel("θ₁ (degrees)", fontsize=12)
    ax1.set_ylabel("θ₂ (degrees)", fontsize=12)
    ax1.set_title("Sensing Contrast Landscape", fontsize=14)
    ax1.grid(True, alpha=0.3)
    cbar1 = plt.colorbar(im1, ax=ax1, label="Contrast")

    # Find and mark maximum contrast
    max_idx = np.unravel_index(np.argmax(contrast_map), contrast_map.shape)
    max_x = P1_deg[max_idx]
    max_y = P2_deg[max_idx]
    max_contrast = contrast_map[max_idx]

    # Mark points on contrast plot
    ax1.plot(
        center_x, center_y, "w+", markersize=15, markeredgewidth=3, label="Center point", zorder=10
    )
    ax1.plot(
        max_x,
        max_y,
        "ro",
        markersize=10,
        markerfacecolor="red",
        markeredgecolor="white",
        markeredgewidth=2,
        label=f"Max = {max_contrast:.6f}",
        zorder=10,
    )
    ax1.legend(loc="upper right", fontsize=10)

    # Plot 2: Detection probability landscape
    im2 = ax2.contourf(P1_deg, P2_deg, detection_map, levels=30, cmap="plasma")
    ax2.set_xlabel("θ₁ (degrees)", fontsize=12)
    ax2.set_ylabel("θ₂ (degrees)", fontsize=12)
    ax2.set_title("Detection Probability Landscape (with photon)", fontsize=14)
    ax2.grid(True, alpha=0.3)
    cbar2 = plt.colorbar(im2, ax=ax2, label="Detection Probability")

    # Mark center point
    ax2.plot(
        center_x, center_y, "w+", markersize=15, markeredgewidth=3, label="Center point", zorder=10
    )
    ax2.legend(loc="upper right", fontsize=10)

    # Adjust layout to leave space at bottom
    plt.tight_layout(rect=(0, 0.12, 1, 1))

    # Create comprehensive system information box
    if (
        exp_params._measurement_times_list is not None
        and len(exp_params._measurement_times_list) > 1
    ):
        meas_times = exp_params._measurement_times_list
        time_intervals = np.diff(meas_times)
        avg_interval = np.mean(time_intervals)
        interval_text = f"{avg_interval:.6f}"
        n_measurements = len(exp_params._measurement_times_list)
    else:
        interval_text = "N/A"
        n_measurements = 0

    # Format chi and noise rates for display (handle list format)
    chi_display = (
        exp_params.chi
        if isinstance(exp_params.chi, (int, float))
        else exp_params.chi[0] if len(exp_params.chi) == 1 else str(exp_params.chi)
    )
    relaxation_display = (
        exp_params.noise_config.relaxation
        if isinstance(exp_params.noise_config.relaxation, (int, float))
        else (
            exp_params.noise_config.relaxation[0]
            if len(exp_params.noise_config.relaxation) == 1
            else str(exp_params.noise_config.relaxation)
        )
    )
    dephasing_display = (
        exp_params.noise_config.dephasing
        if isinstance(exp_params.noise_config.dephasing, (int, float))
        else (
            exp_params.noise_config.dephasing[0]
            if len(exp_params.noise_config.dephasing) == 1
            else str(exp_params.noise_config.dephasing)
        )
    )
    depolarizing_display = (
        exp_params.noise_config.depolarizing
        if isinstance(exp_params.noise_config.depolarizing, (int, float))
        else (
            exp_params.noise_config.depolarizing[0]
            if len(exp_params.noise_config.depolarizing) == 1
            else str(exp_params.noise_config.depolarizing)
        )
    )

    system_info = f"""SYSTEM PARAMETERS AND CONFIGURATION

Physical Constants:
  • Photon-cavity coupling (γ):    {exp_params.photon_cavity_coupling:.6f} rad/time
  • Inverse pulse width (σ):        {exp_params.inverse_pulse_width:.6f} 1/time
  • Dispersive coupling (χ):        {chi_display if isinstance(chi_display, str) else f'{chi_display:.6f}'} rad/time

System Dimensions:
  • Number of qubits: {exp_params.n_qubits}  |  Cavity levels:  {exp_params.cavity_levels}  |  Qubit levels:  {exp_params.qubit_levels}  |  Field levels:  {exp_params.field_levels}

Noise Configuration:
  • Relaxation (γ_relax):   {relaxation_display if isinstance(relaxation_display, str) else f'{relaxation_display:.6f}'} rad/time
  • Dephasing (γ_deph):     {dephasing_display if isinstance(dephasing_display, str) else f'{dephasing_display:.6f}'} rad/time
  • Depolarizing (γ_depol): {depolarizing_display if isinstance(depolarizing_display, str) else f'{depolarizing_display:.6f}'} rad/time

Measurement Protocol:
  • Initial time:     {exp_params.measurement.initial_time:.6f}  |  Final time:  {exp_params.measurement.final_time:.6f}
  • Time interval (Δt):  {exp_params.measurement.time_interval:.6f}
  • Avg. interval between measurements:  {interval_text}  |  Number of measurements:  {n_measurements}

Initial State:  {exp_params.initial_state.state_type.value}

Landscape Statistics:
  • Contrast range:     [{contrast_map.min():.6f}, {contrast_map.max():.6f}]  |  Variation:  {contrast_map.max() - contrast_map.min():.2e}
  • Detection range:    [{detection_map.min():.6f}, {detection_map.max():.6f}]  |  Variation:  {detection_map.max() - detection_map.min():.2e}
  • Maximum at:         θ₁ = {max_x:.2f}°,  θ₂ = {max_y:.2f}°,  Contrast = {max_contrast:.8f}
"""

    # Add text box below plots
    fig.text(
        0.05,
        0.02,
        system_info,
        fontsize=9,
        family="monospace",
        verticalalignment="bottom",
        bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.7, pad=0.8),
    )

    # Save figure if path provided
    if save_path:
        save_path_obj = Path(save_path)
        save_path_obj.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"Landscape plot saved to: {save_path}")

    return fig


def plot_time_interval_landscape(
    landscape_data: Dict[str, Union[np.ndarray, float, str, int]],
    exp_params: "ExperimentalParameters",
    save_path: Optional[str] = None,
    dpi: int = 300,
    show_measurement_count: bool = False,
) -> Figure:
    """
    Plot time interval landscape with system information.

    Creates a comprehensive visualization showing:
    1. Sensing contrast vs time interval
    2. Detection probabilities (with and without photon) vs time interval
    3. (Optional) Number of measurements vs time interval

    Includes system information box showing:
    - Physical constants (coupling strengths, pulse widths)
    - Rotation parameters (θ₁, θ₂)
    - Noise configuration
    - Batch averaging details (if used)
    - Optimal interval statistics

    Args:
        landscape_data: Dictionary from ``compute_time_interval_landscape()`` containing:
            - 'interval_vals': Array of time interval values
            - 'contrast_vals': Array of contrast values
            - 'detection_with': Array of detection probabilities with photon
            - 'detection_without': Array of detection probabilities without photon
            - 'n_measurements': Array of number of measurements per interval
            - 'theta1': Fixed θ₁ value
            - 'theta2': Fixed θ₂ value
            - 'mode': Computation mode ('continuous' or 'discrete')
            - 'batch_size': Batch size used
            - 'initial_time_uncertainty': Uncertainty value
        exp_params: ExperimentalParameters instance with system configuration
        save_path: Optional file path to save figure. If None, figure is not saved.
        dpi: Resolution for saved figure. Default: 300.
        show_measurement_count: Include measurement count subplot when True (default: False)

    Returns:
        matplotlib Figure object

    Example:
        >>> from qsopt.utils import compute_time_interval_landscape, plot_time_interval_landscape
        >>>
        >>> data = compute_time_interval_landscape(
        ...     exp_params,
        ...     theta1=np.pi/2,
        ...     theta2=-np.pi/2,
        ...     resolution=50,
        ...     batch_size=10
        ... )
        >>> fig = plot_time_interval_landscape(
        ...     data,
        ...     exp_params,
        ...     save_path='time_interval_landscape.png'
        ... )
        >>> plt.show()

    Notes:
        - The figure layout adapts to show different features for continuous vs discrete modes
        - Optimal interval is marked with a vertical line and annotation
        - If batch_size > 1, uncertainty information is displayed
    """
    # Create figure with subplots stacked vertically
    if show_measurement_count:
        fig, axes = plt.subplots(3, 1, figsize=(12, 14))
        ax1, ax2, ax3 = axes
    else:
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))
        ax1, ax2 = axes
        ax3 = None

    # Extract data
    interval_vals = np.asarray(landscape_data["interval_vals"])
    contrast_vals = np.asarray(landscape_data["contrast_vals"])
    detection_with = np.asarray(landscape_data["detection_with"])
    detection_without = np.asarray(landscape_data["detection_without"])
    n_measurements = np.asarray(landscape_data["n_measurements"])
    theta1 = float(landscape_data["theta1"])
    theta2 = float(landscape_data["theta2"])
    mode = str(landscape_data["mode"])
    batch_size = int(landscape_data["batch_size"])
    uncertainty = float(landscape_data["initial_time_uncertainty"])
    uncertainty_spec = landscape_data.get("initial_time_uncertainty_spec")

    # Find optimal interval
    optimal_idx = np.argmax(contrast_vals)
    optimal_interval = interval_vals[optimal_idx]
    optimal_contrast = contrast_vals[optimal_idx]
    optimal_n_meas = n_measurements[optimal_idx]

    # Plot 1: Sensing contrast vs time interval
    if mode == "discrete":
        ax1.plot(interval_vals, contrast_vals, "bo-", linewidth=2, markersize=6, label="Contrast")
    else:
        ax1.plot(interval_vals, contrast_vals, "b-", linewidth=2, label="Contrast")

    # Mark optimal point
    ax1.axvline(optimal_interval, color="red", linestyle="--", alpha=0.7, linewidth=1.5)
    ax1.plot(
        optimal_interval,
        optimal_contrast,
        "ro",
        markersize=10,
        markerfacecolor="red",
        markeredgecolor="white",
        markeredgewidth=2,
        label=f"Optimal: Δt={optimal_interval:.4f}",
        zorder=10,
    )

    ax1.set_xlabel("Time Interval (Δt)", fontsize=12)
    ax1.set_ylabel("Sensing Contrast", fontsize=12)
    ax1.set_title(f"Sensing Contrast vs Time Interval ({mode} mode)", fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="best", fontsize=10)

    # Plot 2: Detection probabilities
    if mode == "discrete":
        ax2.plot(
            interval_vals,
            detection_with,
            "go-",
            linewidth=2,
            markersize=5,
            label="With photon",
            alpha=0.8,
        )
        ax2.plot(
            interval_vals,
            detection_without,
            "mo-",
            linewidth=2,
            markersize=5,
            label="Without photon",
            alpha=0.8,
        )
    else:
        ax2.plot(interval_vals, detection_with, "g-", linewidth=2, label="With photon", alpha=0.8)
        ax2.plot(
            interval_vals, detection_without, "m-", linewidth=2, label="Without photon", alpha=0.8
        )

    # Mark optimal point
    ax2.axvline(optimal_interval, color="red", linestyle="--", alpha=0.7, linewidth=1.5)

    ax2.set_xlabel("Time Interval (Δt)", fontsize=12)
    ax2.set_ylabel("Detection Probability", fontsize=12)
    ax2.set_title("Detection Probabilities vs Time Interval", fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="best", fontsize=10)

    # Plot 3: Number of measurements (optional)
    if show_measurement_count and ax3 is not None:
        if mode == "discrete":
            ax3.plot(
                interval_vals,
                n_measurements,
                "ko-",
                linewidth=2,
                markersize=5,
                label="Number of measurements",
            )
        else:
            ax3.plot(
                interval_vals, n_measurements, "k-", linewidth=2, label="Number of measurements"
            )

        # Mark optimal point
        ax3.axvline(optimal_interval, color="red", linestyle="--", alpha=0.7, linewidth=1.5)
        ax3.plot(
            optimal_interval,
            optimal_n_meas,
            "ro",
            markersize=10,
            markerfacecolor="red",
            markeredgecolor="white",
            markeredgewidth=2,
            zorder=10,
        )

        ax3.set_xlabel("Time Interval (Δt)", fontsize=12)
        ax3.set_ylabel("Number of Measurements", fontsize=12)
        ax3.set_title("Measurement Count vs Time Interval", fontsize=14)
        ax3.grid(True, alpha=0.3)
        ax3.legend(loc="best", fontsize=10)

    # Adjust layout to leave space at bottom for info box
    layout_bottom = 0.15 if show_measurement_count else 0.2
    plt.tight_layout(rect=(0, layout_bottom, 1, 1))

    # Create comprehensive system information box
    batch_info = f"  • Batch size: {batch_size} realizations"
    spec_suffix = (
        f" (specified as '{uncertainty_spec}')" if isinstance(uncertainty_spec, str) else ""
    )
    if batch_size > 1 and uncertainty > 0:
        batch_info += f" (uncertainty: ±{uncertainty:.4f}{spec_suffix})"
    elif batch_size == 1 and uncertainty > 0:
        batch_info += f" (uncertainty available: ±{uncertainty:.4f}{spec_suffix}, not used)"

    # Format chi and noise rates for display (handle list format)
    chi_display = (
        exp_params.chi
        if isinstance(exp_params.chi, (int, float))
        else exp_params.chi[0] if len(exp_params.chi) == 1 else str(exp_params.chi)
    )
    relaxation_display = (
        exp_params.noise_config.relaxation
        if isinstance(exp_params.noise_config.relaxation, (int, float))
        else (
            exp_params.noise_config.relaxation[0]
            if len(exp_params.noise_config.relaxation) == 1
            else str(exp_params.noise_config.relaxation)
        )
    )
    dephasing_display = (
        exp_params.noise_config.dephasing
        if isinstance(exp_params.noise_config.dephasing, (int, float))
        else (
            exp_params.noise_config.dephasing[0]
            if len(exp_params.noise_config.dephasing) == 1
            else str(exp_params.noise_config.dephasing)
        )
    )
    depolarizing_display = (
        exp_params.noise_config.depolarizing
        if isinstance(exp_params.noise_config.depolarizing, (int, float))
        else (
            exp_params.noise_config.depolarizing[0]
            if len(exp_params.noise_config.depolarizing) == 1
            else str(exp_params.noise_config.depolarizing)
        )
    )

    system_info = f"""SYSTEM PARAMETERS AND CONFIGURATION

Physical Constants:
  • Photon-cavity coupling (γ):    {exp_params.photon_cavity_coupling:.6f} rad/time
  • Inverse pulse width (σ):        {exp_params.inverse_pulse_width:.6f} 1/time
  • Dispersive coupling (χ):        {chi_display if isinstance(chi_display, str) else f'{chi_display:.6f}'} rad/time

Rotation Parameters:
  • θ₁ (first rotation):   {np.degrees(theta1):>7.2f}° ({theta1:.6f} rad)
  • θ₂ (second rotation):  {np.degrees(theta2):>7.2f}° ({theta2:.6f} rad)

Noise Configuration:
  • Relaxation (γ_relax):   {relaxation_display if isinstance(relaxation_display, str) else f'{relaxation_display:.6f}'} rad/time
  • Dephasing (γ_deph):     {dephasing_display if isinstance(dephasing_display, str) else f'{dephasing_display:.6f}'} rad/time
  • Depolarizing (γ_depol): {depolarizing_display if isinstance(depolarizing_display, str) else f'{depolarizing_display:.6f}'} rad/time

Measurement Protocol:
  • Initial time:     {exp_params.measurement.initial_time:.6f}  |  Final time:  {exp_params.measurement.final_time:.6f}
  • Total evolution:  {exp_params.measurement.final_time - exp_params.measurement.initial_time:.6f}
{batch_info}
  • Computation mode: {mode}

Landscape Statistics:
  • Contrast range:     [{contrast_vals.min():.6f}, {contrast_vals.max():.6f}]  |  Variation:  {contrast_vals.max() - contrast_vals.min():.2e}
  • Optimal interval:   {optimal_interval:.6f}  |  N_measurements:  {optimal_n_meas}  |  Contrast:  {optimal_contrast:.8f}
  • Interval range:     [{interval_vals.min():.6f}, {interval_vals.max():.6f}]
"""

    # Add text box below plots
    fig.text(
        0.05,
        0.01,
        system_info,
        fontsize=9,
        family="monospace",
        verticalalignment="bottom",
        bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.7, pad=0.8),
    )

    # Save figure if path provided
    if save_path:
        save_path_obj = Path(save_path)
        save_path_obj.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"Time interval landscape plot saved to: {save_path}")

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

    # Plot probabilities
    if is_two_qubit:
        # Two-qubit plotting
        linestyles = {"prob_00": "-", "prob_01": "--", "prob_10": "-.", "prob_11": ":"}
        labels = {
            "prob_00": r"$P_{00}$",
            "prob_01": r"$P_{01}$",
            "prob_10": r"$P_{10}$",
            "prob_11": r"$P_{11}$",
        }

        for key in ["prob_00", "prob_01", "prob_10", "prob_11"]:
            if key in probabilities:
                ax.plot(
                    times,
                    probabilities[key],
                    label=labels[key],
                    linestyle=linestyles[key],
                    linewidth=2,
                )
    else:
        # Single-qubit plotting
        if "prob_0" in probabilities:
            ax.plot(times, probabilities["prob_0"], label="P(0)", linewidth=2, linestyle="-")
        if "prob_1" in probabilities:
            ax.plot(times, probabilities["prob_1"], label="P(1)", linewidth=2, linestyle="--")

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


def plot_sweep_results(
    sweep: "SweepResults",
    results_to_plot: Optional[List[str]] = None,
    figsize: Optional[Tuple[int, int]] = None,
    contour_levels: Optional[List[float]] = None,
    label_levels: Optional[List[float]] = None,
    mark_optimal: bool = True,
    save_path: Optional[str] = None,
    dpi: int = 300,
) -> Figure:
    """
    Unified function to plot parameter sweep results.

    This function creates contour plots for any parameter sweep, automatically
    detecting whether it's a chi-gamma sweep, asymmetry sweep, or custom sweep.
    It plots all results contained in the SweepResults object by default, or
    a user-specified subset.

    Args:
        sweep: SweepResults object from any compute_*_sweep function
        results_to_plot: Optional list of result keys to plot. If None, plots all results.
            Common keys: 'contrast_map', 'detection_map', 'detection_without_map',
            'p00', 'p01', 'p10', 'p11'
        figsize: Figure size (width, height) in inches. If None, automatically sized
        contour_levels: Levels for filled contours. Default: [0, 0.1, 0.2, ..., 1.0]
        label_levels: Levels for labeled line contours. Default: [0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 0.99, 1.0]
        mark_optimal: If True and contrast_map exists, mark the optimal point
        save_path: Optional path to save the figure
        dpi: Resolution for saved figure

    Returns:
        matplotlib Figure object

    Example:
        >>> # Chi-gamma sweep
        >>> sweep = compute_chi_gamma_sweep(exp, chi_interval=[0.1, 50], gamma_interval=[0.1, 30])
        >>> plot_sweep_results(sweep)  # Plots all results (contrast, detection maps)
        >>>
        >>> # Two-qubit sweep - plot only probabilities
        >>> sweep = compute_chi_gamma_sweep(exp_2q, ...)
        >>> plot_sweep_results(sweep, results_to_plot=['p00', 'p01', 'p10', 'p11'])
        >>>
        >>> # Asymmetry sweep
        >>> sweep = compute_asymmetry_coupling_sweep(exp_2q, ...)
        >>> plot_sweep_results(sweep, mark_optimal=True)
    """
    # Default contour levels
    if contour_levels is None:
        contour_levels = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0]

    if label_levels is None:
        label_levels = [0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 0.99, 1.0]

    # Determine which results to plot
    if results_to_plot is None:
        results_to_plot = list(sweep.results.keys())

    # Filter to only existing keys
    results_to_plot = [k for k in results_to_plot if k in sweep.results]

    if not results_to_plot:
        raise ValueError("No valid results to plot")

    n_plots = len(results_to_plot)

    # Create meshgrid for plotting
    Param1, Param2 = np.meshgrid(sweep.param1_vals, sweep.param2_vals)

    # Determine optimal point if available
    optimal_param1, optimal_param2 = None, None
    if mark_optimal and "contrast_map" in sweep.results:
        contrast_map = sweep.results["contrast_map"]
        max_idx = np.unravel_index(np.argmax(contrast_map), contrast_map.shape)
        # max_idx[0] is row index (param1), max_idx[1] is column index (param2)
        optimal_param1 = sweep.param1_vals[max_idx[0]]
        optimal_param2 = sweep.param2_vals[max_idx[1]]
    elif mark_optimal and "optimal_idx" in sweep.metadata:
        max_idx = sweep.metadata["optimal_idx"]
        # max_idx[0] is row index (param1), max_idx[1] is column index (param2)
        optimal_param1 = sweep.param1_vals[max_idx[0]]
        optimal_param2 = sweep.param2_vals[max_idx[1]]

    # Determine subplot layout
    if n_plots == 1:
        nrows, ncols = 1, 1
        subplot_size = 8
    elif n_plots == 2:
        nrows, ncols = 1, 2
        subplot_size = 8
    elif n_plots == 3:
        nrows, ncols = 3, 1
        subplot_size = 8
    elif n_plots == 4:
        nrows, ncols = 2, 2
        subplot_size = 6
    else:
        # General case: try to make square-ish
        ncols = int(np.ceil(np.sqrt(n_plots)))
        nrows = int(np.ceil(n_plots / ncols))
        subplot_size = 6

    # Determine figure size
    if figsize is None:
        figsize = (subplot_size * ncols, subplot_size * nrows)

    # Create figure
    fig, axes_array = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    axes = axes_array.flatten()

    # Title mapping for common result types
    title_map = {
        "contrast_map": "Sensing Contrast",
        "detection_map": "P(detection | with photon)",
        "detection_without_map": "P(detection | without photon)",
        "p00": r"$P_{00}$ (Both ground)",
        "p01": r"$P_{01}$ (Q1 ground, Q2 excited)",
        "p10": r"$P_{10}$ (Q1 excited, Q2 ground)",
        "p11": r"$P_{11}$ (Both excited)",
    }

    # Format parameter names for axis labels
    param1_label = (
        sweep.param1_name.replace("_", " ")
        .replace("gamma", "γ")
        .replace("chi", "χ")
        .replace("Delta", "Δ")
    )
    param2_label = (
        sweep.param2_name.replace("_", " ")
        .replace("gamma", "γ")
        .replace("chi", "χ")
        .replace("Delta", "Δ")
    )

    # Plot each result
    for idx, result_key in enumerate(results_to_plot):
        ax = axes[idx]
        result_data = sweep.results[result_key]

        # Transpose data to match param1-param2 orientation
        result_T = result_data.T

        # Filled contours
        cf = ax.contourf(Param1, Param2, result_T, levels=contour_levels, cmap="rainbow")
        cbar = plt.colorbar(cf, ax=ax, fraction=0.04)
        cbar.set_label("Probability" if "p" in result_key else "Value", fontsize=10)

        # Line contours with labels
        cs = ax.contour(Param1, Param2, result_T, levels=label_levels, colors="k", linewidths=0.2)
        ax.clabel(cs, inline=True, fontsize=8)

        # Mark optimal point
        if optimal_param1 is not None and optimal_param2 is not None:
            if result_key == "contrast_map":
                max_val = sweep.results[result_key][
                    np.unravel_index(
                        np.argmax(sweep.results[result_key]), sweep.results[result_key].shape
                    )
                ]
                ax.plot(
                    optimal_param1, optimal_param2, "r*", markersize=15, label=f"Max: {max_val:.4f}"
                )
            else:
                ax.plot(
                    optimal_param1, optimal_param2, "r*", markersize=15, label="At max contrast"
                )
            ax.legend(loc="upper right", fontsize=9)

        # Set scales
        if sweep.param1_scale == "log":
            ax.set_xscale("log")
        if sweep.param2_scale == "log":
            ax.set_yscale("log")

        # Labels and title
        ax.set_xlabel(param1_label, fontsize=11)
        ax.set_ylabel(param2_label, fontsize=11)
        ax.set_title(title_map.get(result_key, result_key), fontsize=12)
        ax.set_aspect("equal", adjustable="box")

    # Hide unused subplots
    for idx in range(n_plots, len(axes)):
        axes[idx].axis("off")

    # Build system characteristics text from metadata
    system_info_lines = []
    if sweep.metadata:
        # Extract common system parameters
        if "n_qubits" in sweep.metadata:
            system_info_lines.append(f"Qubits: {sweep.metadata['n_qubits']}")
        if "cavity_levels" in sweep.metadata:
            system_info_lines.append(f"Cavity levels: {sweep.metadata['cavity_levels']}")
        if "qubit_levels" in sweep.metadata:
            levels = sweep.metadata["qubit_levels"]
            if isinstance(levels, list):
                system_info_lines.append(f"Qubit levels: {levels}")
            else:
                system_info_lines.append(f"Qubit levels: {levels}")
        if "field_levels" in sweep.metadata:
            system_info_lines.append(f"Field levels: {sweep.metadata['field_levels']}")

        # Measurement info
        if "n_measurements" in sweep.metadata:
            system_info_lines.append(f"Measurements: {sweep.metadata['n_measurements']}")
        if "measurement_times" in sweep.metadata:
            times = sweep.metadata["measurement_times"]
            if isinstance(times, (list, np.ndarray)) and len(times) == 2:
                system_info_lines.append(f"Meas. times: [{times[0]:.1f}, {times[1]:.1f}]")
        if "initial_time_uncertainty" in sweep.metadata:
            system_info_lines.append(
                f"Time uncertainty: {sweep.metadata['initial_time_uncertainty']}"
            )

        # Noise parameters
        noise_lines = []
        if "depolarizing_rate" in sweep.metadata and sweep.metadata["depolarizing_rate"] > 0:
            noise_lines.append(f"Depolarizing: {sweep.metadata['depolarizing_rate']}")
        if "dephasing_rate" in sweep.metadata and sweep.metadata["dephasing_rate"] > 0:
            noise_lines.append(f"Dephasing: {sweep.metadata['dephasing_rate']}")
        if "relaxation_rate" in sweep.metadata and sweep.metadata["relaxation_rate"] > 0:
            noise_lines.append(f"Relaxation: {sweep.metadata['relaxation_rate']}")
        if noise_lines:
            system_info_lines.append("Noise: " + ", ".join(noise_lines))
        elif (
            "depolarizing_rate" in sweep.metadata
            or "dephasing_rate" in sweep.metadata
            or "relaxation_rate" in sweep.metadata
        ):
            system_info_lines.append("Noise: None")

        # Initial state
        if "initial_state" in sweep.metadata:
            system_info_lines.append(f"Initial: {sweep.metadata['initial_state']}")

        # Pulse parameters
        if "inverse_pulse_width" in sweep.metadata:
            system_info_lines.append(f"σ⁻¹: {sweep.metadata['inverse_pulse_width']}")

    # Add summary text if optimal point exists
    if (
        optimal_param1 is not None
        and optimal_param2 is not None
        and "max_contrast" in sweep.metadata
    ):
        summary_text = f"""OPTIMAL PARAMETERS
{param2_label}: {optimal_param2:.3f}
{param1_label}: {optimal_param1:.3f}
Contrast: {sweep.metadata['max_contrast']:.6f}"""

        fig.text(
            0.5,
            0.02,
            summary_text,
            fontsize=10,
            family="monospace",
            ha="center",
            va="bottom",
            bbox=dict(boxstyle="round", facecolor="lightgreen", alpha=0.7, pad=0.8),
        )

        # Add system info on the right side if available
        if system_info_lines:
            system_text = "SYSTEM INFO\n" + "\n".join(system_info_lines)
            fig.text(
                0.98,
                0.02,
                system_text,
                fontsize=9,
                family="monospace",
                ha="right",
                va="bottom",
                bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.7, pad=0.8),
            )

        plt.tight_layout(rect=[0, 0.08, 1, 1])
    else:
        # No optimal point, but still show system info if available
        if system_info_lines:
            system_text = "SYSTEM INFO\n" + "\n".join(system_info_lines)
            fig.text(
                0.98,
                0.02,
                system_text,
                fontsize=9,
                family="monospace",
                ha="right",
                va="bottom",
                bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.7, pad=0.8),
            )
            plt.tight_layout(rect=[0, 0.08, 1, 1])
        else:
            plt.tight_layout()

    # Save figure if path provided
    if save_path:
        save_path_obj = Path(save_path)
        save_path_obj.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"Sweep results plot saved to: {save_path}")

    return fig
