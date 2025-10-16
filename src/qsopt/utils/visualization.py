"""
Visualization utilities for quantum sensing optimization results.

This module provides functions for creating comprehensive optimization dashboards
that display key metrics including sensing contrast, gradient evolution, parameter
trajectories, and detection probabilities.
"""

from typing import Optional, Dict, List, Tuple, Union
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from pathlib import Path

from qsopt.core.callback import OptimizationCallback
from qsopt.core.experimental_parameters import ExperimentalParameters

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
    dpi: int = 300
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
        optimization_callback: OptimizationCallback from optimize() method
            Contains history of epochs, contrast, probabilities, and parameters
        reference_callback: Optional SimulationCallback from run_simulation()
            If provided, reference values are shown as horizontal benchmark lines
        show_contrast: Display sensing contrast evolution plot
        show_gradients: Display gradient magnitude evolution plot
        show_parameters: Display parameter evolution plot
        show_trajectory: Display optimization trajectory in parameter space
        show_probabilities: Display detection probabilities plot
        figsize: Figure size as (width, height) in inches
        save_path: Optional path to save the figure (e.g., 'dashboard.pdf')
            If None, figure is displayed but not saved
        dpi: Resolution for saved figure (default: 300)
    
    Returns:
        matplotlib Figure object containing the dashboard
    
    Example:
        >>> # Basic usage with optimization only
        >>> history = experiment.optimize(theta_init=[1.5, -1.3], num_steps=50)
        >>> fig = plot_optimization_dashboard(history)
        >>> 
        >>> # With reference comparison
        >>> results = experiment.run_simulation()
        >>> history = experiment.optimize(theta_init=[1.5, -1.3], num_steps=50)
        >>> fig = plot_optimization_dashboard(history, reference_callback=results,
        ...                                   save_path='opt_dashboard.pdf')
        >>> 
        >>> # Selective plotting
        >>> fig = plot_optimization_dashboard(history, 
        ...                                   show_gradients=False,
        ...                                   show_trajectory=False)
    """
    # Count active plots to determine layout
    active_plots = [show_contrast, show_gradients, show_parameters, 
                   show_trajectory, show_probabilities]
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
    epochs = np.array(history['epochs'])
    contrast = np.array(history['contrast'])
    prob_with = np.array(history['prob_with'])
    prob_without = np.array(history['prob_without'])
    
    # Extract parameter arrays (assuming rotation angles)
    param_arrays = []
    param_names = []
    if history['trainable_params']:
        first_params = history['trainable_params'][0]
        angles = first_params.get_rotation_angles()
        param_names = list(angles.keys())
        
        for tp in history['trainable_params']:
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
        if ref_history['contrast']:
            reference_contrast = ref_history['contrast'][0]
            reference_prob_with = ref_history['prob_with'][0]
            reference_prob_without = ref_history['prob_without'][0]
        
        if ref_history['trainable_params']:
            ref_tp = ref_history['trainable_params'][0]
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
        
        ax.plot(epochs, contrast, 'g-', linewidth=2, alpha=0.8, label='Optimized')
        
        if reference_contrast is not None:
            ax.axhline(y=reference_contrast, color='red', linestyle='--', 
                      linewidth=2, alpha=0.7, label='Reference')
        
        ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
        ax.set_ylabel('Sensing Contrast', fontsize=12, fontweight='bold')
        ax.set_title('Sensing Contrast Evolution', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
    
    # Plot 2: Gradient Magnitude Evolution
    if show_gradients and len(grad_norms) > 0:
        ax = plt.subplot(n_rows, n_cols, plot_idx + 1)
        axes.append(ax)
        plot_idx += 1
        
        ax.semilogy(epochs, grad_norms, 'm-', linewidth=2, alpha=0.8)
        ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
        ax.set_ylabel('Gradient Magnitude', fontsize=12, fontweight='bold')
        ax.set_title('Gradient Evolution (Log Scale)', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    # Plot 3: Parameter Evolution
    if show_parameters and len(param_arrays) > 0:
        ax = plt.subplot(n_rows, n_cols, plot_idx + 1)
        axes.append(ax)
        plot_idx += 1
        
        colors = ['r', 'b', 'g', 'orange', 'purple', 'brown']
        
        for i, name in enumerate(param_names):
            color = colors[i % len(colors)]
            params_deg = param_arrays[:, i] * 180 / np.pi
            ax.plot(epochs, params_deg, '-', linewidth=2, 
                   label=name, color=color, alpha=0.8)
            
            # Add reference line if available
            if reference_params is not None:
                ref_deg = reference_params[i] * 180 / np.pi
                ax.axhline(y=ref_deg, color=color, linestyle='--', 
                          alpha=0.5, linewidth=1.5)
        
        ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
        ax.set_ylabel('Rotation Angle (degrees)', fontsize=12, fontweight='bold')
        ax.set_title('Parameter Evolution', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
    
    # Plot 4: Optimization Trajectory in Parameter Space
    if show_trajectory and len(param_arrays) > 0 and len(param_names) >= 2:
        ax = plt.subplot(n_rows, n_cols, plot_idx + 1)
        axes.append(ax)
        plot_idx += 1
        
        # Use first two parameters for trajectory plot
        theta1_deg = param_arrays[:, 0] * 180 / np.pi
        theta2_deg = param_arrays[:, 1] * 180 / np.pi
        
        # Plot trajectory with color gradient
        scatter = ax.scatter(theta1_deg, theta2_deg, c=epochs, cmap='viridis',
                           s=30, alpha=0.7, edgecolors='black', linewidth=0.5)
        ax.plot(theta1_deg, theta2_deg, 'k-', alpha=0.3, linewidth=1)
        
        # Mark start and end points
        ax.plot(theta1_deg[0], theta2_deg[0], 'ro', markersize=8, 
               label='Start', markeredgecolor='black')
        ax.plot(theta1_deg[-1], theta2_deg[-1], 'gs', markersize=8, 
               label='End', markeredgecolor='black')
        
        # Mark reference point if available
        if reference_params is not None:
            ref_theta1_deg = reference_params[0] * 180 / np.pi
            ref_theta2_deg = reference_params[1] * 180 / np.pi
            ax.plot(ref_theta1_deg, ref_theta2_deg, 'b^', markersize=10,
                   label='Reference', markeredgecolor='black')
        
        ax.set_xlabel(f'{param_names[0]} (degrees)', fontsize=12, fontweight='bold')
        ax.set_ylabel(f'{param_names[1]} (degrees)', fontsize=12, fontweight='bold')
        ax.set_title('Optimization Trajectory', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
        cbar.set_label('Epoch', fontsize=10)
    
    # Plot 5: Detection Probabilities Evolution
    if show_probabilities:
        ax = plt.subplot(n_rows, n_cols, plot_idx + 1)
        axes.append(ax)
        plot_idx += 1
        
        ax.plot(epochs, prob_with, 'g-', linewidth=2,
               label='With Photon (Optimized)', alpha=0.8)
        ax.plot(epochs, prob_without, 'r-', linewidth=2,
               label='Without Photon (Optimized)', alpha=0.8)
        
        # Add reference benchmarks if available
        if reference_prob_with is not None:
            ax.axhline(y=reference_prob_with, color='green', linestyle='--',
                      linewidth=2, alpha=0.6, label='With Photon (Reference)')
        if reference_prob_without is not None:
            ax.axhline(y=reference_prob_without, color='red', linestyle='--',
                      linewidth=2, alpha=0.6, label='Without Photon (Reference)')
        
        ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
        ax.set_ylabel('Detection Probability', fontsize=12, fontweight='bold')
        ax.set_title('Detection Probabilities Evolution', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
    
    # Overall title
    plt.suptitle('Optimization Dashboard', fontsize=18, fontweight='bold')
    plt.tight_layout()
    
    # Save if path provided
    if save_path is not None:
        # Create directory if it doesn't exist
        save_path_obj = Path(save_path)
        save_path_obj.parent.mkdir(parents=True, exist_ok=True)
        
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    
    return fig


def plot_contrast_evolution(
    optimization_callback: OptimizationCallback,
    reference_callback: Optional[OptimizationCallback] = None,
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None,
    dpi: int = 300
) -> Figure:
    """
    Create a standalone plot of sensing contrast evolution.
    
    Args:
        optimization_callback: OptimizationCallback from optimize()
        reference_callback: Optional reference from run_simulation()
        figsize: Figure size as (width, height)
        save_path: Optional path to save figure
        dpi: Resolution for saved figure
    
    Returns:
        matplotlib Figure object
    """
    history = optimization_callback.get_history()
    epochs = np.array(history['epochs'])
    contrast = np.array(history['contrast'])
    
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.plot(epochs, contrast, 'g-', linewidth=2.5, alpha=0.8, 
           label='Optimized', marker='o', markersize=4)
    
    if reference_callback is not None:
        ref_history = reference_callback.get_history()
        if ref_history['contrast']:
            reference_contrast = ref_history['contrast'][0]
            ax.axhline(y=reference_contrast, color='red', linestyle='--',
                      linewidth=2, alpha=0.7, label='Reference')
    
    ax.set_xlabel('Epoch', fontsize=14, fontweight='bold')
    ax.set_ylabel('Sensing Contrast', fontsize=14, fontweight='bold')
    ax.set_title('Sensing Contrast Evolution', fontsize=16, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path is not None:
        # Create directory if it doesn't exist
        save_path_obj = Path(save_path)
        save_path_obj.parent.mkdir(parents=True, exist_ok=True)
        
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    
    return fig


def plot_parameter_trajectory(
    optimization_callback: OptimizationCallback,
    reference_callback: Optional[OptimizationCallback] = None,
    param_indices: Tuple[int, int] = (0, 1),
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[str] = None,
    dpi: int = 300
) -> Figure:
    """
    Create a standalone plot of optimization trajectory in parameter space.
    
    Args:
        optimization_callback: OptimizationCallback from optimize()
        reference_callback: Optional reference from run_simulation()
        param_indices: Tuple of parameter indices to plot (default: (0, 1))
        figsize: Figure size as (width, height)
        save_path: Optional path to save figure
        dpi: Resolution for saved figure
    
    Returns:
        matplotlib Figure object
    """
    history = optimization_callback.get_history()
    epochs = np.array(history['epochs'])
    
    # Extract parameters
    param_arrays = []
    param_names = []
    if history['trainable_params']:
        first_params = history['trainable_params'][0]
        angles = first_params.get_rotation_angles()
        param_names = list(angles.keys())
        
        for tp in history['trainable_params']:
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
    scatter = ax.scatter(theta1_deg, theta2_deg, c=epochs, cmap='viridis',
                        s=50, alpha=0.7, edgecolors='black', linewidth=1)
    ax.plot(theta1_deg, theta2_deg, 'k-', alpha=0.4, linewidth=1.5)
    
    # Mark start and end
    ax.plot(theta1_deg[0], theta2_deg[0], 'ro', markersize=12,
           label='Start', markeredgecolor='black', markeredgewidth=2)
    ax.plot(theta1_deg[-1], theta2_deg[-1], 'gs', markersize=12,
           label='End', markeredgecolor='black', markeredgewidth=2)
    
    # Mark reference if available
    if reference_callback is not None:
        ref_history = reference_callback.get_history()
        if ref_history['trainable_params']:
            ref_tp = ref_history['trainable_params'][0]
            ref_angles = ref_tp.get_rotation_angles()
            ref_params = [ref_angles[name][0] for name in param_names]
            ref_theta1_deg = ref_params[idx1] * 180 / np.pi
            ref_theta2_deg = ref_params[idx2] * 180 / np.pi
            ax.plot(ref_theta1_deg, ref_theta2_deg, 'b^', markersize=14,
                   label='Reference', markeredgecolor='black', markeredgewidth=2)
    
    ax.set_xlabel(f'{param_names[idx1]} (degrees)', fontsize=14, fontweight='bold')
    ax.set_ylabel(f'{param_names[idx2]} (degrees)', fontsize=14, fontweight='bold')
    ax.set_title('Optimization Trajectory', fontsize=16, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Epoch', fontsize=12)
    
    plt.tight_layout()
    
    if save_path is not None:
        # Create directory if it doesn't exist
        save_path_obj = Path(save_path)
        save_path_obj.parent.mkdir(parents=True, exist_ok=True)
        
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"Trajectory plot saved to: {save_path}")
    
    return fig


def plot_parameter_landscape(
    landscape_data: Dict[str, Union[np.ndarray, float]],
    exp_params: 'ExperimentalParameters',
    save_path: Optional[str] = None,
    dpi: int = 300
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
    theta1_vals = landscape_data['theta1_vals']
    theta2_vals = landscape_data['theta2_vals']
    contrast_map = landscape_data['contrast_map']
    detection_map = landscape_data['detection_map']
    center_theta1 = landscape_data['center_theta1']
    center_theta2 = landscape_data['center_theta2']
    
    # Create meshgrid
    P1, P2 = np.meshgrid(theta1_vals, theta2_vals)
    
    # Convert to degrees for display
    P1_deg = np.degrees(P1)
    P2_deg = np.degrees(P2)
    center_x = np.degrees(center_theta1)
    center_y = np.degrees(center_theta2)
    
    # Plot 1: Contrast landscape
    im1 = ax1.contourf(P1_deg, P2_deg, contrast_map, 
                       levels=30, cmap='viridis')
    ax1.set_xlabel('θ₁ (degrees)', fontsize=12)
    ax1.set_ylabel('θ₂ (degrees)', fontsize=12)
    ax1.set_title('Sensing Contrast Landscape', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    cbar1 = plt.colorbar(im1, ax=ax1, label='Contrast')
    
    # Find and mark maximum contrast
    max_idx = np.unravel_index(np.argmax(contrast_map), contrast_map.shape)
    max_x = P1_deg[max_idx]
    max_y = P2_deg[max_idx]
    max_contrast = contrast_map[max_idx]
    
    # Mark points on contrast plot
    ax1.plot(center_x, center_y, 'w+', markersize=15, markeredgewidth=3,
             label='Center point', zorder=10)
    ax1.plot(max_x, max_y, 'ro', markersize=10, markerfacecolor='red',
             markeredgecolor='white', markeredgewidth=2,
             label=f'Max = {max_contrast:.6f}', zorder=10)
    ax1.legend(loc='upper right', fontsize=10)
    
    # Plot 2: Detection probability landscape
    im2 = ax2.contourf(P1_deg, P2_deg, detection_map,
                       levels=30, cmap='plasma')
    ax2.set_xlabel('θ₁ (degrees)', fontsize=12)
    ax2.set_ylabel('θ₂ (degrees)', fontsize=12)
    ax2.set_title('Detection Probability Landscape (with photon)', 
                  fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    cbar2 = plt.colorbar(im2, ax=ax2, label='Detection Probability')
    
    # Mark center point
    ax2.plot(center_x, center_y, 'w+', markersize=15, markeredgewidth=3,
             label='Center point', zorder=10)
    ax2.legend(loc='upper right', fontsize=10)
    
    # Adjust layout to leave space at bottom
    plt.tight_layout(rect=(0, 0.12, 1, 1))
    
    # Create comprehensive system information box
    if exp_params._measurement_times_list is not None and len(exp_params._measurement_times_list) > 1:
        meas_times = exp_params._measurement_times_list
        time_intervals = np.diff(meas_times)
        avg_interval = np.mean(time_intervals)
        interval_text = f"{avg_interval:.6f}"
        n_measurements = len(exp_params._measurement_times_list)
    else:
        interval_text = "N/A"
        n_measurements = 0
    
    system_info = f"""SYSTEM PARAMETERS AND CONFIGURATION

Physical Constants:
  • Photon-cavity coupling (γ):    {exp_params.photon_cavity_coupling:.6f} rad/time
  • Inverse pulse width (σ):        {exp_params.inverse_pulse_width:.6f} 1/time
  • Dispersive coupling (χ):        {exp_params.chi:.6f} rad/time

System Dimensions:
  • Cavity levels:  {exp_params.cavity_levels}  |  Qubit levels:  {exp_params.qubit_levels}  |  Field levels:  {exp_params.field_levels}

Noise Configuration:
  • Relaxation (γ_relax):   {exp_params.noise_config.relaxation:.6f} rad/time
  • Dephasing (γ_deph):     {exp_params.noise_config.dephasing:.6f} rad/time
  • Depolarizing (γ_depol): {exp_params.noise_config.depolarizing:.6f} rad/time

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
    fig.text(0.05, 0.02, system_info, fontsize=9, family='monospace',
             verticalalignment='bottom',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7, pad=0.8))
    
    # Save figure if path provided
    if save_path:
        save_path_obj = Path(save_path)
        save_path_obj.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"Landscape plot saved to: {save_path}")
    
    return fig


def plot_time_interval_landscape(
    landscape_data: Dict[str, Union[np.ndarray, float, str, int]],
    exp_params: 'ExperimentalParameters',
    save_path: Optional[str] = None,
    dpi: int = 300
) -> Figure:
    """
    Plot time interval landscape with system information.
    
    Creates a comprehensive visualization showing:
    1. Sensing contrast vs time interval
    2. Detection probabilities (with and without photon) vs time interval
    3. Number of measurements vs time interval
    
    Includes system information box showing:
    - Physical constants (coupling strengths, pulse widths)
    - Rotation parameters (θ₁, θ₂)
    - Noise configuration
    - Batch averaging details (if used)
    - Optimal interval statistics
    
    Args:
        landscape_data: Dictionary from compute_time_interval_landscape() containing:
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
    # Create figure with three subplots stacked vertically
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 14))
    
    # Extract data
    interval_vals = landscape_data['interval_vals']
    contrast_vals = landscape_data['contrast_vals']
    detection_with = landscape_data['detection_with']
    detection_without = landscape_data['detection_without']
    n_measurements = landscape_data['n_measurements']
    theta1 = landscape_data['theta1']
    theta2 = landscape_data['theta2']
    mode = landscape_data['mode']
    batch_size = landscape_data['batch_size']
    uncertainty = landscape_data['initial_time_uncertainty']
    
    # Find optimal interval
    optimal_idx = np.argmax(contrast_vals)
    optimal_interval = interval_vals[optimal_idx]
    optimal_contrast = contrast_vals[optimal_idx]
    optimal_n_meas = n_measurements[optimal_idx]
    
    # Plot 1: Sensing contrast vs time interval
    if mode == 'discrete':
        ax1.plot(interval_vals, contrast_vals, 'bo-', linewidth=2, markersize=6, label='Contrast')
    else:
        ax1.plot(interval_vals, contrast_vals, 'b-', linewidth=2, label='Contrast')
    
    # Mark optimal point
    ax1.axvline(optimal_interval, color='red', linestyle='--', alpha=0.7, linewidth=1.5)
    ax1.plot(optimal_interval, optimal_contrast, 'ro', markersize=10, 
             markerfacecolor='red', markeredgecolor='white', markeredgewidth=2,
             label=f'Optimal: Δt={optimal_interval:.4f}', zorder=10)
    
    ax1.set_xlabel('Time Interval (Δt)', fontsize=12)
    ax1.set_ylabel('Sensing Contrast', fontsize=12)
    ax1.set_title(f'Sensing Contrast vs Time Interval ({mode} mode)', 
                  fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='best', fontsize=10)
    
    # Plot 2: Detection probabilities
    if mode == 'discrete':
        ax2.plot(interval_vals, detection_with, 'go-', linewidth=2, markersize=5, 
                label='With photon', alpha=0.8)
        ax2.plot(interval_vals, detection_without, 'mo-', linewidth=2, markersize=5,
                label='Without photon', alpha=0.8)
    else:
        ax2.plot(interval_vals, detection_with, 'g-', linewidth=2, 
                label='With photon', alpha=0.8)
        ax2.plot(interval_vals, detection_without, 'm-', linewidth=2,
                label='Without photon', alpha=0.8)
    
    # Mark optimal point
    ax2.axvline(optimal_interval, color='red', linestyle='--', alpha=0.7, linewidth=1.5)
    
    ax2.set_xlabel('Time Interval (Δt)', fontsize=12)
    ax2.set_ylabel('Detection Probability', fontsize=12)
    ax2.set_title('Detection Probabilities vs Time Interval', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='best', fontsize=10)
    
    # Plot 3: Number of measurements
    if mode == 'discrete':
        ax3.plot(interval_vals, n_measurements, 'ko-', linewidth=2, markersize=5,
                label='Number of measurements')
    else:
        ax3.plot(interval_vals, n_measurements, 'k-', linewidth=2,
                label='Number of measurements')
    
    # Mark optimal point
    ax3.axvline(optimal_interval, color='red', linestyle='--', alpha=0.7, linewidth=1.5)
    ax3.plot(optimal_interval, optimal_n_meas, 'ro', markersize=10,
             markerfacecolor='red', markeredgecolor='white', markeredgewidth=2, zorder=10)
    
    ax3.set_xlabel('Time Interval (Δt)', fontsize=12)
    ax3.set_ylabel('Number of Measurements', fontsize=12)
    ax3.set_title('Measurement Count vs Time Interval', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc='best', fontsize=10)
    
    # Adjust layout to leave space at bottom for info box
    plt.tight_layout(rect=(0, 0.15, 1, 1))
    
    # Create comprehensive system information box
    batch_info = f"  • Batch size: {batch_size} realizations"
    if batch_size > 1 and uncertainty > 0:
        batch_info += f" (uncertainty: ±{uncertainty:.4f})"
    elif batch_size == 1 and uncertainty > 0:
        batch_info += f" (uncertainty available: ±{uncertainty:.4f}, not used)"
    
    system_info = f"""SYSTEM PARAMETERS AND CONFIGURATION

Physical Constants:
  • Photon-cavity coupling (γ):    {exp_params.photon_cavity_coupling:.6f} rad/time
  • Inverse pulse width (σ):        {exp_params.inverse_pulse_width:.6f} 1/time
  • Dispersive coupling (χ):        {exp_params.chi:.6f} rad/time

Rotation Parameters:
  • θ₁ (first rotation):   {np.degrees(theta1):>7.2f}° ({theta1:.6f} rad)
  • θ₂ (second rotation):  {np.degrees(theta2):>7.2f}° ({theta2:.6f} rad)

Noise Configuration:
  • Relaxation (γ_relax):   {exp_params.noise_config.relaxation:.6f} rad/time
  • Dephasing (γ_deph):     {exp_params.noise_config.dephasing:.6f} rad/time
  • Depolarizing (γ_depol): {exp_params.noise_config.depolarizing:.6f} rad/time

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
    fig.text(0.05, 0.01, system_info, fontsize=9, family='monospace',
             verticalalignment='bottom',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7, pad=0.8))
    
    # Save figure if path provided
    if save_path:
        save_path_obj = Path(save_path)
        save_path_obj.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"Time interval landscape plot saved to: {save_path}")
    
    return fig


def plot_pulse_shape_with_measurements(
    exp_params: 'ExperimentalParameters',
    save_path: Optional[str] = None,
    dpi: int = 300
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
        
    Returns:
        matplotlib.figure.Figure: Figure object containing the plot
        
    Example:
        >>> from qsopt.core.experimental_parameters import ExperimentalParameters
        >>> from qsopt.core.trainable_parameters import TrainableParameters
        >>> exp_params = ExperimentalParameters(...)
        >>> train_params = TrainableParameters(theta1=0.5, theta2=1.2, time_interval=0.1)
        >>> fig = plot_pulse_shape_with_measurements(
        ...     exp_params, train_params.theta1, train_params.theta2
        ... )
        >>> # Plot shows pulse shape with measurement markers
        
    Note:
        - Pulse shape is computed using the u0() function from quantum_utils
        - Measurement times are extracted from exp_params.measurement
        - The plot window extends beyond the measurement range to show pulse decay
    """
    from ..core.quantum_utils import u0
    
    # Extract measurement times
    initial_time = exp_params.measurement.initial_time
    final_time = exp_params.measurement.final_time
    interval = exp_params.measurement.time_interval
    
    # Generate measurement times
    times = np.arange(initial_time, final_time + interval/2, interval)
    n_measurements = len(times)
    
    # Create time array for plotting pulse (extend beyond measurement range)
    time_range = final_time - initial_time
    t_plot = np.linspace(initial_time - 0.3*time_range, 
                         final_time + 0.3*time_range, 
                         1000)
    
    # Compute pulse envelope using u0
    sigma = exp_params.inverse_pulse_width
    pulse_vals = np.array([float(u0(t, sigma=sigma)) for t in t_plot])
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot pulse envelope
    ax.plot(t_plot, pulse_vals, 'b-', linewidth=2, label='Gaussian pulse envelope')
    
    # Add vertical lines for measurement times
    for i, t_meas in enumerate(times):
        if i == 0:
            ax.axvline(t_meas, color='red', linestyle='--', alpha=0.6, 
                      linewidth=1.5, label='Measurement times')
        else:
            ax.axvline(t_meas, color='red', linestyle='--', alpha=0.6, linewidth=1.5)
    
    # Shade the measurement region
    ax.axvspan(initial_time, final_time, alpha=0.1, color='green', 
               label='Measurement window')
    
    # Formatting
    ax.set_xlabel('Time (1/σ)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Pulse amplitude |u₀(t)|', fontsize=12, fontweight='bold')
    ax.set_title('Gaussian Pulse Shape with Measurement Protocol', 
                fontsize=14, fontweight='bold', pad=15)
    ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
    ax.set_ylim([0, 1.1])
    
    # Add system information
    system_info = f"""PULSE AND MEASUREMENT CONFIGURATION

Physical Parameters:
  • Inverse pulse width (σ):        {exp_params.inverse_pulse_width:.6f} 1/time

Measurement Protocol:
  • Initial time:       {initial_time:.6f}
  • Final time:         {final_time:.6f}
  • Time interval:      {interval:.6f}
  • Total duration:     {time_range:.6f}
  • Number of measurements:  {n_measurements}
"""
    
    # Add text box with system info
    fig.text(0.05, 0.01, system_info, fontsize=9, family='monospace',
             verticalalignment='bottom',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7, pad=0.8))
    
    plt.tight_layout()
    
    # Save figure if path provided
    if save_path:
        save_path_obj = Path(save_path)
        save_path_obj.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"Pulse shape plot saved to: {save_path}")
    
    return fig
