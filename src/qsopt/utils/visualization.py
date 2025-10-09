"""
Visualization utilities for quantum sensing optimization results.

This module provides functions for creating comprehensive optimization dashboards
that display key metrics including sensing contrast, gradient evolution, parameter
trajectories, and detection probabilities.
"""

from typing import Optional, Dict, List, Tuple
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec

from qsopt.core.callback import OptimizationCallback


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
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"Dashboard saved to: {save_path}")
    
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
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"Contrast plot saved to: {save_path}")
    
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
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"Trajectory plot saved to: {save_path}")
    
    return fig
