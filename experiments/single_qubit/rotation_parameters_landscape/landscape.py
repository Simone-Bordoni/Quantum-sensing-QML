"""
Parameter Space Landscape Analysis using qsopt module
======================================================

This script analyzes the parameter space landscape for quantum sensing
optimization using the θ₁, θ₂ parameterization strategy.

Uses run_simulation() with different rotation parameters and plots heatmaps of:
- Sensing contrast landscape
- Detection probability landscape

The analysis uses time-interval based measurements and includes comprehensive
system parameters in the visualization.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import time

# Import qsopt modules
from qsopt.core.experimental_parameters import ExperimentalParameters
from qsopt.core.trainable_parameters import TrainableParameters
from qsopt.core.experiment import SingleQubitExperiment


def create_experiment_setup():
    """
    Create experimental parameters using time-interval based measurements.
    
    System parameters:
    - Cavity levels: 2
    - Photon-cavity coupling (γ): 0.18850
    - Pulse width inverse (σ): 0.01885
    - Dispersive coupling (χ): 0.09425
    - Noise - Relaxation: 0.000628
    - Noise - Dephasing: 0.000628
    - Noise - Depolarizing: 0.0
    - Measurement: initial_time, final_time, time_interval based
    """
    # Create experimental parameters
    exp_params = ExperimentalParameters()
    
    # Physical constants
    gm = 0.03 * 2 * np.pi  # photon-cavity coupling
    sigma = 0.1 * gm  # inverse of the pulse width
    chi = 0.5 * gm  # dispersive coupling
    
    exp_params.photon_cavity_coupling = gm
    exp_params.inverse_pulse_width = sigma
    exp_params.chi = chi
    
    # System dimensions
    exp_params.cavity_levels = 2
    exp_params.qubit_levels = 2
    exp_params.field_levels = 2
    
    # Noise configuration
    gamma_relax = 0.000 * 2 * np.pi
    gamma_dephasing = 0.000 * 2 * np.pi
    gamma_depol = 0.0
    
    exp_params.noise_config.relaxation = gamma_relax
    exp_params.noise_config.dephasing = gamma_dephasing
    exp_params.noise_config.depolarizing = gamma_depol
    
    initial_time = -5.0 / sigma
    final_time = 5.0 / sigma
    time_interval = 10 / sigma  # Interval between consecutive measurements
    
    exp_params.measurement.initial_time = initial_time
    exp_params.measurement.final_time = final_time
    exp_params.measurement.time_interval = time_interval
    
    # Initial state: |1,0,0⟩ (1 photon in input cavity)
    from qsopt.core.experimental_parameters import InitialStateType
    exp_params.initial_state.state_type = InitialStateType.SINGLE_PHOTON
    
    return exp_params


def compute_theta1_theta2_landscape(exp_params, resolution=25, 
                                     center_theta1=np.pi/2, center_theta2=-np.pi/2,
                                     param_range=np.pi/6, verbose=True):
    """
    Compute parameter landscape for θ₁, θ₂ strategy.
    
    Args:
        exp_params: ExperimentalParameters instance
        resolution: Number of points per dimension
        center_theta1: Center value for θ₁ (radians)
        center_theta2: Center value for θ₂ (radians)
        param_range: Range around center (±param_range)
        verbose: Print progress
        
    Returns:
        dict: Contains theta1_vals, theta2_vals, contrast_map, detection_map
    """
    if verbose:
        print("Computing θ₁, θ₂ landscape...")
        print(f"  Resolution: {resolution}×{resolution}")
        print(f"  Center: θ₁={np.degrees(center_theta1):.1f}°, θ₂={np.degrees(center_theta2):.1f}°")
        print(f"  Range: ±{np.degrees(param_range):.1f}°")
    
    # Create parameter grid
    theta1_vals = np.linspace(center_theta1 - param_range, 
                              center_theta1 + param_range, 
                              resolution)
    theta2_vals = np.linspace(center_theta2 - param_range, 
                              center_theta2 + param_range, 
                              resolution)
    
    # Initialize result arrays
    contrast_map = np.zeros((resolution, resolution))
    detection_map = np.zeros((resolution, resolution))
    
    # Create trainable parameters template
    trainable_params = TrainableParameters()
    trainable_params.add_rotation_angles(
        names=['theta1', 'theta2'],
        initial_values=[0.0, 0.0],  # Will be overwritten
        trainable=[False, False]  # Not training, just evaluating
    )
    
    # Create experiment
    exp = SingleQubitExperiment(exp_params, trainable_params)
    
    start_time = time.time()
    total_points = resolution * resolution
    
    # Compute landscape
    for i, theta1 in enumerate(theta1_vals):
        for j, theta2 in enumerate(theta2_vals):
            # Update parameters
            exp.trainable_params.parameters[0].value = theta1
            exp.trainable_params.parameters[1].value = theta2
            
            # Run simulation
            callback = exp.run_simulation(batch_size=1)
            
            # Store results (use history lists)
            contrast_map[j, i] = callback.history['contrast'][-1]  # Note: j,i for correct orientation
            detection_map[j, i] = callback.history['prob_with'][-1]
            
            # Progress update
            if verbose and ((i * resolution + j) % 50 == 0):
                progress = (i * resolution + j) / total_points * 100
                elapsed = time.time() - start_time
                eta = elapsed / max(i * resolution + j, 1) * (total_points - i * resolution - j)
                print(f"  Progress: {progress:.1f}% (ETA: {eta:.1f}s)", end='\r')
    
    if verbose:
        elapsed = time.time() - start_time
        print(f"\n  ✓ Completed in {elapsed:.1f}s ({elapsed/total_points:.3f}s per point)")
    
    return {
        'theta1_vals': theta1_vals,
        'theta2_vals': theta2_vals,
        'contrast_map': contrast_map,
        'detection_map': detection_map,
        'center_theta1': center_theta1,
        'center_theta2': center_theta2
    }


def plot_landscape(data, exp_params, save_path=None):
    """
    Plot contrast and detection probability landscapes for θ₁, θ₂ strategy.
    
    Args:
        data: Dictionary with landscape data
        exp_params: ExperimentalParameters instance for system info
        save_path: Optional path to save figure
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 16))
    
    # θ₁, θ₂ strategy
    theta1_vals = data['theta1_vals']
    theta2_vals = data['theta2_vals']
    P1, P2 = np.meshgrid(theta1_vals, theta2_vals)
    
    # Convert to degrees
    P1_deg = np.degrees(P1)
    P2_deg = np.degrees(P2)
    
    xlabel = 'θ₁ (degrees)'
    ylabel = 'θ₂ (degrees)'
    
    center_x = np.degrees(data['center_theta1'])
    center_y = np.degrees(data['center_theta2'])
    
    # Plot 1: Contrast landscape
    im1 = ax1.contourf(P1_deg, P2_deg, data['contrast_map'], 
                       levels=30, cmap='viridis')
    ax1.set_xlabel(xlabel, fontsize=12)
    ax1.set_ylabel(ylabel, fontsize=12)
    ax1.set_title('Contrast Landscape Around Optimal Region', 
                  fontsize=14)
    ax1.grid(True, alpha=0.3)
    plt.colorbar(im1, ax=ax1, label='Contrast')
    
    # Find and mark maximum contrast
    max_idx = np.unravel_index(np.argmax(data['contrast_map']), 
                               data['contrast_map'].shape)
    max_x = P1_deg[max_idx]
    max_y = P2_deg[max_idx]
    max_contrast = data['contrast_map'][max_idx]
    
    # Mark points on contrast plot
    ax1.plot(center_x, center_y, 'w+', markersize=12, markeredgewidth=3,
             label=f'Center point')
    ax1.plot(max_x, max_y, 'ro', markersize=8, 
             label=f'Max contrast = {max_contrast:.6f}')
    ax1.legend(loc='upper right')
    
    # Plot 2: Detection probability landscape
    im2 = ax2.contourf(P1_deg, P2_deg, data['detection_map'],
                       levels=30, cmap='plasma')
    ax2.set_xlabel(xlabel, fontsize=12)
    ax2.set_ylabel(ylabel, fontsize=12)
    ax2.set_title('Detection Probability Landscape Around Optimal Region',
                  fontsize=14)
    ax2.grid(True, alpha=0.3)
    plt.colorbar(im2, ax=ax2, label='Detection Probability')
    
    # Mark center point
    ax2.plot(center_x, center_y, 'w+', markersize=12, markeredgewidth=3,
             label='Center point')
    ax2.legend(loc='upper right')
    
    plt.tight_layout(rect=(0, 0.12, 1, 1))  # Leave space at bottom for text box
    
    # Create comprehensive system information box below the plots
    # Calculate measurement interval
    if exp_params._measurement_times_list is not None and len(exp_params._measurement_times_list) > 1:
        meas_times = exp_params._measurement_times_list
        time_intervals = np.diff(meas_times)
        avg_interval = np.mean(time_intervals)
        interval_text = f"{avg_interval:.6f}"
    else:
        interval_text = "N/A"
    
    system_info = f"""
Physical Constants:
  • Photon-cavity coupling (γ):    {exp_params.photon_cavity_coupling:.6f} rad/time
  • Inverse pulse width (σ):        {exp_params.inverse_pulse_width:.6f} 1/time
  • Dispersive coupling (χ):        {exp_params.chi:.6f} rad/time
Noise Configuration:
  • Relaxation rate (γ_relax):      {exp_params.noise_config.relaxation:.6f} rad/time
  • Dephasing rate (γ_deph):        {exp_params.noise_config.dephasing:.6f} rad/time
  • Depolarizing rate (γ_depol):    {exp_params.noise_config.depolarizing:.6f} rad/time
Measurement Protocol:
  • Initial time:                    {exp_params.measurement.initial_time:.6f}
  • Final time:                      {exp_params.measurement.final_time:.6f}
  • Time interval (Δt):              {exp_params.measurement.time_interval:.6f}
  • Avg. interval between measurements: {interval_text}
Landscape Statistics:
  • Contrast range:    [{data['contrast_map'].min():.6f}, {data['contrast_map'].max():.6f}]
  • Contrast variation: {data['contrast_map'].max() - data['contrast_map'].min():.2e}
  • Detection range:   [{data['detection_map'].min():.6f}, {data['detection_map'].max():.6f}]
  • Maximum at:        θ₁={max_x:.2f}°, θ₂={max_y:.2f}°, Contrast={max_contrast:.8f}
"""
    
    # Add text box below plots
    fig.text(0.05, 0.02, system_info, fontsize=9, family='monospace',
             verticalalignment='bottom',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.6, pad=0.8))
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.show()


def main():
    """Main execution function."""
    print("Analyzing quantum sensing parameter landscape using qsopt module")
    
    # Create output directory
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    print(f"Output directory: {output_dir.absolute()}\n")
    
    # Setup experimental parameters
    print("Setting up experiment...")
    exp_params = create_experiment_setup()
    
    # Compute θ₁, θ₂ landscape
    print("-"*80)
    data_theta12 = compute_theta1_theta2_landscape(
        exp_params,
        resolution=25,
        center_theta1=np.pi/2,
        center_theta2=-np.pi/2,
        param_range=np.pi/6,
        verbose=True
    )
    
    print("\nGenerating landscape visualization...")
    plot_landscape(
        data_theta12,
        exp_params,
        save_path=output_dir / 'parameter_landscape.png'
    )
    
    # Summary
    print("\n" + "="*80)
    print("✓ Analysis complete!")
    print("="*80)
    
    # Find and display optimal parameters
    max_idx = np.unravel_index(np.argmax(data_theta12['contrast_map']),
                               data_theta12['contrast_map'].shape)
    theta1_opt = data_theta12['theta1_vals'][max_idx[1]]
    theta2_opt = data_theta12['theta2_vals'][max_idx[0]]
    contrast_opt = data_theta12['contrast_map'][max_idx]
    
    print(f"\nOptimal parameters in explored region:")
    print(f"  θ₁ = {np.degrees(theta1_opt):.2f}° ({theta1_opt:.6f} rad)")
    print(f"  θ₂ = {np.degrees(theta2_opt):.2f}° ({theta2_opt:.6f} rad)")
    print(f"  Maximum contrast = {contrast_opt:.8f}")
    print(f"\nFigure saved to: {output_dir / 'parameter_landscape.png'}")


if __name__ == "__main__":
    main()
