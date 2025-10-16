"""
Parameter Landscape Analysis Utilities
=======================================

This module provides functions for computing parameter space landscapes
for quantum sensing optimization.

Functions:
    compute_theta1_theta2_landscape: Compute 2D landscape over rotation parameters
"""

import numpy as np
import time
from typing import Dict, Union, Any
from qsopt.core.trainable_parameters import TrainableParameters
from qsopt.core.experiment import SingleQubitExperiment
from qsopt.core.experimental_parameters import ExperimentalParameters


def compute_theta1_theta2_landscape(
    exp_params: ExperimentalParameters,
    resolution: int = 25, 
    center_theta1: float = np.pi/2,
    center_theta2: float = -np.pi/2,
    param_range: float = np.pi/6,
    batch_size: int = 1,
    verbose: bool = True
) -> Dict[str, Union[np.ndarray, float]]:
    """
    Compute parameter landscape for θ₁, θ₂ rotation strategy.
    
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
    if verbose:
        print("Computing θ₁, θ₂ landscape...")
        print(f"  Resolution: {resolution}×{resolution}")
        print(f"  Center: θ₁={np.degrees(center_theta1):.1f}°, "
              f"θ₂={np.degrees(center_theta2):.1f}°")
        print(f"  Range: ±{np.degrees(param_range):.1f}°")
    
    # Create parameter grid
    theta1_vals = np.linspace(
        center_theta1 - param_range, 
        center_theta1 + param_range, 
        resolution
    )
    theta2_vals = np.linspace(
        center_theta2 - param_range, 
        center_theta2 + param_range, 
        resolution
    )
    
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
            
            # Run simulation with batch averaging for uncertainty
            callback = exp.run_simulation(batch_size=batch_size)
            
            # Store results (j,i indexing for correct orientation in plots)
            contrast_map[j, i] = callback.history['contrast'][-1]
            detection_map[j, i] = callback.history['prob_with'][-1]
            
            # Progress update
            current_point = i * resolution + j
            if verbose and (current_point % 50 == 0):
                progress = current_point / total_points * 100
                elapsed = time.time() - start_time
                eta = elapsed / max(current_point, 1) * (total_points - current_point)
                print(f"  Progress: {progress:.1f}% (ETA: {eta:.1f}s)", end='\r')
    
    if verbose:
        elapsed = time.time() - start_time
        print(f"\nCompleted in {elapsed:.1f}s "
              f"({elapsed/total_points:.3f}s per point)")
    
    return {
        'theta1_vals': theta1_vals,
        'theta2_vals': theta2_vals,
        'contrast_map': contrast_map,
        'detection_map': detection_map,
        'center_theta1': center_theta1,
        'center_theta2': center_theta2
    }


def compute_time_interval_landscape(
    exp_params: ExperimentalParameters,
    theta1: float,
    theta2: float,
    resolution: int = 50,
    mode: str = 'continuous',
    batch_size: int = 1,
    verbose: bool = True
) -> Dict[str, Union[np.ndarray, float, str, int]]:
    """
    Compute contrast landscape vs measurement time interval.
    
    This function evaluates how sensing contrast varies with the time interval
    between measurements, keeping rotation parameters (θ₁, θ₂) fixed. Two modes
    are supported:
    
    1. **Continuous mode**: Time interval varies continuously from a minimum
       value to the full evolution time (final_time - initial_time).
       
    2. **Discrete mode**: Time interval is restricted to integer fractions
       of the full evolution time (e.g., T/2, T/3, T/4, ..., T/N).
    
    The function supports batch averaging to account for initial_time_uncertainty,
    providing more realistic simulations that include timing jitter effects.
    
    Workflow for each time interval:
        1. Set time_interval in exp_params
        2. Recompute measurement times based on initial_time, final_time, interval
        3. Run quantum simulation with batch averaging (if batch_size > 1)
        4. Calculate average sensing contrast across realizations
        5. Store results in 1D array
    
    Args:
        exp_params: Configured experimental parameters. The function will
            modify exp_params.measurement.time_interval temporarily during
            computation but restore the original value at the end.
        theta1: Fixed first rotation angle (radians)
        theta2: Fixed second rotation angle (radians)
        resolution: Number of time interval values to evaluate. Default: 50.
        mode: Computation mode - either 'continuous' or 'discrete'.
            - 'continuous': Linearly spaced intervals from min to max
            - 'discrete': Integer fractions of total time (1/2, 1/3, ..., 1/N)
            Default: 'continuous'.
        batch_size: Number of random realizations to average over for
            measurement uncertainty. Recommended: ≥10 for realistic results
            when initial_time_uncertainty > 0. Default: 1.
        verbose: Print progress information. Default: True.
        
    Returns:
        Dictionary containing:
            - 'interval_vals': 1D array of time interval values (shape: [resolution])
            - 'contrast_vals': 1D array of sensing contrast (shape: [resolution])
            - 'detection_with': 1D array of detection prob with photon (shape: [resolution])
            - 'detection_without': 1D array of detection prob without photon (shape: [resolution])
            - 'n_measurements': 1D array of number of measurements per interval (shape: [resolution])
            - 'theta1': Fixed θ₁ value (float)
            - 'theta2': Fixed θ₂ value (float)
            - 'mode': Computation mode used (str)
            - 'batch_size': Batch size used (int)
            - 'initial_time_uncertainty': Uncertainty value from exp_params (float)
            
    Raises:
        ValueError: If mode is not 'continuous' or 'discrete'
        ValueError: If resolution < 2
        
    Example:
        >>> # Continuous mode with uncertainty
        >>> exp_params = ExperimentalParameters()
        >>> exp_params.measurement.initial_time = -5.0
        >>> exp_params.measurement.final_time = 5.0
        >>> exp_params.measurement.initial_time_uncertainty = 0.1
        >>> 
        >>> data = compute_time_interval_landscape(
        ...     exp_params,
        ...     theta1=np.pi/2,
        ...     theta2=-np.pi/2,
        ...     resolution=50,
        ...     mode='continuous',
        ...     batch_size=20  # Average over 20 realizations
        ... )
        >>> 
        >>> # Find optimal interval
        >>> optimal_idx = np.argmax(data['contrast_vals'])
        >>> optimal_interval = data['interval_vals'][optimal_idx]
        >>> print(f"Optimal interval: {optimal_interval:.4f}")
        
        >>> # Discrete mode (integer fractions)
        >>> data_discrete = compute_time_interval_landscape(
        ...     exp_params,
        ...     theta1=np.pi/2,
        ...     theta2=-np.pi/2,
        ...     resolution=20,
        ...     mode='discrete',
        ...     batch_size=10
        ... )
        
    Notes:
        - When batch_size > 1 and initial_time_uncertainty > 0, each simulation
          point averages over multiple realizations with random timing shifts.
        - The original exp_params.measurement.time_interval is restored after
          computation.
        - In discrete mode, intervals are chosen as T/N where N = 2, 3, ..., resolution+1
        - In continuous mode, the minimum interval ensures at least 2 measurements
    """
    # Validate inputs
    if mode not in ['continuous', 'discrete']:
        raise ValueError(f"mode must be 'continuous' or 'discrete', got '{mode}'")
    if resolution < 2:
        raise ValueError(f"resolution must be >= 2, got {resolution}")
    
    # Store original time interval to restore later
    original_interval = exp_params.measurement.time_interval
    
    # Calculate total evolution time
    initial_time = exp_params.measurement.initial_time
    final_time = exp_params.measurement.final_time
    total_time = final_time - initial_time
    
    if verbose:
        print(f"Computing time interval landscape (mode: {mode})...")
        print(f"  Rotation angles: θ₁={np.degrees(theta1):.1f}°, θ₂={np.degrees(theta2):.1f}°")
        print(f"  Resolution: {resolution} points")
        print(f"  Batch size: {batch_size} realizations")
        print(f"  Total evolution time: {total_time:.4f}")
        if exp_params.measurement.initial_time_uncertainty > 0:
            print(f"  Initial time uncertainty: ±{exp_params.measurement.initial_time_uncertainty:.4f}")
    
    # Generate time interval values based on mode
    if mode == 'continuous':
        # Continuous: linearly spaced from min to max
        # Minimum interval ensures at least 2 measurements
        min_interval = total_time / 100.0  # At least 100 potential measurements
        max_interval = total_time  # Single measurement at end
        interval_vals = np.linspace(min_interval, max_interval, resolution)
    else:  # mode == 'discrete'
        # Discrete: integer fractions T/N for N = 1, 2, ..., resolution
        fractions = np.arange(1, resolution + 1)  # [1, 2, 3, ..., resolution]
        interval_vals = total_time / fractions
    
    # Initialize result arrays
    contrast_vals = np.zeros(resolution)
    detection_with = np.zeros(resolution)
    detection_without = np.zeros(resolution)
    n_measurements = np.zeros(resolution, dtype=int)
    
    # Create trainable parameters with fixed rotation angles
    trainable_params = TrainableParameters()
    trainable_params.add_rotation_angles(
        names=['theta1', 'theta2'],
        initial_values=[theta1, theta2],
        trainable=[False, False]  # Fixed parameters, not training
    )
    
    # Create experiment instance
    exp = SingleQubitExperiment(exp_params, trainable_params)
    
    start_time = time.time()
    
    # Evaluate each time interval
    for i, interval in enumerate(interval_vals):
        # Update time interval in exp_params
        exp_params.measurement.time_interval = interval
        exp_params.measurement.measurement_times = None  # Force recomputation
        exp_params._update_measurement_times()
        
        # Store number of measurements for this interval
        meas_times_list = exp_params._measurement_times_list
        n_measurements[i] = len(meas_times_list) if meas_times_list is not None else 0
        
        # Run simulation with batch averaging
        callback = exp.run_simulation(batch_size=batch_size)
        
        # Store results (averaged over batch)
        contrast_vals[i] = callback.history['contrast'][-1]
        detection_with[i] = callback.history['prob_with'][-1]
        detection_without[i] = callback.history['prob_without'][-1]
        
        # Progress update
        if verbose and ((i + 1) % max(1, resolution // 10) == 0):
            progress = (i + 1) / resolution * 100
            elapsed = time.time() - start_time
            eta = elapsed / (i + 1) * (resolution - i - 1)
            print(f"  Progress: {progress:.1f}% "
                  f"(interval={interval:.4f}, n_meas={n_measurements[i]}, "
                  f"contrast={contrast_vals[i]:.6f}, ETA: {eta:.1f}s)", end='\r')
    
    # Restore original time interval
    exp_params.measurement.time_interval = original_interval
    exp_params.measurement.measurement_times = None
    exp_params._update_measurement_times()
    
    if verbose:
        elapsed = time.time() - start_time
        print(f"\nCompleted in {elapsed:.1f}s "
              f"({elapsed/resolution:.3f}s per point)")
        
        # Report optimal interval
        optimal_idx = np.argmax(contrast_vals)
        optimal_interval = interval_vals[optimal_idx]
        optimal_contrast = contrast_vals[optimal_idx]
        optimal_n_meas = n_measurements[optimal_idx]
        print(f"  Optimal interval: {optimal_interval:.4f} "
              f"(n_meas={optimal_n_meas}, contrast={optimal_contrast:.6f})")
    
    return {
        'interval_vals': interval_vals,
        'contrast_vals': contrast_vals,
        'detection_with': detection_with,
        'detection_without': detection_without,
        'n_measurements': n_measurements,
        'theta1': theta1,
        'theta2': theta2,
        'mode': mode,
        'batch_size': batch_size,
        'initial_time_uncertainty': exp_params.measurement.initial_time_uncertainty
    }
