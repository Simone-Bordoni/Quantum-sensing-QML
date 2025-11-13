"""
Chi-Gamma Parameter Sweep Utilities
====================================

This module provides functions for sweeping over chi (dispersive coupling)
and gamma (cavity decay rate) parameters for quantum sensing experiments.

Functions:
    compute_chi_gamma_sweep: Compute 2D sweep over chi and gamma parameters
"""

import numpy as np
import time
from typing import Dict, Union
from qsopt.core.experiment import SingleQubitExperiment, TwoQubitExperiment
from qsopt.core.experimental_parameters import (
    ExperimentalParameters,
    PhysicalConstants,
)


def compute_chi_gamma_sweep(
    experiment: Union[SingleQubitExperiment, TwoQubitExperiment],
    chi_interval: list = [0.1, 100.0],
    gamma_interval: list = [1.0, 100.0],
    resolution_chi: int = 20,
    resolution_gamma: int = 20,
    chi_scale: str = 'linear',
    gamma_scale: str = 'linear',
    batch_size: int = 1,
    verbose: bool = True
) -> Dict[str, Union[np.ndarray, float, str]]:
    """
    Compute parameter sweep over chi (dispersive coupling) and gamma (cavity decay rate).
    
    This function evaluates sensing contrast and detection probability across a 2D grid
    of chi and gamma values. For each parameter combination, it creates a new experiment
    with updated physical constants and runs the quantum simulation.
    
    The workflow for each parameter point:
        1. Update chi and gamma in PhysicalConstants
        2. Create new ExperimentalParameters with updated constants
        3. Recreate experiment with new parameters
        4. Run quantum simulation with and without photon
        5. Calculate sensing contrast and detection probability
        6. Store results in 2D arrays
    
    Args:
        experiment: Configured single or two-qubit experiment. This is used as a template
            to extract trainable parameters and measurement configuration.
        chi_interval: List [min, max] for chi values (dispersive coupling strength). Default: [0.1, 100.0].
        gamma_interval: List [min, max] for gamma values (cavity decay rate). Default: [1.0, 100.0].
        resolution_chi: Number of chi points to evaluate. Default: 20.
        resolution_gamma: Number of gamma points to evaluate. Default: 20.
        chi_scale: Scale type for chi axis: 'linear' or 'log'. Default: 'linear'.
        gamma_scale: Scale type for gamma axis: 'linear' or 'log'. Default: 'linear'.
        batch_size: Number of random realizations to average over. Default: 1.
        verbose: If True, print progress information. Default: True.
        
    Returns:
        Dictionary containing:
            - 'chi_vals': Array of chi values evaluated (length=resolution_chi)
            - 'gamma_vals': Array of gamma values evaluated (length=resolution_gamma)
            - 'contrast_map': 2D array of sensing contrast values (shape: resolution_gamma × resolution_chi)
            - 'detection_map': 2D array of detection probability with photon (shape: resolution_gamma × resolution_chi)
            - 'detection_without_map': 2D array of detection probability without photon (shape: resolution_gamma × resolution_chi)
            - 'chi_scale': Scale type used for chi axis ('linear' or 'log')
            - 'gamma_scale': Scale type used for gamma axis ('linear' or 'log')
            
    Example:
        >>> from qsopt.core.experiment import SingleQubitExperiment
        >>> from qsopt.utils import compute_chi_gamma_sweep
        >>> 
        >>> # Create experiment
        >>> exp = SingleQubitExperiment(exp_params, trainable_params)
        >>> 
        >>> # Compute sweep
        >>> results = compute_chi_gamma_sweep(
        ...     exp,
        ...     chi_interval=[0.1, 50.0],
        ...     gamma_interval=[1.0, 50.0],
        ...     resolution_chi=15,
        ...     resolution_gamma=15,
        ...     chi_scale='log',
        ...     gamma_scale='linear'
        ... )
        >>> 
        >>> # Find optimal parameters
        >>> max_idx = np.unravel_index(
        ...     np.argmax(results['contrast_map']),
        ...     results['contrast_map'].shape
        ... )
        >>> optimal_chi = results['chi_vals'][max_idx[1]]
        >>> optimal_gamma = results['gamma_vals'][max_idx[0]]
        
    Notes:
        - For two-qubit experiments, chi is set equal for both qubits
        - Computation time scales as O(n_chi × n_gamma)
        - Each point requires a full quantum dynamics simulation
        - Results stored in row-major order: contrast_map[j, i] corresponds to
          (chi_vals[i], gamma_vals[j])
        
    See Also:
        plot_chi_gamma_sweep: Visualize the computed sweep
    """
    # Validate scale parameters
    if chi_scale not in ['linear', 'log']:
        raise ValueError(f"chi_scale must be 'linear' or 'log', got '{chi_scale}'")
    if gamma_scale not in ['linear', 'log']:
        raise ValueError(f"gamma_scale must be 'linear' or 'log', got '{gamma_scale}'")
    
    if verbose:
        print("Computing χ-γ parameter sweep...")
        print(f"  Resolution: {resolution_chi}×{resolution_gamma} = {resolution_chi * resolution_gamma} points")
        print(f"  χ range: [{chi_interval[0]:.2f}, {chi_interval[1]:.2f}] ({chi_scale} scale)")
        print(f"  γ range: [{gamma_interval[0]:.2f}, {gamma_interval[1]:.2f}] ({gamma_scale} scale)")
    
    # Create parameter grids with specified scales
    if chi_scale == 'log':
        chi_vals = np.logspace(np.log10(chi_interval[0]), np.log10(chi_interval[1]), resolution_chi)
    else:
        chi_vals = np.linspace(chi_interval[0], chi_interval[1], resolution_chi)
    
    if gamma_scale == 'log':
        gamma_vals = np.logspace(np.log10(gamma_interval[0]), np.log10(gamma_interval[1]), resolution_gamma)
    else:
        gamma_vals = np.linspace(gamma_interval[0], gamma_interval[1], resolution_gamma)
    
    # Initialize result arrays
    contrast_map = np.zeros((resolution_gamma, resolution_chi))
    detection_map = np.zeros((resolution_gamma, resolution_chi))
    detection_without_map = np.zeros((resolution_gamma, resolution_chi))
    
    # Extract base parameters from experiment
    base_exp_params = experiment.experimental_params
    trainable_params = experiment.trainable_params
    
    # Determine if single or two-qubit experiment
    is_two_qubit = isinstance(experiment, TwoQubitExperiment)
    n_qubits = 2 if is_two_qubit else 1
    
    # For two-qubit experiments, also track individual probabilities
    if is_two_qubit:
        prob_maps = {
            'p00': np.zeros((resolution_gamma, resolution_chi)),
            'p01': np.zeros((resolution_gamma, resolution_chi)),
            'p10': np.zeros((resolution_gamma, resolution_chi)),
            'p11': np.zeros((resolution_gamma, resolution_chi))
        }
    
    start_time = time.time()
    total_points = resolution_chi * resolution_gamma
    
    # Compute sweep
    for i, chi in enumerate(chi_vals):
        for j, gamma_val in enumerate(gamma_vals):
            # Update physical constants with new chi and gamma
            # For two-qubit: chi is same for both qubits
            chi_list = [chi] * n_qubits
            
            new_phys_const = PhysicalConstants(
                n_qubits=n_qubits,
                chi=chi_list,
                photon_cavity_coupling=gamma_val,
                inverse_pulse_width=base_exp_params.physical_constants.inverse_pulse_width
            )
            
            # Create new experimental parameters with updated constants
            new_exp_params = ExperimentalParameters(
                physical_constants=new_phys_const,
                system_dims=base_exp_params.system_dims,
                measurement=base_exp_params.measurement,
                initial_state=base_exp_params.initial_state,
                noise_config=base_exp_params.noise_config,
                random_seed=base_exp_params.random_seed
            )
            
            # Create new experiment with updated parameters
            if is_two_qubit:
                temp_exp = TwoQubitExperiment(new_exp_params, trainable_params)
                # For two-qubit, get full probability information
                results = temp_exp.run_simulation_with_probabilities()
                
                # Store detection and contrast results
                contrast_map[j, i] = results['contrast']
                detection_map[j, i] = results['detection_with']
                detection_without_map[j, i] = results['detection_without']
                
                # Store individual probability maps
                for key in ['p00', 'p01', 'p10', 'p11']:
                    prob_maps[key][j, i] = results['probs_with'][key]
            else:
                temp_exp = SingleQubitExperiment(new_exp_params, trainable_params)
                # Run simulation with batch averaging
                callback = temp_exp.run_simulation(batch_size=batch_size)
                
                # Store results (j,i indexing for correct orientation in plots)
                contrast_map[j, i] = callback.history['contrast'][-1]
                detection_map[j, i] = callback.history['prob_with'][-1]
                detection_without_map[j, i] = callback.history['prob_without'][-1]
            
            # Progress indicator
            if verbose and (i * resolution_gamma + j + 1) % max(1, total_points // 10) == 0:
                elapsed = time.time() - start_time
                progress = (i * resolution_gamma + j + 1) / total_points
                print(f"  Progress: {progress*100:.1f}% | "
                      f"Elapsed: {elapsed:.1f}s")
    
    if verbose:
        total_time = time.time() - start_time
        print(f"✓ Sweep completed in {total_time:.1f}s")
        print(f"  Max contrast: {np.max(contrast_map):.6f}")
        max_idx = np.unravel_index(np.argmax(contrast_map), contrast_map.shape)
        print(f"  Optimal χ: {chi_vals[max_idx[1]]:.3f}")
        print(f"  Optimal γ: {gamma_vals[max_idx[0]]:.3f}")
    
    result = {
        'chi_vals': chi_vals,
        'gamma_vals': gamma_vals,
        'contrast_map': contrast_map,
        'detection_map': detection_map,
        'detection_without_map': detection_without_map,
        'chi_scale': chi_scale,
        'gamma_scale': gamma_scale
    }
    
    # Add probability maps for two-qubit experiments
    if is_two_qubit:
        result['prob_maps'] = prob_maps
    
    return result


# Backward compatibility alias
compute_chi_lambda_sweep = compute_chi_gamma_sweep
