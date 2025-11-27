"""
Parameter Sweep Utilities
==========================

This module provides functions for sweeping over various quantum sensing
parameters including chi (dispersive coupling), gamma (cavity decay rate),
and qubit coupling asymmetry.

Functions:
    compute_chi_gamma_sweep: Compute 2D sweep over chi and gamma parameters
    compute_asymmetry_coupling_sweep: Sweep over coupling asymmetry and qubit-qubit interaction
    compute_asymmetry_gamma_sweep: Sweep over coupling asymmetry and gamma
"""

import numpy as np
import time
from typing import Dict, Union
from qsopt.core.experiment import SingleQubitExperiment, TwoQubitExperiment
from qsopt.core.experimental_parameters import (
    ExperimentalParameters,
    PhysicalConstants,
    QubitInteraction,
    InteractionType,
)
from qsopt.utils.results import SweepResults


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
        plot_sweep_results: Visualize the computed sweep
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
            
            # Use copy method to preserve all properties including interactions
            new_exp_params = base_exp_params.copy(
                physical_constants={
                    'chi': chi_list,
                    'photon_cavity_coupling': gamma_val
                }
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
        print(f"Sweep completed in {total_time:.1f}s")
        print(f"  Max contrast: {np.max(contrast_map):.6f}")
        max_idx = np.unravel_index(np.argmax(contrast_map), contrast_map.shape)
        print(f"  Optimal χ: {chi_vals[max_idx[1]]:.3f}")
        print(f"  Optimal γ: {gamma_vals[max_idx[0]]:.3f}")
    
    # Prepare results dictionary
    results_dict = {
        'contrast_map': contrast_map,
        'detection_map': detection_map,
        'detection_without_map': detection_without_map
    }
    
    # Add probability maps for two-qubit experiments
    if is_two_qubit:
        results_dict.update(prob_maps)
    
    # Prepare metadata
    max_idx = np.unravel_index(np.argmax(contrast_map), contrast_map.shape)
    
    # Get measurement times, handling None case
    meas_times = base_exp_params.measurement.measurement_times
    n_measurements = len(meas_times) if meas_times is not None else 0
    
    # Get noise rates (could be list or float)
    depol = base_exp_params.noise_config.depolarizing
    depol_val = depol[0] if isinstance(depol, list) else depol
    dephasing = base_exp_params.noise_config.dephasing
    dephasing_val = dephasing[0] if isinstance(dephasing, list) else dephasing
    relax = base_exp_params.noise_config.relaxation
    relax_val = relax[0] if isinstance(relax, list) else relax
    
    metadata = {
        'optimal_chi': chi_vals[max_idx[1]],
        'optimal_gamma': gamma_vals[max_idx[0]],
        'max_contrast': contrast_map[max_idx],
        'optimal_idx': max_idx,
        # System characteristics
        'n_qubits': n_qubits,
        'cavity_levels': base_exp_params.system_dims.cavity_levels,
        'qubit_levels': base_exp_params.system_dims.qubit_levels,
        'field_levels': base_exp_params.system_dims.field_levels,
        'n_measurements': n_measurements,
        'measurement_times': meas_times,
        'initial_time_uncertainty': base_exp_params.measurement.initial_time_uncertainty,
        'depolarizing_rate': depol_val,
        'dephasing_rate': dephasing_val,
        'relaxation_rate': relax_val,
        'initial_state': base_exp_params.initial_state.state_type.name,
        'inverse_pulse_width': base_exp_params.physical_constants.inverse_pulse_width
    }
    
    return SweepResults(
        param1_name='gamma',
        param1_vals=gamma_vals,
        param1_scale=gamma_scale,
        param2_name='chi',
        param2_vals=chi_vals,
        param2_scale=chi_scale,
        results=results_dict,
        metadata=metadata
    )


def compute_asymmetry_coupling_sweep(
    experiment: TwoQubitExperiment,
    asymmetry_interval: list = [-8.0, 8.0],
    coupling_interval: list = [0.0, 10.0],
    resolution_asymmetry: int = 30,
    resolution_coupling: int = 30,
    chi_mean_factor: float = 10.0,
    gamma: float = 10.0,
    interaction_type: InteractionType = InteractionType.XX,
    batch_size: int = 1,
    verbose: bool = True
) -> Dict[str, Union[np.ndarray, float, str]]:
    """
    Sweep over qubit coupling asymmetry (Δχ) and qubit-qubit interaction strength (χ₁₂).
    
    This mimics the analysis from the reference notebook where:
    - χ_mean = (χ₁ + χ₂) / 2 is kept constant (chi_mean_factor * gamma)
    - Δχ = χ₁ - χ₂ varies (asymmetry)
    - χ₁₂ (qubit-qubit coupling) varies
    
    The individual chi values are computed as:
        χ₁ = chi_mean + Δχ/2
        χ₂ = chi_mean - Δχ/2
    
    Args:
        experiment: Template two-qubit experiment for configuration
        asymmetry_interval: [min, max] for Δχ/γ (chi asymmetry relative to gamma)
        coupling_interval: [min, max] for χ₁₂/γ (qubit-qubit coupling relative to gamma)
        resolution_asymmetry: Number of asymmetry points
        resolution_coupling: Number of coupling points
        chi_mean_factor: Ratio of mean chi to gamma, (χ₁+χ₂)/(2γ). Default: 10.0
        gamma: Fixed gamma value. Default: 10.0
        interaction_type: Type of qubit-qubit interaction (XX, YY, ZZ, XXYY). Default: XX
        batch_size: Number of averaging realizations
        verbose: Print progress information
        
    Returns:
        Dictionary with:
            - 'asymmetry_vals': Δχ/γ values
            - 'coupling_vals': χ₁₂/γ values  
            - 'prob_maps': Dict with 'p00', 'p01', 'p10', 'p11' 2D arrays
            - 'contrast_map': 2D contrast array
            - Other metadata
            
    Example:
        >>> # Sweep asymmetry vs coupling with chi_mean = 10*gamma
        >>> results = compute_asymmetry_coupling_sweep(
        ...     exp_2q,
        ...     asymmetry_interval=[-8, 8],
        ...     coupling_interval=[0, 10],
        ...     resolution_asymmetry=30,
        ...     resolution_coupling=30,
        ...     chi_mean_factor=10.0,
        ...     gamma=10.0
        ... )
        >>> 
        >>> # Plot P11
        >>> import matplotlib.pyplot as plt
        >>> plt.contourf(results['asymmetry_vals'], results['coupling_vals'], 
        ...              results['prob_maps']['p11'].T, cmap='rainbow')
        >>> plt.xlabel('Asymmetry Δχ/γ')
        >>> plt.ylabel('Coupling χ₁₂/γ')
    """
    if not isinstance(experiment, TwoQubitExperiment):
        raise ValueError("This sweep requires a TwoQubitExperiment")
    
    if verbose:
        print("Computing asymmetry-coupling parameter sweep...")
        print(f"  Resolution: {resolution_asymmetry}×{resolution_coupling} = "
              f"{resolution_asymmetry * resolution_coupling} points")
        print(f"  Δχ/γ range: [{asymmetry_interval[0]:.2f}, {asymmetry_interval[1]:.2f}]")
        print(f"  χ₁₂/γ range: [{coupling_interval[0]:.2f}, {coupling_interval[1]:.2f}]")
        print(f"  χ_mean/γ = {chi_mean_factor:.1f}, γ = {gamma:.1f}")
    
    # Create parameter grids
    asymmetry_vals = np.linspace(asymmetry_interval[0], asymmetry_interval[1], resolution_asymmetry)
    coupling_vals = np.linspace(coupling_interval[0], coupling_interval[1], resolution_coupling)
    
    # Initialize result arrays
    prob_maps = {
        'p00': np.zeros((resolution_coupling, resolution_asymmetry)),
        'p01': np.zeros((resolution_coupling, resolution_asymmetry)),
        'p10': np.zeros((resolution_coupling, resolution_asymmetry)),
        'p11': np.zeros((resolution_coupling, resolution_asymmetry))
    }
    contrast_map = np.zeros((resolution_coupling, resolution_asymmetry))
    detection_map = np.zeros((resolution_coupling, resolution_asymmetry))
    detection_without_map = np.zeros((resolution_coupling, resolution_asymmetry))
    
    # Extract base parameters
    base_exp_params = experiment.experimental_params
    trainable_params = experiment.trainable_params
    
    chi_mean = chi_mean_factor * gamma
    
    start_time = time.time()
    total_points = resolution_asymmetry * resolution_coupling
    
    # Compute sweep
    for i, delta_chi_rel in enumerate(asymmetry_vals):
        for j, chi12_rel in enumerate(coupling_vals):
            # Compute individual chi values
            delta_chi = delta_chi_rel * gamma
            chi1 = chi_mean + delta_chi / 2
            chi2 = chi_mean - delta_chi / 2
            chi12 = chi12_rel * gamma
            
            # Create qubit interaction
            if chi12 > 0:
                interaction = QubitInteraction(
                    qubit_indices=(0, 1),
                    chi=chi12,
                    interaction_type=interaction_type
                )
                interactions = [interaction]
            else:
                interactions = []
            
            # Use copy method to preserve all properties and update specific ones
            new_exp_params = base_exp_params.copy(
                physical_constants={
                    'chi': [chi1, chi2],
                    'photon_cavity_coupling': gamma,
                    'qubit_interactions': interactions
                }
            )
            
            temp_exp = TwoQubitExperiment(new_exp_params, trainable_params)
            results = temp_exp.run_simulation_with_probabilities()
            
            # Store results (j,i indexing for plotting)
            contrast_map[j, i] = results['contrast']
            detection_map[j, i] = results['detection_with']
            detection_without_map[j, i] = results['detection_without']
            
            for key in ['p00', 'p01', 'p10', 'p11']:
                prob_maps[key][j, i] = results['probs_with'][key]
            
            # Progress
            if verbose and (i * resolution_coupling + j + 1) % max(1, total_points // 10) == 0:
                elapsed = time.time() - start_time
                progress = (i * resolution_coupling + j + 1) / total_points
                print(f"  Progress: {progress*100:.1f}% | Elapsed: {elapsed:.1f}s")
    
    if verbose:
        print(f"✓ Sweep completed in {time.time() - start_time:.1f}s")
        max_idx = np.unravel_index(np.argmax(contrast_map), contrast_map.shape)
        print(f"  Max contrast: {contrast_map[max_idx]:.6f}")
        print(f"  Optimal Δχ/γ: {asymmetry_vals[max_idx[1]]:.3f}")
        print(f"  Optimal χ₁₂/γ: {coupling_vals[max_idx[0]]:.3f}")
    
    # Prepare results dictionary
    results_dict = {
        'contrast_map': contrast_map,
        'detection_map': detection_map,
        'detection_without_map': detection_without_map
    }
    results_dict.update(prob_maps)
    
    # Prepare metadata
    max_idx = np.unravel_index(np.argmax(contrast_map), contrast_map.shape)
    
    # Get measurement times, handling None case
    meas_times = base_exp_params.measurement.measurement_times
    n_measurements = len(meas_times) if meas_times is not None else 0
    
    # Get noise rates (could be list or float)
    depol = base_exp_params.noise_config.depolarizing
    depol_val = depol[0] if isinstance(depol, list) else depol
    dephasing = base_exp_params.noise_config.dephasing
    dephasing_val = dephasing[0] if isinstance(dephasing, list) else dephasing
    relax = base_exp_params.noise_config.relaxation
    relax_val = relax[0] if isinstance(relax, list) else relax
    
    metadata = {
        'optimal_asymmetry': asymmetry_vals[max_idx[1]],
        'optimal_coupling': coupling_vals[max_idx[0]],
        'max_contrast': contrast_map[max_idx],
        'optimal_idx': max_idx,
        'chi_mean_factor': chi_mean_factor,
        'gamma': gamma,
        # System characteristics
        'n_qubits': 2,
        'cavity_levels': base_exp_params.system_dims.cavity_levels,
        'qubit_levels': base_exp_params.system_dims.qubit_levels,
        'field_levels': base_exp_params.system_dims.field_levels,
        'n_measurements': n_measurements,
        'measurement_times': meas_times,
        'initial_time_uncertainty': base_exp_params.measurement.initial_time_uncertainty,
        'depolarizing_rate': depol_val,
        'dephasing_rate': dephasing_val,
        'relaxation_rate': relax_val,
        'initial_state': base_exp_params.initial_state.state_type.name,
        'inverse_pulse_width': base_exp_params.physical_constants.inverse_pulse_width
    }
    
    return SweepResults(
        param1_name='chi12/gamma',
        param1_vals=coupling_vals,
        param1_scale='linear',
        param2_name='Delta_chi/gamma',
        param2_vals=asymmetry_vals,
        param2_scale='linear',
        results=results_dict,
        metadata=metadata
    )


def compute_asymmetry_gamma_sweep(
    experiment: TwoQubitExperiment,
    asymmetry_interval: list = [-8.0, 8.0],
    gamma_interval: list = [0.1, 20.0],
    resolution_asymmetry: int = 30,
    resolution_gamma: int = 30,
    chi_mean_factor: float = 10.0,
    chi12_factor: float = 0.0,
    interaction_type: InteractionType = InteractionType.XX,
    batch_size: int = 1,
    verbose: bool = True
) -> Dict[str, Union[np.ndarray, float, str]]:
    """
    Sweep over qubit coupling asymmetry (Δχ) and gamma (γ).
    
    Similar to asymmetry_coupling_sweep but varies gamma instead of χ₁₂.
    The relationships are:
        - χ_mean = chi_mean_factor * γ (scales with gamma)
        - χ₁ = χ_mean + Δχ/2
        - χ₂ = χ_mean - Δχ/2
        - χ₁₂ = chi12_factor * γ (fixed ratio, scales with gamma)
    
    Args:
        experiment: Template two-qubit experiment
        asymmetry_interval: [min, max] for Δχ/γ  
        gamma_interval: [min, max] for γ values
        resolution_asymmetry: Number of asymmetry points
        resolution_gamma: Number of gamma points
        chi_mean_factor: Ratio (χ₁+χ₂)/(2γ). Default: 10.0
        chi12_factor: Ratio χ₁₂/γ. Default: 0.0 (no coupling)
        interaction_type: Type of qubit-qubit interaction (XX, YY, ZZ, XXYY). Default: XX
        batch_size: Number of averaging realizations
        verbose: Print progress information
        
    Returns:
        Dictionary with swept results including prob_maps
        
    Example:
        >>> # Sweep asymmetry vs gamma
        >>> results = compute_asymmetry_gamma_sweep(
        ...     exp_2q,
        ...     asymmetry_interval=[-8, 8],
        ...     gamma_interval=[0.1, 20],
        ...     chi_mean_factor=10.0,
        ...     chi12_factor=10.0  # With coupling
        ... )
    """
    if not isinstance(experiment, TwoQubitExperiment):
        raise ValueError("This sweep requires a TwoQubitExperiment")
    
    if verbose:
        print("Computing asymmetry-gamma parameter sweep...")
        print(f"  Resolution: {resolution_asymmetry}×{resolution_gamma}")
        print(f"  Δχ/γ range: [{asymmetry_interval[0]:.2f}, {asymmetry_interval[1]:.2f}]")
        print(f"  γ range: [{gamma_interval[0]:.2f}, {gamma_interval[1]:.2f}]")
        print(f"  χ_mean/γ = {chi_mean_factor:.1f}, χ₁₂/γ = {chi12_factor:.1f}")
    
    # Create grids
    asymmetry_vals = np.linspace(asymmetry_interval[0], asymmetry_interval[1], resolution_asymmetry)
    gamma_vals = np.linspace(gamma_interval[0], gamma_interval[1], resolution_gamma)
    
    # Initialize arrays
    prob_maps = {
        'p00': np.zeros((resolution_gamma, resolution_asymmetry)),
        'p01': np.zeros((resolution_gamma, resolution_asymmetry)),
        'p10': np.zeros((resolution_gamma, resolution_asymmetry)),
        'p11': np.zeros((resolution_gamma, resolution_asymmetry))
    }
    contrast_map = np.zeros((resolution_gamma, resolution_asymmetry))
    detection_map = np.zeros((resolution_gamma, resolution_asymmetry))
    detection_without_map = np.zeros((resolution_gamma, resolution_asymmetry))
    
    base_exp_params = experiment.experimental_params
    trainable_params = experiment.trainable_params
    
    start_time = time.time()
    total_points = resolution_asymmetry * resolution_gamma
    
    for i, delta_chi_rel in enumerate(asymmetry_vals):
        for j, gamma in enumerate(gamma_vals):
            # Compute parameters
            chi_mean = chi_mean_factor * gamma
            delta_chi = delta_chi_rel * gamma
            chi1 = chi_mean + delta_chi / 2
            chi2 = chi_mean - delta_chi / 2
            chi12 = chi12_factor * gamma
            
            # Create interaction if chi12 > 0
            if chi12 > 0:
                interaction = QubitInteraction(
                    qubit_indices=(0, 1),
                    chi=chi12,
                    interaction_type=interaction_type
                )
                interactions = [interaction]
            else:
                interactions = []
            
            # Use copy method to preserve all properties and update specific ones
            new_exp_params = base_exp_params.copy(
                physical_constants={
                    'chi': [chi1, chi2],
                    'photon_cavity_coupling': gamma,
                    'qubit_interactions': interactions
                }
            )
            
            temp_exp = TwoQubitExperiment(new_exp_params, trainable_params)
            results = temp_exp.run_simulation_with_probabilities()
            
            contrast_map[j, i] = results['contrast']
            detection_map[j, i] = results['detection_with']
            detection_without_map[j, i] = results['detection_without']
            
            for key in ['p00', 'p01', 'p10', 'p11']:
                prob_maps[key][j, i] = results['probs_with'][key]
            
            if verbose and (i * resolution_gamma + j + 1) % max(1, total_points // 10) == 0:
                elapsed = time.time() - start_time
                progress = (i * resolution_gamma + j + 1) / total_points
                print(f"  Progress: {progress*100:.1f}% | Elapsed: {elapsed:.1f}s")
    
    if verbose:
        print(f"✓ Sweep completed in {time.time() - start_time:.1f}s")
        max_idx = np.unravel_index(np.argmax(contrast_map), contrast_map.shape)
        print(f"  Max contrast: {contrast_map[max_idx]:.6f}")
        print(f"  Optimal Δχ/γ: {asymmetry_vals[max_idx[1]]:.3f}")
        print(f"  Optimal γ: {gamma_vals[max_idx[0]]:.3f}")
    
    # Prepare results dictionary
    results_dict = {
        'contrast_map': contrast_map,
        'detection_map': detection_map,
        'detection_without_map': detection_without_map
    }
    results_dict.update(prob_maps)
    
    # Prepare metadata
    max_idx = np.unravel_index(np.argmax(contrast_map), contrast_map.shape)
    
    # Get measurement times, handling None case
    meas_times = base_exp_params.measurement.measurement_times
    n_measurements = len(meas_times) if meas_times is not None else 0
    
    # Get noise rates (could be list or float)
    depol = base_exp_params.noise_config.depolarizing
    depol_val = depol[0] if isinstance(depol, list) else depol
    dephasing = base_exp_params.noise_config.dephasing
    dephasing_val = dephasing[0] if isinstance(dephasing, list) else dephasing
    relax = base_exp_params.noise_config.relaxation
    relax_val = relax[0] if isinstance(relax, list) else relax
    
    metadata = {
        'optimal_asymmetry': asymmetry_vals[max_idx[1]],
        'optimal_gamma': gamma_vals[max_idx[0]],
        'max_contrast': contrast_map[max_idx],
        'optimal_idx': max_idx,
        'chi_mean_factor': chi_mean_factor,
        'chi12_factor': chi12_factor,
        # System characteristics
        'n_qubits': 2,
        'cavity_levels': base_exp_params.system_dims.cavity_levels,
        'qubit_levels': base_exp_params.system_dims.qubit_levels,
        'field_levels': base_exp_params.system_dims.field_levels,
        'n_measurements': n_measurements,
        'measurement_times': meas_times,
        'initial_time_uncertainty': base_exp_params.measurement.initial_time_uncertainty,
        'depolarizing_rate': depol_val,
        'dephasing_rate': dephasing_val,
        'relaxation_rate': relax_val,
        'initial_state': base_exp_params.initial_state.state_type.name,
        'inverse_pulse_width': base_exp_params.physical_constants.inverse_pulse_width
    }
    
    return SweepResults(
        param1_name='gamma',
        param1_vals=gamma_vals,
        param1_scale='linear',
        param2_name='Delta_chi/gamma',
        param2_vals=asymmetry_vals,
        param2_scale='linear',
        results=results_dict,
        metadata=metadata
    )

