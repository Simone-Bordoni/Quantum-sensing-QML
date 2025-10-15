"""
Comprehensive Example: Trainable Time Interval Optimization
============================================================

This example demonstrates the complete workflow for optimizing the time_interval
parameter along with rotation angles in a quantum sensing experiment.
"""

import numpy as np
from qsopt.core.experimental_parameters import (
    ExperimentalParameters,
    PhysicalConstants,
    SystemDimensions,
    MeasurementProtocol,
    InitialStateConfig,
    NoiseConfiguration
)
from qsopt.core.trainable_parameters import TrainableParameters
from qsopt.core.experiment import SingleQubitExperiment

print("="*80)
print("COMPREHENSIVE EXAMPLE: Trainable Time Interval Optimization")
print("="*80)
print()

# =============================================================================
# STEP 1: Configure Physical System
# =============================================================================
print("STEP 1: Configure Physical System")
print("-" * 80)

physical_constants = PhysicalConstants(
    chi=0.5,                    # Dispersive coupling strength
    photon_cavity_coupling=1.0,  # Photon-cavity coupling
    inverse_pulse_width=0.1      # Inverse pulse width
)

system_dims = SystemDimensions(
    cavity_levels=4,   # 4 cavity Fock states
    qubit_levels=2,    # 2-level qubit
    field_levels=2     # 2 field levels
)

# IMPORTANT: Use interval-based measurement protocol for trainable time_interval
measurement_protocol = MeasurementProtocol(
    measurement_times=None,     # Must be None for interval mode
    initial_time=-10.0,         # Start time
    final_time=10.0,            # End time  
    time_interval=2.0,          # Initial interval (will be optimized)
    initial_time_uncertainty=0.0
)

initial_state = InitialStateConfig()

# Add some noise for realism
noise_config = NoiseConfiguration(
    depolarizing=0.001,
    dephasing=0.0005,
    relaxation=0.0002
)

exp_params = ExperimentalParameters(
    physical_constants=physical_constants,
    system_dims=system_dims,
    measurement=measurement_protocol,
    initial_state=initial_state,
    noise_config=noise_config
)

print(f"  Physical constants: χ={physical_constants.chi}, g={physical_constants.photon_cavity_coupling}")
print(f"  System dimensions: {system_dims.cavity_levels}×{system_dims.qubit_levels}×{system_dims.field_levels}")
print(f"  Measurement protocol: interval mode")
print(f"    Initial time_interval: {measurement_protocol.time_interval}")
print(f"    Initial measurement count: {len(exp_params.measurement_times)}")
print()

# =============================================================================
# STEP 2: Configure Trainable Parameters
# =============================================================================
print("STEP 2: Configure Trainable Parameters")
print("-" * 80)

trainable_params = TrainableParameters()

# Add rotation angles (always needed)
trainable_params.add_rotation_angles(
    names=['theta1', 'theta2'],
    initial_values=[0.0, np.pi/2],
    trainable=[True, True]  # Both trainable
)

# Add trainable time_interval
# NOTE: This will OVERRIDE the value from experimental_parameters
trainable_params.add_measurement_interval(
    names='time_interval',
    initial_values=1.5,      # Starting value for optimization
    min_interval=0.1,        # Minimum allowed value (must be > 0)
    trainable=True           # Enable optimization
)

print(f"  Rotation angles:")
print(f"    theta1: {trainable_params.parameters[0].value:.6f} rad (trainable)")
print(f"    theta2: {trainable_params.parameters[1].value:.6f} rad (trainable)")
print(f"  Time interval:")
print(f"    time_interval: {trainable_params.parameters[2].value:.6f} (trainable)")
print(f"    min_interval: {trainable_params.constraints[2].min_value:.6f}")
print()

# =============================================================================
# STEP 3: Create Experiment
# =============================================================================
print("STEP 3: Create Experiment")
print("-" * 80)

experiment = SingleQubitExperiment(exp_params, trainable_params)

# Update exp_params with the trainable parameter value
exp_params.measurement.time_interval = trainable_params.parameters[2].value
exp_params._update_measurement_times()

print(f"  Experiment created successfully")
print(f"  Initial measurement count: {len(exp_params.measurement_times)}")
print(f"  Hilbert space dimension: {exp_params.cavity_levels * exp_params.qubit_levels * exp_params.field_levels}")
print()

# =============================================================================
# STEP 4: Run Initial Simulation
# =============================================================================
print("STEP 4: Run Initial Simulation")
print("-" * 80)

initial_callback = experiment.run_simulation(batch_size=1)
initial_contrast = initial_callback.best_metrics['contrast']

print(f"  Initial parameters:")
print(f"    theta1 = {trainable_params.parameters[0].value:.6f} rad")
print(f"    theta2 = {trainable_params.parameters[1].value:.6f} rad")
print(f"    Δt = {trainable_params.parameters[2].value:.6f}")
print(f"  Results:")
print(f"    P(with photon) = {initial_callback.best_metrics['prob_with']:.8f}")
print(f"    P(without photon) = {initial_callback.best_metrics['prob_without']:.8f}")
print(f"    Contrast = {initial_contrast:.8f}")
print()

# =============================================================================
# STEP 5: Run Optimization
# =============================================================================
print("STEP 5: Run Optimization")
print("-" * 80)
print()

history = experiment.optimize(
    num_steps=30,
    batch_size=1,
    tolerance=1e-5,
    verbose=True,
    verbose_step=5
)

print()

# =============================================================================
# STEP 6: Analyze Results
# =============================================================================
print("STEP 6: Analyze Results")
print("-" * 80)

final_theta1 = trainable_params.parameters[0].value
final_theta2 = trainable_params.parameters[1].value
final_dt = trainable_params.parameters[2].value
final_contrast = history.best_contrast

print(f"  Optimization Summary:")
print(f"    Converged: {history.converged}")
print(f"    Final gradient norm: {history.final_grad_norm:.2e}")
print(f"    Total iterations: {history.epoch}")
print()

print(f"  Parameter Changes:")
print(f"    theta1: {0.0:.6f} → {final_theta1:.6f} rad (Δ={final_theta1-0.0:.6f})")
print(f"    theta2: {np.pi/2:.6f} → {final_theta2:.6f} rad (Δ={final_theta2-np.pi/2:.6f})")
print(f"    Δt: {1.5:.6f} → {final_dt:.6f} (Δ={final_dt-1.5:.6f})")
print()

print(f"  Contrast Improvement:")
print(f"    Initial: {initial_contrast:.8f}")
print(f"    Final: {final_contrast:.8f}")
print(f"    Improvement: {final_contrast - initial_contrast:.8f} ({100*(final_contrast-initial_contrast)/max(abs(initial_contrast), 1e-10):.2f}%)")
print()

print(f"  Measurement Protocol:")
print(f"    Initial measurement count: 11")
print(f"    Final measurement count: {len(exp_params.measurement_times)}")
print(f"    Time range: [{exp_params.measurement.initial_time:.1f}, {exp_params.measurement.final_time:.1f}]")
print(f"    Final Δt: {final_dt:.6f}")
print()

# =============================================================================
# STEP 7: Compare with Fixed Time Interval
# =============================================================================
print("STEP 7: Compare with Fixed Time Interval")
print("-" * 80)

# Create new parameters with fixed time_interval
trainable_params_fixed = TrainableParameters()
trainable_params_fixed.add_rotation_angles(
    names=['theta1', 'theta2'],
    initial_values=[0.0, np.pi/2]
)
trainable_params_fixed.add_measurement_interval(
    names='time_interval',
    initial_values=1.5,
    trainable=False  # FIXED
)

exp_params_fixed = ExperimentalParameters(
    physical_constants=physical_constants,
    system_dims=system_dims,
    measurement=measurement_protocol,
    initial_state=initial_state,
    noise_config=noise_config
)

experiment_fixed = SingleQubitExperiment(exp_params_fixed, trainable_params_fixed)

print(f"\n  Running optimization with FIXED time_interval = 1.5...\n")

history_fixed = experiment.optimize(
    num_steps=30,
    batch_size=1,
    tolerance=1e-5,
    verbose=False
)

print(f"  Results with FIXED time_interval:")
print(f"    Final contrast: {history_fixed.best_contrast:.8f}")
print(f"    Final Δt: {trainable_params_fixed.parameters[2].value:.6f} (unchanged)")
print()

print(f"  Comparison:")
print(f"    Trainable Δt contrast: {final_contrast:.8f}")
print(f"    Fixed Δt contrast: {history_fixed.best_contrast:.8f}")
if final_contrast > history_fixed.best_contrast:
    improvement = final_contrast - history_fixed.best_contrast
    print(f"    ✓ Trainable Δt gives {improvement:.8f} better contrast!")
else:
    print(f"    → Fixed Δt performs similarly (contrast is insensitive to Δt)")
print()

# =============================================================================
# CONCLUSION
# =============================================================================
print("="*80)
print("CONCLUSION")
print("="*80)
print()
print("This example demonstrated:")
print("  1. Setting up interval-based measurement protocol")
print("  2. Adding trainable time_interval parameter")
print("  3. Optimizing time_interval with rotation angles")
print("  4. Dynamic recomputation of measurement times")
print("  5. Comparison with fixed time_interval")
print()
print("Key Takeaways:")
print("  • time_interval must be > 0 (strictly positive)")
print("  • Gradient descent respects this constraint")
print("  • Measurement times are recomputed after each update")
print("  • Can be fixed by setting trainable=False")
print()
print("Example completed successfully!")
print("="*80)
