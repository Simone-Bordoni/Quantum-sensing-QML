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

# Import qsopt modules
from qsopt.core.experimental_parameters import ExperimentalParameters, InitialStateType, PhysicalConstants, SystemDimensions, MeasurementProtocol, NoiseConfiguration, InitialStateConfig
from qsopt.utils import compute_theta1_theta2_landscape, plot_parameter_landscape

gm = 0.03 * 2 * np.pi
inverse_pulse_width = 0.1 * gm

# Define custom physical constants
custom_constants = PhysicalConstants(
    chi = 0.5 * gm,                    # Dispersive coupling
    photon_cavity_coupling = gm,  # Photon-cavity coupling
    inverse_pulse_width = inverse_pulse_width      # Inverse pulse width
)

# Define custom system dimensions
custom_dims = SystemDimensions(
    cavity_levels=2,
    qubit_levels=2,
    field_levels=2
)

# Define measurement protocol
custom_measurement = MeasurementProtocol(
    measurement_times=None,  # Use interval mode
    initial_time=-5.0/inverse_pulse_width,
    final_time=5.0/inverse_pulse_width,
    time_interval=5.0/inverse_pulse_width,
    initial_time_uncertainty=0.0/inverse_pulse_width
)

# Define initial state configuration (SINGLE_PHOTON)
initial_state = InitialStateConfig(
    state_type=InitialStateType.SINGLE_PHOTON
)

# Define noise configuration
noise_config = NoiseConfiguration(
    depolarizing = 0.0005,  
    dephasing = 0.0005,      
    relaxation = 0.0005
)

# Create parameters with custom configuration
exp_parameters = ExperimentalParameters(
    physical_constants=custom_constants,
    system_dims=custom_dims,
    measurement=custom_measurement,
    initial_state=initial_state,
    noise_config=noise_config,
    random_seed=42
)

data_theta12 = compute_theta1_theta2_landscape(
    exp_parameters,
    resolution=30,
    center_theta1=np.pi/2,
    center_theta2=-np.pi/2,
    param_range=np.pi/8,
    verbose=True
)

fig = plot_parameter_landscape(
    data_theta12,
    exp_parameters,
    save_path=str(Path('experiments/single_qubit/rotation_parameters_landscape/results/parameter_landscape_noisy.png'))
)
plt.show()
