"""
Parameter Space Landscape Analysis using qsopt module
======================================================

This script analyzes the parameter space landscape for quantum sensing
optimization using the θ₁, θ₂ parameterization strategy.

Uses run_simulation() with different rotation parameters and plots heatmaps of:
- Metric values landscape
- Detection measures landscape

The analysis uses time-interval based measurements and includes comprehensive
system parameters in the visualization.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Import qsopt modules
from qsopt.core.circuit import create_ry_circuit
from qsopt.core.experiment import Experiment
from qsopt.core.experimental_parameters import (
    ExperimentalParameters,
    InitialStateType,
    PhysicalSetup,
    SystemDimensions,
    MeasurementProtocol,
    NoiseConfiguration,
    InitialStateConfig,
)
from qsopt.utils import plot_parameter_landscape

gm = 0.03 * 2 * np.pi
inverse_pulse_width = 0.1 * gm

# Define custom physical setup
custom_setup = PhysicalSetup(
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
    physical_setup=custom_setup,
    system_dims=custom_dims,
    measurement=custom_measurement,
    initial_state=initial_state,
    noise_config=noise_config,
    random_seed=42
)

resolution = 30
center_theta1 = np.pi / 2
center_theta2 = -np.pi / 2
param_range = np.pi / 8

theta1_vals = np.linspace(center_theta1 - param_range, center_theta1 + param_range, resolution)
theta2_vals = np.linspace(center_theta2 - param_range, center_theta2 + param_range, resolution)

initial_circuit = create_ry_circuit(n_qubits=1, theta_values=center_theta1)
final_circuit = create_ry_circuit(n_qubits=1, theta_values=center_theta2)
experiment = Experiment(exp_parameters, initial_circuit, final_circuit)

metric_map = np.zeros((resolution, resolution))
detection_map = np.zeros((resolution, resolution))

for i, theta1 in enumerate(theta1_vals):
    for j, theta2 in enumerate(theta2_vals):
        experiment.initial_circuit.set_trainable_parameters([theta1])
        experiment.final_circuit.set_trainable_parameters([theta2])
        callback = experiment.run_simulation(batch_size=1)

        # Keep the same orientation expected by plot_parameter_landscape.
        metric_map[j, i] = callback.history["metric"][-1]
        detection_map[j, i] = callback.history["detection_with"][-1]

data_theta12 = {
    "theta1_vals": theta1_vals,
    "theta2_vals": theta2_vals,
    "metric_map": metric_map,
    "detection_map": detection_map,
    "center_theta1": center_theta1,
    "center_theta2": center_theta2,
}

fig = plot_parameter_landscape(
    data_theta12,
    exp_parameters,
    save_path=str(Path('experiments/single_qubit/rotation_parameters_landscape/results/parameter_landscape_noisy.png'))
)
plt.show()
