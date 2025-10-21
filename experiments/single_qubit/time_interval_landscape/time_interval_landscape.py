"""
Time Interval Landscape Analysis Example
=========================================

This script demonstrates the time-interval landscape analysis functionality
with batch averaging to account for measurement uncertainty.

The analysis shows how sensing contrast varies with the time interval between
measurements while keeping rotation parameters fixed.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Import qsopt modules
from qsopt.core.experimental_parameters import ExperimentalParameters, InitialStateType, PhysicalConstants, SystemDimensions, MeasurementProtocol, NoiseConfiguration, InitialStateConfig
from qsopt.utils import compute_time_interval_landscape, plot_time_interval_landscape

gm = 0.03 * 2 * np.pi
inverse_pulse_width = 0.1 * gm

def create_experiment_setup():
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
        initial_time=-9.0/inverse_pulse_width,
        final_time=9.0/inverse_pulse_width,
        time_interval=5.0/inverse_pulse_width,
        initial_time_uncertainty=2.0/inverse_pulse_width
    )

    # Define initial state configuration (SINGLE_PHOTON)
    initial_state = InitialStateConfig(
        state_type=InitialStateType.SINGLE_PHOTON
    )

    # Define noise configuration
    noise_config = NoiseConfiguration(
        depolarizing = 0.0001,  
        dephasing = 0.0001,      
        relaxation = 0.0001
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
    return exp_parameters


def main():
    
    # Create output directory
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    print(f"Output directory: {output_dir.absolute()}\n")

    exp_params = create_experiment_setup()
    
    # Define fixed rotation angles (optimal from parameter landscape)
    #theta1 = np.pi / 2  # 90 degrees
    theta1 = 1.5614751446622785
    #theta2 = -np.pi / 2  # -90 degrees
    theta2 = -1.5556073614887143

    data = compute_time_interval_landscape(
        exp_params,
        theta1=theta1,
        theta2=theta2,
        resolution=7,
        min_interval=40,
        max_interval=150,
        mode='discrete',
        batch_size=30, 
        verbose=True
    )
    
    fig = plot_time_interval_landscape(
        data,
        exp_params,
        save_path=str(output_dir / 'time_interval_landscape_theta_optimized.png'),
        show_measurement_count=True
    )


if __name__ == "__main__":
    main()
