from qsopt import * 
import numpy as np
import jax.numpy as jnp
import optax

gm = 0.03 * 2 * np.pi

# Define custom physical constants
custom_constants = PhysicalConstants(
    chi = 0.5 * gm,                    # Dispersive coupling
    photon_cavity_coupling = gm,  # Photon-cavity coupling
    inverse_pulse_width = 0.1 * gm      # Inverse pulse width
)

# Define custom system dimensions
custom_dims = SystemDimensions(
    cavity_levels=2,
    qubit_levels=2,
    field_levels=2
)

# Define measurement protocol
custom_measurement = MeasurementProtocol(
    measurement_times = list(np.array([-5.0, -1., 0.0, 1., 5.0])/(0.1 * gm)) # Specific measurement times
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
    noise_config=noise_config
)

parameters = TrainableParameters()
parameters.add_rotation_angles(['ry1', 'ry2'], [np.pi/2, -np.pi/2], optimizer=optax.sgd(0.5))

experiment = SingleQubitExperiment(exp_parameters, parameters)

benchmark_results = experiment.run_simulation()

history = experiment.optimize_rotations(
    theta_init=[1.5, -1.5],
    num_steps=70,
    verbose=True,
    verbose_step=20,
    tolerance=1e-7
)

fig = plot_optimization_dashboard(
    optimization_callback=history,
    reference_callback=benchmark_results,
    save_path='experiments/single_qubit/rotation_opt/results/opt_dashboard.pdf'
)

experiment.save_experiment_report(save_path='experiments/single_qubit/rotation_opt/results/experiment_report.json')